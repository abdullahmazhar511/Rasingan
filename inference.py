# Force IPv4 for HuggingFace Hub and other network connections
import socket
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import os
import sys


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda:0"
dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

CONFIG = {
    # Two different models
    "explainer_model_id": "Qwen/Qwen3-4B-Instruct-2507", 
    "classifier_model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "classifier_weights": "/home/umairai/faith/faith/classification_output_stable_v5",
    "embedding_model": "all-MiniLM-L6-v2",
}

LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
NUM_CLASSES = 5
CARE_LABELS=[
    "Non-Judgmental Language", "Warmth and Encouragement", 
    "Respect for Autonomy", "Active Listening", 
    "Reflecting Feelings", "Situational Appropriateness"
]

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state)
        w[attention_mask == 0] = float('-inf')
        weights = torch.softmax(w, dim=1)
        pooled = torch.sum(last_hidden_state * weights, dim=1)
        return pooled

class QwenHierarchicalClassifier(nn.Module):
    def __init__(self, model_id, class_weights=None, binary_weights=None):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cuda:0"},  # Load on single GPU instead of auto
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=16, lora_alpha=32, lora_dropout=0.1, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.backbone = get_peft_model(base_model, peft_config)
        self.pooler = AttentionPooling(self.config.hidden_size)
        self.norm = nn.LayerNorm(self.config.hidden_size)
        
        self.main_heads = nn.ModuleList()
        self.binary_heads = nn.ModuleList()
        
        for _ in range(6):
            self.main_heads.append(nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_size, 64),
                nn.ReLU(),
                nn.LayerNorm(64), 
                nn.Linear(64, NUM_CLASSES)
            ))
            self.binary_heads.append(nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_size, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, 1) 
            ))
            
        self._init_head_weights()
        # Move all components to the appropriate device and dtype
        self.backbone.to(device=device, dtype=torch_dtype)
        self.pooler.to(device=device, dtype=torch_dtype)
        self.norm.to(device=device, dtype=torch_dtype)
        self.main_heads.to(device=device, dtype=torch_dtype)
        self.binary_heads.to(device=device, dtype=torch_dtype)
        
        self.class_weights = class_weights
        self.binary_weights = binary_weights

    def _init_head_weights(self):
        for head in self.main_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        for head in self.binary_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = self.pooler(last_hidden, attention_mask)
        pooled = self.norm(pooled)
        
        main_logits_list = []
        binary_logits_list = []
        for i in range(6):
            main_logits_list.append(self.main_heads[i](pooled))
            binary_logits_list.append(self.binary_heads[i](pooled))
            
        main_logits = torch.stack(main_logits_list, dim=1)
        binary_logits = torch.stack(binary_logits_list, dim=1).squeeze(-1)
        
        loss = None
        if labels is not None:
            total_loss = 0
            binary_targets = (labels < 2).float()
            for i in range(6):
                w_c = self.class_weights[i] if self.class_weights is not None else None
                loss_main = nn.CrossEntropyLoss(weight=w_c)(main_logits[:, i, :], labels[:, i])
                w_b = self.binary_weights[i] if self.binary_weights is not None else None
                loss_bin = nn.BCEWithLogitsLoss(pos_weight=w_b)(binary_logits[:, i], binary_targets[:, i])
                total_loss += loss_main + loss_bin
            loss = total_loss / 6.0
            
        return {"loss": loss, "logits": main_logits, "binary_logits": binary_logits}


class CareModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["classifier_model_id"], trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        with torch.no_grad():
            self.model = QwenHierarchicalClassifier(CONFIG["classifier_model_id"])
            print("Loading model weights...")
            state_dict = torch.load(
                os.path.join(CONFIG["classifier_weights"], "best_classifier.pt"),
                map_location=device
            )
            self.model.load_state_dict(state_dict)
            self.model.to(device, dtype=torch.float16)
            self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        self.max_length = 2048 # Set reasonable max length to avoid OOM
        self.analysis_labels = None
        
        #load explanations - use absolute path
        explanations_path = "/home/umairai/faithfulness_emnlp/Rasingan/explanations/explanations_16.csv"
        self.explanations=pd.read_csv(explanations_path)
        # Load embedding model on GPU for faster inference
        embedding_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=embedding_device)
        
        self.init_explanations()

    def init_explanations(self):
        # Organize samples by dimension with positive (Polarity='positive') and negative (Polarity='negative') samples
        self.dimension_samples = {}
        for dimension in self.explanations['Dimension'].unique():
            dim_data = self.explanations[self.explanations['Dimension'] == dimension]
            positive_samples = dim_data[dim_data['Polarity'] == 'positive']
            negative_samples = dim_data[dim_data['Polarity'] == 'negative']
            
            # Store one positive and one negative sample with their embeddings
            pos_text = positive_samples['Therapist Response'].iloc[0] if len(positive_samples) > 0 else ""
            neg_text = negative_samples['Therapist Response'].iloc[0] if len(negative_samples) > 0 else ""
            
            # Encode embeddings on GPU for faster computation
            pos_embedding = self.embedding_model.encode(pos_text, convert_to_tensor=True) if pos_text else None
            neg_embedding = self.embedding_model.encode(neg_text, convert_to_tensor=True) if neg_text else None
            
            self.dimension_samples[dimension] = {
                'positive': {
                    'text': pos_text,
                    'embedding': pos_embedding,
                    'row': positive_samples.iloc[0] if len(positive_samples) > 0 else None
                },
                'negative': {
                    'text': neg_text,
                    'embedding': neg_embedding,
                    'row': negative_samples.iloc[0] if len(negative_samples) > 0 else None
                }
            }

    def get_analysis(self, context, utterance):

        explanations = self.get_explanations(utterance)
        analysis = ""
        for lbl in CARE_LABELS:
            # row = str(explanations.get(lbl, "")).strip() or "No info."
            analysis += f"{lbl}:\n"
            for polarity in ['positive', 'negative']:
                expl_text = explanations.get(lbl, {}).get(polarity, {}).get('explanation', "No info.")
                expl_text = str(expl_text).strip() or "No info."
                expl_text = f"{polarity.capitalize()}: {expl_text}"
                analysis += f"{lbl}: {expl_text}\n"
            analysis += "\n"
        return analysis
    
    def _get_analysis_batch(self, contexts, utterances):
        """
        Efficiently generate analysis for multiple utterances in parallel.
        Returns list of analysis strings.
        """
        # Batch encode utterances for similarity computation
        text_embeddings = self.embedding_model.encode(utterances, convert_to_tensor=True)
        
        all_explanations = []
        for text_embedding in text_embeddings:
            results = {}
            for dimension, samples in self.dimension_samples.items():
                results[dimension] = {}
                
                # Calculate similarity with positive sample
                if samples['positive']['embedding'] is not None:
                    pos_similarity = cos_sim(text_embedding, samples['positive']['embedding']).item()
                    results[dimension]['positive'] = {
                        'similarity': pos_similarity,
                        'explanation': samples['positive']['row']['Explanation']
                    }
                else:
                    results[dimension]['positive'] = {'similarity': None, 'row': None}
                
                # Calculate similarity with negative sample
                if samples['negative']['embedding'] is not None:
                    neg_similarity = cos_sim(text_embedding, samples['negative']['embedding']).item()
                    results[dimension]['negative'] = {
                        'similarity': neg_similarity,
                        'explanation': samples['negative']['row']['Explanation']
                    }
                else:
                    results[dimension]['negative'] = {'similarity': None, 'row': None}
            all_explanations.append(results)
        
        # Generate analysis strings for each sample
        analyses = []
        for explanations in all_explanations:
            analysis = ""
            for lbl in CARE_LABELS:
                analysis += f"{lbl}:\n"
                for polarity in ['positive', 'negative']:
                    expl_text = explanations.get(lbl, {}).get(polarity, {}).get('explanation', "No info.")
                    expl_text = str(expl_text).strip() or "No info."
                    expl_text = f"{polarity.capitalize()}: {expl_text}"
                    analysis += f"{lbl}: {expl_text}\n"
                analysis += "\n"
            analyses.append(analysis)
        
        return analyses
    
    def get_explanations(self, text):
        """
        Match input text with positive and negative samples for each dimension.
        Returns similarity scores for each dimension with one positive and one negative sample.
        
        Args:
            text: Input text to match against samples
            
        Returns:
            dict: Dictionary with dimension -> {'positive': {'similarity': float, 'row': pd.Series}, 
                                              'negative': {'similarity': float, 'row': pd.Series}}
        """
        # Encode input text on GPU
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        
        results = {}
        for dimension, samples in self.dimension_samples.items():
            results[dimension] = {}
            
            # Calculate similarity with positive sample
            if samples['positive']['embedding'] is not None:
                pos_similarity = cos_sim(text_embedding, samples['positive']['embedding']).item()
                results[dimension]['positive'] = {
                    'similarity': pos_similarity,
                    'explanation': samples['positive']['row']['Explanation']
                }
            else:
                results[dimension]['positive'] = {'similarity': None, 'row': None}
            
            # Calculate similarity with negative sample
            if samples['negative']['embedding'] is not None:
                neg_similarity = cos_sim(text_embedding, samples['negative']['embedding']).item()
                results[dimension]['negative'] = {
                    'similarity': neg_similarity,
                    'explanation': samples['negative']['row']['Explanation']
                }
            else:
                results[dimension]['negative'] = {'similarity': None, 'row': None}
        # print(results)
        return results

    def predict(self, context, utterance, include_analysis=True):
        """
        Single sample inference with optional analysis.
        """
        if include_analysis:
            analysis = self.get_analysis(context, utterance)
        else:
            analysis = ""
            
        text_input = (
            f"Context:\n{context}\n"
            f"Therapist: \"{utterance}\"\n"
            f"Analysis:\n{analysis}\n"
            "Classify the clinical traits."
        )
        
        tokenized_input = self.tokenizer(
            text_input,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                output = self.model(
                    input_ids=tokenized_input.input_ids,
                    attention_mask=tokenized_input.attention_mask
                )
            logits = output["logits"]
            preds = torch.argmax(logits, dim=2)
            preds_idx = preds.cpu().numpy()
        
        # Convert predictions to labels
        preds_real = np.vectorize(IDX_TO_LABEL.get)(preds_idx)
        
        # Return predictions with labels
        result = {}
        for i, label in enumerate(CARE_LABELS):
            result[label] = preds_real[0][i].item()
        
        return result
    
    def batch_predict(self, contexts, utterances, batch_size=8, include_analysis=True):
        """
        Batch inference for fast processing of multiple samples.
        
        Args:
            contexts: List of context strings
            utterances: List of utterance strings
            batch_size: Number of samples to process at once
            include_analysis: Whether to include analysis (slower but more accurate)
        
        Returns:
            List of prediction dictionaries
        """
        assert len(contexts) == len(utterances), "contexts and utterances must have same length"
        
        all_results = []
        num_samples = len(utterances)
        
        # Process analyses in batch if needed
        if include_analysis:
            all_analyses = self._get_analysis_batch(contexts, utterances)
        else:
            all_analyses = [""] * num_samples
        
        # Process predictions in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_contexts = contexts[batch_start:batch_end]
            batch_utterances = utterances[batch_start:batch_end]
            batch_analyses = all_analyses[batch_start:batch_end]
            
            # Build batch inputs
            batch_inputs = []
            for context, utterance, analysis in zip(batch_contexts, batch_utterances, batch_analyses):
                text_input = (
                    f"Context:\n{context}\n"
                    f"Therapist: \"{utterance}\"\n"
                    f"Analysis:\n{analysis}\n"
                    "Classify the clinical traits."
                )
                batch_inputs.append(text_input)
            
            # Tokenize batch
            tokenized_batch = self.tokenizer(
                batch_inputs,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Inference with mixed precision
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    output = self.model(
                        input_ids=tokenized_batch.input_ids,
                        attention_mask=tokenized_batch.attention_mask
                    )
                logits = output["logits"]
                preds = torch.argmax(logits, dim=2)
                preds_batch = preds.cpu().numpy()
            
            # Convert predictions to labels
            preds_real = np.vectorize(IDX_TO_LABEL.get)(preds_batch)
            
            # Build results for this batch
            for i in range(len(batch_inputs)):
                result = {}
                for j, label in enumerate(CARE_LABELS):
                    result[label] = preds_real[i][j].item()
                all_results.append(result)
            
            # Clear cache to avoid OOM
            torch.cuda.empty_cache()
        
        return all_results
        
# Dataset = MHCoPilot_Dataset("/home/umairai/faith_data/")
# Dataset.get_data()
# train_dataset = Dataset.train_dataset

# model=CareModel()


# print(train_dataset)
# #print only utterance and type
# # print(train_dataset[2])
# model.predict(train_dataset[2]['context'], train_dataset[2]['Utterance'])
