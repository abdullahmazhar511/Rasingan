# Force IPv4 for HuggingFace Hub and other network connections
import socket
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import os
import json
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARE_DIR = os.environ.get("CARE_DIR", os.path.join(BASE_DIR, "CARE"))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

CONFIG = {
    # Two different models
    "explainer_model_id": "Qwen/Qwen3-4B-Instruct-2507", 
    "classifier_model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "classifier_weights": os.path.join(CARE_DIR, "care_checkpoint", "best_classifier.pt"),
    "embedding_model": "all-MiniLM-L6-v2",
    "train_explanations_json": os.path.join(CARE_DIR, "rag_cache", "train_processed.json"),
    "rag_index_cache": os.path.join(CARE_DIR, "rag_cache", "rag_index.pt"),
    "top_k": 3,
    "max_len": 1536,
}

LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
NUM_CLASSES = 5
# Used by return_expected=True path: E[label] = sum(p_c * c).
LABEL_VALUES = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
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
            device_map={"": device},
            torch_dtype=torch_dtype, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
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
        self.pooler.to(device=device, dtype=torch.float32)
        self.norm.to(device=device, dtype=torch.float32)
        self.main_heads.to(device=device, dtype=torch.float32)
        self.binary_heads.to(device=device, dtype=torch.float32)
        
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
        pooled = self.pooler(last_hidden.to(dtype=torch.float32), attention_mask)
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
            state_dict = torch.load(CONFIG["classifier_weights"], map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device=device)
            self.model.backbone.to(device=device, dtype=torch_dtype)
            self.model.pooler.to(device=device, dtype=torch.float32)
            self.model.norm.to(device=device, dtype=torch.float32)
            self.model.main_heads.to(device=device, dtype=torch.float32)
            self.model.binary_heads.to(device=device, dtype=torch.float32)
            self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        self.max_length = CONFIG["max_len"]
        self.analysis_labels = CARE_LABELS
        self.dimension_samples = {}

        embedding_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(CONFIG["embedding_model"], device=embedding_device)
        self.ideal_sets, self.train_embeddings, self.trait_explanations = self._load_or_build_rag_index()

    def _load_or_build_rag_index(self):
        cache_path = CONFIG["rag_index_cache"]
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False, map_location="cpu")
            embeddings = cache["embeddings"].to(device)
            return cache["ideal_sets"], embeddings, cache["trait_explanations"]

        train_json = CONFIG["train_explanations_json"]
        if not os.path.exists(train_json):
            raise FileNotFoundError(
                f"Missing training explanations JSON at {train_json}. "
                "Run CARE scoring setup to generate rag_cache assets first."
            )

        with open(train_json, "r") as f:
            train_data = json.load(f)

        utterances = [str(item.get("Utterance", "")) for item in train_data]
        embeddings = self.embedding_model.encode(utterances, convert_to_tensor=True, show_progress_bar=True)

        ideal_sets = {label: {"Pos": [], "Neg": []} for label in CARE_LABELS}
        used_indices = set()
        candidates = {}

        for label in CARE_LABELS:
            label_values = []
            for item in train_data:
                try:
                    label_values.append(int(item.get(label, 0)))
                except (TypeError, ValueError):
                    label_values.append(0)
            values_np = np.array(label_values)
            pos_idxs = np.where(values_np > 0)[0].tolist()
            neg_idxs = np.where(values_np < 0)[0].tolist()
            pos_idxs.sort(key=lambda i: values_np[i], reverse=True)
            neg_idxs.sort(key=lambda i: abs(values_np[i]), reverse=True)
            candidates[label] = {"Pos": pos_idxs, "Neg": neg_idxs}

        keep_going = True
        while keep_going:
            added_this_round = False
            for label in CARE_LABELS:
                for polarity in ("Pos", "Neg"):
                    while candidates[label][polarity]:
                        idx = candidates[label][polarity].pop(0)
                        if idx not in used_indices:
                            ideal_sets[label][polarity].append(idx)
                            used_indices.add(idx)
                            added_this_round = True
                            break
            if not added_this_round:
                keep_going = False

        trait_explanations = {}
        for label in CARE_LABELS:
            trait_explanations[label] = [
                str((item.get("Explanations") or {}).get(label, "")).strip()
                for item in train_data
            ]

        torch.save(
            {
                "ideal_sets": ideal_sets,
                "embeddings": embeddings.cpu(),
                "trait_explanations": trait_explanations,
            },
            cache_path,
        )

        return ideal_sets, embeddings.to(device), trait_explanations

    def _retrieve_trait_texts(self, query_embedding, trait_label, polarity):
        pool_idxs = self.ideal_sets.get(trait_label, {}).get(polarity, [])
        if not pool_idxs:
            return []

        pool_embeddings = self.train_embeddings[pool_idxs]
        scores = util.cos_sim(query_embedding, pool_embeddings)[0]
        k = min(CONFIG["top_k"], len(pool_idxs))
        _, topk_inds = torch.topk(scores, k=k)

        texts = []
        for j in topk_inds.detach().cpu().numpy().tolist():
            idx = pool_idxs[int(j)]
            expl = self.trait_explanations[trait_label][idx]
            if expl:
                texts.append(expl)
        return texts

    def get_analysis(self, context, utterance):
        explanations = self.get_explanations(utterance)
        analysis = ""
        for lbl in CARE_LABELS:
            expl_text = str(explanations.get(lbl, "")).strip() or "No info."
            analysis += f"{lbl}: {expl_text}\n"
        return analysis
    
    def _get_analysis_batch(self, contexts, utterances):
        text_embeddings = self.embedding_model.encode(utterances, convert_to_tensor=True)

        all_analyses = []
        for text_embedding in text_embeddings:
            analysis = ""
            for label in CARE_LABELS:
                pos_texts = self._retrieve_trait_texts(text_embedding, label, "Pos")
                neg_texts = self._retrieve_trait_texts(text_embedding, label, "Neg")
                expl_text = "\n".join(pos_texts + neg_texts).strip() or "No info."
                analysis += f"{label}: {expl_text}\n"
            all_analyses.append(analysis)

        return all_analyses
    
    def get_explanations(self, text):
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)

        results = {}
        for label in CARE_LABELS:
            pos_texts = self._retrieve_trait_texts(text_embedding, label, "Pos")
            neg_texts = self._retrieve_trait_texts(text_embedding, label, "Neg")
            results[label] = "\n".join(pos_texts + neg_texts).strip() or "No info."

        return results

    def predict(self, context, utterance, include_analysis=True, return_expected=False):
        """
        Single sample inference with optional analysis.

        Args:
            return_expected: If False (default), each CARE dim is an integer in
                {-2,-1,0,1,2} via argmax — matches dataset annotations, use
                for eval. If True, each CARE dim is the expected value
                E[label] = sum_c p_c * c over the softmaxed class
                distribution — continuous in [-2, 2], use for RL reward.
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
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = self.model(
                input_ids=tokenized_input.input_ids,
                attention_mask=tokenized_input.attention_mask
            )
            logits = output["logits"]               # (1, 6, 5)
            preds_idx = torch.argmax(logits, dim=2).cpu().numpy()
            preds_argmax = np.vectorize(IDX_TO_LABEL.get)(preds_idx)
            if return_expected:
                probs = torch.softmax(logits, dim=-1)
                vals = LABEL_VALUES.to(probs.device)
                preds_expected = (probs * vals).sum(dim=-1).cpu().numpy()

        # Primary result = expected when requested, else argmax.
        result = {}
        for i, label in enumerate(CARE_LABELS):
            if return_expected:
                result[label] = float(preds_expected[0][i])
            else:
                result[label] = float(preds_argmax[0][i])

        if return_expected:
            # Also return argmax for downstream logging (free from same forward pass).
            argmax_dict = {label: float(preds_argmax[0][i]) for i, label in enumerate(CARE_LABELS)}
            return {"expected": result, "argmax": argmax_dict}
        return result
    
    def batch_predict(self, contexts, utterances, batch_size=8, include_analysis=True, return_expected=False):
        """
        Batch inference for fast processing of multiple samples.

        Args:
            contexts: List of context strings
            utterances: List of utterance strings
            batch_size: Number of samples to process at once
            include_analysis: Whether to include analysis (slower but more accurate)
            return_expected: See predict() — argmax integer vs E[label] float.
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
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Inference with mixed precision
            with torch.no_grad():
                output = self.model(
                    input_ids=tokenized_batch.input_ids,
                    attention_mask=tokenized_batch.attention_mask
                )
                logits = output["logits"]               # (B, 6, 5)
                preds_batch = torch.argmax(logits, dim=2).cpu().numpy()
                preds_argmax = np.vectorize(IDX_TO_LABEL.get)(preds_batch)
                if return_expected:
                    probs = torch.softmax(logits, dim=-1)
                    vals = LABEL_VALUES.to(probs.device)
                    preds_expected = (probs * vals).sum(dim=-1).cpu().numpy()

            # Build results for this batch
            for i in range(len(batch_inputs)):
                if return_expected:
                    expected = {label: float(preds_expected[i][j]) for j, label in enumerate(CARE_LABELS)}
                    argmax   = {label: float(preds_argmax[i][j])   for j, label in enumerate(CARE_LABELS)}
                    all_results.append({"expected": expected, "argmax": argmax})
                else:
                    result = {label: float(preds_argmax[i][j]) for j, label in enumerate(CARE_LABELS)}
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
