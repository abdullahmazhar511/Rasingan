import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
import numpy as np


LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
NUM_CLASSES = 5


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
            device_map="auto",
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
        self.tokenizer=None
        self.max_length=None
        self.model=None
        self.analysis_labels = None

    def get_analysis(self, context, utterance):
        pass

    def predict(self, context, utterance):
        analysis = self.get_analysis(context, utterance)
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
            all_preds_idx = []

            loss,logits,binary_logits = self.model(
                input_ids=tokenized_input.input_ids,
                attention_mask=tokenized_input.attention_mask
            )
            preds = torch.argmax(logits, dim=2)
            all_preds_idx.extend(preds.cpu().numpy())

            all_preds_idx = np.array(all_preds_idx)
            all_preds_real = np.vectorize(IDX_TO_LABEL.get)(all_preds_idx)
        