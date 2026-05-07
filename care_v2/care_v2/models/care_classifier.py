"""
CARE_v2 Hierarchical Classifier.

Architecture:
  Qwen Backbone (with PEFT/LoRA)
      |
  AttentionPooling
      |
  LayerNorm
      |
  6 Classification Heads → 5-class softmax each
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from care_v2.configs.config import CAREv2Config, NUM_CLASSES

log = logging.getLogger(__name__)


# =====================================================================
# Attention Pooling
# =====================================================================

class AttentionPooling(nn.Module):
    """Weighted sum of token hidden states using a learned attention scorer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        w = self.attn(hidden_states)                    # [B, T, 1]
        w[attention_mask == 0] = float("-inf")
        weights = torch.softmax(w, dim=1)               # [B, T, 1]
        pooled = (hidden_states * weights).sum(dim=1)   # [B, H]
        return pooled


# =====================================================================
# Classification Head (per dimension)
# =====================================================================

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = NUM_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# CARE_v2 Full Model
# =====================================================================

class CAREv2Classifier(nn.Module):
    """
    Full CARE_v2 model.

    LoRA is applied on top of the Qwen backbone.
    Classification heads are separate float32 modules.
    Supports staged training (freeze/unfreeze LoRA) via callbacks.
    """

    NUM_DIMENSIONS = 6

    def __init__(self, config: CAREv2Config, class_weights: Optional[torch.Tensor] = None, device: Optional[str] = None):
        super().__init__()

        self.config_care = config

        # 1. Backbone
        hf_config = AutoConfig.from_pretrained(config.classifier_model_id, trust_remote_code=True)
        self.hidden_size = hf_config.hidden_size

        # Load base Qwen model
        log.info(f"Loading backbone: {config.classifier_model_id}")
        
        # Determine device map
        import os
        is_distributed = "RANK" in os.environ or "WORLD_SIZE" in os.environ
        
        if device:
            # Use the explicitly provided device
            device_map = {"": device}
            log.info(f"Using explicit device_map: {device_map}")
        elif torch.cuda.is_available() and not is_distributed:
            device_map = "auto"
            log.info("Single-GPU detected: using device_map='auto'")
        else:
            device_map = None
            log.info(f"Distributed mode detected or no GPU: disabling device_map")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.classifier_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map=device_map,
            tie_word_embeddings=False,
        )

        # 2. Wrap with LoRA
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.backbone = get_peft_model(base_model, peft_cfg)
        self.backbone.print_trainable_parameters()

        # 3. Pooling + Norm (float32 for numerical stability)
        self.pooler = AttentionPooling(self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

        # 4. Classification heads
        self.heads = nn.ModuleList([
            ClassificationHead(self.hidden_size, NUM_CLASSES)
            for _ in range(self.NUM_DIMENSIONS)
        ])

        # Move float32 modules to same device as backbone
        _device = next(self.backbone.parameters()).device
        self.pooler.to(device=_device, dtype=torch.float32)
        self.norm.to(device=_device, dtype=torch.float32)
        self.heads.to(device=_device, dtype=torch.float32)

        # 5. Loss weights
        self.class_weights = class_weights  # [6, 5] or None

    # ------------------------------------------------------------------
    # Staged Training Helpers
    # ------------------------------------------------------------------

    def freeze_lora(self):
        """Freezes all LoRA adapter parameters (Phase 2)."""
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad_(False)
        log.info("LoRA adapters FROZEN – entering Phase 2 (heads only).")

    def unfreeze_lora(self):
        """Re-enables LoRA adapter parameters (Phase 1)."""
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
        log.info("LoRA adapters UNFROZEN – Phase 1 active.")

    # ------------------------------------------------------------------
    # Gradient Checkpointing Support
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate gradient checkpointing disable to the backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

     # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Backbone forward (causal LM, we use hidden states)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1].to(dtype=torch.float32)

        # Pool + Normalise
        pooled = self.pooler(last_hidden, attention_mask)
        pooled = self.norm(pooled)

        # Classification heads → [B, 6, 5]
        logits_list = [head(pooled) for head in self.heads]
        logits = torch.stack(logits_list, dim=1)

        # Loss
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        return {"loss": loss, "logits": logits}

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Hybrid Loss: alpha * MSE(expected_val, true_val) + beta * CE(probs, true_idx)
        This jointly models ordinal distance and categorical accuracy.
        """
        total_loss = 0.0
        alpha = self.config_care.loss_alpha
        beta = self.config_care.loss_beta

        # Mapping indices [0, 1, 2, 3, 4] to therapeutic values [-2, -1, 0, 1, 2]
        target_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=logits.device, dtype=torch.float32)

        for i in range(self.NUM_DIMENSIONS):
            # 1. Categorical Accuracy (Weighted Cross-Entropy)
            w = self.class_weights[i] if self.class_weights is not None else None
            ce_loss = nn.CrossEntropyLoss(weight=w)(logits[:, i, :], labels[:, i])

            # 2. Ordinal Distance (MSE on Expected Value)
            # Map index labels to real values
            y_true = target_values[labels[:, i]]
            
            # Softmax to get probabilities, then compute expected value: sum(p * value)
            probs = torch.softmax(logits[:, i, :], dim=-1)
            y_pred = (probs * target_values).sum(dim=-1)
            
            mse_loss = nn.MSELoss()(y_pred, y_true)

            # Combine per head
            total_loss += (alpha * mse_loss) + (beta * ce_loss)

        return total_loss / self.NUM_DIMENSIONS

    # ------------------------------------------------------------------
    # Checkpoint helpers (heads only)
    # ------------------------------------------------------------------

    def save_heads(self, path: str):
        """Saves only the classification heads (compatible with LoRA save)."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "pooler": self.pooler.state_dict(),
                "norm": self.norm.state_dict(),
                "heads": self.heads.state_dict(),
            },
            f"{path}/heads.pt",
        )
        log.info(f"Saved classification heads to {path}/heads.pt")

    def load_heads(self, path: str):
        """Loads pooler, norm, and heads from a saved checkpoint."""
        ckpt = torch.load(f"{path}/heads.pt", map_location="cpu")
        self.pooler.load_state_dict(ckpt["pooler"])
        self.norm.load_state_dict(ckpt["norm"])
        self.heads.load_state_dict(ckpt["heads"])
        log.info(f"Loaded classification heads from {path}/heads.pt")
