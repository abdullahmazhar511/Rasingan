from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ========================================================
# LABEL MAPPING
# ========================================================

LABELS = [
    "Non-Judgmental Language",
    "Warmth and Encouragement",
    "Respect for Autonomy",
    "Active Listening",
    "Reflecting Feelings",
    "Situational Appropriateness",
]

LABEL_TO_IDX = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = 5


# ========================================================
# MAIN CONFIG
# ========================================================

@dataclass
class CAREv2Config:

    # Model
    classifier_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    embedding_model_id: str = "BAAI/bge-large-en-v1.5"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Two-Stage Training
    phase1_epochs: int = 3   # LoRA + heads train together
    phase2_epochs: int = 4   # Heads only (frozen LoRA)

    # Hyperparameters
    batch_size: int = 32
    grad_accum: int = 2
    dataloader_num_workers: int = 2
    gradient_checkpointing: bool = False
    remove_unused_columns: bool = True
    lr_phase1: float = 5e-5
    lr_phase2: float = 1e-4
    max_len: int = 1536
    loss_alpha: float = 0.5    # MSE weight
    loss_beta: float = 0.5     # CE weight
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    bf16: bool = True

    # Data
    train_csv_folder: str = "faith_data/RESPAIR_augmented/train"
    val_csv_folder: str = "faith_data/RESPAIR_augmented/val"
    test_csv_folder: str = "faith_data/RESPAIR_augmented/test"
    context_window: int = 4

    # Prototype Index
    prototype_dir: str = "faith_data/05_generated_explanations"
    prototype_k: str = "k_12"  # Exactly one k-value per experiment (e.g., k_4, k_12, k_16)

    # Output
    output_dir: str = "care_v2_outputs_k12"
    resume_checkpoint: Optional[str] = None

    # Labels (imported from global)
    labels: List[str] = field(default_factory=lambda: LABELS)

    # --------------------------------------------------------
    # Ablation switches (for paper experiments)
    # --------------------------------------------------------
    use_context: bool = True           # Include conversation history
    use_explanations: bool = True      # Include prototype explanations
    use_positive_refs: bool = True     # Include positive prototype references
    use_negative_refs: bool = True     # Include negative prototype references

    # --------------------------------------------------------
    # Retrieval settings
    # --------------------------------------------------------
    debug_retrieval: bool = False      # Print retrieved prototypes for validation
    embed_batch_size: int = 64         # Batch size for utterance embedding cache


# Global default instance
DEFAULT_CONFIG = CAREv2Config()
