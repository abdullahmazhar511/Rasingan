"""
Training callbacks for CARE_v2 two-stage training.

Stage Transitions:
  Phase 1 (epochs 1..N):     LoRA + heads train together
  Phase 2 (epochs N+1..end): LoRA frozen, heads-only training with higher LR
"""

import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from care_v2.configs.config import CAREv2Config

log = logging.getLogger(__name__)


class TwoStageTrainingCallback(TrainerCallback):
    """
    Automatically transitions between Phase 1 and Phase 2.

    Phase 1: Both LoRA adapters and classification heads are trained.
    Phase 2: LoRA is frozen; only the classification heads learn.
    """

    def __init__(self, model, config: CAREv2Config):
        self.model = model
        self.phase1_epochs = config.phase1_epochs
        self.phase2_lr = config.lr_phase2
        self._in_phase2 = False

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        current_epoch = int(state.epoch) if state.epoch else 0

        if current_epoch == self.phase1_epochs and not self._in_phase2:
            log.info(
                f"\n{'='*60}\n"
                f"Epoch {current_epoch+1}: Transitioning to PHASE 2\n"
                f"  Freezing LoRA adapters.\n"
                f"  Increasing LR to {self.phase2_lr:.2e}\n"
                f"{'='*60}"
            )
            self.model.freeze_lora()
            self._in_phase2 = True

            # Adjust optimizer LR for remaining training
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = self.phase2_lr

        return control


class BestModelLoggingCallback(TrainerCallback):
    """Logs QWK improvements and triggers extra head saves."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_qwk = -1.0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        if metrics is None:
            return control

        current_qwk = metrics.get("eval_avg_qwk", -1.0)
        if current_qwk > self.best_qwk:
            self.best_qwk = current_qwk
            log.info(
                f"[QWK Improved] New best QWK: {current_qwk:.4f} "
                f"(epoch {state.epoch:.1f})"
            )

        return control
