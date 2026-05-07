"""
Custom Trainer for CARE_v2.

Overrides compute_loss() to handle the model's dict output format
and optional class weighting. Compatible with HuggingFace Trainer
including checkpointing, evaluation, and gradient accumulation.
"""

import torch
from transformers import Trainer


class CAREv2Trainer(Trainer):
    """
    Extends Trainer to handle CAREv2Classifier's dict output.

    The model returns {"loss": ..., "logits": ...} and the Trainer
    expects this to be forwarded correctly.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to extract logits from dict output correctly."""
        with torch.no_grad():
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs["input_ids"].to(model.device if hasattr(model, 'device') else next(model.parameters()).device),
                attention_mask=inputs["attention_mask"].to(next(model.parameters()).device),
                labels=labels.to(next(model.parameters()).device) if labels is not None else None,
            )

        loss = outputs["loss"]
        logits = outputs["logits"]

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)
