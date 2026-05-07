"""
CARE_v2 main training entry-point.

Run:
    python -m care_v2.train.py
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments

from care_v2.configs.config import CAREv2Config
from care_v2.data.preprocessing import load_csv_data
from care_v2.datasets.care_dataset import CAREDataset
from care_v2.models.care_classifier import CAREv2Classifier
from care_v2.retrieval.prototype_retriever import PrototypeRetriever
from care_v2.training.callbacks import TwoStageTrainingCallback, BestModelLoggingCallback
from care_v2.training.metrics import compute_metrics
from care_v2.training.trainer import CAREv2Trainer
from care_v2.utils.io_utils import setup_logging, compute_class_weights

log = logging.getLogger(__name__)


def main():
    config = CAREv2Config()
    setup_logging(config.output_dir)

    log.info("=" * 60)
    log.info("CARE_v2 Training Pipeline")
    log.info("=" * 60)
    log.info(f"Phase 1 epochs: {config.phase1_epochs}")
    log.info(f"Phase 2 epochs: {config.phase2_epochs}")
    log.info(f"Total epochs  : {config.phase1_epochs + config.phase2_epochs}")

    # ----------------------------------------------------------------
    # 1. Tokenizer
    # ----------------------------------------------------------------
    log.info(f"Loading tokenizer: {config.classifier_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.classifier_model_id, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ----------------------------------------------------------------
    # 2. Prototype Retriever
    # ----------------------------------------------------------------
    log.info("Initialising PrototypeRetriever (loading embeddings)...")
    retriever = PrototypeRetriever(config)

    # ----------------------------------------------------------------
    # 3. Datasets
    # ----------------------------------------------------------------
    log.info("Loading training data...")
    train_df = load_csv_data(config.train_csv_folder, config)
    val_df = load_csv_data(config.val_csv_folder, config)
    log.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_ds = CAREDataset(train_df, retriever, tokenizer, config)
    val_ds = CAREDataset(val_df, retriever, tokenizer, config)

    # ----------------------------------------------------------------
    # 4. Class Weights & Model
    # ----------------------------------------------------------------
    log.info("Computing class weights...")
    class_weights = compute_class_weights(train_ds, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = class_weights.to(device)

    log.info(f"Initialising CAREv2Classifier ({config.classifier_model_id})...")
    model = CAREv2Classifier(config, class_weights=class_weights)

    if config.resume_checkpoint:
        log.info(f"Resuming heads from: {config.resume_checkpoint}")
        model.load_heads(config.resume_checkpoint)

    # ----------------------------------------------------------------
    # 5. Training Arguments
    # ----------------------------------------------------------------
    total_epochs = config.phase1_epochs + config.phase2_epochs
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=total_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr_phase1,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16 and torch.cuda.is_bf16_supported(),
        fp16=not (config.bf16 and torch.cuda.is_bf16_supported()),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_avg_qwk",
        greater_is_better=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        remove_unused_columns=True,
    )

    # ----------------------------------------------------------------
    # 6. Callbacks
    # ----------------------------------------------------------------
    callbacks = [
        TwoStageTrainingCallback(model, config),
        BestModelLoggingCallback(config.output_dir),
    ]

    # ----------------------------------------------------------------
    # 7. Trainer
    # ----------------------------------------------------------------
    trainer = CAREv2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ----------------------------------------------------------------
    # 8. Train
    # ----------------------------------------------------------------
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.resume_checkpoint)

    # ----------------------------------------------------------------
    # 9. Final Evaluation & Confusion Matrix (Best Model)
    # ----------------------------------------------------------------
    log.info("Training complete. Loading best model for final evaluation...")
    
    # The Trainer already loads the best model at the end because of load_best_model_at_end=True
    predictions = trainer.predict(val_ds)
    logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    labels = predictions.label_ids
    
    # preds: [N, 6]
    preds = np.argmax(logits, axis=-1)
    
    log.info(f"Generating confusion matrices in: {config.output_dir}/plots")
    from care_v2.utils.io_utils import save_confusion_matrices
    save_confusion_matrices(preds, labels, config.output_dir)

    log.info("Pipeline complete.")

    # ----------------------------------------------------------------
    # 10. Save final artefacts
    # ----------------------------------------------------------------
    final_dir = Path(config.output_dir) / "final"
    log.info(f"Saving final model to {final_dir}...")

    # LoRA adapters
    model.backbone.save_pretrained(str(final_dir / "lora_adapters"))
    tokenizer.save_pretrained(str(final_dir / "tokenizer"))

    # Classification heads
    model.save_heads(str(final_dir))

    log.info("Training complete.")


if __name__ == "__main__":
    main()
