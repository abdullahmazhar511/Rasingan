"""
CARE_v2 evaluation entry-point.

Runs full evaluation on the test set with per-dimension reporting.

Run:
    python -m care_v2.evaluate
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from care_v2.configs.config import CAREv2Config, LABELS, IDX_TO_LABEL
from care_v2.data.preprocessing import load_csv_data
from care_v2.datasets.care_dataset import CAREDataset
from care_v2.models.care_classifier import CAREv2Classifier
from care_v2.retrieval.prototype_retriever import PrototypeRetriever
from care_v2.utils.io_utils import setup_logging

log = logging.getLogger(__name__)


import os

def main():
    config = CAREv2Config()
    setup_logging(config.output_dir)

    # Detect explicit device from environment
    device_id = os.environ.get("DEVICE_ID") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    primary_gpu = device_id.split(",")[0].strip()
    device = f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu"
    
    log.info(f"Using device: {device}")

    # ----------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------
    log.info("Loading tokenizer + model...")
    final_dir = Path(config.output_dir) / "final"
    tokenizer = AutoTokenizer.from_pretrained(str(final_dir / "tokenizer"), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    retriever = PrototypeRetriever(config, device=device)
    model = CAREv2Classifier(config, device=device)
    model.load_heads(str(final_dir))
    model.to(device)
    model.eval()

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------
    log.info("Loading test data...")
    test_df = load_csv_data(config.test_csv_folder, config)
    test_ds = CAREDataset(test_df, retriever, tokenizer, config)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    all_preds_idx = []
    all_trues_idx = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            out = model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out["logits"], dim=2).cpu().numpy()

            all_preds_idx.extend(preds)
            all_trues_idx.extend(labels.numpy())

    all_preds_idx = np.array(all_preds_idx)     # [N, 6]
    all_trues_idx = np.array(all_trues_idx)

    all_preds = np.vectorize(IDX_TO_LABEL.get)(all_preds_idx)
    all_trues = np.vectorize(IDX_TO_LABEL.get)(all_trues_idx)

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    score_range = [-2, -1, 0, 1, 2]
    final_metrics = {}
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    print("\n" + "=" * 70)
    print(f"{'Dimension':<35} {'QWK':>6} {'Acc':>6} {'F1':>6}")
    print("=" * 70)

    for i, dim in enumerate(LABELS):
        p = all_preds[:, i]
        t = all_trues[:, i]
        p_idx = all_preds_idx[:, i]
        t_idx = all_trues_idx[:, i]

        qwk = cohen_kappa_score(t, p, weights="quadratic")
        acc = accuracy_score(t, p)
        f1 = f1_score(t_idx, p_idx, average="weighted", zero_division=0)

        final_metrics[dim] = {"QWK": round(qwk, 4), "Acc": round(acc, 4), "F1": round(f1, 4)}
        print(f"{dim:<35} {qwk:>6.3f} {acc:>6.3f} {f1:>6.3f}")

        cm = confusion_matrix(t, p, labels=score_range)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=score_range, yticklabels=score_range)
        axes[i].set_title(f"{dim}\nQWK: {qwk:.3f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    avg_qwk = np.mean([v["QWK"] for v in final_metrics.values()])
    avg_acc = np.mean([v["Acc"] for v in final_metrics.values()])
    print("=" * 70)
    print(f"{'AVERAGE':<35} {avg_qwk:>6.3f} {avg_acc:>6.3f}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    out_dir = Path(config.output_dir)
    plt.tight_layout()
    plt.savefig(out_dir / "test_confusion_matrices.png", dpi=150)
    log.info(f"Confusion matrices saved.")

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
