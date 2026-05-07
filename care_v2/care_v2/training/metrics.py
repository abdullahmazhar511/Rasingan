"""
Metrics for CARE_v2.

Computes per-dimension and average:
- Quadratic Weighted Kappa (QWK)
- Accuracy
- Macro F1
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from transformers import EvalPrediction
from care_v2.configs.config import LABELS, IDX_TO_LABEL


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    HuggingFace Trainer-compatible compute_metrics function.

    eval_pred.predictions: numpy array [N, 6, 5]
    eval_pred.label_ids:   numpy array [N, 6]
    """
    logits, labels = eval_pred

    # Handle tuple output (loss, logits) from some trainer versions
    if isinstance(logits, tuple):
        logits = logits[0]

    # logits: [N, 6, 5] → preds: [N, 6]
    preds = np.argmax(logits, axis=-1)

    metrics = {}
    qwks, accs, f1s_macro, f1s_weighted = [], [], [], []

    for i, dim_name in enumerate(LABELS):
        y_true = labels[:, i]
        y_pred = preds[:, i]

        # Convert idx → real labels for QWK
        y_true_real = np.vectorize(IDX_TO_LABEL.get)(y_true)
        y_pred_real = np.vectorize(IDX_TO_LABEL.get)(y_pred)

        try:
            qwk = cohen_kappa_score(y_true_real, y_pred_real, weights="quadratic")
        except Exception:
            qwk = 0.0

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Short key for HuggingFace logging
        safe_key = dim_name.replace(" ", "_")
        metrics[f"qwk_{safe_key}"] = round(qwk, 4)
        metrics[f"acc_{safe_key}"] = round(acc, 4)
        metrics[f"f1_macro_{safe_key}"] = round(f1_macro, 4)
        metrics[f"f1_weighted_{safe_key}"] = round(f1_weighted, 4)

        qwks.append(qwk)
        accs.append(acc)
        f1s_macro.append(f1_macro)
        f1s_weighted.append(f1_weighted)

    metrics["avg_qwk"] = round(float(np.mean(qwks)), 4)
    metrics["avg_acc"] = round(float(np.mean(accs)), 4)
    metrics["avg_f1_macro"] = round(float(np.mean(f1s_macro)), 4)
    metrics["avg_f1_weighted"] = round(float(np.mean(f1s_weighted)), 4)

    return metrics
