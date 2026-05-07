"""
Utility functions for CARE_v2.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from care_v2.configs.config import CAREv2Config, LABEL_TO_IDX, NUM_CLASSES


def setup_logging(output_dir: str = "."):
    """Configures unified logging to console + file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{output_dir}/care_v2.log", encoding="utf-8"),
        ],
    )


def compute_class_weights(dataset: Dataset, config: CAREv2Config) -> torch.Tensor:
    """
    Computes log-smoothed inverse-frequency class weights for each dimension.

    Returns: Tensor of shape [6, 5] (dimensions × classes)
    """
    log = logging.getLogger(__name__)
    log.info("Computing class weights from training data...")
    df = dataset.df

    weights_list = []
    for lbl in config.labels:
        vals = [LABEL_TO_IDX[int(x)] for x in df[lbl].tolist()]
        counts = np.bincount(vals, minlength=NUM_CLASSES).astype(float)
        total = len(vals)
        safe_counts = np.maximum(counts, 1)
        w = 1.0 + np.log(total / safe_counts)
        w = w / w.mean()
        w = np.clip(w, 0.5, 3.0)
        weights_list.append(torch.tensor(w, dtype=torch.float32))

    return torch.stack(weights_list)   # [6, 5]


def save_confusion_matrices(preds: np.ndarray, labels: np.ndarray, output_dir: str):
    """
    Generates and saves confusion matrix plots for each therapeutic dimension.
    """
    import os
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib or seaborn not installed. Skipping confusion matrix plots.")
        return

    from sklearn.metrics import confusion_matrix
    from care_v2.configs.config import LABELS

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for i, dim_name in enumerate(LABELS):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1, 2, 3, 4])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['-2', '-1', '0', '1', '2'],
                    yticklabels=['-2', '-1', '0', '1', '2'])
        plt.title(f"Confusion Matrix: {dim_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filename = dim_name.replace(" ", "_").lower() + "_cm.png"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        plt.close()
        logging.getLogger(__name__).info(f"Saved confusion matrix: {save_path}")
