from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_binary_classification_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    predicted = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = ["auc", "average_precision", "brier", "acc", "precision", "recall", "f1"]
    return " | ".join(f"{key}={metrics[key]:.4f}" for key in ordered if key in metrics)
