from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")

    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_true, y_pred, zero_division=0)
    )

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered_keys = ["auc", "acc", "precision", "recall", "f1"]
    parts = []
    for k in ordered_keys:
        if k in metrics:
            v = metrics[k]
            if np.isnan(v):
                parts.append(f"{k}=nan")
            else:
                parts.append(f"{k}={v:.4f}")
    return " | ".join(parts)