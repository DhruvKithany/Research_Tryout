"""Classification metrics: Top-1, Top-k, Balanced Accuracy, Macro F1, and Cohen's Kappa."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def topKAccuracy(yTrue: np.ndarray, yProba: np.ndarray, k: int = 3) -> float:
    """Checks if the ground-truth emotion is inside the model's top-k highest probability picks."""
    if yProba is None or yProba.ndim != 2:
        return float(np.nan)

    # Sort probabilities to get the indices of the top k highest guesses
    topKPreds = np.argsort(yProba, axis=1)[:, -k:]
    # Check if the true label appears in any of those k spots
    correct = np.any(topKPreds == yTrue[:, None], axis=1)
    return float(np.mean(correct))


def computeClassificationMetrics(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    yProba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes a full set of classification scores (Accuracy, Top-3, Top-5, Macro F1, Kappa)."""
    metrics = {
        "accuracy": float(accuracy_score(yTrue, yPred)),
        "balanced_accuracy": float(balanced_accuracy_score(yTrue, yPred)),
        "macro_f1": float(f1_score(yTrue, yPred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yTrue, yPred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yTrue, yPred)),
    }

    if yProba is not None and yProba.ndim == 2:
        nClasses = yProba.shape[1]
        if nClasses >= 3:
            metrics["top_3_accuracy"] = topKAccuracy(yTrue, yProba, k=3)
        if nClasses >= 5:
            metrics["top_5_accuracy"] = topKAccuracy(yTrue, yProba, k=5)

    return metrics


# Aliases for backward compatibility
top_k_accuracy = topKAccuracy
compute_classification_metrics = computeClassificationMetrics
