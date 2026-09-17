"""Model benchmarking harness across linear, ensemble, metric, and neural classifiers."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..data.preprocessor import EEGPreprocessor
from ..data.split import SubjectGroupSplitter
from .metrics import computeClassificationMetrics


def runModelBenchmark(
    models: Dict[str, BaseEstimator],
    xTrain: np.ndarray,
    yTrain: np.ndarray,
    xTest: np.ndarray,
    yTest: np.ndarray,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Trains and tests multiple models on the exact same data split.

    Args:
        models: Dictionary of models, like {'RandomForest': model1, 'MLP': model2}.
        xTrain: Training features from 74 people.
        yTrain: Training emotion labels.
        xTest: Test features from 14 brand new people.
        yTest: Test emotion labels.
        verbose: Print progress as models finish.

    Returns:
        resultsDf: A summary table sorted by Macro F1 score.
        detailedResults: Dictionary with predictions, probabilities, and latencies.
    """
    records = []
    detailedResults = {}

    for name, model in models.items():
        if verbose:
            print(f"--> Training and evaluating {name}...")

        # Measure training time
        t0 = time.time()
        model.fit(xTrain, yTrain)
        fitTime = time.time() - t0

        # Measure how fast it predicts a test sample
        t1 = time.time()
        yPred = model.predict(xTest)
        inferenceTime = time.time() - t1

        # Extract prediction probabilities (for Top-3 and Top-5 accuracy)
        yProba = None
        if hasattr(model, "predict_proba"):
            try:
                yProba = model.predict_proba(xTest)
            except Exception:
                yProba = None
        elif hasattr(model, "decision_function"):
            try:
                dfunc = model.decision_function(xTest)
                if dfunc.ndim == 2:
                    # Convert raw scores to probabilities using softmax
                    expDf = np.exp(dfunc - np.max(dfunc, axis=1, keepdims=True))
                    yProba = expDf / np.sum(expDf, axis=1, keepdims=True)
            except Exception:
                yProba = None

        # Compute accuracy, Top-3, Top-5, and F1 scores
        metrics = computeClassificationMetrics(yTest, yPred, yProba)
        metrics["fit_time_s"] = round(fitTime, 2)
        metrics["inference_time_s"] = round(inferenceTime, 4)
        metrics["model"] = name

        records.append(metrics)
        detailedResults[name] = {
            "model": model,
            "y_pred": yPred,
            "y_proba": yProba,
            "metrics": metrics,
        }

    resultsDf = pd.DataFrame(records)
    resultsDf = resultsDf.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    return resultsDf, detailedResults


# Aliases for backward compatibility
run_model_benchmark = runModelBenchmark
