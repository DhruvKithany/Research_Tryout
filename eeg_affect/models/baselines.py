"""Baseline classifiers: Ridge, Logistic Regression, and Cosine Nearest Centroid."""

from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import NearestCentroid


class CosineNearestCentroid(BaseEstimator, ClassifierMixin):
    """Classifies a sample by picking whichever emotion average has the highest cosine similarity (smallest angle)."""

    def __init__(self, shrinkThreshold: Optional[float] = None, **kwargs):
        self.shrinkThreshold = kwargs.get("shrink_threshold", shrinkThreshold)
        self.shrink_threshold = self.shrinkThreshold
        self.classes_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CosineNearestCentroid":
        """Calculates the average feature center for each emotion class."""
        self.classes_ = np.unique(y)
        centroids = []
        for c in self.classes_:
            centroids.append(np.mean(X[y == c], axis=0))
        self.centroids_ = np.array(centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns each test point to the nearest emotion centroid by cosine similarity."""
        proba = self.predictProba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        """Converts cosine similarities into class probabilities using softmax."""
        # Normalize vectors to unit length
        normCentroids = self.centroids_ / np.linalg.norm(self.centroids_, axis=1, keepdims=True)
        normX = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # Dot product of normalized vectors is cosine similarity in [-1, 1]
        sims = np.dot(normX, normCentroids.T)

        # Scaled softmax
        expSims = np.exp(sims * 5.0)
        return expSims / np.sum(expSims, axis=1, keepdims=True)

    predict_proba = predictProba


def buildBaselineModels(randomState: int = 42, **kwargs) -> Dict[str, BaseEstimator]:
    """Creates a dictionary of basic linear and centroid models for benchmarking."""
    rState = kwargs.get("random_state", randomState)
    return {
        "Ridge": RidgeClassifier(alpha=1.0, random_state=rState),
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=500,
            solver="lbfgs",
            random_state=rState,
        ),
        "NearestCentroid_Euclidean": NearestCentroid(metric="euclidean"),
        "NearestCentroid_Cosine": CosineNearestCentroid(),
    }


# Aliases for backward compatibility
build_baseline_models = buildBaselineModels
