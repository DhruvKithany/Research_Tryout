"""Ensemble classifiers: Random Forest, Extra-Trees, and Histogram Gradient Boosting."""

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


def buildEnsembleModels(nJobs: int = -1, randomState: int = 42) -> Dict[str, BaseEstimator]:
    """Creates our non-linear tree-based ensemble classifiers."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            max_features="sqrt",
            n_jobs=nJobs,
            random_state=randomState,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            max_features="sqrt",
            n_jobs=nJobs,
            random_state=randomState,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=randomState,
        ),
    }


# Aliases for backward compatibility
build_ensemble_models = buildEnsembleModels
