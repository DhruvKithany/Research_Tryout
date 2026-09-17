"""Machine learning and generative models for EEG emotion decoding."""

from .baselines import (
    buildBaselineModels,
    CosineNearestCentroid,
    build_baseline_models,
)
from .ensembles import (
    buildEnsembleModels,
    build_ensemble_models,
)
from .neural import (
    buildNeuralModel,
    build_neural_model,
)
from .generator import EEGFeatureGenerator

__all__ = [
    "buildBaselineModels",
    "CosineNearestCentroid",
    "build_baseline_models",
    "buildEnsembleModels",
    "build_ensemble_models",
    "buildNeuralModel",
    "build_neural_model",
    "EEGFeatureGenerator",
]
