"""Geometric and Topological Manifold Analysis of Affective EEG Space."""

from .manifold import EmotionManifoldEmbedder
from .distance import (
    computeEmotionCentroids,
    computePairwiseGeometry,
    compareEegVsPsychologyGeometry,
    hierarchicalClusteringEmotions,
    compute_emotion_centroids,
    compute_pairwise_geometry,
    compare_eeg_vs_psychology_geometry,
    hierarchical_clustering_emotions,
)
from .intrinsic_dim import (
    estimateTwoNnDimension,
    estimatePcaVarianceDimensions,
    estimateIntrinsicDimension,
    estimate_two_nn_dimension,
    estimate_pca_variance_dimensions,
    estimate_intrinsic_dimension,
)

__all__ = [
    "EmotionManifoldEmbedder",
    "computeEmotionCentroids",
    "computePairwiseGeometry",
    "compareEegVsPsychologyGeometry",
    "hierarchicalClusteringEmotions",
    "compute_emotion_centroids",
    "compute_pairwise_geometry",
    "compare_eeg_vs_psychology_geometry",
    "hierarchical_clustering_emotions",
    "estimateTwoNnDimension",
    "estimatePcaVarianceDimensions",
    "estimateIntrinsicDimension",
    "estimate_two_nn_dimension",
    "estimate_pca_variance_dimensions",
    "estimate_intrinsic_dimension",
]
