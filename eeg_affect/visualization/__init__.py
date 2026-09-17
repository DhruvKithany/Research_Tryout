"""Publication-grade visualization utilities for EEG neurobiology and manifold geometry."""

from .topomap import (
    plotTopomap,
    plotMultibandTopomaps,
    bandTitle,
    plot_topomap,
    plot_multiband_topomaps,
    band_title,
)
from .manifold_plots import (
    plotManifoldEmbedding,
    plotDistanceMatrix,
    plot_manifold_embedding,
    plot_distance_matrix,
)
from .neuro_plots import (
    plotFrontalAlphaAsymmetry,
    plotRegionalBandpowers,
    plot_frontal_alpha_asymmetry,
    plot_regional_bandpowers,
)
from .confusion import (
    plotConfusionMatrix,
    plotEmotionDendrogram,
    plot_confusion_matrix,
    plot_emotion_dendrogram,
)

__all__ = [
    "plotTopomap",
    "plotMultibandTopomaps",
    "bandTitle",
    "plotManifoldEmbedding",
    "plotDistanceMatrix",
    "plotFrontalAlphaAsymmetry",
    "plotRegionalBandpowers",
    "plotConfusionMatrix",
    "plotEmotionDendrogram",
    "plot_topomap",
    "plot_multiband_topomaps",
    "band_title",
    "plot_manifold_embedding",
    "plot_distance_matrix",
    "plot_frontal_alpha_asymmetry",
    "plot_regional_bandpowers",
    "plot_confusion_matrix",
    "plot_emotion_dendrogram",
]
