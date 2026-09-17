"""Confusion matrix heatmaps and hierarchical dendrogram visualizers.

Methodology:
- Normalized confusion matrices reveal misclassification patterns and cross-talk between semantically
  adjacent emotional categories (e.g. Adoration vs Joy, Horror vs Fear).
- Supports both 4-class Valence-Arousal quadrants and full 27-class Cowen taxonomy.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plotConfusionMatrix(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    labels: List[str],
    normalize: str = "true",
    title: str = "Normalized Confusion Matrix",
    cmap: str = "Blues",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots a normalized confusion matrix with emotion category labels.

    Args:
        yTrue: Ground truth labels.
        yPred: Model predicted labels.
        labels: String category names.
        normalize: Normalization mode ('true', 'pred', 'all', or None).
        title: Figure title.
        cmap: Color palette name.
        savePath: Optional output path.

    Returns:
        plt.Figure: Rendered matplotlib figure.
    """
    sPath = kwargs.get("save_path", savePath)
    cm = confusion_matrix(yTrue, yPred, normalize=normalize)

    figsize = (10, 8) if len(labels) <= 10 else (14, 12)
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        annot=len(labels) <= 10,
        fmt=".2f" if len(labels) <= 10 else "",
        ax=ax,
        cbar_kws={"label": "Normalized Proportion"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Emotion", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Emotion", fontsize=11, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig


def plotEmotionDendrogram(
    linkageMatrix: np.ndarray,
    labels: List[str],
    title: str = "Hierarchical Clustering of Neural Emotion States (Ward Linkage)",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots hierarchical dendrogram showing how emotion states cluster in EEG feature space.

    Args:
        linkageMatrix: Hierarchical clustering linkage matrix.
        labels: Emotion category labels.
        title: Figure title.
        savePath: Optional file path.

    Returns:
        plt.Figure: Rendered matplotlib figure.
    """
    sPath = kwargs.get("save_path", savePath)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    sch.dendrogram(
        linkageMatrix,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=0.7 * np.max(linkageMatrix[:, 2]),
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("Ward Distance", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig

# Backward-compatible snake_case aliases
plot_confusion_matrix = plotConfusionMatrix
plot_emotion_dendrogram = plotEmotionDendrogram
