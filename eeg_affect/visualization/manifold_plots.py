"""Manifold embedding scatter plots, centroid trajectories, and distance heatmaps.

Theoretical Framework:
Visualizes low-dimensional projections of the electrophysiological emotion manifold:
- Diffusion Maps / Spectral Embeddings colored by Russell Circumplex Quadrants.
- Pairwise Euclidean distance matrices across 27 emotion centroids.
- Hierarchical agglomerative dendrograms revealing intrinsic affective clusters.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns

from ..config import COWEN_27_EMOTIONS, COWEN_VALENCE_AROUSAL, QUADRANT_NAMES


def plotManifoldEmbedding(
    coords: np.ndarray,
    labels: np.ndarray,
    labelType: str = "quadrant",
    title: str = "EEG Emotion Manifold",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots 2D manifold coordinates colored by affective label (quadrant or emotion).

    Args:
        coords: (n_samples, 2) coordinates from manifold reduction.
        labels: Integer category labels.
        labelType: 'quadrant' (4 classes) or 'cowen' (27 classes).
        title: Figure title.
        savePath: Optional output path.

    Returns:
        plt.Figure: Rendered matplotlib figure.
    """
    sPath = kwargs.get("save_path", savePath)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    if labelType == "quadrant":
        palette = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
        for qIdx in range(4):
            mask = (labels == qIdx)
            if np.sum(mask) > 0:
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    label=QUADRANT_NAMES[qIdx],
                    color=palette[qIdx],
                    alpha=0.6,
                    s=25,
                    edgecolors="none",
                )
        ax.legend(title="Valence-Arousal Quadrant", fontsize=9, loc="best", framealpha=0.9)

    elif labelType == "cowen":
        cmap = plt.cm.get_cmap("tab20", 27)
        for cid in sorted(np.unique(labels)):
            mask = (labels == cid)
            name = COWEN_27_EMOTIONS.get(cid, f"E_{cid}")
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=name,
                alpha=0.5,
                s=20,
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=2)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Manifold Dimension 1", fontsize=10)
    ax.set_ylabel("Manifold Dimension 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig


def plotPairwiseDistanceHeatmap(
    distMatrix: np.ndarray,
    emotionNames: List[str],
    title: str = "Pairwise Euclidean Distance Between Emotion Centroids",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots a 27x27 symmetric distance matrix heatmap between emotion prototypes.

    Args:
        distMatrix: (27, 27) distance matrix.
        emotionNames: List of 27 emotion names.
        title: Figure title.
        savePath: Optional save path.

    Returns:
        plt.Figure: Rendered heatmap.
    """
    sPath = kwargs.get("save_path", savePath)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=200)
    sns.heatmap(
        distMatrix,
        xticklabels=emotionNames,
        yticklabels=emotionNames,
        cmap="mako_r",
        ax=ax,
        cbar_kws={"label": "Euclidean Distance in Feature Space"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig


def plotHierarchicalDendrogram(
    linkageMatrix: np.ndarray,
    emotionNames: List[str],
    title: str = "Hierarchical Clustering Dendrogram (Ward Linkage)",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots hierarchical clustering dendrogram of emotional categories.

    Args:
        linkageMatrix: Scipy linkage matrix.
        emotionNames: Emotion labels for dendrogram leaves.
        title: Figure title.
        savePath: Optional file path.

    Returns:
        plt.Figure: Rendered dendrogram.
    """
    sPath = kwargs.get("save_path", savePath)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sch.dendrogram(
        linkageMatrix,
        labels=emotionNames,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=0.7 * np.max(linkageMatrix[:, 2]),
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Ward Cluster Dissimilarity Distance", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig

# Backward-compatible aliases
plotDistanceMatrix = plotPairwiseDistanceHeatmap
plot_distance_matrix = plotPairwiseDistanceHeatmap
plot_manifold_embedding = plotManifoldEmbedding
plot_pairwise_distance_heatmap = plotPairwiseDistanceHeatmap
plot_hierarchical_dendrogram = plotHierarchicalDendrogram
