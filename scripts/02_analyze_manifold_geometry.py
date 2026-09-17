"""Manifold geometry, RSA, hierarchical clustering, and intrinsic dimensionality."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
projectRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projectRoot))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_affect.config import FIGURES_DIR
from eeg_affect.data.loader import loadEegFeatures
from eeg_affect.data.preprocessor import EEGPreprocessor
from eeg_affect.geometry.distance import (
    compareEegVsPsychologyGeometry,
    computeEmotionCentroids,
    computePairwiseGeometry,
    hierarchicalClusteringEmotions,
)
from eeg_affect.geometry.intrinsic_dim import estimateIntrinsicDimension
from eeg_affect.geometry.manifold import EmotionManifoldEmbedder
from eeg_affect.visualization.confusion import plotEmotionDendrogram
from eeg_affect.visualization.manifold_plots import (
    plotDistanceMatrix,
    plotManifoldEmbedding,
)


def main():
    print("Running Script 02: Manifold Geometry and Representational Similarity Analysis...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load feature dataset
    print("Loading extracted EEG features...")
    X_df, yCowen, groups, metadata = loadEegFeatures(target="cowen")
    print(f"Dataset shape: {X_df.shape} across {len(np.unique(groups))} participants.")

    # 2. Preprocess features (Robust scaling)
    print("Normalizing features with EEGPreprocessor (RobustScaler)...")
    preprocessor = EEGPreprocessor(scaler="robust")
    X_scaled = preprocessor.fitTransform(X_df)

    # 3. Compute Emotion Centroids & Distance Matrix
    print("Computing 27 emotion centroids and pairwise neural geometry...")
    # Map 0-indexed yCowen to 1-indexed Cowen IDs
    cowenIds1Based = yCowen + 1
    centroids, uniqueEids, emotionNames = computeEmotionCentroids(X_scaled, cowenIds1Based)
    distMatrix = computePairwiseGeometry(centroids, metric="euclidean")

    distFigPath = FIGURES_DIR / "02_neural_emotion_distance_matrix.png"
    figDist = plotDistanceMatrix(
        distMatrix,
        emotionNames,
        title="Pairwise Euclidean Distance in EEG Neural Space (27 Cowen Emotions)",
        save_path=str(distFigPath),
    )
    plt.close(figDist)
    print(f"Saved neural distance matrix -> {distFigPath}")

    # 4. Representational Similarity Analysis (RSA) vs Psychological Circumplex
    print("Computing Representational Similarity Analysis (RSA) vs Psychological Space...")
    rsaResults = compareEegVsPsychologyGeometry(centroids, uniqueEids)
    print(f"--> Pearson r:    {rsaResults['pearson_r']:.4f} (p = {rsaResults['pearson_p']:.4e})")
    print(f"--> Spearman rho: {rsaResults['spearman_rho']:.4f} (p = {rsaResults['spearman_p']:.4e})")

    # Hierarchical Clustering Dendrogram
    linkage = hierarchicalClusteringEmotions(distMatrix, method="ward")
    dendroPath = FIGURES_DIR / "02_emotion_hierarchical_dendrogram.png"
    figDendro = plotEmotionDendrogram(
        linkage,
        emotionNames,
        title="Hierarchical Clustering of EEG Neural Affective Space (Ward Linkage)",
        save_path=str(dendroPath),
    )
    plt.close(figDendro)
    print(f"Saved hierarchical clustering dendrogram -> {dendroPath}")

    # Intrinsic Dimensionality
    print("Estimating Intrinsic Dimensionality of Affective Manifold...")
    idResults = estimateIntrinsicDimension(X_scaled)
    for k, v in idResults.items():
        print(f"--> {k}: {v}")

    # 5. Low-Dimensional Manifold Projections
    print("Computing manifold embeddings (Diffusion Maps & PCA)...")
    # Subsample for clear visualization (50 samples per emotion)
    sampleIndices = []
    rng = np.random.RandomState(42)
    for cid in range(27):
        cIdx = np.where(yCowen == cid)[0]
        chosen = rng.choice(cIdx, size=min(60, len(cIdx)), replace=False)
        sampleIndices.extend(chosen)
    sampleIndices = np.array(sampleIndices)

    X_sub = X_scaled[sampleIndices]
    yCowenSub = yCowen[sampleIndices]
    yQuadSub = metadata["Quadrant"].values[sampleIndices]

    # Diffusion / Spectral Embedding
    print("Computing Diffusion / Laplacian Eigenmaps...")
    diffEmbedder = EmotionManifoldEmbedder(method="diffusion", n_components=2, n_neighbors=25)
    coordsDiff = diffEmbedder.fitTransform(X_sub)

    diffFigPath = FIGURES_DIR / "02_manifold_diffusion_quadrants.png"
    figDiff = plotManifoldEmbedding(
        coordsDiff,
        yQuadSub,
        label_type="quadrant",
        title="EEG Affective Manifold: Diffusion / Laplacian Eigenmap Embedding",
        save_path=str(diffFigPath),
    )
    plt.close(figDiff)
    print(f"Saved diffusion embedding plot -> {diffFigPath}")

    # PCA Embedding
    print("Computing PCA Projection...")
    pcaEmbedder = EmotionManifoldEmbedder(method="pca", n_components=2)
    coordsPca = pcaEmbedder.fitTransform(X_sub)

    pcaFigPath = FIGURES_DIR / "02_manifold_pca_quadrants.png"
    figPca = plotManifoldEmbedding(
        coordsPca,
        yQuadSub,
        label_type="quadrant",
        title="EEG Affective Space: Principal Component Analysis (PC1 vs PC2)",
        save_path=str(pcaFigPath),
    )
    plt.close(figPca)
    print(f"Saved PCA embedding plot -> {pcaFigPath}")

    # Save metrics json
    metricsOut = {
        "rsa": rsaResults,
        "intrinsic_dimensionality": idResults,
    }
    jsonPath = FIGURES_DIR / "02_manifold_metrics.json"
    with open(jsonPath, "w") as f:
        json.dump(metricsOut, f, indent=4)
    print(f"\nSaved manifold analysis metrics -> {jsonPath}")
    print("\nManifold geometry analysis completed successfully!")


if __name__ == "__main__":
    main()
