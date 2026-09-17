"""Pairwise distance matrices, hierarchical clustering, and representational similarity."""

from typing import Dict, List, Tuple

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import scipy.stats as stats

from ..config import COWEN_27_EMOTIONS, COWEN_VALENCE_AROUSAL


def computeEmotionCentroids(
    X: np.ndarray,
    yCowen: np.ndarray,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Finds the average (mean) feature vector for each of the 27 emotion classes.

    Returns:
        centroids: (27, 462) matrix where each row is the average brainwave for one emotion.
        uniqueEmotions: List of emotion IDs (1 to 27).
        names: List of emotion names ('Joy', 'Sadness', etc.).
    """
    uniqueEmotions = sorted(np.unique(yCowen))
    centroids = []
    names = []

    for eid in uniqueEmotions:
        mask = (yCowen == eid)
        # Average all samples for this specific emotion
        centroid = np.mean(X[mask], axis=0)
        centroids.append(centroid)
        names.append(COWEN_27_EMOTIONS.get(eid, f"Emotion_{eid}"))

    return np.array(centroids), uniqueEmotions, names


def computePairwiseGeometry(centroids: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Builds a symmetric 27x27 distance table showing how far apart each pair of emotions is in brain space."""
    return spd.squareform(spd.pdist(centroids, metric=metric))


def compareEegVsPsychologyGeometry(
    centroids: np.ndarray,
    emotionIds: List[int],
) -> Dict[str, float]:
    """Tests if brain distances match psychological distances (Representational Similarity Analysis).

    Steps:
    1. Measure distances between all pairs of emotions using EEG brain features.
    2. Measure distances between all pairs of emotions using standard psychology coordinates (Valence & Arousal).
    3. Run a correlation (Pearson and Spearman) between the two sets of distances.
    """
    # 1. Neural distance vector (upper triangle of the 27x27 distance table)
    neuralDist = spd.pdist(centroids, metric="euclidean")

    # 2. Psychological distance vector (distances on the 2D Pleasure vs Energy chart)
    psychCoords = np.array([COWEN_VALENCE_AROUSAL[eid] for eid in emotionIds])
    psychDist = spd.pdist(psychCoords, metric="euclidean")

    # 3. Check if they correlate
    pearsonR, pearsonP = stats.pearsonr(neuralDist, psychDist)
    spearmanRho, spearmanP = stats.spearmanr(neuralDist, psychDist)

    return {
        "pearson_r": float(pearsonR),
        "pearson_p": float(pearsonP),
        "spearman_rho": float(spearmanRho),
        "spearman_p": float(spearmanP),
    }


def hierarchicalClusteringEmotions(
    distanceMatrix: np.ndarray,
    method: str = "ward",
) -> np.ndarray:
    """Builds an emotion family tree (dendrogram) by repeatedly grouping the two closest emotions together."""
    condensedDist = spd.squareform(distanceMatrix)
    linkage = sch.linkage(condensedDist, method=method)
    return linkage


# Aliases for backward compatibility
compute_emotion_centroids = computeEmotionCentroids
compute_pairwise_geometry = computePairwiseGeometry
compare_eeg_vs_psychology_geometry = compareEegVsPsychologyGeometry
hierarchical_clustering_emotions = hierarchicalClusteringEmotions
