"""Unit tests for manifold embeddings, distance matrices, and intrinsic dimension."""

import numpy as np

from eeg_affect.geometry.distance import (
    compare_eeg_vs_psychology_geometry,
    compute_emotion_centroids,
    compute_pairwise_geometry,
    hierarchical_clustering_emotions,
)
from eeg_affect.geometry.intrinsic_dim import (
    estimate_intrinsic_dimension,
    estimate_two_nn_dimension,
)
from eeg_affect.geometry.manifold import EmotionManifoldEmbedder


def test_manifold_embedder():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20)

    # PCA
    pca_emb = EmotionManifoldEmbedder(method="pca", n_components=2)
    c_pca = pca_emb.fit_transform(X)
    assert c_pca.shape == (100, 2)

    # Diffusion / Spectral
    diff_emb = EmotionManifoldEmbedder(method="diffusion", n_components=2, n_neighbors=10)
    c_diff = diff_emb.fit_transform(X)
    assert c_diff.shape == (100, 2)


def test_centroids_and_pairwise_distance():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = rng.choice([1, 2, 3, 4, 5], size=100)

    centroids, unique_eids, names = compute_emotion_centroids(X, y)
    assert centroids.shape == (5, 10)
    assert len(unique_eids) == 5

    dist_matrix = compute_pairwise_geometry(centroids, metric="euclidean")
    assert dist_matrix.shape == (5, 5)
    assert np.allclose(np.diag(dist_matrix), 0.0)
    assert np.allclose(dist_matrix, dist_matrix.T)

    linkage = hierarchical_clustering_emotions(dist_matrix, method="ward")
    assert linkage.shape[0] == 4  # N - 1 merges


def test_two_nn_intrinsic_dimension():
    rng = np.random.RandomState(42)
    # Generate points on a 3D sphere embedded in 20D space
    pts_3d = rng.randn(200, 3)
    pts_3d /= np.linalg.norm(pts_3d, axis=1, keepdims=True)
    matrix = rng.randn(3, 20)
    X = np.dot(pts_3d, matrix)

    dim_est = estimate_two_nn_dimension(X)
    # 2D surface of a sphere, should estimate around 2 +/- 1
    assert 1.0 <= dim_est <= 5.0
