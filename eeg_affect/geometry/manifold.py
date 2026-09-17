"""Manifold learning and non-linear dimensionality reduction for EEG feature spaces."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding


class EmotionManifoldEmbedder:
    """Compresses 462-dimensional EEG samples down to 2D or 3D coordinates."""

    def __init__(
        self,
        method: str = "diffusion",
        nComponents: int = 2,
        randomState: int = 42,
        **kwargs,
    ):
        """Initializes the dimensionality reduction model.

        Args:
            method: 'diffusion', 'pca', 'tsne', 'isomap', or 'mds'.
            nComponents: Target dimension (default 2 for simple 2D x,y plotting).
            randomState: Random seed for reproducible plots.
        """
        self.method = method.lower()
        self.nComponents = kwargs.pop("n_components", nComponents)
        self.n_components = self.nComponents
        self.randomState = kwargs.pop("random_state", randomState)
        self.random_state = self.randomState
        self.kwargs = kwargs
        self.model = None

    def fitTransform(self, X: np.ndarray) -> np.ndarray:
        """Fits the compression algorithm on the features and returns the (N, 2) coordinates."""
        if self.method == "pca":
            self.model = PCA(
                n_components=self.nComponents,
                random_state=self.randomState,
                **self.kwargs,
            )
            return self.model.fit_transform(X)

        elif self.method == "tsne":
            # For TSNE, compress down with PCA first to remove extreme noise
            if X.shape[1] > 50:
                xIn = PCA(n_components=50, random_state=self.randomState).fit_transform(X)
            else:
                xIn = X
            perplexity = self.kwargs.get("perplexity", min(30.0, max(5.0, len(X) / 5.0)))
            self.model = TSNE(
                n_components=self.nComponents,
                perplexity=perplexity,
                random_state=self.randomState,
                init="pca",
                learning_rate="auto",
            )
            return self.model.fit_transform(xIn)

        elif self.method in ("diffusion", "spectral", "laplacian"):
            # Builds a nearest-neighbor graph and finds the lowest vibrating frequencies of the graph
            nNeighbors = self.kwargs.get("n_neighbors", min(20, max(3, len(X) // 10)))
            self.model = SpectralEmbedding(
                n_components=self.nComponents,
                affinity="nearest_neighbors",
                n_neighbors=nNeighbors,
                random_state=self.randomState,
            )
            return self.model.fit_transform(X)

        elif self.method == "isomap":
            nNeighbors = self.kwargs.get("n_neighbors", min(15, max(3, len(X) // 10)))
            self.model = Isomap(
                n_components=self.nComponents,
                n_neighbors=nNeighbors,
            )
            return self.model.fit_transform(X)

        elif self.method == "mds":
            self.model = MDS(
                n_components=self.nComponents,
                random_state=self.randomState,
                normalized_stress="auto",
            )
            return self.model.fit_transform(X)

        else:
            raise ValueError(f"Unknown manifold method: {self.method}")

    # Backward-compatible alias
    fit_transform = fitTransform
