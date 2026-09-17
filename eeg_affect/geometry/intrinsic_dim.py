"""Intrinsic dimensionality estimation via Two-NN and linear PCA variance thresholds."""

from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def estimateTwoNnDimension(X: np.ndarray) -> float:
    """Estimates the true non-linear dimension of data points using distance to 1st and 2nd neighbors."""
    # Find the 2 nearest neighbors for every single sample
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)

    # distances[:, 0] is distance to self (0)
    # distances[:, 1] is distance to closest neighbor (r1)
    # distances[:, 2] is distance to 2nd closest neighbor (r2)
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Filter out duplicate points or zero distances to avoid dividing by 0
    valid = (r1 > 1e-10) & (r2 > r1)
    if np.sum(valid) < 10:
        return float(np.nan)

    # Compute ratio mu = r2 / r1
    mu = r2[valid] / r1[valid]
    muSorted = np.sort(mu)
    n = len(muSorted)

    # Cumulative distribution curve
    fEmp = np.arange(1, n + 1) / n

    # The mathematical law: -log(1 - F(mu)) = d * log(mu)
    # The slope 'd' is the intrinsic dimension!
    xVal = np.log(muSorted[:-1])
    yVal = -np.log(1.0 - fEmp[:-1] + 1e-12)

    # Fit a simple line through the origin to get the slope d
    dEst = float(np.sum(xVal * yVal) / (np.sum(xVal**2) + 1e-12))
    return max(1.0, dEst)


def estimatePcaVarianceDimensions(
    X: np.ndarray,
    thresholds: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Dict[str, int]:
    """Checks how many flat, straight-line PCA directions are needed to capture 80%, 90%, 95% of data spread."""
    pca = PCA().fit(X)
    cumVar = np.cumsum(pca.explained_variance_ratio_)

    dims = {}
    for th in thresholds:
        idx = int(np.searchsorted(cumVar, th)) + 1
        dims[f"pca_var_{int(th * 100)}pct"] = idx
    return dims


def estimateIntrinsicDimension(X: np.ndarray) -> Dict[str, float]:
    """Runs both Two-NN (curved shape dimension) and PCA (flat dimension) and returns a summary."""
    twoNnDim = estimateTwoNnDimension(X)
    pcaDims = estimatePcaVarianceDimensions(X)

    res = {"two_nn_dimension": twoNnDim}
    res.update(pcaDims)
    return res


# Aliases for backward compatibility
estimate_two_nn_dimension = estimateTwoNnDimension
estimate_pca_variance_dimensions = estimatePcaVarianceDimensions
estimate_intrinsic_dimension = estimateIntrinsicDimension
