"""Class-conditional generative models with Ledoit-Wolf shrinkage and Fréchet Distance evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf


class EEGFeatureGenerator:
    """Creates synthetic EEG feature vectors for any of the 27 emotions."""

    def __init__(self, randomState: int = 42, **kwargs):
        """Initializes the synthetic generator.

        Args:
            randomState: Random seed so synthetic numbers can be reproduced.
        """
        self.randomState = kwargs.get("random_state", randomState)
        self.random_state = self.randomState
        self.rng = np.random.RandomState(self.randomState)
        self.classMeans_: Dict[int, np.ndarray] = {}
        self.classCovs_: Dict[int, np.ndarray] = {}
        self.classes_: List[int] = []
        self.nFeatures_: Optional[int] = None
        self.class_means_ = self.classMeans_
        self.class_covs_ = self.classCovs_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGFeatureGenerator":
        """Learns the average numbers and stabilized spread for each emotion class."""
        self.classes_ = sorted(list(np.unique(y)))
        self.nFeatures_ = X.shape[1]
        self.n_features_ = self.nFeatures_

        for c in self.classes_:
            mask = (y == c)
            Xc = X[mask]

            meanC = np.mean(Xc, axis=0)
            # Apply Ledoit-Wolf shrinkage to stabilize the covariance matrix
            lw = LedoitWolf().fit(Xc)
            covC = lw.covariance_

            self.classMeans_[c] = meanC
            self.classCovs_[c] = covC

        return self

    def sample(self, c: int, nSamples: int = 10, **kwargs) -> np.ndarray:
        """Generates nSamples fake feature vectors for a specific emotion class."""
        n = kwargs.get("n_samples", nSamples)
        if c not in self.classMeans_:
            raise ValueError(f"Emotion class {c} was not seen during fitting.")

        mean = self.classMeans_[c]
        cov = self.classCovs_[c]

        # Draw random points from the stabilized multi-dimensional bell curve
        return self.rng.multivariate_normal(mean, cov, size=n)

    def sampleAll(self, nSamplesPerClass: int = 20, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a balanced fake dataset with equal samples for all 27 emotions."""
        n = kwargs.get("n_samples_per_class", nSamplesPerClass)
        xSynthList = []
        ySynthList = []

        for c in self.classes_:
            samples = self.sample(c, nSamples=n)
            xSynthList.append(samples)
            ySynthList.append(np.full(n, c, dtype=int))

        return np.vstack(xSynthList), np.concatenate(ySynthList)

    def computeFrechetDistance(self, c: int, xReal: Optional[np.ndarray] = None, **kwargs) -> float:
        """Measures how close the fake data cloud is to the real data cloud (lower is better).

        Fréchet distance compares both the difference in averages and the difference in spread.
        """
        import scipy.linalg

        real = kwargs.get("X_real", xReal)
        if real is None:
            raise ValueError("xReal (or X_real) must be provided.")

        if c not in self.classMeans_:
            raise ValueError(f"Class {c} not observed during fitting.")

        mu1 = self.classMeans_[c]
        sigma1 = self.classCovs_[c]

        mu2 = np.mean(real, axis=0)
        lw2 = LedoitWolf().fit(real)
        sigma2 = lw2.covariance_

        # 1. Difference in average points
        diff = mu1 - mu2
        meanTerm = float(np.dot(diff, diff))

        # 2. Difference in shape / covariance spread
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        covTerm = float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
        return float(np.sqrt(max(0.0, meanTerm + covTerm)))

    # Backward-compatible aliases
    sample_all = sampleAll
    compute_frechet_distance = computeFrechetDistance
