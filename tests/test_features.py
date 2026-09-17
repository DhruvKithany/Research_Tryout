"""Unit tests for Frontal Alpha Asymmetry and band feature groupings."""

import numpy as np
import pandas as pd

from eeg_affect.data.loader import load_eeg_features
from eeg_affect.features.asymmetry import compute_frontal_alpha_asymmetry
from eeg_affect.features.bands import group_features_by_band, group_features_by_region
from eeg_affect.features.statistics import compute_hjorth_parameters, compute_shannon_entropy


def test_frontal_alpha_asymmetry():
    X, y, groups, meta = load_eeg_features(target="cowen")
    faa = compute_frontal_alpha_asymmetry(X, "AF3", "AF4")
    assert len(faa) == len(X)
    assert not np.any(np.isnan(faa))
    assert not np.any(np.isinf(faa))


def test_feature_groupings():
    X, y, groups, meta = load_eeg_features(target="cowen")
    feat_cols = list(X.columns)

    band_map = group_features_by_band(feat_cols)
    assert "alpha" in band_map
    assert "beta" in band_map
    assert len(band_map["alpha"]) == 14  # 1 per channel

    region_map = group_features_by_region(feat_cols)
    assert "Prefrontal" in region_map
    assert "Frontal" in region_map
    assert "Occipital" in region_map


def test_signal_statistics():
    rng = np.random.RandomState(42)
    sig = rng.randn(512)

    act, mob, comp = compute_hjorth_parameters(sig)
    assert act > 0
    assert mob > 0
    assert comp > 0

    ent = compute_shannon_entropy(sig)
    assert ent > 0
