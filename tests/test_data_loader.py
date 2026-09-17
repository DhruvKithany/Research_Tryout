"""Unit tests for data loading and subject-independent splitting."""

import numpy as np
import pandas as pd

from eeg_affect.data.loader import (
    loadChannelLocations,
    loadEegFeatures,
    loadParticipantsInfo,
    load_channel_locations,
    load_eeg_features,
    load_participants_info,
)
from eeg_affect.data.split import (
    SubjectGroupSplitter,
    createCrossValidationSplits,
    create_cross_validation_splits,
)


def test_load_channel_locations():
    ced = loadChannelLocations()
    assert len(ced) == 14
    assert "labels" in ced.columns
    assert "AF3" in ced["labels"].values
    assert "AF4" in ced["labels"].values


def test_load_participants_info():
    pinfo = load_participants_info()
    assert len(pinfo) >= 88
    assert "Participant_ID" in pinfo.columns


def test_load_eeg_features_cowen():
    X, y, groups, meta = load_eeg_features(target="cowen")
    assert len(X) > 1000
    assert X.shape[1] == 462  # 33 features x 14 channels
    assert len(np.unique(y)) == 27
    assert len(np.unique(groups)) == 88
    assert len(meta) == len(X)


def test_load_eeg_features_quadrant():
    X, y, groups, meta = load_eeg_features(target="quadrant")
    assert set(np.unique(y)).issubset({0, 1, 2, 3})


def test_subject_independent_split():
    X, y, groups, meta = load_eeg_features(target="quadrant")
    splitter = SubjectGroupSplitter(test_size=0.15, val_size=0.10, random_state=42)
    splits = splitter.split(X, y, groups)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # Ensure mutual exclusivity of subjects
    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    test_subjects = set(groups[test_idx])

    assert train_subjects.isdisjoint(test_subjects)
    assert train_subjects.isdisjoint(val_subjects)
    assert val_subjects.isdisjoint(test_subjects)
