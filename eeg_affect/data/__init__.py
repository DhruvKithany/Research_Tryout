"""Data loading, subject-independent splitting, and preprocessing."""

from .loader import (
    loadEegFeatures,
    loadParticipantsInfo,
    loadRawEeg,
    loadChannelLocations,
    findFeaturesFile,
    load_eeg_features,
    load_participants_info,
    load_raw_eeg,
    load_channel_locations,
    find_features_file,
)
from .split import (
    SubjectGroupSplitter,
    createCrossValidationSplits,
    create_cross_validation_splits,
)
from .preprocessor import EEGPreprocessor

__all__ = [
    "loadEegFeatures",
    "loadParticipantsInfo",
    "loadRawEeg",
    "loadChannelLocations",
    "findFeaturesFile",
    "load_eeg_features",
    "load_participants_info",
    "load_raw_eeg",
    "load_channel_locations",
    "find_features_file",
    "SubjectGroupSplitter",
    "createCrossValidationSplits",
    "create_cross_validation_splits",
    "EEGPreprocessor",
]
