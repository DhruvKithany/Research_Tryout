"""Data loaders for EEG features, metadata, and raw time-series."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import (
    CHANNELS_14,
    DATA_DIR,
    RAW_REPO_DIR,
    COWEN_27_EMOTIONS,
    get_quadrant_label,
)


def findFeaturesFile(customPath: Optional[Union[str, Path]] = None) -> Path:
    """Finds the path to eeg_features_extracted.csv."""
    if customPath:
        p = Path(customPath)
        if p.exists():
            return p
        raise FileNotFoundError(f"Feature CSV not found at: {customPath}")

    candidates = [
        DATA_DIR / "eeg_features_extracted.csv",
        RAW_REPO_DIR / "training" / "eeg_features_extracted.csv",
        Path("eeg_features_extracted.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not locate eeg_features_extracted.csv in any candidate path: {candidates}"
    )

find_features_file = findFeaturesFile


def loadEegFeatures(
    csvPath: Optional[Union[str, Path]] = None,
    target: str = "cowen",
    includeDemographics: bool = False,
    subsampleParticipants: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads extracted EEG features and prepares features (X), labels (y), groups, and metadata.

    Args:
        csv_path: Optional path to the CSV file.
        target: Target label type:
            - 'cowen': 27 fine-grained emotion classes (1 to 27).
            - 'ekman': 5 coarse Ekman classes.
            - 'quadrant': 4 Valence-Arousal quadrants (0: HVHA, 1: HVLA, 2: LVHA, 3: LVLA).
            - 'valence': Binary valence (0: Negative, 1: Positive).
            - 'arousal': Binary arousal (0: Low, 1: High).
        include_demographics: If True, keep Age, Gender, Nation as input features.
        subsample_participants: If specified, only keep data for the first N participants.

    Returns:
        X: pd.DataFrame of EEG feature columns.
        y: np.ndarray of target labels (0-indexed integers).
        groups: np.ndarray of ParticipantID for group-aware splitting.
        metadata: pd.DataFrame with sample-level annotations (ParticipantID, EmotionName, etc.).
    """
    filePath = findFeaturesFile(csvPath)
    df = pd.read_csv(filePath)

    if subsampleParticipants is not None:
        uniquePids = sorted(df["ParticipantID"].unique())[:subsampleParticipants]
        df = df[df["ParticipantID"].isin(uniquePids)].copy()

    # Targets & metadata
    cowenIds = df["Emo_Label_Cowen(27)"].values
    groups = df["ParticipantID"].values

    if target == "cowen":
        # Zero-index cowen (1..27 -> 0..26)
        y = cowenIds - 1
    elif target == "ekman":
        y = df["Emo_Label_Ekman(6)"].values - 1
    elif target == "quadrant":
        y = np.array([get_quadrant_label(e) for e in cowenIds], dtype=int)
    elif target == "valence":
        quads = [get_quadrant_label(e) for e in cowenIds]
        y = np.array([1 if q in (0, 1) else 0 for q in quads], dtype=int)
    elif target == "arousal":
        quads = [get_quadrant_label(e) for e in cowenIds]
        y = np.array([1 if q in (0, 2) else 0 for q in quads], dtype=int)
    else:
        raise ValueError(f"Unknown target specification: {target}")

    # Build metadata dataframe
    metadata = pd.DataFrame({
        "ParticipantID": groups,
        "CowenID": cowenIds,
        "EmotionName": [COWEN_27_EMOTIONS.get(cid, f"Emotion_{cid}") for cid in cowenIds],
        "Quadrant": [get_quadrant_label(cid) for cid in cowenIds],
        "Age": df["Age"].values if "Age" in df.columns else np.nan,
        "Gender": df["Gender"].values if "Gender" in df.columns else np.nan,
        "Nation": df["Nation"].values if "Nation" in df.columns else np.nan,
    })

    # Drop non-feature columns
    dropCols = [
        "eeg_component_number",
        "Emo_Label_Ekman(6)",
        "Emo_Label_Cowen(27)",
        "ParticipantID",
    ]
    if not includeDemographics:
        dropCols.extend(["Age", "Gender", "Nation"])

    featureCols = [c for c in df.columns if c not in dropCols]
    X = df[featureCols].copy()

    return X, y, groups, metadata

load_eeg_features = loadEegFeatures


def loadParticipantsInfo(csvPath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads participants_info.csv demographic summary."""
    candidates = [
        DATA_DIR / "participants_info.csv",
        RAW_REPO_DIR / "participants_info.csv",
    ]
    if csvPath:
        candidates.insert(0, Path(csvPath))

    for c in candidates:
        if c.exists():
            return pd.read_csv(c)

    raise FileNotFoundError("participants_info.csv could not be found.")

load_participants_info = loadParticipantsInfo


def loadChannelLocations(cedPath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads emotivX_channels_location.ced electrode positions."""
    candidates = [
        DATA_DIR / "emotivX_channels_location.ced",
        RAW_REPO_DIR / "emotivX_channels_location.ced",
    ]
    if cedPath:
        candidates.insert(0, Path(cedPath))

    for c in candidates:
        if c.exists():
            return pd.read_csv(c, sep=r"\s+")

    raise FileNotFoundError("emotivX_channels_location.ced could not be found.")

load_channel_locations = loadChannelLocations


def loadRawEeg(
    participantId: int,
    emotionId: int,
    rawDir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, float]:
    """Loads a single raw 14-channel EEG recording file.

    File format is {Participant_ID}_{Emotion_ID}.0.txt with 14 tab-separated values per line.
    Sampling rate is 256 Hz.

    Returns:
        signals: np.ndarray of shape (n_samples, 14).
        sampling_rate: 256.0 Hz.
    """
    candidates = [
        DATA_DIR / "sample_raw",
        RAW_REPO_DIR / "eeg_raw",
    ]
    if rawDir:
        candidates.insert(0, Path(rawDir))

    filename = f"{participantId}_{emotionId}.0.txt"
    foundPath = None
    for d in candidates:
        candidateFile = d / filename
        if candidateFile.exists():
            foundPath = candidateFile
            break

    if not foundPath:
        raise FileNotFoundError(
            f"Raw EEG recording {filename} not found in searched directories: {candidates}"
        )

    signals = np.loadtxt(foundPath, delimiter="\t")
    return signals, 256.0

load_raw_eeg = loadRawEeg
