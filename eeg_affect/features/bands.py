"""Spectral band definitions, cortical region groupings, and feature name mapping."""

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import CHANNEL_INDEX, CHANNELS_14

# Grouping the 14 sensors by their physical location on the head
CORTICAL_REGIONS: Dict[str, List[str]] = {
    "Prefrontal": ["AF3", "AF4"],
    "Frontal": ["F7", "F3", "FC5", "FC6", "F4", "F8"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["P7", "P8"],
    "Occipital": ["O1", "O2"],
}

# Prefix names in the dataset for each wave speed
BAND_PREFIXES = {
    "delta": "bpd",
    "theta": "bpt",
    "alpha": "bpa",
    "beta":  "bpb",
    "gamma": "bpg",
    "ratio_beta_alpha": "rba",
}


def extractBandPowerDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the table to keep only the frequency power columns (Delta through Gamma)."""
    cols = []
    for prefix in BAND_PREFIXES.values():
        for chIdx in range(1, 15):
            cname = f"{prefix}_{chIdx}"
            if cname in df.columns:
                cols.append(cname)
    return df[cols].copy()


def groupFeaturesByBand(featureNames: List[str]) -> Dict[str, List[str]]:
    """Sorts a list of feature names into buckets by wave speed (delta, theta, alpha, beta, gamma)."""
    bandMap = {b: [] for b in BAND_PREFIXES.keys()}
    bandMap["other"] = []

    prefixToBand = {v: k for k, v in BAND_PREFIXES.items()}

    for f in featureNames:
        prefix = f.split("_")[0]
        if prefix in prefixToBand:
            bandMap[prefixToBand[prefix]].append(f)
        else:
            bandMap["other"].append(f)

    return bandMap


def groupFeaturesByRegion(featureNames: List[str]) -> Dict[str, List[str]]:
    """Sorts a list of feature names into buckets by head location (Prefrontal, Frontal, etc.)."""
    regionMap = {r: [] for r in CORTICAL_REGIONS.keys()}
    regionMap["demographics"] = []

    # Map each sensor number (1 to 14) to its head area
    idxToRegion = {}
    for region, chList in CORTICAL_REGIONS.items():
        for ch in chList:
            for idx, cname in CHANNEL_INDEX.items():
                if cname == ch:
                    idxToRegion[idx] = region

    for f in featureNames:
        parts = f.split("_")
        try:
            chIdx = int(parts[-1])
            if chIdx in idxToRegion:
                regionMap[idxToRegion[chIdx]].append(f)
        except ValueError:
            regionMap["demographics"].append(f)

    return regionMap


# Aliases for backward compatibility
extract_band_power_dataframe = extractBandPowerDataFrame
group_features_by_band = groupFeaturesByBand
group_features_by_region = groupFeaturesByRegion
