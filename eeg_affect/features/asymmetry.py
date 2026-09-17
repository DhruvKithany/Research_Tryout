"""Frontal Alpha Asymmetry (FAA) and hemispheric lateralization metrics."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ASYMMETRY_PAIRS, CHANNEL_NAME_TO_INDEX, COWEN_27_EMOTIONS


def computeFrontalAlphaAsymmetry(
    df: pd.DataFrame,
    leftCh: str = "AF3",
    rightCh: str = "AF4",
    eps: float = 1e-6,
) -> pd.Series:
    """Calculates Left vs Right forehead alpha wave difference (FAA).

    AF3 is the electrode on the left forehead (channel 1).
    AF4 is the electrode on the right forehead (channel 14).

    Formula:
        FAA = log(AF4) - log(AF3)

    Returns:
        pd.Series: Positive numbers mean positive/approach feelings,
                   negative numbers mean avoidance/negative feelings.
    """
    leftIdx = CHANNEL_NAME_TO_INDEX[leftCh]
    rightIdx = CHANNEL_NAME_TO_INDEX[rightCh]

    # In our dataset, alpha power columns are named like 'bpa_1', 'bpa_14'
    leftCol = f"bpa_{leftIdx}"
    rightCol = f"bpa_{rightIdx}"

    if leftCol not in df.columns or rightCol not in df.columns:
        raise ValueError(f"Missing columns {leftCol} or {rightCol} in data.")

    # Guard against dividing or taking log of zero
    leftPower = np.maximum(df[leftCol].values, eps)
    rightPower = np.maximum(df[rightCol].values, eps)

    # Difference of logs is the log of the ratio: log(Right / Left)
    faa = np.log(rightPower) - np.log(leftPower)
    return pd.Series(faa, index=df.index, name=f"FAA_{leftCh}_{rightCh}")


def computeAllAsymmetries(
    df: pd.DataFrame,
    band: str = "bpa",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Calculates left-vs-right balance across all 7 electrode pairs on the head.

    For each pair of sensors (one on left side, one on right side):
        Asymmetry = log(Right) - log(Left)
    """
    asymDict = {}
    for leftCh, rightCh in ASYMMETRY_PAIRS:
        lIdx = CHANNEL_NAME_TO_INDEX[leftCh]
        rIdx = CHANNEL_NAME_TO_INDEX[rightCh]
        lCol = f"{band}_{lIdx}"
        rCol = f"{band}_{rIdx}"

        if lCol in df.columns and rCol in df.columns:
            lVal = np.maximum(df[lCol].values, eps)
            rVal = np.maximum(df[rCol].values, eps)
            asymDict[f"Asym_{band.upper()}_{leftCh}_{rightCh}"] = np.log(rVal) - np.log(lVal)

    return pd.DataFrame(asymDict, index=df.index)


def summarizeFaaByEmotion(df: pd.DataFrame) -> pd.DataFrame:
    """Averages the Frontal Alpha Asymmetry score for each of the 27 emotion classes."""
    faaAf = computeFrontalAlphaAsymmetry(df, "AF3", "AF4")
    faaF = computeFrontalAlphaAsymmetry(df, "F3", "F4")

    workDf = pd.DataFrame({
        "CowenID": df["Emo_Label_Cowen(27)"],
        "FAA_AF": faaAf,
        "FAA_F": faaF,
    })
    workDf["Emotion"] = workDf["CowenID"].map(COWEN_27_EMOTIONS)

    # Group by emotion and compute mean, standard deviation, and standard error
    summary = workDf.groupby(["CowenID", "Emotion"])[["FAA_AF", "FAA_F"]].agg(["mean", "std", "sem"])
    return summary


# Aliases for backward compatibility
compute_frontal_alpha_asymmetry = computeFrontalAlphaAsymmetry
compute_all_asymmetries = computeAllAsymmetries
summarize_faa_by_emotion = summarizeFaaByEmotion
