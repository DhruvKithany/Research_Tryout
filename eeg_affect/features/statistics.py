"""Signal statistics, Hjorth parameters, and spectral power extraction for raw EEG."""

from typing import Dict, Tuple

import numpy as np
import scipy.signal
import scipy.stats

from ..config import CHANNELS_14, FREQUENCY_BANDS


def computeHjorthParameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """Calculates Hjorth parameters: Activity (energy), Mobility (average frequency), and Complexity (shape changes).

    Args:
        signal: 1D array of voltage numbers from one sensor over time.

    Returns:
        (Activity, Mobility, Complexity)
    """
    # 1. Activity is simply the variance of the signal
    activity = float(np.var(signal))
    if activity < 1e-12:
        return 0.0, 0.0, 0.0

    # 2. First derivative (rate of change between consecutive time points)
    d1 = np.diff(signal)
    varD1 = float(np.var(d1))
    # Mobility is the ratio of derivative variance to signal variance
    mobility = float(np.sqrt(varD1 / activity))

    # 3. Second derivative (acceleration of the signal wave)
    d2 = np.diff(d1)
    varD2 = float(np.var(d2))
    if varD1 < 1e-12:
        return activity, mobility, 0.0

    mobilityD1 = float(np.sqrt(varD2 / varD1))
    # Complexity compares mobility of the slope to mobility of the wave
    complexity = float(mobilityD1 / (mobility + 1e-12))

    return activity, mobility, complexity


def computeShannonEntropy(signal: np.ndarray, nBins: int = 50) -> float:
    """Measures how unpredictable or disordered the signal voltage distribution is in bits."""
    counts, _ = np.histogram(signal, bins=nBins, density=True)
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts + 1e-12)))


def computeBandPowers(
    signal: np.ndarray,
    fs: float = 256.0,
) -> Dict[str, float]:
    """Calculates how much power is in each frequency band using Welch's periodogram."""
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))

    powers = {}
    totalPower = np.sum(psd) + 1e-12
    for bandName, (low, high) in FREQUENCY_BANDS.items():
        idxBand = np.logical_and(freqs >= low, freqs <= high)
        bandPower = float(np.sum(psd[idxBand]))
        powers[f"power_{bandName}"] = bandPower
        powers[f"rel_power_{bandName}"] = float(bandPower / totalPower)

    # Beta to Alpha ratio (higher = more alert/stimulated, lower = more relaxed)
    powers["ratio_beta_alpha"] = float(powers["power_beta"] / (powers["power_alpha"] + 1e-12))
    return powers


def extractTimeseriesSummaryFeatures(
    rawMatrix: np.ndarray,
    fs: float = 256.0,
) -> Dict[str, float]:
    """Extracts a full dictionary of statistical, Hjorth, and frequency features across all 14 sensors."""
    feats = {}
    for chIdx, chName in enumerate(CHANNELS_14):
        sig = rawMatrix[:, chIdx]
        feats[f"mean_{chName}"] = float(np.mean(sig))
        feats[f"std_{chName}"] = float(np.std(sig))
        feats[f"skew_{chName}"] = float(scipy.stats.skew(sig))
        feats[f"kurt_{chName}"] = float(scipy.stats.kurtosis(sig))

        act, mob, comp = computeHjorthParameters(sig)
        feats[f"hjorth_act_{chName}"] = act
        feats[f"hjorth_mob_{chName}"] = mob
        feats[f"hjorth_comp_{chName}"] = comp

        bp = computeBandPowers(sig, fs=fs)
        for k, v in bp.items():
            feats[f"{k}_{chName}"] = v

    return feats


# Aliases for backward compatibility
computeSignalFeatures = extractTimeseriesSummaryFeatures
compute_signal_features = computeSignalFeatures
compute_hjorth_parameters = computeHjorthParameters
compute_shannon_entropy = computeShannonEntropy
compute_band_powers = computeBandPowers
extract_timeseries_summary_features = extractTimeseriesSummaryFeatures
