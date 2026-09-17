"""Visualizing scalp topomaps, FAA asymmetry, and raw EEG traces."""

import sys
from pathlib import Path

# Add project root to sys.path
projectRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projectRoot))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_affect.config import (
    CHANNELS_14,
    COWEN_27_EMOTIONS,
    DATA_DIR,
    FIGURES_DIR,
)
from eeg_affect.data.loader import (
    loadEegFeatures,
    loadRawEeg,
)
from eeg_affect.features.asymmetry import (
    computeFrontalAlphaAsymmetry,
    summarizeFaaByEmotion,
)
from eeg_affect.features.bands import BAND_PREFIXES
from eeg_affect.visualization.neuro_plots import plotFrontalAlphaAsymmetry
from eeg_affect.visualization.topomap import plotMultibandTopomaps, plotTopomap


def main():
    print("Running Script 01: Neurobiological Visualizations...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load feature dataset
    print("Loading extracted feature dataset...")
    X, y, groups, metadata = loadEegFeatures(target="cowen")
    print(f"Loaded {len(X)} samples from {len(np.unique(groups))} participants across 27 emotions.")

    # 2. Frontal Alpha Asymmetry (FAA)
    print("Computing Frontal Alpha Asymmetry (AF4 vs AF3)...")
    # In features, AF3 is ch 1 (bpa_1) and AF4 is ch 14 (bpa_14)
    faa = computeFrontalAlphaAsymmetry(X, "AF3", "AF4")
    faaFigPath = FIGURES_DIR / "01_frontal_alpha_asymmetry.png"
    figFaa = plotFrontalAlphaAsymmetry(
        faa,
        metadata["CowenID"],
        title="Frontal Alpha Asymmetry (AF4 - AF3) Across 27 Emotional States",
        save_path=str(faaFigPath),
    )
    plt.close(figFaa)
    print(f"Saved FAA plot -> {faaFigPath}")

    # 3. Multi-band scalp topomaps for contrasting emotions
    print("Generating multi-band scalp topomaps...")
    # Contrasting emotions: Joy (20), Calmness (10), Anger (5), Sadness (24)
    contrastEmotions = [
        (20, "Joy_HVHA"),
        (10, "Calmness_HVLA"),
        (5, "Anger_LVHA"),
        (24, "Sadness_LVLA"),
    ]

    for eid, ename in contrastEmotions:
        mask = (metadata["CowenID"] == eid)
        subX = X[mask]

        bandDict = {}
        for bname, prefix in BAND_PREFIXES.items():
            if bname == "ratio_beta_alpha":
                continue
            chVals = {}
            for chIdx, chName in enumerate(CHANNELS_14, start=1):
                col = f"{prefix}_{chIdx}"
                if col in subX.columns:
                    chVals[chName] = float(np.mean(subX[col]))
            bandDict[bname] = chVals

        topoPath = FIGURES_DIR / f"01_topomap_multiband_{ename.lower()}.png"
        figTopo = plotMultibandTopomaps(
            bandDict,
            emotion_name=f"{COWEN_27_EMOTIONS[eid]} ({ename})",
            save_path=str(topoPath),
        )
        plt.close(figTopo)
        print(f"Saved multi-band topomap for {ename} -> {topoPath}")

    # 4. Raw EEG Traces and Power Spectral Density from curated sample
    sampleRawFile = DATA_DIR / "sample_raw" / "1_20.0.txt"
    if sampleRawFile.exists():
        print("\nPlotting raw EEG time-series and PSD for Participant 1 (Joy)...")
        rawSig, fs = loadRawEeg(1, 20)
        # Take first 4 seconds (1024 samples)
        t = np.arange(1024) / fs
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=200)

        # Plot 4 channels: AF3, AF4, T7, O1
        selectedChannels = [("AF3", 0), ("AF4", 13), ("T7", 4), ("O1", 6)]
        for chName, chIdx in selectedChannels:
            # Demean for visualization
            sigSegment = rawSig[:1024, chIdx] - np.mean(rawSig[:1024, chIdx])
            ax1.plot(t, sigSegment, label=chName, linewidth=1.1, alpha=0.85)

        ax1.set_title("Raw EEG Multi-Channel Time Series (First 4 Seconds)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (seconds)", fontsize=10)
        ax1.set_ylabel(r"Amplitude ($\mu$V)", fontsize=10)
        ax1.legend(loc="upper right", ncol=4, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # PSD Welch
        import scipy.signal
        for chName, chIdx in selectedChannels:
            freqs, psd = scipy.signal.welch(rawSig[:, chIdx], fs=fs, nperseg=int(fs * 2))
            idxFreq = (freqs >= 1.0) & (freqs <= 45.0)
            ax2.semilogy(freqs[idxFreq], psd[idxFreq], label=chName, linewidth=1.2)

        # Highlight bands
        ax2.axvspan(1, 4, color="#3498db", alpha=0.1, label="Delta (1-4Hz)")
        ax2.axvspan(4, 8, color="#2ecc71", alpha=0.1, label="Theta (4-8Hz)")
        ax2.axvspan(8, 13, color="#f1c40f", alpha=0.15, label="Alpha (8-13Hz)")
        ax2.axvspan(13, 30, color="#e67e22", alpha=0.1, label="Beta (13-30Hz)")
        ax2.axvspan(30, 45, color="#e74c3c", alpha=0.1, label="Gamma (30-45Hz)")

        ax2.set_title("Power Spectral Density (Welch PSD, 1-45 Hz)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontsize=10)
        ax2.set_ylabel(r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)", fontsize=10)
        ax2.legend(loc="upper right", ncol=3, fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        rawFigPath = FIGURES_DIR / "01_raw_eeg_traces_psd.png"
        fig.savefig(rawFigPath, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved raw EEG time-series and PSD plot -> {rawFigPath}")

    print("\nNeurobiological visualization suite completed successfully!")


if __name__ == "__main__":
    main()
