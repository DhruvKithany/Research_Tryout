"""Class-conditional synthetic EEG feature generation and fidelity evaluation."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
projectRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projectRoot))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from eeg_affect.config import COWEN_27_EMOTIONS, FIGURES_DIR
from eeg_affect.data.loader import loadEegFeatures
from eeg_affect.data.preprocessor import EEGPreprocessor
from eeg_affect.data.split import SubjectGroupSplitter
from eeg_affect.models.generator import EEGFeatureGenerator


def main():
    print("Running Script 04: Class-Conditional EEG Feature Generation & Fidelity...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading features and setting up subject-independent split...")
    X_df, yCowen, groups, metadata = loadEegFeatures(target="cowen")

    splitter = SubjectGroupSplitter(test_size=0.15, val_size=0.0, random_state=42)
    splitIndices = splitter.split(X_df, yCowen, groups)
    trainIdx, testIdx = splitIndices["train"], splitIndices["test"]

    prep = EEGPreprocessor(scaler="robust")
    X_train = prep.fitTransform(X_df.iloc[trainIdx])
    X_test = prep.transform(X_df.iloc[testIdx])
    y_train = yCowen[trainIdx]
    y_test = yCowen[testIdx]

    # 2. Fit class-conditional generator
    print("Fitting class-conditional generator with Ledoit-Wolf shrinkage...")
    generator = EEGFeatureGenerator(random_state=42)
    generator.fit(X_train, y_train)

    # 3. Generate synthetic data & evaluate Fréchet Distance
    print("Generating synthetic samples and evaluating Fréchet Distance...")
    evalEmotions = [
        (19, "Joy"),       # 0-indexed 19 = Cowen 20 (Joy)
        (9,  "Calmness"),  # 0-indexed 9 = Cowen 10 (Calmness)
        (4,  "Anger"),     # 0-indexed 4 = Cowen 5 (Anger)
        (23, "Sadness"),   # 0-indexed 23 = Cowen 24 (Sadness)
    ]

    fdMetrics = {}
    for cIdx, eName in evalEmotions:
        maskTest = (y_test == cIdx)
        X_test_c = X_test[maskTest]
        if len(X_test_c) > 5:
            fd = generator.computeFrechetDistance(cIdx, X_test_c)
            fdMetrics[eName] = round(fd, 3)
            print(f"--> Fréchet Distance for {eName:<10}: {fd:.3f}")

    # 4. Manifold Overlap Visualization (Real vs. Synthetic)
    print("Visualizing Real vs. Synthetic Feature Manifolds...")
    selectedCIndices = [c for c, _ in evalEmotions]
    testMask = np.isin(y_test, selectedCIndices)
    X_real_subset = X_test[testMask]
    y_real_subset = y_test[testMask]

    pca = PCA(n_components=2, random_state=42).fit(X_real_subset)
    real2D = pca.transform(X_real_subset)

    # Sample equal number of synthetic points per emotion
    synthSamples = []
    synthLabels = []
    for cIdx, eName in evalEmotions:
        nPts = np.sum(y_real_subset == cIdx)
        sC = generator.sample(cIdx, n_samples=nPts)
        synthSamples.append(sC)
        synthLabels.append(np.full(nPts, cIdx, dtype=int))

    X_synth_all = np.vstack(synthSamples)
    y_synth_all = np.concatenate(synthLabels)
    synth2D = pca.transform(X_synth_all)

    # Plot Two-Panel Comparison: Real vs Synthetic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    colorMap = {
        19: "#2ecc71",  # Joy: Green
        9:  "#3498db",  # Calmness: Blue
        4:  "#e74c3c",  # Anger: Red
        23: "#9b59b6",  # Sadness: Purple
    }

    # Left: Real Test Samples
    for cIdx, eName in evalEmotions:
        mask = (y_real_subset == cIdx)
        ax1.scatter(
            real2D[mask, 0],
            real2D[mask, 1],
            c=colorMap[cIdx],
            label=eName,
            alpha=0.6,
            s=35,
            edgecolors="none",
        )
    ax1.set_title("Empirical EEG Features (Held-Out Test Subjects)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("PC 1", fontsize=10)
    ax1.set_ylabel("PC 2", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(title="Emotion", fontsize=9)

    # Right: Synthetic Generated Samples
    for cIdx, eName in evalEmotions:
        mask = (y_synth_all == cIdx)
        ax2.scatter(
            synth2D[mask, 0],
            synth2D[mask, 1],
            c=colorMap[cIdx],
            label=f"{eName} (Synth)",
            marker="^",
            alpha=0.6,
            s=35,
            edgecolors="none",
        )
    ax2.set_title("Synthetic Conditional Generated Features", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PC 1", fontsize=10)
    ax2.set_ylabel("PC 2", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(title="Emotion", fontsize=9)

    plt.tight_layout()
    genFigPath = FIGURES_DIR / "04_generative_feature_fidelity.png"
    fig.savefig(genFigPath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Generative Fidelity Plot -> {genFigPath}")

    # Save metrics JSON
    outJson = {
        "frechet_distances": fdMetrics,
        "n_features": generator.n_features_,
        "n_classes_fitted": len(generator.classes_),
    }
    jsonPath = FIGURES_DIR / "04_generative_metrics.json"
    with open(jsonPath, "w") as f:
        json.dump(outJson, f, indent=4)
    print(f"Saved Generative Metrics JSON -> {jsonPath}")
    print("\nGenerative EEG modeling completed successfully!")


if __name__ == "__main__":
    main()
