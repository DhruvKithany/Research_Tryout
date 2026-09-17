"""Subject-independent decoding benchmark and feature importance."""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier

from eeg_affect.config import (
    CHANNELS_14,
    COWEN_27_EMOTIONS,
    FIGURES_DIR,
    QUADRANT_NAMES,
)
from eeg_affect.data.loader import loadEegFeatures
from eeg_affect.data.preprocessor import EEGPreprocessor
from eeg_affect.data.split import SubjectGroupSplitter
from eeg_affect.evaluation.benchmark import runModelBenchmark
from eeg_affect.features.bands import groupFeaturesByBand, groupFeaturesByRegion
from eeg_affect.models.baselines import CosineNearestCentroid
from eeg_affect.visualization.confusion import plotConfusionMatrix


def main():
    print("Running Script 03: Subject-Independent Decoding Benchmark...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Benchmark Task A: 4-Class Valence-Arousal Quadrants
    print("Benchmarking Task A: 4-Class Valence-Arousal Quadrant Decoding...")

    X_df, yQuad, groups, metaQuad = loadEegFeatures(target="quadrant")
    print(f"Loaded {len(X_df)} samples across {len(np.unique(groups))} participants.")

    # Subject-independent split (leave participants out)
    splitter = SubjectGroupSplitter(test_size=0.15, val_size=0.0, random_state=42)
    splitIndices = splitter.split(X_df, yQuad, groups)
    trainIdx, testIdx = splitIndices["train"], splitIndices["test"]

    print(f"Subject-Independent Split: Train {len(trainIdx)} samples ({len(np.unique(groups[trainIdx]))} subjects), "
          f"Test {len(testIdx)} samples ({len(np.unique(groups[testIdx]))} subjects).")

    # Robust scaling
    preprocessor = EEGPreprocessor(scaler="robust")
    X_train = preprocessor.fitTransform(X_df.iloc[trainIdx])
    X_test = preprocessor.transform(X_df.iloc[testIdx])
    y_train = yQuad[trainIdx]
    y_test = yQuad[testIdx]

    # Baseline models and neural network
    quadModels = {
        "NearestCentroid_Cosine": CosineNearestCentroid(),
        "Ridge": RidgeClassifier(alpha=10.0, random_state=42),
        "LogisticRegression": LogisticRegression(C=0.05, max_iter=300, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True, random_state=42),
    }

    dfResultsQuad, detailedQuad = runModelBenchmark(
        quadModels, X_train, y_train, X_test, y_test, verbose=True
    )
    print("\nTask A Results (Valence-Arousal Quadrants):")
    print(dfResultsQuad.to_string(index=False))

    # Plot normalized confusion matrix for best quadrant model
    bestQuadModelName = dfResultsQuad.iloc[0]["model"]
    yPredBestQuad = detailedQuad[bestQuadModelName]["y_pred"]
    cmQuadPath = FIGURES_DIR / "03_confusion_matrix_quadrants.png"
    figCmQ = plotConfusionMatrix(
        y_test,
        yPredBestQuad,
        labels=["HVHA", "HVLA", "LVHA", "LVLA"],
        title=f"Normalized Confusion Matrix: {bestQuadModelName} (Valence-Arousal)",
        save_path=str(cmQuadPath),
    )
    plt.close(figCmQ)
    print(f"Saved Quadrant Confusion Matrix -> {cmQuadPath}")

    # 2. Benchmark Task B: 27-Class Fine-Grained Cowen Emotion Classification
    print("Benchmarking Task B: 27-Class Fine-Grained Cowen Emotion Classification...")

    X_df_cowen, yCowen, groupsCowen, metaCowen = loadEegFeatures(target="cowen")
    splitCowen = splitter.split(X_df_cowen, yCowen, groupsCowen)
    trainCIdx, testCIdx = splitCowen["train"], splitCowen["test"]

    prepCowen = EEGPreprocessor(scaler="robust")
    X_train_c = prepCowen.fitTransform(X_df_cowen.iloc[trainCIdx])
    X_test_c = prepCowen.transform(X_df_cowen.iloc[testCIdx])
    y_train_c = yCowen[trainCIdx]
    y_test_c = yCowen[testCIdx]

    cowenModels = {
        "NearestCentroid_Cosine": CosineNearestCentroid(),
        "Ridge": RidgeClassifier(alpha=10.0, random_state=42),
        "LogisticRegression": LogisticRegression(C=0.05, max_iter=300, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
        "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100, early_stopping=True, random_state=42),
    }

    dfResultsCowen, detailedCowen = runModelBenchmark(
        cowenModels, X_train_c, y_train_c, X_test_c, y_test_c, verbose=True
    )
    print("\nTask B Results (27 Cowen Emotions):")
    print(dfResultsCowen.to_string(index=False))

    # Plot normalized confusion matrix for best 27-class model
    bestCowenModelName = dfResultsCowen.iloc[0]["model"]
    yPredBestCowen = detailedCowen[bestCowenModelName]["y_pred"]
    cmCowenPath = FIGURES_DIR / "03_confusion_matrix_cowen27.png"
    cowenLabels = [COWEN_27_EMOTIONS[i + 1] for i in range(27)]
    figCmC = plotConfusionMatrix(
        y_test_c,
        yPredBestCowen,
        labels=cowenLabels,
        title=f"Normalized Confusion Matrix: {bestCowenModelName} (27 Cowen Emotions)",
        save_path=str(cmCowenPath),
    )
    plt.close(figCmC)
    print(f"Saved 27-Class Confusion Matrix -> {cmCowenPath}")

    # 3. Feature Importance Analysis across Bands and Lobes
    print("Extracting Feature Importance across Bands and Lobes...")

    # Random Forest Gini feature importances
    rfModel = detailedCowen["RandomForest"]["model"]
    importances = rfModel.feature_importances_
    featNames = list(prepCowen.feature_names_)

    # Group by Spectral Band
    bandGrouped = groupFeaturesByBand(featNames)
    bandImp = {}
    for bname, fList in bandGrouped.items():
        if fList:
            indices = [featNames.index(f) for f in fList]
            bandImp[bname] = float(np.mean(importances[indices]))

    # Group by Cortical Region
    regionGrouped = groupFeaturesByRegion(featNames)
    regionImp = {}
    for rname, fList in regionGrouped.items():
        if fList and rname != "demographics":
            indices = [featNames.index(f) for f in fList]
            regionImp[rname] = float(np.mean(importances[indices]))

    # Two-panel barplot (Bands & Lobes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Panel 1: Spectral Bands
    bandsSorted = sorted(bandImp.items(), key=lambda x: x[1], reverse=True)
    bKeys = [k.capitalize() for k, _ in bandsSorted]
    bVals = [v for _, v in bandsSorted]
    sns.barplot(x=bKeys, y=bVals, hue=bKeys, ax=ax1, palette="mako", legend=False)
    ax1.set_title("Mean Feature Importance by Spectral Band", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Gini Impurity Reduction", fontsize=10)
    ax1.set_xlabel("Frequency Band", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: Cortical Lobes
    regionsSorted = sorted(regionImp.items(), key=lambda x: x[1], reverse=True)
    rKeys = [k for k, _ in regionsSorted]
    rVals = [v for _, v in regionsSorted]
    sns.barplot(x=rKeys, y=rVals, hue=rKeys, ax=ax2, palette="crest", legend=False)
    ax2.set_title("Mean Feature Importance by Cortical Lobe", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Gini Impurity Reduction", fontsize=10)
    ax2.set_xlabel("Cortical Region", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    impFigPath = FIGURES_DIR / "03_feature_importance_bands_lobes.png"
    fig.savefig(impFigPath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Feature Importance plot -> {impFigPath}")

    # Save benchmark records to json
    allBenchmarks = {
        "task_a_quadrants": dfResultsQuad.to_dict(orient="records"),
        "task_b_cowen27": dfResultsCowen.to_dict(orient="records"),
        "feature_importance": {
            "by_band": bandImp,
            "by_region": regionImp,
        },
    }
    benchJsonPath = FIGURES_DIR / "03_benchmark_results.json"
    with open(benchJsonPath, "w") as f:
        json.dump(allBenchmarks, f, indent=4)
    print(f"Saved full benchmark results JSON -> {benchJsonPath}")
    print("\nBenchmark and feature evaluation completed successfully!")


if __name__ == "__main__":
    main()
