"""Visualizations for Frontal Alpha Asymmetry (FAA) and regional spectral profiles.

Theoretical Framework:
- Frontal Alpha Asymmetry (AF4 - AF3) indexes approach vs withdrawal motivational tendency.
- Ranked horizontal bar plots with standard error bars illustrate emotion-specific cortical signatures.
- Regional spectral profiles aggregate band power across anatomical lobes.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import COWEN_27_EMOTIONS


def plotFrontalAlphaAsymmetry(
    faaSeries: pd.Series,
    cowenIds: pd.Series,
    title: str = "Frontal Alpha Asymmetry (AF4 - AF3) Across 27 Emotional States",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    sPath = kwargs.get("save_path", savePath)
    """Plots ranked Frontal Alpha Asymmetry across the 27 Cowen emotions.

    Positive FAA = relatively greater left frontal activity (Approach / Positive affect).
    Negative FAA = relatively greater right frontal activity (Withdrawal / Negative affect).

    Args:
        faaSeries: pd.Series of computed FAA values per trial.
        cowenIds: pd.Series of integer Cowen emotion IDs (1..27).
        title: Figure title string.
        savePath: Optional path to save high-resolution PNG image.

    Returns:
        plt.Figure: Rendered matplotlib figure.
    """
    df = pd.DataFrame({"CowenID": cowenIds, "FAA": faaSeries})
    df["Emotion"] = df["CowenID"].map(COWEN_27_EMOTIONS)

    # Compute mean and standard error of the mean (SEM) per emotion
    summary = df.groupby("Emotion")["FAA"].agg(["mean", "sem"]).reset_index()
    summary = summary.sort_values(by="mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Diverging color palette: Red for withdrawal (<0), Green for approach (>0)
    colors = ["#e74c3c" if m < 0 else "#2ecc71" for m in summary["mean"]]

    ax.barh(
        summary["Emotion"],
        summary["mean"],
        xerr=summary["sem"],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(
        r"Frontal Alpha Asymmetry: $\ln(\mathrm{Alpha_{AF4}}) - \ln(\mathrm{Alpha_{AF3}})$ [$\pm$ SEM]",
        fontsize=10,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Annotate approach vs withdrawal biological interpretations
    ax.text(
        0.02,
        0.02,
        r"$\leftarrow$ Withdrawal / Avoidance (Right Frontal Dominance)",
        transform=ax.transAxes,
        fontsize=9,
        color="#c0392b",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.02,
        r"Approach / Positive Valence (Left Frontal Dominance) $\rightarrow$",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        color="#27ae60",
        fontweight="bold",
    )

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig


def plotRegionalBandpowers(
    dfRegional: pd.DataFrame,
    title: str = "Cortical Regional Spectral Power Profile",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots barplot of spectral power across cortical regions (Prefrontal, Frontal, etc.).

    Args:
        dfRegional: DataFrame with cortical region columns.
        title: Figure title.
        savePath: Optional output path.

    Returns:
        plt.Figure: Rendered matplotlib figure.
    """
    sPath = kwargs.get("save_path", savePath)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    sns.barplot(data=dfRegional, ax=ax, palette="mako")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Normalized Spectral Power", fontsize=10)
    plt.xticks(rotation=25, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig

# Backward-compatible snake_case aliases
plot_frontal_alpha_asymmetry = plotFrontalAlphaAsymmetry
plot_regional_bandpowers = plotRegionalBandpowers
