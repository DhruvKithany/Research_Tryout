"""Publication-quality 10-20 EEG Scalp Topography map generator.

Technical Implementation:
- Uses the standard 10-20 international electrode placement system adapted for the Emotiv 14-channel montage.
- Performs 2D cubic / Clough-Tocher radial interpolation across a dense Cartesian head grid.
- Masks the interpolated potential field with a circular head outline (radius = 0.52).
- Draws nose, ears, and electrode markers with custom anatomical annotations.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from ..config import CHANNELS_14, ELECTRODE_2D_COORDS


def bandTitle(band: str) -> str:
    """Returns frequency range string for a given spectral band name."""
    from ..config import FREQUENCY_BANDS
    if band in FREQUENCY_BANDS:
        low, high = FREQUENCY_BANDS[band]
        return f"{int(low)}-{int(high)} Hz"
    return ""


def plotTopomap(
    channelValues: Union[Dict[str, float], np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    res: int = 150,
    showNames: bool = True,
    showColorbar: bool = True,
    cbarLabel: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a 2D interpolated scalp topomap for the 14 Emotiv electrodes.

    Args:
        channelValues: Dict mapping channel name -> float value, OR 14-element array.
        ax: Existing matplotlib Axes, or None to create a new figure.
        title: Subplot title.
        cmap: Matplotlib colormap name (default 'RdBu_r' diverging for EEG).
        res: Cartesian grid interpolation resolution.
        showNames: Whether to print channel labels on markers.
        showColorbar: Whether to draw colorbar.
        cbarLabel: Colorbar title/unit.
        vmin, vmax: Color scale limits.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    else:
        fig = ax.get_figure()

    # Parse channel values
    if isinstance(channelValues, (list, np.ndarray)):
        valDict = {ch: float(channelValues[i]) for i, ch in enumerate(CHANNELS_14)}
    else:
        valDict = channelValues

    # Extract 2D sensor positions and associated scalar potentials
    xCoords = []
    yCoords = []
    values = []
    labels = []

    for ch in CHANNELS_14:
        if ch in valDict and ch in ELECTRODE_2D_COORDS:
            x, y = ELECTRODE_2D_COORDS[ch]
            xCoords.append(x)
            yCoords.append(y)
            values.append(valDict[ch])
            labels.append(ch)

    xCoords = np.array(xCoords)
    yCoords = np.array(yCoords)
    values = np.array(values)

    headRadius = 0.52
    xi = np.linspace(-headRadius, headRadius, res)
    yi = np.linspace(-headRadius, headRadius, res)
    Xi, Yi = np.meshgrid(xi, yi)

    # 2D Grid interpolation
    try:
        rbf = scipy.interpolate.RBFInterpolator(
            np.column_stack([xCoords, yCoords]),
            values,
            kernel="thin_plate_spline",
            smoothing=0.01,
        )
        gridZ = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    except Exception:
        # Fallback to linear / nearest griddata
        gridZ = scipy.interpolate.griddata(
            (xCoords, yCoords), values, (Xi, Yi), method="cubic", fill_value=np.mean(values)
        )

    # Circular mask for head outline
    mask = (Xi**2 + Yi**2) > (headRadius**2)
    gridZ[mask] = np.nan

    if vmin is None:
        vmin = np.nanpercentile(gridZ, 2)
    if vmax is None:
        vmax = np.nanpercentile(gridZ, 98)

    # Plot interpolated potential contours
    im = ax.imshow(
        gridZ,
        origin="lower",
        extent=(-headRadius, headRadius, -headRadius, headRadius),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Anatomical Head Outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(headRadius * np.cos(theta), headRadius * np.sin(theta), color="black", linewidth=2.0)

    # Nose marker
    noseX = [headRadius * -0.12, 0.0, headRadius * 0.12]
    noseY = [headRadius * 0.98, headRadius * 1.15, headRadius * 0.98]
    ax.plot(noseX, noseY, color="black", linewidth=2.0)

    # Left Ear
    earL_y = np.linspace(-0.08, 0.08, 30)
    earL_x = -np.sqrt(np.maximum(0, 0.08**2 - earL_y**2)) * 0.5 - headRadius
    ax.plot(earL_x, earL_y, color="black", linewidth=1.5)

    # Right Ear
    earR_y = np.linspace(-0.08, 0.08, 30)
    earR_x = np.sqrt(np.maximum(0, 0.08**2 - earR_y**2)) * 0.5 + headRadius
    ax.plot(earR_x, earR_y, color="black", linewidth=1.5)

    # Electrode markers
    ax.scatter(xCoords, yCoords, color="black", s=32, zorder=4, edgecolor="white", linewidth=0.8)

    if showNames:
        for x, y, label in zip(xCoords, yCoords, labels):
            ax.text(
                x,
                y + 0.035,
                label,
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#111111",
                zorder=5,
            )

    ax.set_xlim(-0.68, 0.68)
    ax.set_ylim(-0.62, 0.68)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    if showColorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, pad=0.04)
        if cbarLabel:
            cbar.set_label(cbarLabel, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    return fig, ax


def plotMultibandTopomaps(
    bandValuesDict: Dict[str, Dict[str, float]],
    emotionName: str = "",
    savePath: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plots a 5-panel row of scalp topographies across Delta, Theta, Alpha, Beta, Gamma bands.

    Args:
        bandValuesDict: Dict mapping band name -> dict of channel scalar values.
        emotionName: Optional emotion label for the figure title.
        savePath: Optional filepath to save the rendered figure.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    sPath = kwargs.get("save_path", savePath)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, axes = plt.subplots(1, len(bands), figsize=(16, 3.5), dpi=200)

    for i, band in enumerate(bands):
        valDict = bandValuesDict.get(band, {})
        title = f"{band.capitalize()} ({bandTitle(band)})"
        plotTopomap(
            valDict,
            ax=axes[i],
            title=title,
            showColorbar=True,
            showNames=False,
            cmap="viridis",
        )

    if emotionName:
        fig.suptitle(
            f"Scalp Spectral Power Distribution: {emotionName}",
            fontsize=13,
            fontweight="bold",
            y=1.05,
        )

    plt.tight_layout()
    if sPath:
        fig.savefig(sPath, bbox_inches="tight", dpi=300)
    return fig

# Backward-compatible snake_case aliases
plot_topomap = plotTopomap
plot_multiband_topomaps = plotMultibandTopomaps
band_title = bandTitle
