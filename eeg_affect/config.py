"""Configuration, constants, and metadata for EEG Affect project.

Includes:
- Emotiv 14-channel electrode coordinates (10-20 system).
- Cowen 27 fine-grained emotion dictionary and taxonomy.
- Valence-Arousal circumplex mappings.
- Spectral frequency band definitions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RAW_REPO_DIR = PROJECT_ROOT / "raw_repo"

# Electrode Configuration (Emotiv EPOC / Emotiv X 14 channels)
CHANNELS_14: List[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# Channel index (1-based to match dataset feature suffixes like _1 ... _14)
CHANNEL_INDEX: Dict[int, str] = {i + 1: ch for i, ch in enumerate(CHANNELS_14)}
CHANNEL_NAME_TO_INDEX: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(CHANNELS_14)}

# Frontal asymmetry pairs (Left vs Right homologues)
ASYMMETRY_PAIRS: List[Tuple[str, str]] = [
    ("AF3", "AF4"),  # Prefrontal (channel 1 vs 14)
    ("F3", "F4"),    # Frontal (channel 3 vs 12)
    ("FC5", "FC6"),  # Fronto-central (channel 4 vs 11)
    ("F7", "F8"),    # Antero-temporal (channel 2 vs 13)
    ("T7", "T8"),    # Temporal (channel 5 vs 10)
    ("P7", "P8"),    # Parietal (channel 6 vs 9)
    ("O1", "O2"),    # Occipital (channel 7 vs 8)
]

def _load_dynamic_electrode_coords() -> Dict[str, Tuple[float, float]]:
    """Dynamically parses electrode 2D coordinates from emotivX_channels_location.ced."""
    candidates = [
        DATA_DIR / "emotivX_channels_location.ced",
        RAW_REPO_DIR / "emotivX_channels_location.ced",
    ]
    for c in candidates:
        if c.exists():
            try:
                import pandas as pd
                import numpy as np
                ced = pd.read_csv(c, sep=r"\s+")
                coords = {}
                for _, row in ced.iterrows():
                    ch = str(row["labels"]).strip()
                    theta_rad = np.radians(float(row["theta"]))
                    r = float(row["radius"])
                    coords[ch] = (
                        round(float(r * np.sin(theta_rad)), 4),
                        round(float(r * np.cos(theta_rad)), 4),
                    )
                return coords
            except Exception:
                pass
    # Verified 10-20 mathematical projection fallback
    return {
        "AF3": (-0.1606, 0.3783), "F7": (-0.4134, 0.3004), "F3": (-0.2096, 0.2588),
        "FC5": (-0.3678, 0.1412), "T7": (-0.5110, 0.0000), "P7": (-0.4134, -0.3004),
        "O1": (-0.1579, -0.4860), "O2": (0.1579, -0.4860),  "P8": (0.4134, -0.3004),
        "T8": (0.5110, 0.0000),  "FC6": (0.3678, 0.1412), "F4": (0.2096, 0.2588),
        "F8": (0.4134, 0.3004),  "AF4": (0.1606, 0.3783),
    }

ELECTRODE_2D_COORDS: Dict[str, Tuple[float, float]] = _load_dynamic_electrode_coords()

# Spectral Bands (Hz)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Cowen & Keltner (2017) 27 Emotion Labels
COWEN_27_EMOTIONS: Dict[int, str] = {
    1: "Admiration",
    2: "Adoration",
    3: "Aesthetic Appreciation",
    4: "Amusement",
    5: "Anger",
    6: "Anxiety",
    7: "Awe",
    8: "Awkwardness",
    9: "Boredom",
    10: "Calmness",
    11: "Confusion",
    12: "Craving",
    13: "Disgust",
    14: "Empathic Pain",
    15: "Entrancement",
    16: "Excitement",
    17: "Fear",
    18: "Horror",
    19: "Interest",
    20: "Joy",
    21: "Nostalgia",
    22: "Relief",
    23: "Romance",
    24: "Sadness",
    25: "Satisfaction",
    26: "Sexual Desire",
    27: "Surprise",
}

# Mapping of Cowen 27 emotions to 2D Circumplex space: (Valence [-1, 1], Arousal [-1, 1])
# Based on affective psychology normative ratings (Cowen & Keltner 2017, Russell 1980)
COWEN_VALENCE_AROUSAL: Dict[int, Tuple[float, float]] = {
    1:  (0.70, 0.30),   # Admiration: pos, moderate
    2:  (0.80, -0.10),  # Adoration: pos, low
    3:  (0.60, -0.30),  # Aesthetic Appreciation: pos, low
    4:  (0.85, 0.70),   # Amusement: pos, high
    5:  (-0.75, 0.80),  # Anger: neg, high
    6:  (-0.60, 0.75),  # Anxiety: neg, high
    7:  (0.65, 0.60),   # Awe: pos, high
    8:  (-0.40, 0.30),  # Awkwardness: neg, moderate
    9:  (-0.50, -0.60), # Boredom: neg, low
    10: (0.75, -0.70),  # Calmness: pos, low
    11: (-0.30, 0.40),  # Confusion: neg, moderate
    12: (0.50, 0.65),   # Craving: pos, high
    13: (-0.80, 0.50),  # Disgust: neg, moderate/high
    14: (-0.65, 0.35),  # Empathic Pain: neg, moderate
    15: (0.55, -0.20),  # Entrancement: pos, low/mod
    16: (0.85, 0.85),   # Excitement: pos, high
    17: (-0.80, 0.85),  # Fear: neg, high
    18: (-0.90, 0.90),  # Horror: neg, high
    19: (0.60, 0.40),   # Interest: pos, moderate
    20: (0.90, 0.80),   # Joy: pos, high
    21: (0.30, -0.25),  # Nostalgia: mixed/pos, low/mod
    22: (0.70, -0.50),  # Relief: pos, low
    23: (0.75, 0.30),   # Romance: pos, moderate
    24: (-0.85, -0.50), # Sadness: neg, low
    25: (0.80, -0.40),  # Satisfaction: pos, low
    26: (0.70, 0.75),   # Sexual Desire: pos, high
    27: (0.20, 0.80),   # Surprise: neutral/pos, high
}

# Categorical Quadrant Mapping (High/Low Valence x High/Low Arousal)
# HVHA: 0, HVLA: 1, LVHA: 2, LVLA: 3
def getQuadrantLabel(emotionId: int) -> int:
    val, aro = COWEN_VALENCE_AROUSAL.get(emotionId, (0.0, 0.0))
    if val >= 0 and aro >= 0:
        return 0  # HVHA
    elif val >= 0 and aro < 0:
        return 1  # HVLA
    elif val < 0 and aro >= 0:
        return 2  # LVHA
    else:
        return 3  # LVLA

get_quadrant_label = getQuadrantLabel

QUADRANT_NAMES = ["HVHA (Positive Excited)", "HVLA (Positive Calm)", "LVHA (Negative Agitated)", "LVLA (Negative Depressed)"]
