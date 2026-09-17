"""EEG Affect: Geometry of Affect, Decoding and Manifold Analysis of 27 Fine-Grained Emotions."""

__version__ = "0.1.0"
__author__ = "Dhruv Kithany"

from . import config
from . import data
from . import features
from . import geometry
from . import models
from . import evaluation
from . import visualization

__all__ = [
    "config",
    "data",
    "features",
    "geometry",
    "models",
    "evaluation",
    "visualization",
]
