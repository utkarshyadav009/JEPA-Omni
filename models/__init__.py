"""JEPA-Omni model package — public API.

Re-exports the names that train_m1.py / eval_m1.py import as
`from models import SpineConfig, SpineM1, info_nce`.
"""

from .vision_encoder import VisionEncoder
from .text_target import TextTarget
from .predictor import Predictor
from .losses import info_nce, compute_siglip_loss
from .spine_m1 import SpineConfig, SpineM1

__all__ = [
    "VisionEncoder",
    "TextTarget",
    "Predictor",
    "info_nce",
    "compute_siglip_loss",
    "SpineConfig",
    "SpineM1",
]
