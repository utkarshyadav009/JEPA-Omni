"""Model components for the JEPA-Omni M1 video-text alignment spine.

This package exposes the public interface consumed by the training and
evaluation entry points (``train_m1.py`` / ``eval_m1.py``) and by the
data pipeline (``data/video_text_dataset.py``):

- :class:`SpineConfig`     -- dataclass configuration for the spine.
- :class:`SpineM1`         -- the M1 video/text alignment model.
- :class:`VisionEncoder`   -- SigLIP2 vision tower (``.encode`` takes a *list*).
- :class:`TextEncoder`     -- SigLIP2 text tower.
"""

from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder
from .spine import SpineConfig, SpineM1

__all__ = [
    "SpineConfig",
    "SpineM1",
    "VisionEncoder",
    "TextEncoder",
]
