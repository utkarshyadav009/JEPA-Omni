"""Data pipeline for the JEPA-Omni M1 spine."""

from .video_text_dataset import (
    MSRVTTVideoTextDataset,
    collate_fn,
    build_dataset,
)

__all__ = ["MSRVTTVideoTextDataset", "collate_fn", "build_dataset"]
