"""Cached-feature dataset: loads pre-computed VisionEncoder features from disk.

Used when ``data.feature_cache_dir`` is set in the config.  Each sample is a
``(features, caption)`` pair where ``features`` is a ``[N, 1024]`` bfloat16
tensor loaded from ``{cache_dir}/{shard}/{video_id}.pt``.

The annotation parsing is reused from :mod:`data.video_text_dataset` so
train/eval splits and caption selection are identical.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .video_text_dataset import (
    MSRVTTVideoTextDataset,
    Sample,
)


def _shard_dir(video_id: str) -> str:
    """First 2 characters of the video_id, used as a shard directory."""
    return video_id[:2]


def _feature_path(cache_dir: str, video_id: str) -> str:
    return os.path.join(cache_dir, _shard_dir(video_id), f"{video_id}.pt")


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-computed encoder features instead of raw video.

    Parameters
    ----------
    cache_dir:
        Root of the feature cache (contains ``manifest.json`` and shard dirs).
    samples:
        List of :class:`Sample` (video_id, caption) — reused from the video
        dataset's annotation parsing.
    """

    def __init__(self, cache_dir: str, samples: List[Sample]) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        sample = self.samples[index]
        feat_path = _feature_path(self.cache_dir, sample.video_id)
        feats = torch.load(feat_path, map_location="cpu", weights_only=True)
        return feats, sample.caption

    # ------------------------------------------------------------------ #
    # Construction from config (mirrors MSRVTTVideoTextDataset.from_config)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg,
        split: str,
        *,
        limit: Optional[int] = None,
    ) -> "CachedFeatureDataset":
        """Build a cached-feature dataset for *split* from a config.

        Reuses :class:`MSRVTTVideoTextDataset` to parse annotations and
        produce the sample list, but never touches video files.
        """
        from utils import cfg_get

        cache_dir = str(cfg_get(cfg, "data.feature_cache_dir"))
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(
                f"Feature cache directory does not exist: {cache_dir!r}"
            )

        # Build the video dataset just for its sample list (no decoding).
        video_ds = MSRVTTVideoTextDataset.from_config(
            cfg, split, limit=limit, decode_device="cpu",
        )
        return cls(cache_dir=cache_dir, samples=video_ds.samples)


def cached_collate_fn(
    batch: Sequence[Tuple[Tensor, str]],
) -> Tuple[Tensor, List[str]]:
    """Stack features into ``[B, N, D]`` and collect captions."""
    feats = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return feats, captions


def validate_manifest(cache_dir: str, cfg) -> Dict:
    """Load and validate manifest.json against the live config.

    Raises ``ValueError`` if the cache was produced with a different encoder,
    resolution, num_frames, or dtype — preventing silent result corruption.
    """
    from utils import cfg_get

    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No manifest.json in feature cache {cache_dir!r}. "
            "Run scripts/extract_features.py first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    live_repo = str(cfg_get(cfg, "model.vision_repo", default="facebook/vjepa2-vitl-fpc64-256"))
    live_frames = int(cfg_get(cfg, "data.num_frames", "num_frames", default=64))
    live_res = int(cfg_get(cfg, "data.resolution", "resolution", default=256))

    errors = []
    if manifest.get("encoder_repo") != live_repo:
        errors.append(
            f"encoder_repo: cache={manifest.get('encoder_repo')!r} vs config={live_repo!r}"
        )
    if manifest.get("num_frames") != live_frames:
        errors.append(
            f"num_frames: cache={manifest.get('num_frames')} vs config={live_frames}"
        )
    if manifest.get("resolution") != live_res:
        errors.append(
            f"resolution: cache={manifest.get('resolution')} vs config={live_res}"
        )
    if manifest.get("dtype") != "bfloat16":
        errors.append(
            f"dtype: cache={manifest.get('dtype')!r}, expected 'bfloat16'"
        )
    if errors:
        raise ValueError(
            "Feature cache manifest mismatch — stale or wrong cache!\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
    return manifest
