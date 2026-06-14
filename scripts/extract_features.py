"""Extract frozen VisionEncoder features for all videos in the dataset.

Run once before cached-feature training::

    python scripts/extract_features.py --config configs/m1_scale.yaml
    python scripts/extract_features.py --config configs/m1_scale.yaml --limit 32  # smoke test

Produces ``{feature_cache_dir}/{shard}/{video_id}.pt`` files (bfloat16) and a
``manifest.json`` that train/eval scripts validate on startup.

Resume-safe: skips videos whose ``.pt`` already exists.  Atomic writes prevent
truncated files from a killed extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# Add project root to path so imports work when run as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.video_text_dataset import (
    MSRVTTVideoTextDataset,
    Sample,
    _uniform_frame_indices,
)
from models.vision_encoder import VisionEncoder
from utils import AttrDict, cfg_get, load_config


def _shard_dir(video_id: str) -> str:
    return video_id[:2]


def _feature_path(cache_dir: str, video_id: str) -> str:
    return os.path.join(cache_dir, _shard_dir(video_id), f"{video_id}.pt")


def collect_unique_videos(
    cfg: AttrDict, limit: int | None,
) -> Dict[str, str]:
    """Return {video_id: video_path} for all unique videos across splits."""
    unique: Dict[str, str] = {}
    for split in ("train", "eval"):
        try:
            ds = MSRVTTVideoTextDataset.from_config(
                cfg, split, limit=limit, decode_device="cpu",
            )
        except Exception as exc:
            print(f"[extract] WARNING: could not build {split} dataset: {exc}")
            continue
        for s in ds.samples:
            if s.video_id not in unique:
                unique[s.video_id] = s.video_path
    return unique


import signal

class _DecodeTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _DecodeTimeout("Video decode timed out")


def decode_clip(
    path: str, num_frames: int, resolution: int, timeout_sec: int = 60,
) -> Tensor:
    """Decode a single video clip — mirrors MSRVTTVideoTextDataset._decode_clip.

    Deterministic uniform sampling via ``_uniform_frame_indices`` (pure
    function of clip length and num_frames, zero RNG).

    A ``timeout_sec`` alarm prevents corrupt videos from hanging forever.
    """
    from torchcodec.decoders import VideoDecoder

    # --- Kaggle filename fallback (same as dataset) ---
    if not os.path.exists(path):
        basename = os.path.basename(path)
        stem, ext = os.path.splitext(basename)
        yt_id = stem[:11]
        fallback_path = os.path.join(os.path.dirname(path), yt_id + ext)
        if os.path.exists(fallback_path):
            path = fallback_path

    # Set alarm so hung decoders don't block extraction forever.
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        decoder = VideoDecoder(path, device="cpu")
        num_total = getattr(decoder.metadata, "num_frames", None)
        if not num_total:
            num_total = len(decoder)
        indices = _uniform_frame_indices(int(num_total), num_frames)

        unique_sorted = sorted(set(indices))
        remap = {orig: i for i, orig in enumerate(unique_sorted)}
        batch = decoder.get_frames_at(indices=unique_sorted)
        decoded = batch.data
        gather_idx = torch.tensor([remap[i] for i in indices], dtype=torch.long)
        frames = decoded.index_select(0, gather_idx.to(decoded.device))

        # Resize to target resolution (same logic as dataset._resize)
        if frames.shape[-2] != resolution or frames.shape[-1] != resolution:
            x = frames.float()
            x = F.interpolate(
                x,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            frames = x.round_().clamp_(0, 255).to(torch.uint8)
    finally:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)

    return frames  # [T, C, H, W] uint8


def extract(
    cfg: AttrDict,
    limit: int | None = None,
    micro_batch: int = 4,
) -> None:
    cache_dir = str(cfg_get(cfg, "data.feature_cache_dir"))
    num_frames = int(cfg_get(cfg, "data.num_frames", "num_frames", default=64))
    resolution = int(cfg_get(cfg, "data.resolution", "resolution", default=256))
    vision_repo = str(cfg_get(cfg, "model.vision_repo", default="facebook/vjepa2-vitl-fpc64-256"))

    os.makedirs(cache_dir, exist_ok=True)

    # ---- Collect unique videos ----
    print("[extract] Collecting unique videos from annotations...", flush=True)
    unique_videos = collect_unique_videos(cfg, limit)
    print(f"[extract] Found {len(unique_videos)} unique videos.", flush=True)

    # ---- Check which are already extracted ----
    todo: List[Tuple[str, str]] = []
    for vid, path in unique_videos.items():
        if not os.path.exists(_feature_path(cache_dir, vid)):
            todo.append((vid, path))
    print(
        f"[extract] {len(unique_videos) - len(todo)} already cached, "
        f"{len(todo)} to extract.",
        flush=True,
    )
    if not todo:
        print("[extract] Nothing to do.", flush=True)
        _write_manifest(cache_dir, vision_repo, num_frames, resolution)
        return

    # ---- Load encoder ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] Loading VisionEncoder ({vision_repo}) on {device}...", flush=True)
    encoder = VisionEncoder(vision_repo, dtype=torch.bfloat16, device=device)
    hidden_size = encoder.hidden_size
    print(f"[extract] Encoder ready. hidden_size={hidden_size}", flush=True)

    # ---- Extract in micro-batches ----
    t0 = time.time()
    done = 0
    failed = 0
    for batch_start in range(0, len(todo), micro_batch):
        batch_items = todo[batch_start : batch_start + micro_batch]

        # Decode all clips in this micro-batch
        decoded_clips: List[Tensor] = []
        valid_items: List[Tuple[str, str]] = []
        for vid, path in batch_items:
            try:
                frames = decode_clip(path, num_frames, resolution)
                decoded_clips.append(frames)
                valid_items.append((vid, path))
            except Exception as exc:
                print(f"[extract] SKIP {vid}: {exc}", flush=True)
                failed += 1

        if not decoded_clips:
            continue

        # Encode batch
        with torch.no_grad():
            feats = encoder.encode(decoded_clips)  # [B, N, hidden]

        # Save each
        for i, (vid, _) in enumerate(valid_items):
            feat = feats[i].to(torch.bfloat16).cpu()  # [N, hidden]
            out_dir = os.path.join(cache_dir, _shard_dir(vid))
            os.makedirs(out_dir, exist_ok=True)
            out_path = _feature_path(cache_dir, vid)
            tmp_path = out_path + ".tmp"
            torch.save(feat, tmp_path)
            os.rename(tmp_path, out_path)

        done += len(valid_items)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done - failed) / rate if rate > 0 else 0
        if done % 20 == 0 or done == len(todo):
            print(
                f"[extract] {done}/{len(todo)} done, {failed} failed, "
                f"{rate:.1f} vid/s, ETA {eta/60:.0f}m",
                flush=True,
            )

    # ---- Write manifest ----
    _write_manifest(cache_dir, vision_repo, num_frames, resolution, hidden_size)
    print(
        f"[extract] DONE. {done} extracted, {failed} failed, "
        f"{time.time() - t0:.0f}s total.",
        flush=True,
    )


def _write_manifest(
    cache_dir: str,
    encoder_repo: str,
    num_frames: int,
    resolution: int,
    hidden_size: int = 1024,
) -> None:
    manifest = {
        "encoder_repo": encoder_repo,
        "num_frames": num_frames,
        "resolution": resolution,
        "dtype": "bfloat16",
        "sampling_rule": "uniform_linspace_deterministic",
        "hidden_size": hidden_size,
    }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_path, manifest_path)
    print(f"[extract] Wrote {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen VisionEncoder features to disk.",
    )
    parser.add_argument("--config", default="configs/m1_scale.yaml")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples per split (for smoke testing).",
    )
    parser.add_argument(
        "--micro-batch", type=int, default=4,
        help="Number of videos to encode in one GPU batch.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    extract(cfg, limit=args.limit, micro_batch=args.micro_batch)


if __name__ == "__main__":
    main()
