"""scripts/extract_features_ego4d_train.py — item 5: extract feature cache
for the Ego4D train split (EGO4D_TRAIN_SPLIT_FILEDISJOINT_V2.json, 17,140
windows / 946 files, file-disjoint from the v2 held-out gate) into the
SAME cache schema scripts/extract_features_av.py uses for VGGSound, so
data/av_cached_dataset.py::AVCachedDataset can read it unmodified.

Writes to a NEW, ISOLATED cache directory (never the persistent VGGSound
cache at /mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k) -- zero
risk of ever touching that corpus. clip_id convention: "ego4d_{source_id}_
w{window_idx:04d}" (window_idx = int(start_sec // WINDOW_SEC), matching
data/source_disjoint_batch_sampler.py's expected naming so
SourceDisjointBatchSampler correctly groups windows by source file).

Per-clip tensors match extract_features_av.py's schema exactly:
  "vision"          : (32, 16, 1024) bf16
  "ambient_base"     : (T_base, 768)  bf16
  "ambient_nat"      : (T_nat,  768)  bf16
  "vision_ts"        : (32, 2)        f32
  "ambient_base_ts"  : (T_base, 2)    f32
  "ambient_nat_ts"   : (T_nat, 2)     f32
  "clip_duration_s"  : float

Item 2c fix applied: zero-length audio RAISES (window recorded as failed
and skipped), never silently substituted with zeros.

Multi-GPU: run one process per GPU with --shard-idx/--num-shards (plain
CUDA_VISIBLE_DEVICES + arg sharding, no torchrun needed -- this script
does one simple decode-then-encode pass per window, no DDP).

Usage (per-GPU shard):
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_ego4d_train.py --shard-idx 0 --num-shards 4
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_features_ego4d_train.py --shard-idx 1 --num-shards 4
    CUDA_VISIBLE_DEVICES=2 python scripts/extract_features_ego4d_train.py --shard-idx 2 --num-shards 4
    CUDA_VISIBLE_DEVICES=3 python scripts/extract_features_ego4d_train.py --shard-idx 3 --num-shards 4
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import (
    VISION_REPO, VISION_SPAT, VISION_DIM, AUDIO_SR, CLIP_DURATION_S,
    VISION_TOKEN_RATE, _spatial_pool, _vision_ts, _audio_ts,
)
from models.audio_encoder import WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO

TRAIN_MANIFEST = "checkpoints/vjepa21_shelved/EGO4D_TRAIN_SPLIT_FILEDISJOINT_V2.json"
CACHE_DIR = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_ego4d_train_v1"
WINDOW_SEC = 10.0
VIDEO_FPS = 30.0  # confirmed via ffprobe across all v1 (81) and v2 (350) gallery files -- both
                  # Ego4D sources (video_540ss, clips) are exactly 30/1
AUDIO_DIM = 768


def _shard(vid: str) -> str:
    return vid[:2]


def _feat_path(cache_dir: str, vid: str) -> str:
    return os.path.join(cache_dir, _shard(vid), f"{vid}.pt")


def decode_video(video_path, start_sec, device):
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices
    t0 = start_sec
    t1 = start_sec + WINDOW_SEC
    dec = VideoDecoder(video_path, device="cpu")
    n_total = getattr(dec.metadata, "num_frames", None) or len(dec)
    f0 = int(round(t0 * VIDEO_FPS)); f1 = int(round(t1 * VIDEO_FPS))
    f1 = min(f1, n_total); f0 = min(f0, f1 - 1)
    idx = _uniform_frame_indices(f1 - f0, 64)
    abs_idx = sorted(set(i + f0 for i in idx))
    batch = dec.get_frames_at(indices=abs_idx)
    decoded = batch.data
    remap = {o: i for i, o in enumerate(abs_idx)}
    gi = torch.tensor([remap[i + f0] for i in idx], dtype=torch.long)
    frames = decoded.index_select(0, gi)
    if frames.shape[-2] != 256 or frames.shape[-1] != 256:
        x = frames.float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames, t0, t1


def decode_audio(video_path, t0, t1):
    """item 2c fix: raises on zero-length audio, never silently zeroed."""
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
           "-i", video_path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    if sr != AUDIO_SR:
        raise ValueError(f"sample rate mismatch: got {sr}, expected {AUDIO_SR}")
    if audio.shape[0] < 1:
        raise ValueError(f"zero-length audio decoded from {video_path} @ [{t0:.2f},{t1:.2f}]")
    return torch.from_numpy(audio), audio.shape[0] / AUDIO_SR


def window_idx_from_start(start_sec: float) -> int:
    return int(round(start_sec / WINDOW_SEC))


def write_manifest(cache_dir: str, base_hz: float, nat_hz: float) -> None:
    m = {
        "vision_repo": VISION_REPO,
        "wavjepa_base_repo": WAVJEPA_BASE_REPO,
        "wavjepa_nat_repo": WAVJEPA_NAT_REPO,
        "vision_spatial_pool": VISION_SPAT,
        "video_fps": round(VIDEO_FPS, 4),
        "video_n_frames": 64,
        "video_resolution": 256,
        "vision_token_rate_hz": round(VISION_TOKEN_RATE, 4),
        "audio_sample_rate": AUDIO_SR,
        "wavjepa_base_token_rate_hz": round(base_hz, 4),
        "wavjepa_nat_token_rate_hz": round(nat_hz, 4),
        "vision_out_shape_per_clip": [32, VISION_SPAT, VISION_DIM],
        "audio_out_dim": AUDIO_DIM,
        "dtype": "bfloat16",
        "timestamp_format": "[start_s, end_s] per token",
        "sampling_rule": "ego4d_10s_window_uniform_frame_sampling",
        "source": "scripts/extract_features_ego4d_train.py",
    }
    path = os.path.join(cache_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.rename(tmp, path)
    print(f"[ego4d-extract] manifest -> {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--limit", type=int, default=None, help="smoke-test: only process the first N windows of this shard")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ego4d-extract] shard {args.shard_idx}/{args.num_shards} device={device}", flush=True)

    with open(TRAIN_MANIFEST) as f:
        train_windows = json.load(f)
    my_windows = [w for i, w in enumerate(train_windows) if i % args.num_shards == args.shard_idx]
    if args.limit is not None:
        my_windows = my_windows[:args.limit]
    print(f"[ego4d-extract] shard {args.shard_idx}: {len(my_windows)}/{len(train_windows)} windows", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    os.makedirs(args.cache_dir, exist_ok=True)
    if args.shard_idx == 0:
        write_manifest(args.cache_dir, base_hz=99.6, nat_hz=99.6)  # placeholder rate, overwritten below once known

    t_start = time.time()
    n_done, n_failed, n_skipped = 0, 0, 0
    base_hz_observed, nat_hz_observed = None, None

    for i, m in enumerate(my_windows):
        vid = f"ego4d_{m['source_id']}_w{window_idx_from_start(m['start_sec']):04d}"
        out_path = _feat_path(args.cache_dir, vid)
        if os.path.isfile(out_path):
            n_skipped += 1
            continue
        try:
            frames, t0, t1 = decode_video(m["path"], m["start_sec"], device)
            audio, true_dur = decode_audio(m["path"], t0, t1)

            with torch.no_grad():
                raw = vision_enc.encode(frames.unsqueeze(0).to(device))  # (1, 8192, 1024)
            raw = raw[0]
            vis_full = raw.view(32, 256, VISION_DIM)
            vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16)  # (32, 16, 1024)
            vis_ts = _vision_ts()  # dur=CLIP_DURATION_S default -- matches training's literal convention

            audio_mono = audio
            wav1 = audio_mono.unsqueeze(0).unsqueeze(0)
            wav2 = audio_mono.unsqueeze(0).expand(2, -1).unsqueeze(0)
            with torch.no_grad():
                base_feat = base_enc.encode(wav1.to(device))[0].to(torch.bfloat16)
                nat_feat = nat_enc.encode(wav2.to(device))[0].to(torch.bfloat16)
            base_ts = _audio_ts(base_feat.shape[0], true_dur)
            nat_ts = _audio_ts(nat_feat.shape[0], true_dur)
            if base_hz_observed is None:
                base_hz_observed = base_feat.shape[0] / true_dur
                nat_hz_observed = nat_feat.shape[0] / true_dur

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({
                "vision": vis_pooled.cpu(),
                "ambient_base": base_feat.cpu(),
                "ambient_nat": nat_feat.cpu(),
                "vision_ts": vis_ts,
                "ambient_base_ts": base_ts,
                "ambient_nat_ts": nat_ts,
                "clip_duration_s": true_dur,
            }, out_path + ".tmp")
            os.rename(out_path + ".tmp", out_path)
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[ego4d-extract] shard {args.shard_idx}: window {i} ({m['path']}@{m['start_sec']}) "
                  f"FAILED (item 2c: recorded as failed, skipped, not zeroed): {e!r}", flush=True)

        if (i + 1) % 100 == 0 or i == len(my_windows) - 1:
            elapsed = time.time() - t_start
            print(f"[ego4d-extract] shard {args.shard_idx}: {i+1}/{len(my_windows)} "
                  f"(done={n_done}, failed={n_failed}, skipped={n_skipped}), "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    if args.shard_idx == 0 and base_hz_observed is not None:
        write_manifest(args.cache_dir, base_hz=base_hz_observed, nat_hz=nat_hz_observed)

    print(f"[ego4d-extract] shard {args.shard_idx} DONE: done={n_done} failed={n_failed} "
          f"skipped={n_skipped}", flush=True)


if __name__ == "__main__":
    main()
