"""scripts/extract_features_audioset.py — RUN-2 data prep: extract feature
cache for the visually-gated AudioSet-Strong kept set
(checkpoints/vjepa21_shelved/audioset_visual_gate_result_kept.json, 8,588
whole short clips that passed BOTH the audio AV-relevance filter (P4 stage
1) and the visual admissibility gate (P4 stage 2, staticness + CLIP
real-photo check)) into the SAME cache schema
scripts/extract_features_av.py uses for VGGSound / scripts/
extract_features_ego4d_train.py uses for Ego4D, so data/av_cached_dataset.py
::AVCachedDataset can read it unmodified.

AudioSet-Strong clips are WHOLE short clips (1-10s+, variable duration),
not fixed-duration windows into a longer video -- reuses
extract_features_av.py's _decode_video_raw (uniform 64-frame sampling over
whatever the real duration is) but does NOT reuse its _decode_audio_raw,
which silently zeros on decode failure. Mirrors extract_features_ego4d_
train.py's item-2c convention instead: audio decode failures RAISE and the
clip is recorded as failed/skipped, never silently zeroed. Real per-clip
duration (not the assumed CLIP_DURATION_S=10.0) is used for _vision_ts/
_audio_ts, matching the Ego4D script's true_dur convention -- required
here since AudioSet-Strong durations are genuinely variable, unlike
VGGSound's fixed ~10s.

Per-clip tensors match the existing cache schema exactly:
  "vision"          : (32, 16, 1024) bf16
  "ambient_base"     : (T_base, 768)  bf16
  "ambient_nat"      : (T_nat,  768)  bf16
  "vision_ts"        : (32, 2)        f32
  "ambient_base_ts"  : (T_base, 2)    f32
  "ambient_nat_ts"   : (T_nat, 2)     f32
  "clip_duration_s"  : float

clip_id convention: the mp4 filename stem itself (e.g.
"kM9QvdZela4_250_260"), already unique per file -- no ytid/start/end
parsing needed (AudioSet-Strong ytids can themselves contain underscores,
so parsing the stem would need a non-trivial regex; the raw stem sidesteps
that entirely since we never need to recover the parts).

Multi-GPU: run one process per GPU with --shard-idx/--num-shards (plain
CUDA_VISIBLE_DEVICES + arg sharding, matching extract_features_ego4d_train.py).

Usage (per-GPU shard):
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_audioset.py --shard-idx 0 --num-shards 4
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_features_audioset.py --shard-idx 1 --num-shards 4
    CUDA_VISIBLE_DEVICES=2 python scripts/extract_features_audioset.py --shard-idx 2 --num-shards 4
    CUDA_VISIBLE_DEVICES=3 python scripts/extract_features_audioset.py --shard-idx 3 --num-shards 4
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import (
    VISION_REPO, VISION_SPAT, VISION_DIM, AUDIO_SR, NUM_FRAMES, RESOLUTION,
    VISION_TOKEN_RATE, _spatial_pool, _vision_ts, _audio_ts, _decode_video_raw,
)
from models.audio_encoder import WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO

KEPT_MANIFEST = "checkpoints/vjepa21_shelved/audioset_visual_gate_result_kept.json"
CACHE_DIR = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_audioset_strong_v1"
AUDIO_DIM = 768


def _shard(vid: str) -> str:
    return vid[:2]


def _feat_path(cache_dir: str, vid: str) -> str:
    return os.path.join(cache_dir, _shard(vid), f"{vid}.pt")


def decode_audio(path: str) -> tuple[torch.Tensor, float]:
    """Same item-2c convention as extract_features_ego4d_train.py: raises on
    zero-length / failed decode, never silently substituted with zeros."""
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    if sr != AUDIO_SR:
        raise ValueError(f"sample rate mismatch: got {sr}, expected {AUDIO_SR}")
    if audio.shape[0] < AUDIO_SR // 2:
        raise ValueError(f"audio too short: {audio.shape[0]} samples from {path}")
    return torch.from_numpy(audio), audio.shape[0] / AUDIO_SR


def write_manifest(cache_dir: str, base_hz: float, nat_hz: float) -> None:
    m = {
        "vision_repo": VISION_REPO,
        "wavjepa_base_repo": WAVJEPA_BASE_REPO,
        "wavjepa_nat_repo": WAVJEPA_NAT_REPO,
        "vision_spatial_pool": VISION_SPAT,
        "video_fps": None,  # AudioSet-Strong source videos have variable per-clip fps (unlike
                             # Ego4D's uniform 30fps); _decode_video_raw uniformly samples frame
                             # INDICES (not fps-derived timestamps), so no single fps value
                             # applies here. Key present (required by load_and_validate_manifest)
                             # but intentionally null -- never read numerically downstream.
        "video_n_frames": NUM_FRAMES,
        "video_resolution": RESOLUTION,
        "vision_token_rate_hz": round(VISION_TOKEN_RATE, 4),
        "audio_sample_rate": AUDIO_SR,
        "wavjepa_base_token_rate_hz": round(base_hz, 4),
        "wavjepa_nat_token_rate_hz": round(nat_hz, 4),
        "vision_out_shape_per_clip": [32, VISION_SPAT, VISION_DIM],
        "audio_out_dim": AUDIO_DIM,
        "dtype": "bfloat16",
        "timestamp_format": "[start_s, end_s] per token, TRUE per-clip duration (variable, not assumed 10.0s)",
        "sampling_rule": "audioset_strong_wholeclip_uniform_64frame_sampling",
        "source": "scripts/extract_features_audioset.py",
        "input_manifest": KEPT_MANIFEST,
    }
    path = os.path.join(cache_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.rename(tmp, path)
    print(f"[audioset-extract] manifest -> {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audioset-extract] shard {args.shard_idx}/{args.num_shards} device={device}", flush=True)

    with open(KEPT_MANIFEST) as f:
        kept = json.load(f)
    my_clips = [c for i, c in enumerate(kept) if i % args.num_shards == args.shard_idx]
    if args.limit is not None:
        my_clips = my_clips[:args.limit]
    print(f"[audioset-extract] shard {args.shard_idx}: {len(my_clips)}/{len(kept)} clips", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    os.makedirs(args.cache_dir, exist_ok=True)
    if args.shard_idx == 0:
        write_manifest(args.cache_dir, base_hz=99.6, nat_hz=99.6)  # placeholder, overwritten once real rate known

    t_start = time.time()
    n_done, n_failed, n_skipped = 0, 0, 0
    base_hz_observed, nat_hz_observed = None, None

    for i, c in enumerate(my_clips):
        path = c["path"]
        vid = os.path.splitext(os.path.basename(path))[0]
        out_path = _feat_path(args.cache_dir, vid)
        if os.path.isfile(out_path):
            n_skipped += 1
            continue
        try:
            frames = _decode_video_raw(path, NUM_FRAMES, RESOLUTION)
            audio, true_dur = decode_audio(path)

            with torch.no_grad():
                raw = vision_enc.encode(frames.unsqueeze(0).to(device))
            raw = raw[0]
            vis_full = raw.view(32, 256, VISION_DIM)
            vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16)
            vis_ts = _vision_ts(dur=true_dur)

            wav1 = audio.unsqueeze(0).unsqueeze(0)
            wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0)
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
            print(f"[audioset-extract] shard {args.shard_idx}: clip {i} ({path}) "
                  f"FAILED (recorded as failed, skipped, not zeroed): {e!r}", flush=True)

        if (i + 1) % 200 == 0 or i == len(my_clips) - 1:
            elapsed = time.time() - t_start
            print(f"[audioset-extract] shard {args.shard_idx}: {i+1}/{len(my_clips)} "
                  f"(done={n_done}, failed={n_failed}, skipped={n_skipped}), "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    if args.shard_idx == 0 and base_hz_observed is not None:
        write_manifest(args.cache_dir, base_hz=base_hz_observed, nat_hz=nat_hz_observed)

    print(f"[audioset-extract] shard {args.shard_idx} DONE: done={n_done} failed={n_failed} "
          f"skipped={n_skipped}", flush=True)


if __name__ == "__main__":
    main()
