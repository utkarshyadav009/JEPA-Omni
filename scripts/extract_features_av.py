"""scripts/extract_features_av.py — M2 audio-visual feature extraction.

Do NOT touch scripts/extract_features.py (M1 script — leave it alone).

Mirrors M1's proven two-phase decode-then-encode pattern:
  Phase 1: decode all video+audio in subprocesses (pre-CUDA, safe fork)
  Phase 2: load encoders, encode, write atomically

Produces THREE tensors per clip in /dev/shm/jepa_m2_cache/:
  {shard}/{video_id}.pt  →  dict:
    "vision"           : (32, 16, 1024) bf16  — V-JEPA2, spatial-pooled
    "ambient_base"     : (T_base, 768)  bf16  — wavjepa-base
    "ambient_nat"      : (T_nat,  768)  bf16  — wavjepa-nat-base (ch-pooled)
    "vision_ts"        : (32, 2)        f32   — [start_s, end_s] per token
    "ambient_base_ts"  : (T_base, 2)    f32   — [start_s, end_s] per token
    "ambient_nat_ts"   : (T_nat,  2)    f32   — [start_s, end_s] per token
    "clip_duration_s"  : float

Plus /dev/shm/jepa_m2_cache/manifest.json.

Usage (smoke test first!):
    python scripts/extract_features_av.py --limit 32

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        scripts/extract_features_av.py --limit 50000

ENV:
    NGPU                 — number of GPUs (default 2)
    CUDA_VISIBLE_DEVICES — physical GPU ids (default "0,1")
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# ── project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import (
    AudioEncoder,
    WAVJEPA_BASE_REPO,
    WAVJEPA_NAT_REPO,
    WAVJEPA_AUDIO_SAMPLE_RATE,
)

# ── constants ────────────────────────────────────────────────────────────────
CACHE_DIR         = "/dev/shm/jepa_m2_cache"
MIN_SHM_GB        = 120
VISION_REPO       = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES        = 64          # V-JEPA2 default
RESOLUTION        = 256         # px
VISION_TEMP       = 32          # temporal tokens (64 frames / tubelet-2)
VISION_SPAT       = 16          # spatial tokens after 256→16 pool
VISION_DIM        = 1024        # ViT-L hidden size
AUDIO_DIM         = 768         # WavJEPA hidden size
AUDIO_SR          = WAVJEPA_AUDIO_SAMPLE_RATE   # 16 000 Hz
CLIP_DURATION_S   = 10.0        # assumed clip duration for timestamps
# Token rate for vision: 32 temporal tokens over 10s clip
VISION_TOKEN_RATE = VISION_TEMP / CLIP_DURATION_S   # 3.2 Hz
# Audio token rates (measured, not assumed):
BASE_TOKEN_RATE_HZ = 99.6
NAT_TOKEN_RATE_HZ  = 99.6


# ── helpers ──────────────────────────────────────────────────────────────────
def _shard(vid: str) -> str:
    return vid[:2]

def _feat_path(cache_dir: str, vid: str) -> str:
    return os.path.join(cache_dir, _shard(vid), f"{vid}.pt")

def _check_shm(min_gb: float = MIN_SHM_GB) -> None:
    stat  = os.statvfs("/dev/shm")
    free  = stat.f_bavail * stat.f_frsize / 1024**3
    if free < min_gb:
        print(
            f"[STOP] /dev/shm only {free:.1f} GB free — need >= {min_gb} GB. "
            "Free space first.",
            flush=True,
        )
        sys.exit(1)
    print(f"[shm] {free:.1f} GB free — OK", flush=True)

def _shm_used_gb() -> float:
    stat = os.statvfs("/dev/shm")
    return (stat.f_blocks - stat.f_bavail) * stat.f_frsize / 1024**3


# ── spatial pool: (32, 256, 1024) → (32, 16, 1024) ─────────────────────────
def _spatial_pool(feat: Tensor, out_spat: int = VISION_SPAT) -> Tensor:
    """Mean-pool 256 spatial tokens → 16 (4×4 grid from 16×16 grid)."""
    T, S, D = feat.shape   # (32, 256, 1024)
    assert S == 256 and out_spat == 16, f"Unexpected shape {feat.shape}"
    grid_in  = 16
    grid_out = 4
    pool_k   = grid_in // grid_out   # 4

    x = feat.view(T, grid_in, grid_in, D).permute(0, 3, 1, 2)  # (T, D, 16, 16)
    x = x.reshape(T * D, 1, grid_in, grid_in).float()
    x = F.avg_pool2d(x, kernel_size=pool_k, stride=pool_k)     # (T*D, 1, 4, 4)
    x = x.view(T, D, grid_out, grid_out).permute(0, 2, 3, 1)   # (T, 4, 4, D)
    x = x.reshape(T, out_spat, D)                               # (T, 16, D)
    return x.to(feat.dtype)


# ── timestamps ───────────────────────────────────────────────────────────────
def _vision_ts(n_temp: int = VISION_TEMP, dur: float = CLIP_DURATION_S,
               n_frames: int = NUM_FRAMES) -> Tensor:
    """(n_temp, 2) float32 [start_s, end_s] per temporal token."""
    fpg = n_frames / n_temp          # frames per temporal group (= 2.0)
    fd  = dur / n_frames             # seconds per raw frame
    ts  = torch.zeros(n_temp, 2)
    for t in range(n_temp):
        ts[t, 0] = t * fpg * fd
        ts[t, 1] = (t + 1) * fpg * fd
    return ts

def _audio_ts(n_tokens: int, dur: float) -> Tensor:
    """(n_tokens, 2) float32 [start_s, end_s] per audio token."""
    td = dur / n_tokens
    ts = torch.zeros(n_tokens, 2)
    for t in range(n_tokens):
        ts[t, 0] = t * td
        ts[t, 1] = (t + 1) * td
    return ts


# ── video/audio decode (subprocess, pre-CUDA) ─────────────────────────────
def _decode_video_raw(path: str, num_frames: int, resolution: int) -> Tensor:
    """Decode video frames; mirrors M1 script."""
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices

    if not os.path.exists(path):
        stem, ext = os.path.splitext(os.path.basename(path))
        fallback  = os.path.join(os.path.dirname(path), stem[:11] + ext)
        if os.path.exists(fallback):
            path = fallback

    decoder   = VideoDecoder(path, device="cpu")
    num_total = getattr(decoder.metadata, "num_frames", None) or len(decoder)
    indices   = _uniform_frame_indices(int(num_total), num_frames)

    unique   = sorted(set(indices))
    remap    = {o: i for i, o in enumerate(unique)}
    batch    = decoder.get_frames_at(indices=unique)
    decoded  = batch.data
    gi       = torch.tensor([remap[i] for i in indices], dtype=torch.long)
    frames   = decoded.index_select(0, gi.to(decoded.device))

    if frames.shape[-2] != resolution or frames.shape[-1] != resolution:
        x = frames.float()
        x = F.interpolate(x, size=(resolution, resolution),
                          mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames   # (T, C, H, W) uint8


def _decode_audio_raw(path: str, target_sr: int = AUDIO_SR) -> Tensor:
    """Decode audio from video; return (n_samples,) float32 mono."""
    try:
        from torchcodec.decoders import AudioDecoder
        dec    = AudioDecoder(path, sample_rate=target_sr)
        frames = dec.get_all_samples()
        wav    = frames.data                         # (channels, n_samples) float32
        if wav.shape[0] > 1:
            wav = wav.mean(0)
        else:
            wav = wav[0]
        return wav
    except Exception:
        return torch.zeros(int(CLIP_DURATION_S * target_sr))


def _worker(path: str, num_frames: int, resolution: int, save_to: str) -> None:
    frames = _decode_video_raw(path, num_frames, resolution)
    audio  = _decode_audio_raw(path)
    torch.save({"frames": frames, "audio": audio}, save_to)


def _decode_timeout(vid: str, path: str, tmp_dir: str,
                    timeout: int = 30) -> str:
    tmp = os.path.join(tmp_dir, f"{vid}.avdec.tmp")
    p   = mp.Process(target=_worker,
                     args=(path, NUM_FRAMES, RESOLUTION, tmp))
    p.start(); p.join(timeout=timeout)
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive(): p.kill(); p.join()
        if os.path.exists(tmp): os.unlink(tmp)
        raise TimeoutError(f"decode timed out after {timeout}s")
    if p.exitcode != 0:
        if os.path.exists(tmp): os.unlink(tmp)
        raise RuntimeError(f"decode subprocess exited {p.exitcode}")
    return tmp


# ── manifest ─────────────────────────────────────────────────────────────────
def _write_manifest(
    cache_dir: str,
    base_hz: float,
    nat_hz: float,
    video_fps: float = VIDEO_FPS if "VIDEO_FPS" in dir() else VISION_TEMP / CLIP_DURATION_S,
) -> None:
    m = {
        "vision_repo":              VISION_REPO,
        "wavjepa_base_repo":        WAVJEPA_BASE_REPO,
        "wavjepa_nat_repo":         WAVJEPA_NAT_REPO,
        "vision_spatial_pool":      VISION_SPAT,
        "video_fps":                round(video_fps, 4),
        "video_n_frames":           NUM_FRAMES,
        "video_resolution":         RESOLUTION,
        "vision_token_rate_hz":     round(VISION_TOKEN_RATE, 4),
        "audio_sample_rate":        AUDIO_SR,
        "wavjepa_base_token_rate_hz": round(base_hz, 4),
        "wavjepa_nat_token_rate_hz":  round(nat_hz, 4),
        "vision_out_shape_per_clip": [VISION_TEMP, VISION_SPAT, VISION_DIM],
        "audio_out_dim":            AUDIO_DIM,
        "dtype":                    "bfloat16",
        "timestamp_format":         "[start_s, end_s] per token",
        "sampling_rule":            "uniform_linspace_deterministic",
    }
    path = os.path.join(cache_dir, "manifest.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.rename(tmp, path)
    print(f"[av-extract] manifest → {path}\n{json.dumps(m, indent=2)}", flush=True)


# ── collect clips ─────────────────────────────────────────────────────────────
def _collect_clips(video_dir: str, limit: Optional[int] = None,
                   rank: int = 0, train_only: bool = False,
                   clip_list: Optional[List[str]] = None) -> Dict[str, str]:
    """Return {video_id: path} from data/train.csv (+ data/test.csv unless train_only).

    If clip_list is given, only those video IDs are returned (ignoring CSV order/limit).

    CSV format (VGGSound — no header): <filename>,<label>
    e.g.  OxPnZzn1_L8_000883.mp4,scuba diving
    """
    import csv
    clip_set = set(clip_list) if clip_list is not None else None
    unique: Dict[str, str] = {}
    csvs = ("train.csv",) if train_only else ("train.csv", "test.csv")
    for csv_name in csvs:
        csv_path = os.path.join(PROJECT_ROOT, "data", csv_name)
        if not os.path.exists(csv_path):
            if rank == 0:
                print(f"[av-extract] WARNING: {csv_path} not found", flush=True)
            continue
        with open(csv_path, "r", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                fname = row[0].strip()          # e.g. OxPnZzn1_L8_000883.mp4
                # Strip extension to get video_id
                vid   = os.path.splitext(fname)[0]   # OxPnZzn1_L8_000883
                if not vid:
                    continue
                # Try full fname first (already includes extension), then vid+ext
                vpath = None
                direct = os.path.join(video_dir, fname)
                if os.path.exists(direct):
                    vpath = direct
                else:
                    for ext in (".mp4", ".webm", ".mkv", ".avi"):
                        p = os.path.join(video_dir, vid + ext)
                        if os.path.exists(p):
                            vpath = p
                            break
                if vpath and vid not in unique:
                    if clip_set is None or vid in clip_set:
                        unique[vid] = vpath
    if clip_set is None and limit:
        unique = dict(list(unique.items())[:limit])
    return unique


# ── main extraction ───────────────────────────────────────────────────────────
def extract(
    video_dir: str,
    cache_dir: str = CACHE_DIR,
    limit: Optional[int] = None,
    decode_timeout: int = 30,
    rank: int = 0,
    world_size: int = 1,
    train_only: bool = False,
    clip_list: Optional[List[str]] = None,
) -> None:
    # ── /dev/shm gate ─────────────────────────────────────────────────────
    if rank == 0:
        _check_shm()

    os.makedirs(cache_dir, exist_ok=True)
    tmp_dir = os.path.join(cache_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    if rank == 0:
        print(f"[av-extract] Collecting clips from {video_dir} ...", flush=True)
    all_clips = _collect_clips(video_dir, limit=limit, rank=rank,
                               train_only=train_only, clip_list=clip_list)
    if rank == 0:
        print(f"[av-extract] {len(all_clips)} clips total", flush=True)

    # ── shard across ranks ───────────────────────────────────────────────
    items      = sorted(all_clips.items())
    rank_items = items[rank::world_size]

    todo = [(v, p) for v, p in rank_items
            if not os.path.exists(_feat_path(cache_dir, v))]
    if rank == 0:
        print(
            f"[av-extract rank={rank}] {len(rank_items)-len(todo)} cached, "
            f"{len(todo)} to do.",
            flush=True,
        )
    if not todo:
        if rank == 0:
            _write_manifest(cache_dir, BASE_TOKEN_RATE_HZ, NAT_TOKEN_RATE_HZ)
        return

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — Decode (pre-CUDA, subprocess)
    # ════════════════════════════════════════════════════════════════════
    print(f"[av-extract rank={rank}] PHASE 1: decoding {len(todo)} clips...",
          flush=True)
    t0 = time.time()
    decoded: List[Tuple[str, str]] = []
    n_fail_dec = 0
    for vid, path in todo:
        try:
            tmp = _decode_timeout(vid, path, tmp_dir, decode_timeout)
            decoded.append((vid, tmp))
        except Exception as exc:
            print(f"[av-extract] SKIP {vid}: {exc}", flush=True)
            n_fail_dec += 1
        done = len(decoded) + n_fail_dec
        if done % 50 == 0 or done == len(todo):
            el   = time.time() - t0
            rate = done / el if el > 0 else 0
            eta  = (len(todo) - done) / rate if rate > 0 else 0
            print(f"[av-extract rank={rank}] dec {len(decoded)}/{len(todo)} "
                  f"ok, {n_fail_dec} fail, {rate:.1f} v/s, ETA {eta/60:.0f}m",
                  flush=True)

    print(f"[av-extract rank={rank}] PHASE 1 done: {len(decoded)} decoded, "
          f"{n_fail_dec} failed, {time.time()-t0:.0f}s", flush=True)
    if not decoded:
        return

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — Encode (CUDA OK now)
    # ════════════════════════════════════════════════════════════════════
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"[av-extract rank={rank}] PHASE 2: loading encoders on {device}...",
          flush=True)

    vis_enc  = VisionEncoder(VISION_REPO, dtype=torch.bfloat16, device=device)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=device)
    nat_enc  = AudioEncoder(WAVJEPA_NAT_REPO,  n_channels=2, device=device)

    base_hz = base_enc.token_rate_hz   # measured at load
    nat_hz  = nat_enc.token_rate_hz    # measured at load

    print(
        f"[av-extract rank={rank}] encoders ready  "
        f"vision={VISION_TOKEN_RATE:.2f}Hz  "
        f"base={base_hz:.2f}Hz  nat={nat_hz:.2f}Hz",
        flush=True,
    )

    # Precompute static vision timestamps (same for every clip)
    vis_ts = _vision_ts()   # (32, 2) f32

    t1 = time.time()
    n_enc = 0; n_fail_enc = 0

    for vid, tmp_path in decoded:
        try:
            payload = torch.load(tmp_path, weights_only=True)
            frames: Tensor = payload["frames"]   # (T, C, H, W) uint8
            audio:  Tensor = payload["audio"]    # (n_samples,) f32

            clip_dur = audio.shape[0] / AUDIO_SR

            # ── Vision ────────────────────────────────────────────────
            with torch.no_grad():
                raw = vis_enc.encode([frames])   # (1, 8192, 1024)
            raw = raw[0]   # (8192, 1024)
            assert raw.shape[0] == VISION_TEMP * 256, \
                f"Expected {VISION_TEMP*256} vis tokens, got {raw.shape[0]}"
            vis_full   = raw.view(VISION_TEMP, 256, VISION_DIM)
            vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16).cpu()  # (32,16,1024)

            # ── Audio — base (1ch) ────────────────────────────────────
            wav1 = audio.unsqueeze(0).unsqueeze(0)   # (1, 1, n_s)
            with torch.no_grad():
                bf = base_enc.encode(wav1.to(device))   # (1, T_b, 768) bf16
            base_feat = bf[0].cpu()   # (T_b, 768)

            # ── Audio — nat (2ch, mono duplicated) ────────────────────
            wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0)  # (1, 2, n_s)
            with torch.no_grad():
                nf = nat_enc.encode(wav2.to(device))    # (1, T_n, 768) bf16
            nat_feat = nf[0].cpu()   # (T_n, 768)

            # ── Timestamps ────────────────────────────────────────────
            base_ts = _audio_ts(base_feat.shape[0], clip_dur)
            nat_ts  = _audio_ts(nat_feat.shape[0],  clip_dur)

            # ── Atomic write ──────────────────────────────────────────
            out_dir  = os.path.join(cache_dir, _shard(vid))
            os.makedirs(out_dir, exist_ok=True)
            dst = _feat_path(cache_dir, vid)
            tmp = dst + ".tmp"
            torch.save({
                "vision":          vis_pooled,   # (32,16,1024) bf16
                "ambient_base":    base_feat,    # (T_b,768)    bf16
                "ambient_nat":     nat_feat,     # (T_n,768)    bf16
                "vision_ts":       vis_ts,       # (32,2)       f32
                "ambient_base_ts": base_ts,      # (T_b,2)      f32
                "ambient_nat_ts":  nat_ts,       # (T_n,2)      f32
                "clip_duration_s": clip_dur,
            }, tmp)
            os.rename(tmp, dst)
            n_enc += 1

        except Exception as exc:
            import traceback
            print(f"[av-extract] ENC FAIL {vid}: {exc}", flush=True)
            traceback.print_exc()
            n_fail_enc += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        done = n_enc + n_fail_enc
        if done % 50 == 0 or done == len(decoded):
            el   = time.time() - t1
            rate = n_enc / el if el > 0 else 0
            eta  = (len(decoded) - done) / rate if rate > 0 else 0
            print(f"[av-extract rank={rank}] enc {n_enc}/{len(decoded)} "
                  f"ok, {n_fail_enc} fail, {rate:.1f} v/s, ETA {eta/60:.0f}m",
                  flush=True)

    print(f"[av-extract rank={rank}] PHASE 2 done: {n_enc} enc, "
          f"{n_fail_enc} fail, {time.time()-t1:.0f}s", flush=True)

    if rank == 0:
        _write_manifest(cache_dir, base_hz, nat_hz)
        print(f"[av-extract] /dev/shm used: {_shm_used_gb():.2f} GB", flush=True)

        # Sample shape report
        shown = 0
        for vid, _ in items:
            p = _feat_path(cache_dir, vid)
            if os.path.exists(p):
                d = torch.load(p, weights_only=True)
                print(
                    f"  {vid}: vision={tuple(d['vision'].shape)} "
                    f"ambient_base={tuple(d['ambient_base'].shape)} "
                    f"ambient_nat={tuple(d['ambient_nat'].shape)}",
                    flush=True,
                )
                shown += 1
                if shown >= 3:
                    break


# ── distributed setup ────────────────────────────────────────────────────────
def _init_dist() -> Tuple[int, int]:
    rank      = int(os.environ.get("LOCAL_RANK", 0))
    world     = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    return rank, world

def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default=None,
                        help="Path to video directory (overrides config).")
    parser.add_argument("--config", default="configs/m1.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit clips (32 for smoke, 50000 for subset).")
    parser.add_argument("--decode-timeout", type=int, default=30)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--train-only", action="store_true",
                        help="Use only data/train.csv (exclude test clips).")
    parser.add_argument("--clip-list", default=None,
                        help="Text file with one clip_id per line; extract only those clips.")
    args = parser.parse_args()

    rank, world = _init_dist()

    # Resolve video_dir
    video_dir = args.video_dir
    if not video_dir:
        from utils import load_config, cfg_get
        cfg       = load_config(args.config)
        video_dir = str(cfg_get(cfg, "data.video_dir"))

    clip_list: Optional[List[str]] = None
    if args.clip_list:
        with open(args.clip_list) as _f:
            clip_list = [l.strip() for l in _f if l.strip()]

    extract(
        video_dir=video_dir,
        cache_dir=args.cache_dir,
        limit=args.limit,
        decode_timeout=args.decode_timeout,
        rank=rank,
        world_size=world,
        train_only=args.train_only,
        clip_list=clip_list,
    )
    _cleanup_dist()


if __name__ == "__main__":
    main()
