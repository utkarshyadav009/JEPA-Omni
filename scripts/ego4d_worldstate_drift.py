"""scripts/ego4d_worldstate_drift.py — item 7: staleness moved from the
vision-refresh path to the generation path once the decision head stopped
consuming World-State (2026-07-26 diff). M3 has never been evaluated on a
stale World-State, and at max observed staleness (5.58s), generation would
be grounding on vision that's 5.6-15.6s old relative to the tick it's
narrating. Cheap first pass: measure real World-State cosine drift at lag
0/2/4/6s on continuous Ego4D footage (long-form video_540ss, real
continuous recordings, not EasyCom's 60s chunks) -- escalate to a full M3
grounding eval only if drift is substantial.

Method: for N sampled (file, reference_time) points, decode a REAL 10s
vision window ending at t0, and REAL 10s windows ending at t0-2/4/6s
(same file, same decode/pooling convention as the rest of this project),
compute REAL M2 World-State for each (vision-only -- audio not needed for
this specific measurement), report mean/median cosine similarity and mean
abs-diff at each lag, averaged across all sampled points.

Usage:
    python scripts/ego4d_worldstate_drift.py --n-points 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor

WINDOW_SEC = 10.0
LAGS = [0, 2, 4, 6]
VIDEO_FPS = 30.0
N_FRAMES = 64
RESOLUTION = 256


def decode_window_ending_at(path, decoder_cache, t_end, device):
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices

    if path not in decoder_cache:
        decoder_cache[path] = VideoDecoder(path, device="cpu")
    dec = decoder_cache[path]
    fps = VIDEO_FPS
    t_start = max(0.0, t_end - WINDOW_SEC)
    f0 = int(round(t_start * fps))
    f1 = int(round(t_end * fps))
    num_total = getattr(dec.metadata, "num_frames", None) or len(dec)
    f1 = min(f1, num_total)
    f0 = min(f0, f1 - 1)
    idx = _uniform_frame_indices(f1 - f0, N_FRAMES)
    abs_idx = sorted(set(i + f0 for i in idx))
    batch = dec.get_frames_at(indices=abs_idx)
    decoded = batch.data
    remap = {o: i for i, o in enumerate(abs_idx)}
    gi = torch.tensor([remap[i + f0] for i in idx], dtype=torch.long)
    frames = decoded.index_select(0, gi)
    if frames.shape[-2] != RESOLUTION or frames.shape[-1] != RESOLUTION:
        x = frames.float()
        x = F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-points", type=int, default=30)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/ego4d_worldstate_drift.json")
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[drift] device={device}", flush=True)

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    def compute_ws(frames):
        with torch.no_grad():
            v = vision_enc.encode(frames.unsqueeze(0).to(device))
            n_tok = v.shape[1]
            bin_idx = torch.linspace(0, predictor_cfg.max_tdm_bins - 1, n_tok, device=device).round().long()
            feats = {"vision": v.float()}
            tbins = {"vision": bin_idx.unsqueeze(0)}
            ws = predictor.encode_world_state(feats, tbins)[0].cpu()
        return ws

    files = sorted(glob.glob("/mnt/Raid-Storage-2/utkarsh-data/ego4d_probe/v2/video_540ss/*.mp4"))
    rng = random.Random(args.seed)
    rng.shuffle(files)

    # need reference times with at least 6+10=16s of runway from t=0
    import subprocess
    def duration(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", path],
                              capture_output=True, text=True, timeout=15).stdout.strip()
        return float(out)

    points = []
    for f in files:
        if len(points) >= args.n_points:
            break
        try:
            dur = duration(f)
        except Exception:
            continue
        if dur < 60.0:
            continue
        t0 = rng.uniform(30.0, dur - 10.0)
        points.append((f, t0))
    print(f"[drift] {len(points)} sample points across {len(set(p[0] for p in points))} files", flush=True)

    lag_cossim = {lag: [] for lag in LAGS}
    lag_absdiff = {lag: [] for lag in LAGS}
    decoder_cache: dict = {}

    for i, (f, t0) in enumerate(points):
        try:
            ws_ref = compute_ws(decode_window_ending_at(f, decoder_cache, t0, device))
        except Exception as e:
            print(f"[drift] point {i} ref decode FAILED: {e}", flush=True)
            continue
        for lag in LAGS:
            try:
                t_lag = t0 - lag
                if t_lag < 0:
                    continue
                ws_lag = compute_ws(decode_window_ending_at(f, decoder_cache, t_lag, device))
                cos = F.cosine_similarity(ws_ref.unsqueeze(0), ws_lag.unsqueeze(0)).item()
                absdiff = (ws_ref - ws_lag).abs().mean().item()
                lag_cossim[lag].append(cos)
                lag_absdiff[lag].append(absdiff)
            except Exception as e:
                print(f"[drift] point {i} lag={lag} FAILED: {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"[drift] {i+1}/{len(points)} points done", flush=True)
        decoder_cache.clear()  # avoid holding too many open decoders

    result = {}
    print("\n[drift] === World-State drift curve (real Ego4D continuous footage) ===", flush=True)
    for lag in LAGS:
        cs = lag_cossim[lag]
        ad = lag_absdiff[lag]
        if not cs:
            continue
        result[f"lag_{lag}s"] = {
            "n": len(cs),
            "mean_cosine_sim": float(np.mean(cs)),
            "median_cosine_sim": float(np.median(cs)),
            "mean_abs_diff": float(np.mean(ad)),
        }
        print(f"lag={lag}s  n={len(cs)}  mean_cos_sim={np.mean(cs):.4f}  "
              f"median_cos_sim={np.median(cs):.4f}  mean_abs_diff={np.mean(ad):.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[drift] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
