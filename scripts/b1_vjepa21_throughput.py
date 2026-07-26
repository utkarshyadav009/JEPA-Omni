"""scripts/b1_vjepa21_throughput.py — B1: measure real V-JEPA 2.1 ViT-B
encoder throughput on 1000 real VGGSound clips, mercury (RTX PRO 6000
Blackwell).

Loads the architecture via torch.hub (pretrained=False -- the repo's own
pretrained-loading path is broken, see PROVENANCE note: backbones.py's
VJEPA_BASE_URL is hardcoded to http://localhost:8300, a left-over test
stub, not the real https://dl.fbaipublicfiles.com/vjepa2 URL), then loads
our already-downloaded, already-verified checkpoint
(checkpoints/vjepa21_shelved/vjepa2_1_vitb_dist_vitG_384.pt) manually with
the exact same key-cleaning logic backbones.py itself uses
(state_dict["ema_encoder"], strip "module."/"backbone." prefixes).

Usage:
    python scripts/b1_vjepa21_throughput.py --n-clips 1000
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

VGGSOUND_DIR = "/home/utkarsh/data/vggsound"
NUM_FRAMES = 64
IMG_SIZE = 384
TUBELET = 2
CLIP_DURATION_S = 10.0


def _clean_backbone_key(state_dict):
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def load_vjepa21_vitb(ckpt_path: str, device: torch.device):
    encoder, _predictor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_1_vit_base_384", pretrained=False
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _clean_backbone_key(ckpt["ema_encoder"])
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"[b1] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[b1]   missing[:5]={missing[:5]}", flush=True)
    if unexpected:
        print(f"[b1]   unexpected[:5]={unexpected[:5]}", flush=True)
    encoder = encoder.to(device).eval()
    return encoder


def collect_vggsound_clips(n: int):
    paths = []
    for csv_name in ("train.csv", "test.csv"):
        csv_path = os.path.join(PROJECT_ROOT, "data", csv_name)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                fname = row[0].strip()
                vpath = os.path.join(VGGSOUND_DIR, fname)
                if os.path.isfile(vpath):
                    paths.append(vpath)
                if len(paths) >= n:
                    return paths
    return paths


def decode_clip_384(video_path: str) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices

    decoder = VideoDecoder(video_path, device="cpu")
    num_total = getattr(decoder.metadata, "num_frames", None) or len(decoder)
    idx = _uniform_frame_indices(num_total, NUM_FRAMES)
    batch = decoder.get_frames_at(indices=sorted(set(idx)))
    decoded = batch.data
    remap = {o: i for i, o in enumerate(sorted(set(idx)))}
    gi = torch.tensor([remap[i] for i in idx], dtype=torch.long)
    frames = decoded.index_select(0, gi)
    if frames.shape[-2] != IMG_SIZE or frames.shape[-1] != IMG_SIZE:
        x = frames.float()
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames  # (T,C,H,W) uint8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clips", type=int, default=1000)
    p.add_argument("--ckpt", default="checkpoints/vjepa21_shelved/vjepa2_1_vitb_dist_vitG_384.pt")
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/B1_THROUGHPUT_PROVENANCE.txt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[b1] device={dev_name}", flush=True)

    print("[b1] loading V-JEPA 2.1 ViT-B (local checkpoint, torch.hub arch only)...", flush=True)
    encoder = load_vjepa21_vitb(args.ckpt, device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[b1] encoder params: {n_params/1e6:.1f}M  embed_dim={encoder.embed_dim}", flush=True)

    print(f"[b1] collecting {args.n_clips} real VGGSound clip paths...", flush=True)
    clip_paths = collect_vggsound_clips(args.n_clips)
    print(f"[b1] got {len(clip_paths)} clip paths", flush=True)

    decode_times = []
    fwd_times = []
    n_ok = 0
    t_wall_start = time.time()
    for i, vp in enumerate(clip_paths):
        try:
            td0 = time.time()
            frames = decode_clip_384(vp).to(device)
            td1 = time.time()
            with torch.no_grad():
                # VideoDecoder returns (T,C,H,W) uint8 -> encoder expects (B,C,T,H,W) float
                x = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).float()
                torch.cuda.synchronize() if device.type == "cuda" else None
                tf0 = time.time()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    out = encoder(x)
                torch.cuda.synchronize() if device.type == "cuda" else None
                tf1 = time.time()
            decode_times.append(td1 - td0)
            fwd_times.append(tf1 - tf0)
            n_ok += 1
        except Exception as e:
            print(f"[b1] clip {i} failed: {e}", flush=True)
            continue
        if (i + 1) % 100 == 0:
            print(f"[b1] {i+1}/{len(clip_paths)} done, n_ok={n_ok}", flush=True)
    t_wall_end = time.time()

    import statistics as st
    total_wall_s = t_wall_end - t_wall_start
    result = {
        "device": dev_name,
        "n_requested": args.n_clips,
        "n_attempted": len(clip_paths),
        "n_ok": n_ok,
        "total_wall_s": total_wall_s,
        "sec_per_clip_wall_mean": total_wall_s / max(1, n_ok),
        "decode_s_mean": st.mean(decode_times) if decode_times else None,
        "decode_s_median": st.median(decode_times) if decode_times else None,
        "fwd_s_mean": st.mean(fwd_times) if fwd_times else None,
        "fwd_s_median": st.median(fwd_times) if fwd_times else None,
        "encoder_params_M": n_params / 1e6,
        "embed_dim": encoder.embed_dim,
    }
    print(json.dumps(result, indent=2), flush=True)

    vggsound_total = 199176  # confirmed on-disk count, see report
    proj_vggsound_days = vggsound_total * result["sec_per_clip_wall_mean"] / 86400.0
    result["vggsound_total_clips_ondisk"] = vggsound_total
    result["projected_full_vggsound_extraction_days"] = proj_vggsound_days

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out.replace(".txt", ".json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[b1] wrote {args.out.replace('.txt', '.json')}", flush=True)


if __name__ == "__main__":
    main()
