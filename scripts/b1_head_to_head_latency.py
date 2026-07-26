"""scripts/b1_head_to_head_latency.py — clean, no-contention, forward-only
head-to-head: current V-JEPA2 ViT-L (256x256) vs V-JEPA 2.1 ViT-B (384x384)
vs V-JEPA 2.1 ViT-L (384x384), same machine, same script, same batch
size, warmed up, n>=20. Decides whether ViT-B's smaller param count
actually buys lower latency once the 384-resolution token-count penalty
(2.25x tokens vs 256) is accounted for -- it does not follow automatically
from param count alone, per the smoke-test finding this script exists to
verify cleanly.

Uses random input tensors (NOT decoded video) -- this isolates encoder
forward cost specifically, deliberately excluding decode, matching the
instruction "forward-only latency and peak activation memory."

Usage:
    python scripts/b1_head_to_head_latency.py --n 20 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _clean_backbone_key(state_dict):
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def load_vjepa21(model_name: str, ckpt_path: str, device):
    encoder, _predictor = torch.hub.load("facebookresearch/vjepa2", model_name, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _clean_backbone_key(ckpt["ema_encoder"])
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"key mismatch: missing={missing} unexpected={unexpected}"
    return encoder.to(device).eval()


def bench(name, fwd_fn, make_input, device, n, warmup):
    for _ in range(warmup):
        with torch.no_grad():
            fwd_fn(make_input())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    starts, ends = [], []
    for _ in range(n):
        x = make_input()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        with torch.no_grad():
            fwd_fn(x)
        e1.record()
        torch.cuda.synchronize()
        starts.append(e0.elapsed_time(e1))  # ms
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    import statistics as st
    return {
        "name": name,
        "n": n,
        "latency_ms_mean": st.mean(starts),
        "latency_ms_median": st.median(starts),
        "latency_ms_min": min(starts),
        "latency_ms_max": max(starts),
        "peak_activation_mb": peak_mb,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default="checkpoints/vjepa21_shelved/HEAD_TO_HEAD_LATENCY.json")
    args = p.parse_args()

    assert torch.cuda.is_available(), "this benchmark requires a CUDA device"
    device = torch.device("cuda")
    print(f"[h2h] device={torch.cuda.get_device_name(0)}", flush=True)

    results = []

    # 1. Current V-JEPA2 ViT-L, 256x256, 64 frames, bf16 (matches models/vision_encoder.py exactly)
    print("[h2h] loading current V-JEPA2 ViT-L (256x256, bf16, our production encoder)...", flush=True)
    from models.vision_encoder import VisionEncoder
    v2_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)

    def make_input_v2():
        return torch.randint(0, 256, (args.batch_size, 64, 3, 256, 256), dtype=torch.uint8, device=device)

    def fwd_v2(x):
        return v2_enc.encode(x)

    results.append(bench("vjepa2_vitl_256_bf16_CURRENT_PRODUCTION", fwd_v2, make_input_v2, device, args.n, args.warmup))
    del v2_enc
    torch.cuda.empty_cache()

    # 2. V-JEPA 2.1 ViT-B, 384x384, 64 frames, autocast bf16 (matches scripts/b1_vjepa21_throughput.py)
    print("[h2h] loading V-JEPA 2.1 ViT-B (384x384)...", flush=True)
    vitb = load_vjepa21("vjepa2_1_vit_base_384", "checkpoints/vjepa21_shelved/vjepa2_1_vitb_dist_vitG_384.pt", device)

    def make_input_384(bs=args.batch_size):
        return torch.randn(bs, 3, 64, 384, 384, device=device)  # (B,C,T,H,W)

    def fwd_vitb(x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return vitb(x)

    results.append(bench("vjepa2_1_vitb_384_autocast_bf16", fwd_vitb, make_input_384, device, args.n, args.warmup))
    del vitb
    torch.cuda.empty_cache()

    # 3. V-JEPA 2.1 ViT-L, 384x384, 64 frames, autocast bf16
    vitl_ckpt = "checkpoints/vjepa21_shelved/vjepa2_1_vitl_dist_vitG_384.pt"
    if os.path.isfile(vitl_ckpt):
        print("[h2h] loading V-JEPA 2.1 ViT-L (384x384)...", flush=True)
        vitl = load_vjepa21("vjepa2_1_vit_large_384", vitl_ckpt, device)

        def fwd_vitl(x):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return vitl(x)

        results.append(bench("vjepa2_1_vitl_384_autocast_bf16", fwd_vitl, make_input_384, device, args.n, args.warmup))
        del vitl
        torch.cuda.empty_cache()
    else:
        print(f"[h2h] SKIPPING V-JEPA 2.1 ViT-L: {vitl_ckpt} not downloaded yet", flush=True)

    print(json.dumps(results, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[h2h] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
