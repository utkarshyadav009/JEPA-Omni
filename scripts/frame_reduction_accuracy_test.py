"""scripts/frame_reduction_accuracy_test.py — real accuracy-vs-latency test for
reducing ViT-L's input frame count (64 -> 32 -> 16), per direct user request
following the Jetson latency investigation (2026-08-04 falsifier_tracking.md
entry: 64-frame ViT-L forward = 1907ms, 81% of total pipeline latency;
TensorRT/onnxsim ruled out as fixes, both hit the same NvMap OOM materializing
the full 8192-token attention matrix; empirically confirmed the 32/16/8-frame
forward passes run WITHOUT error or resampling, giving 2.32x/4.77x/7.98x real
latency speedup on Jetson -- but that only proves it runs, not that the
resulting embeddings are still good for retrieval).

Reuses the EXACT held-out VGGSound split and eval logic
(train_m3.build_splits + train_m2_embed_predictor.retrieval_eval's approach)
so results are directly comparable to the already-reported baseline
(step 799 best.pt: VGGSound pred->text_R@1=34.3, text->pred_R@1=34.6, n=1000).
Does LIVE decode+encode at each frame count (not the cache, which is
64-frames-only) so all three conditions go through IDENTICAL code, isolating
frame count as the only variable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.predictor import Predictor
from models.text_target import TextTarget
from scripts.extract_features_av import (
    _decode_video_raw, _decode_audio_raw, _spatial_pool,
    _vision_ts, _audio_ts, RESOLUTION, CLIP_DURATION_S,
)
from data.av_cached_dataset import _ts_to_tdm_bins
from train_m3 import build_splits

VIDEO_DIR = "/home/utkarsh/data/vggsound"
M2_CKPT = "checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt"
PREDICTOR_CKPT = "checkpoints/m2_embed_predictor_mlp_ddp_gradcache_bs16384/best.pt"
FIELD = "gpt_action_detailed"


def find_video_path(clip_id: str) -> str:
    p = os.path.join(VIDEO_DIR, clip_id + ".mp4")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(clip_id)


def get_held_out_1000(field: str = FIELD, seed: int = 0) -> List[Tuple[str, str, str]]:
    """Mirrors train_m2_embed_predictor.main()'s EXACT rng sequence so this
    reproduces the identical held-out set used to report the 64-frame
    baseline numbers (step 799 best.pt), regardless of --n-clips (truncation
    happens after the shuffle, so it doesn't affect rng consumption order)."""
    rng = random.Random(seed)
    train_pairs, test_pairs = build_splits(field)
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    return test_pairs[:1000]


@torch.no_grad()
def encode_clip_av(vision_enc, wavjepa_base, wavjepa_nat, video_path: str,
                    num_frames: int, device) -> Dict:
    frames = _decode_video_raw(video_path, num_frames, RESOLUTION).to(device)
    audio = _decode_audio_raw(video_path).to(device)

    vis_full = vision_enc.encode(frames.unsqueeze(0))  # (1, N, 1024)
    n_temp = num_frames // 2
    B, N, D = vis_full.shape
    assert N == n_temp * 256, f"unexpected token count {N} for {num_frames} frames"
    vis_full = vis_full.view(n_temp, 256, D)
    vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16).cpu()  # (n_temp, 16, 1024)
    vis_ts = _vision_ts(n_temp=n_temp, dur=CLIP_DURATION_S, n_frames=num_frames)

    wav1 = audio.unsqueeze(0).unsqueeze(0)
    wav2 = audio.unsqueeze(0).expand(2, -1).unsqueeze(0)
    base = wavjepa_base.encode(wav1).squeeze(0).to(torch.bfloat16).cpu()
    nat = wavjepa_nat.encode(wav2).squeeze(0).to(torch.bfloat16).cpu()
    clip_dur = CLIP_DURATION_S
    base_ts = _audio_ts(base.shape[0], clip_dur)
    nat_ts = _audio_ts(nat.shape[0], clip_dur)

    return {
        "vision": vis_pooled, "vision_ts": vis_ts,
        "ambient_base": base, "ambient_base_ts": base_ts,
        "ambient_nat": nat, "ambient_nat_ts": nat_ts,
        "clip_duration_s": clip_dur,
    }


def to_flat_feats_tbins(d: Dict, max_tdm_bins: int = 512) -> Dict:
    """Mirrors AVCachedDataset.__getitem__ exactly, just from an in-memory
    dict instead of a loaded .pt file."""
    vis = d["vision"]
    T_v, S_v, D_v = vis.shape
    vis_flat = vis.reshape(T_v * S_v, D_v)
    vis_ts_exp = d["vision_ts"].unsqueeze(1).expand(T_v, S_v, 2).reshape(T_v * S_v, 2)
    vis_bins = _ts_to_tdm_bins(vis_ts_exp, d["clip_duration_s"], max_tdm_bins)

    base, nat = d["ambient_base"], d["ambient_nat"]
    if base.shape[0] == nat.shape[0]:
        aud = (base.float() + nat.float()).mul_(0.5).to(torch.bfloat16)
        aud_ts = d["ambient_base_ts"]
    else:
        aud, aud_ts = base, d["ambient_base_ts"]
    aud_bins = _ts_to_tdm_bins(aud_ts, d["clip_duration_s"], max_tdm_bins)

    return {"feats": {"vision": vis_flat, "ambient": aud},
            "tbins": {"vision": vis_bins, "ambient": aud_bins}}


def collate(items: List[Dict]) -> Dict:
    B = len(items)
    max_v = max(it["feats"]["vision"].shape[0] for it in items)
    max_a = max(it["feats"]["ambient"].shape[0] for it in items)
    D_v = items[0]["feats"]["vision"].shape[-1]
    D_a = items[0]["feats"]["ambient"].shape[-1]
    vis = torch.zeros(B, max_v, D_v, dtype=torch.bfloat16)
    aud = torch.zeros(B, max_a, D_a, dtype=torch.bfloat16)
    vis_bins = torch.zeros(B, max_v, dtype=torch.long)
    aud_bins = torch.zeros(B, max_a, dtype=torch.long)
    for i, it in enumerate(items):
        nv = it["feats"]["vision"].shape[0]
        na = it["feats"]["ambient"].shape[0]
        vis[i, :nv] = it["feats"]["vision"]
        aud[i, :na] = it["feats"]["ambient"]
        vis_bins[i, :nv] = it["tbins"]["vision"]
        aud_bins[i, :na] = it["tbins"]["ambient"]
    return {"feats": {"vision": vis, "ambient": aud},
            "tbins": {"vision": vis_bins, "ambient": aud_bins}}


@torch.no_grad()
def run_condition(num_frames: int, held_out: List[Tuple[str, str, str]],
                   vision_enc, wavjepa_base, wavjepa_nat, m2, predictor, text_target,
                   device, batch_size: int = 8, max_clips: int = 1000) -> Dict[str, float]:
    zp_all, zt_all = [], []
    n_done, n_failed = 0, 0
    t0 = time.time()
    buf_items, buf_caps = [], []

    def flush():
        nonlocal zp_all, zt_all
        if not buf_items:
            return
        batch = collate(buf_items)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pre_pool = m2.encode_pre_pool_tokens(feats, tbins)
        z_p = predictor(pre_pool)
        z_t = text_target.encode_text(buf_caps)
        zp_all.append(z_p.float().cpu()); zt_all.append(z_t.float().cpu())
        buf_items.clear(); buf_caps.clear()

    for cid, field, caption in held_out[:max_clips]:
        try:
            path = find_video_path(cid)
            d = encode_clip_av(vision_enc, wavjepa_base, wavjepa_nat, path, num_frames, device)
        except Exception as e:
            n_failed += 1
            continue
        buf_items.append(to_flat_feats_tbins(d))
        buf_caps.append(caption)
        n_done += 1
        if len(buf_items) >= batch_size:
            flush()
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [frames={num_frames}] {n_done}/{max_clips} done, {n_failed} failed, "
                  f"{elapsed:.0f}s elapsed ({elapsed/max(n_done,1):.2f}s/clip)", flush=True)
    flush()

    z_p = torch.cat(zp_all, 0)
    z_t = torch.cat(zt_all, 0)
    N = z_p.shape[0]
    gt = torch.arange(N)
    sim = z_p @ z_t.T
    results = {}
    for name, ranked in [("pred→text", (-sim).argsort(1)), ("text→pred", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
            results[f"{name}_R@{k}"] = round(hits * 100, 2)
    results["n_clips"] = float(N)
    results["n_failed"] = float(n_failed)
    results["elapsed_s"] = round(time.time() - t0, 1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-counts", type=int, nargs="+", default=[64, 32, 16])
    ap.add_argument("--max-clips", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="checkpoints/frame_reduction_accuracy_results.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    print(f"[setup] device={device}", flush=True)

    held_out = get_held_out_1000()
    print(f"[setup] held-out VGGSound eval set: {len(held_out)} clips (should match the "
          f"1000-clip set used to report step799 best.pt's R@1=34.3)", flush=True)

    print("[setup] loading frozen encoders + trained heads...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    m2_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2_cfg).to(device)
    m2ckpt = torch.load(M2_CKPT, map_location=device, weights_only=False)
    m2.load_state_dict(m2ckpt["model"], strict=True)
    m2.eval()

    predictor = Predictor(in_dim=m2_cfg.d_model, shared_dim=1536, mode="mlp").to(device)
    text_target = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False, device=str(device))
    pckpt = torch.load(PREDICTOR_CKPT, map_location=device, weights_only=False)
    predictor.load_state_dict(pckpt["predictor"], strict=True)
    text_target.proj.load_state_dict(pckpt["text_target_proj"], strict=True)
    predictor.eval(); text_target.base.eval()
    print(f"[setup] loaded predictor+text_target from step={pckpt['step']} "
          f"(reported baseline R@1: vgg={pckpt['results_log'][-1]['vggsound']})", flush=True)

    all_results = {}
    for nf in args.frame_counts:
        print(f"\n=== Running frame_count={nf} ({args.max_clips} clips, batch={args.batch_size}) ===", flush=True)
        res = run_condition(nf, held_out, vision_enc, wavjepa_base, wavjepa_nat, m2, predictor,
                             text_target, device, batch_size=args.batch_size, max_clips=args.max_clips)
        print(f"[result] frames={nf}: {json.dumps(res)}", flush=True)
        all_results[str(nf)] = res
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n[DONE] results written to {args.out}", flush=True)
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
