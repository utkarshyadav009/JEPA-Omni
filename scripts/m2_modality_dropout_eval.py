"""scripts/m2_modality_dropout_eval.py — M2 fusion robustness gate: World-State
self-retrieval when one modality is FULLY ZEROED at test time.

Context: the existing contrastive retrieval eval (vision->ambient / ambient->
vision R@1) is architecturally ALREADY modality-isolated per direction --
pool_and_project() calls encode_source_tokens(), which runs a masked backbone
pass per modality where every OTHER modality is entirely mask-token'd before
the pass. So that number never told us anything new about zeroing a modality.

What it does NOT test is the FUSED World-State (encode_world_state(): full
unmasked sequence, both modalities genuinely present, single joint pass --
this is what SIGReg/eff_rank/the M3 probe read from). This script tests
exactly that representation's robustness to a modality vanishing at inference:

  1. gallery_ws  = encode_world_state(feats, tbins)                    (clean, both modalities)
  2. vision_only_ws = encode_world_state(feats with ambient zeroed, tbins)
  3. audio_only_ws  = encode_world_state(feats with vision zeroed, tbins)

Then: does vision_only_ws[i] / audio_only_ws[i] still rank gallery_ws[i]
highest among the full gallery (self-retrieval, cosine similarity)? L2-
normalisation here is eval-only (cosine ranking) -- does not touch training,
where World-State must stay un-normalised for SIGReg.

Usage:
    python scripts/m2_modality_dropout_eval.py --ckpt checkpoints/m2_fusion_fullscale/step19000.pt \\
        --cache-dir /home/utkarsh/raid2-data/feature_cache_vgg51k \\
        --eval-subset data/vggsound_eval_1545.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn


@torch.no_grad()
def extract_world_states(
    predictor: AVJepaPredictor,
    cache_dir: str,
    clip_ids: List[str],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Returns (clean_ws, vision_only_ws, audio_only_ws, clip_ids_seen), each (N, d)."""
    ds = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids, max_tdm_bins=512, audio_mode="mean")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         collate_fn=av_collate_fn, drop_last=False)
    predictor.eval()
    clean_all, vonly_all, aonly_all = [], [], []
    seen_ids: List[str] = []
    t0 = time.time()
    n_done = 0
    for batch in loader:
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}

        feats_vision_only = dict(feats)
        feats_vision_only["ambient"] = torch.zeros_like(feats["ambient"])
        feats_audio_only = dict(feats)
        feats_audio_only["vision"] = torch.zeros_like(feats["vision"])

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            clean_ws = predictor.encode_world_state(feats, tbins)
            vonly_ws = predictor.encode_world_state(feats_vision_only, tbins)
            aonly_ws = predictor.encode_world_state(feats_audio_only, tbins)

        clean_all.append(clean_ws.float().cpu())
        vonly_all.append(vonly_ws.float().cpu())
        aonly_all.append(aonly_ws.float().cpu())
        seen_ids.extend(batch["clip_ids"])
        n_done += clean_ws.shape[0]
        if n_done % (batch_size * 5) == 0:
            print(f"[m2-dropout] extracted {n_done}/{len(clip_ids)}  "
                  f"{n_done / (time.time() - t0):.1f} clips/s", flush=True)
    print(f"[m2-dropout] extracted {n_done}/{len(clip_ids)} total, {time.time()-t0:.1f}s", flush=True)
    return (torch.cat(clean_all, 0), torch.cat(vonly_all, 0),
            torch.cat(aonly_all, 0), seen_ids)


def self_retrieval(query: torch.Tensor, gallery: torch.Tensor) -> Dict[str, float]:
    """R@1/5/10 for query[i] retrieving gallery[i] among the full gallery."""
    q = F.normalize(query, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.T
    N = sim.shape[0]
    gt = torch.arange(N)
    ranked = (-sim).argsort(1)
    out: Dict[str, float] = {}
    for k in (1, 5, 10):
        hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    out["n_clips"] = float(N)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--eval-subset", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.eval_subset) as f:
        eval_clip_ids = [l.strip() for l in f if l.strip()]
    print(f"[m2-dropout] eval subset: {len(eval_clip_ids)} clips requested", flush=True)

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                  max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    print(f"[m2-dropout] loaded {args.ckpt}  (step={ckpt.get('step')})", flush=True)

    clean_ws, vonly_ws, aonly_ws, seen_ids = extract_world_states(
        predictor, args.cache_dir, eval_clip_ids, device, batch_size=args.batch_size)
    assert len(seen_ids) == 1545, f"clips_seen={len(seen_ids)} != 1545 -- gallery is not the full fixed set"

    vision_only_res = self_retrieval(vonly_ws, clean_ws)
    audio_only_res = self_retrieval(aonly_ws, clean_ws)

    print(f"[m2-dropout] VISION-ONLY query (ambient zeroed) vs clean gallery: {vision_only_res}", flush=True)
    print(f"[m2-dropout] AUDIO-ONLY  query (vision zeroed)  vs clean gallery: {audio_only_res}", flush=True)
    avg_r1 = (vision_only_res["R@1"] + audio_only_res["R@1"]) / 2
    print(f"[m2-dropout] DONE. avg modality-dropout self-retrieval R@1 = {avg_r1:.2f}%  "
          f"(vision_only={vision_only_res['R@1']:.2f}%  audio_only={audio_only_res['R@1']:.2f}%)",
          flush=True)


if __name__ == "__main__":
    main()
