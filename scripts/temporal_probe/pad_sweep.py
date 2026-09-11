"""scripts/temporal_probe/pad_sweep.py — P0.2 diagnostic.

Question: the padding-corrected gallery eval scores ~23 R@1 points BELOW the
leaked one, and batch-size-1 (where no padding exists and the two code paths are
mathematically identical) agrees with the CORRECTED number. So padding does not
merely perturb the result -- its presence is worth +23 points. Two possible causes:

  H1 INFORMATION LEAK. The pad COUNT is clip-specific (max_Ta(batch) - T_a(i)),
     so it smuggles the clip's audio length into both z_v and z_a, letting the
     retrieval match on length. Falsified already for the simple form: the
     gallery has only 53 distinct T_a values and 2.3% unique
     (docs/artifacts/temporal_probe/duration_only_control.json), far too coarse
     to drive 53% R@1.

  H2 TRAIN/TEST DISTRIBUTION SHIFT. Training also used av_collate_fn with
     padding unmasked, so the model learned to USE pad tokens (attention sinks /
     registers). Removing them at eval is an intervention the weights never saw.

This script separates them. Every clip is run ALONE (batch size 1) with exactly
P appended zero-feature ambient pad tokens at bin 0. P is IDENTICAL for every
clip, so the pad count carries no clip-specific information whatsoever -- H1
predicts no recovery, H2 predicts R@1 climbs back toward the leaked value.

Usage:
    python scripts/temporal_probe/pad_sweep.py --pads 0,10,50,100,200,400 \
        --out docs/artifacts/temporal_probe/pad_sweep.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset
from train_m2 import pool_and_project
from utils import load_config, cfg_get

EXPECTED_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


@torch.no_grad()
def run(predictor, vp, ap, ds, device, P: int, rand_pad: int = 0, seed: int = 0) -> Dict[str, float]:
    """P: a FIXED number of pad tokens appended to every clip (uniform -- carries
    no clip-specific information).

    rand_pad > 0: instead, give each clip a pad count drawn uniformly from
    [0, rand_pad). The count is drawn ONCE PER CLIP and is therefore IDENTICAL
    for that clip's vision pass and its ambient pass, while being UNRELATED to
    anything about the clip. This is the decisive test for a shared-nuisance
    leak: a per-clip variable injected identically into both views inflates
    retrieval even when it is pure noise. If R@1 rises, the batch-padding gain
    is a leak, not a representation property."""
    g = torch.Generator().manual_seed(seed)
    pad_counts = (torch.randint(0, rand_pad, (len(ds),), generator=g).tolist()
                  if rand_pad > 0 else None)
    zv, za = [], []
    for i in range(len(ds)):
        s = ds[i]
        v = s["feats"]["vision"].unsqueeze(0).to(device).float()
        a = s["feats"]["ambient"].unsqueeze(0).to(device).float()
        vb = s["tbins"]["vision"].unsqueeze(0).to(device)
        ab = s["tbins"]["ambient"].unsqueeze(0).to(device)
        n_pad = pad_counts[i] if pad_counts is not None else P
        if n_pad > 0:
            P = n_pad
            # EXACTLY the tokens av_collate_fn would append: zero features, bin 0.
            a = torch.cat([a, torch.zeros(1, P, a.shape[-1], device=device)], 1)
            ab = torch.cat([ab, torch.zeros(1, P, dtype=ab.dtype, device=device)], 1)
        feats = {"vision": v, "ambient": a}
        tbins = {"vision": vb, "ambient": ab}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            z_v, z_a = pool_and_project(predictor, vp, ap, feats, tbins)
        zv.append(z_v.float().cpu()); za.append(z_a.float().cpu())
    z_v = torch.cat(zv, 0); z_a = torch.cat(za, 0)
    N = z_v.shape[0]; gt = torch.arange(N); sim = z_v @ z_a.T
    out = {}
    for name, ranked in [("vision->ambient", (-sim).argsort(1)),
                         ("ambient->vision", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            out[f"{name}_R@{k}"] = round((ranked[:, :k] == gt[:, None]).any(1).float().mean().item() * 100, 2)
    out["matched_cos_sim"] = round(sim.diagonal().mean().item(), 4)
    out["n_clips"] = N
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m2.yaml")
    p.add_argument("--ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--cache-dir", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k")
    p.add_argument("--eval-subset", default="data/vggsound_eval_1545.txt")
    p.add_argument("--pads", default="0,10,50,100,200,400")
    p.add_argument("--rand-pads", default="",
                   help="comma-separated upper bounds; each clip gets a RANDOM pad count "
                        "in [0,bound), identical for its vision and ambient pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sha = sha256_file(args.ckpt)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"ABORT: ckpt sha256 {sha} != {EXPECTED_SHA}")

    device = torch.device("cuda:0")
    cfg = load_config(args.config)
    pcfg = AVJepaConfig(
        d_model=int(cfg_get(cfg, "model.d_model", default=1024)),
        depth=int(cfg_get(cfg, "model.depth", default=8)),
        heads=int(cfg_get(cfg, "model.heads", default=8)),
        mlp_ratio=float(cfg_get(cfg, "model.mlp_ratio", default=4.0)),
        max_tdm_bins=int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
        dropout=float(cfg_get(cfg, "model.dropout", default=0.0)))
    predictor = AVJepaPredictor(pcfg).to(device)
    vp = nn.Linear(pcfg.d_model, 256).to(device)
    ap = nn.Linear(pcfg.d_model, 256).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ck["model"]); vp.load_state_dict(ck["vision_proj"])
    ap.load_state_dict(ck["ambient_proj"])
    predictor.eval(); vp.eval(); ap.eval()

    ids = [l.strip() for l in open(args.eval_subset) if l.strip()]
    ds = AVCachedDataset(cache_dir=args.cache_dir, clip_ids=ids,
                         max_tdm_bins=pcfg.max_tdm_bins, audio_mode="mean")
    assert len(ds) == 1545

    runs = []
    for P in [int(x) for x in args.pads.split(",") if x.strip()]:
        t0 = time.time()
        m = run(predictor, vp, ap, ds, device, P)
        m.update({"uniform_pad_tokens": P, "rand_pad_bound": 0, "batch_size": 1,
                  "wall_s": round(time.time() - t0, 1)})
        runs.append(m)
        print(f"[pad_sweep] uniform P={P:<5} {json.dumps(m)}", flush=True)
    for R in [int(x) for x in args.rand_pads.split(",") if x.strip()]:
        t0 = time.time()
        m = run(predictor, vp, ap, ds, device, 0, rand_pad=R, seed=args.seed)
        m.update({"uniform_pad_tokens": 0, "rand_pad_bound": R, "seed": args.seed,
                  "batch_size": 1, "wall_s": round(time.time() - t0, 1)})
        runs.append(m)
        print(f"[pad_sweep] RANDOM per-clip pad in [0,{R}) {json.dumps(m)}", flush=True)

    json.dump({"script": "scripts/temporal_probe/pad_sweep.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ckpt": args.ckpt, "ckpt_sha256": sha, "gallery": args.eval_subset,
               "gallery_size": len(ds), "runs": runs},
              open(args.out, "w"), indent=2)
    print(f"[pad_sweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
