"""scripts/temporal_probe/bin_scramble_eval.py

Phase 0 / Phase 1 of the temporal-structure probe.

Eval-time-only intervention on the TDM bin INDEX VALUES fed to the
predictor's learned temporal embedding. Token features, token ORDER in the
sequence, and the modality embedding are all untouched -- only the integer
index used to look up `predictor.temporal_emb` is changed.

The metric path is byte-identical in structure to
train_m2.contrastive_retrieval_eval / train_m2.pool_and_project, which is
what produced every published R@k on this gallery. It is reimplemented here
(rather than imported) ONLY so the hook can sit between the dataloader and
pool_and_project; pool_and_project itself is IMPORTED, not copied.

In addition to the retrieval head, this script reports the effect on the
fused WORLD-STATE vector (predictor.encode_world_state) -- which the
retrieval eval never touches -- as cosine to the unintervened world-state.

Usage (one arm):
    torchrun --nproc_per_node=1 scripts/temporal_probe/bin_scramble_eval.py \
        --arm A2 --seed 0 \
        --ckpt checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt \
        --cache-dir /mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k \
        --out docs/artifacts/temporal_probe/phase1_A2_seed0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor, effective_rank
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import pool_and_project
from scripts.temporal_probe.padded_eval import (
    pool_and_project_masked, encode_world_state_masked)
from utils import load_config, cfg_get

EXPECTED_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
MID_BIN = 256   # A8: a single MID-RANGE bin, to separate "position carries signal" from
                # A5's confound (A5 uses bin 0, the same row padding tokens receive)
VISION_TEMP = 32
VISION_SPAT = 16


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- hook ----
def _perm_vision_groups(bins_row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """A2: permute the 32 GROUP bin values among the 32 groups; the 16 spatial
    tokens inside a group keep sharing one value (grouping intact, order gone)."""
    v = bins_row.view(VISION_TEMP, VISION_SPAT)
    group_vals = v[:, 0]                                     # (32,)
    perm = torch.randperm(VISION_TEMP, generator=g)
    return group_vals[perm].unsqueeze(1).expand(VISION_TEMP, VISION_SPAT).reshape(-1)


def _perm_vision_all(bins_row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """A3: permute all 512 vision bin values independently (grouping AND order gone)."""
    perm = torch.randperm(bins_row.numel(), generator=g)
    return bins_row[perm]


def _cyclic_vision(bins_row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """A4: cyclic shift of the 32 group values by a random offset in [1, 31]
    (offset 0 excluded -- it would be a no-op arm). Absolute position destroyed,
    relative order preserved except at the single wrap point."""
    v = bins_row.view(VISION_TEMP, VISION_SPAT)
    group_vals = v[:, 0]
    k = int(torch.randint(1, VISION_TEMP, (1,), generator=g).item())
    rolled = torch.roll(group_vals, shifts=k, dims=0)
    return rolled.unsqueeze(1).expand(VISION_TEMP, VISION_SPAT).reshape(-1)


def apply_intervention(
    tbins: Dict[str, torch.Tensor],
    pad: Dict[str, torch.Tensor],
    arm: str,
    g: torch.Generator,
    max_bins: int = 512,
) -> Dict[str, torch.Tensor]:
    """Returns a NEW tbins dict. Per-clip independent randomness (a single
    gallery-wide permutation would merely relabel the bin axis self-consistently,
    which the model could be invariant to for a trivial reason)."""
    if arm == "A0":
        return tbins
    vb = tbins["vision"].clone()
    ab = tbins["ambient"].clone()
    B = vb.shape[0]

    for i in range(B):
        if arm == "A1":
            perm = torch.randperm(VISION_TEMP, generator=g)   # drawn, then NOT applied:
            _ = perm                                          # identity => hook executes, no-op
        elif arm == "A2":
            vb[i] = _perm_vision_groups(vb[i], g)
        elif arm == "A3":
            vb[i] = _perm_vision_all(vb[i], g)
        elif arm == "A4":
            vb[i] = _cyclic_vision(vb[i], g)
        elif arm == "A5":
            vb[i] = torch.zeros_like(vb[i])
        elif arm == "A8":
            # same "collapse to one bin" as A5 but at a mid-range row. A5 and A8 differ
            # ONLY in which row; if they agree, A5's effect is about losing 32 DISTINCT
            # embeddings, not about absolute temporal position.
            vb[i] = torch.full_like(vb[i], MID_BIN)
        elif arm == "A10":
            # A9 CORRECTED. A9's arithmetic reflection (b -> max_bin-1-b) does NOT preserve
            # the multiset: vision uses rows {0,16,...,496} and reflection lands on
            # {15,31,...,511}, DISJOINT from the trained set -- so A9 is confounded exactly
            # like A5/A8 (out-of-distribution embedding rows), and cannot isolate direction.
            # A10 reverses the ORDER of the existing values instead: group i takes group
            # (n-1-i)'s value, and ambient token t takes token (T-1-t)'s value. The multiset
            # of embedding rows is preserved EXACTLY in both modalities; only temporal
            # direction changes. This is the clean direction test.
            v = vb[i].view(VISION_TEMP, VISION_SPAT)
            gv = v[:, 0].flip(0)
            vb[i] = gv.unsqueeze(1).expand(VISION_TEMP, VISION_SPAT).reshape(-1)
            real = (~pad["ambient"][i]).nonzero(as_tuple=True)[0]
            ab[i, real] = ab[i, real].flip(0)
        elif arm == "A9":
            # TIME REVERSAL, both modalities, consistently. Preserves the MULTISET of bin
            # embeddings exactly (so it defeats the multiset-invariance explanation for
            # A2/A3/A4) while reversing temporal direction. Padded ambient slots are left
            # alone so the arm changes direction, not padding.
            mx = int(max_bins) - 1
            vb[i] = mx - vb[i]
            real = (~pad["ambient"][i]).nonzero(as_tuple=True)[0]
            ab[i, real] = mx - ab[i, real]
        elif arm in ("A6", "A7"):
            if arm == "A7":
                vb[i] = _perm_vision_groups(vb[i], g)
            real = (~pad["ambient"][i]).nonzero(as_tuple=True)[0]
            if real.numel() > 1:
                perm = torch.randperm(real.numel(), generator=g)
                ab[i, real] = ab[i, real][perm]
        else:
            raise ValueError(f"unknown arm {arm}")
    return {"vision": vb, "ambient": ab}


# ---------------------------------------------------------------- eval ----
@torch.no_grad()
def run_arm(predictor, vision_proj, ambient_proj, loader, device, arm, seed,
            want_world_state: bool, fix_padding: bool = False, max_bins: int = 512) -> Dict:
    predictor.eval(); vision_proj.eval(); ambient_proj.eval()
    g = torch.Generator().manual_seed(seed)
    zv_all, za_all, ws_all, ids = [], [], [], []
    n_batches = 0
    for batch in loader:
        pad = batch["padding_mask"]
        tb_raw = batch["tbins"]
        assert tb_raw["vision"].shape[1] == VISION_TEMP * VISION_SPAT, \
            f"vision token count {tb_raw['vision'].shape[1]} != 512; staircase assumption broken"
        assert not pad["vision"].any(), "vision stream unexpectedly padded"
        tb = apply_intervention(tb_raw, pad, arm, g, max_bins=max_bins)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in tb.items()}
        padd = {k: v.to(device) for k, v in pad.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            if fix_padding:
                z_v, z_a = pool_and_project_masked(predictor, vision_proj, ambient_proj,
                                                   feats, tbins, padd)
                ws = (encode_world_state_masked(predictor, feats, tbins, padd)
                      if want_world_state else None)
            else:
                z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
                ws = predictor.encode_world_state(feats, tbins) if want_world_state else None
        zv_all.append(z_v.float().cpu()); za_all.append(z_a.float().cpu())
        if want_world_state:
            ws_all.append(ws.float().cpu())
        ids.extend(batch["clip_ids"])
        n_batches += 1

    z_v = torch.cat(zv_all, 0); z_a = torch.cat(za_all, 0)
    N = z_v.shape[0]
    gt = torch.arange(N)
    sim = z_v @ z_a.T
    res: Dict[str, float] = {}
    for name, ranked in [("vision->ambient", (-sim).argsort(1)),
                         ("ambient->vision", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            res[f"{name}_R@{k}"] = round(
                (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item() * 100, 2)

    res["matched_cos_sim"] = round(sim.diagonal().mean().item(), 4)
    # DETERMINISTIC shuffled pairing (the production eval used an unseeded
    # torch.randperm here; fixing the seed makes this column comparable across arms)
    pg = torch.Generator().manual_seed(12345)
    perm = torch.randperm(N, generator=pg)
    for i in range(N):
        if perm[i] == i:
            j = (i + 1) % N
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    res["shuffled_cos_sim"] = round(sim[torch.arange(N), perm].mean().item(), 4)
    res["shuffle_sanity_gap"] = round(res["matched_cos_sim"] - res["shuffled_cos_sim"], 4)
    res["n_clips"] = int(N)
    res["n_batches"] = n_batches

    out = {"metrics": res, "clip_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    if want_world_state:
        W = torch.cat(ws_all, 0)
        out["world_state"] = {
            "mean_l2_norm": round(W.norm(dim=-1).mean().item(), 4),
            "effective_rank": round(effective_rank(W), 2),
            "_tensor_sha256": hashlib.sha256(W.numpy().tobytes()).hexdigest(),
        }
        out["_W"] = W
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m2.yaml")
    p.add_argument("--ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--cache-dir", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k")
    p.add_argument("--eval-subset", default="data/vggsound_eval_1545.txt")
    p.add_argument("--arms", default="A0", help="comma-separated arm list")
    p.add_argument("--seeds", default="0", help="comma-separated seed list")
    p.add_argument("--contrast-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument("--no-world-state", action="store_true")
    p.add_argument("--fix-padding", action="store_true",
                   help="P0.1: apply av_collate_fn's padding_mask to both the backbone "
                        "self-attention and the pooling. OFF reproduces the published path.")
    p.add_argument("--batch-order-seed", type=int, default=None,
                   help="If set, permute clip order with this seed BEFORE batching. "
                        "Reproduces the shuffle=True batching that "
                        "scripts/eval_checkpoint_gallery.py uses (its loader is built "
                        "with sampler=None -> shuffle=True), which changes batch "
                        "composition and therefore the per-batch ambient padding length.")
    args = p.parse_args()

    sha = sha256_file(args.ckpt)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"ABORT: checkpoint sha256 {sha} != expected {EXPECTED_SHA}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    pcfg = AVJepaConfig(
        d_model=int(cfg_get(cfg, "model.d_model", default=1024)),
        depth=int(cfg_get(cfg, "model.depth", default=8)),
        heads=int(cfg_get(cfg, "model.heads", default=8)),
        mlp_ratio=float(cfg_get(cfg, "model.mlp_ratio", default=4.0)),
        max_tdm_bins=int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
        dropout=float(cfg_get(cfg, "model.dropout", default=0.0)),
    )
    predictor = AVJepaPredictor(pcfg).to(device)
    vision_proj = nn.Linear(pcfg.d_model, args.contrast_dim).to(device)
    ambient_proj = nn.Linear(pcfg.d_model, args.contrast_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(ck["model"])
    vision_proj.load_state_dict(ck["vision_proj"])
    ambient_proj.load_state_dict(ck["ambient_proj"])

    with open(args.eval_subset) as f:
        eval_ids = [l.strip() for l in f if l.strip()]
    if args.batch_order_seed is not None:
        g0 = torch.Generator().manual_seed(args.batch_order_seed)
        eval_ids = [eval_ids[i] for i in torch.randperm(len(eval_ids), generator=g0).tolist()]
    ds = AVCachedDataset(cache_dir=args.cache_dir, clip_ids=eval_ids,
                         max_tdm_bins=pcfg.max_tdm_bins, audio_mode="mean")
    assert len(ds) == 1545, f"dataset_len={len(ds)} != 1545"
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=8,
        collate_fn=av_collate_fn, drop_last=False, pin_memory=True)

    want_ws = not args.no_world_state
    runs: List[Dict] = []
    W0 = None
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for arm in arms:
        arm_seeds = [0] if arm == "A0" else seeds     # A0 has no randomness
        for seed in arm_seeds:
            t0 = time.time()
            r = run_arm(predictor, vision_proj, ambient_proj, loader, device, arm, seed, want_ws,
                        fix_padding=args.fix_padding, max_bins=pcfg.max_tdm_bins)
            W = r.pop("_W", None)
            if arm == "A0" and W0 is None:
                W0 = W
            if want_ws and W is not None and W0 is not None:
                cs_raw = F.cosine_similarity(W, W0, dim=-1)
                r["world_state"]["cos_to_A0_mean"] = round(cs_raw.mean().item(), 8)
                r["world_state"]["cos_to_A0_std"] = round(cs_raw.std().item(), 8)
                Wn, W0n = F.normalize(W, dim=-1), F.normalize(W0, dim=-1)
                r["world_state"]["l2norm_cos_to_A0_mean"] = round(
                    F.cosine_similarity(Wn, W0n, dim=-1).mean().item(), 8)
                r["world_state"]["mean_relative_L2_change"] = round(
                    ((W - W0).norm(dim=-1) / W0.norm(dim=-1)).mean().item(), 6)
            r.update({"arm": arm, "seed": seed, "gallery": args.eval_subset,
                      "gallery_size": len(ds), "cache_dir": args.cache_dir,
                      "ckpt": args.ckpt, "ckpt_sha256": sha,
                      "batch_size": args.batch_size,
                      "batch_order_seed": args.batch_order_seed,
                      "fix_padding": bool(args.fix_padding),
                      "wall_s": round(time.time() - t0, 1)})
            runs.append(r)
            print(f"[{arm} seed={seed}] {json.dumps(r['metrics'])}", flush=True)
            if want_ws:
                print(f"[{arm} seed={seed}] world_state {json.dumps(r['world_state'])}", flush=True)

    payload = {
        "script": "scripts/temporal_probe/bin_scramble_eval.py",
        "command": " ".join([sys.executable] + sys.argv),
        "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                   capture_output=True, text=True).stdout.strip() or "MISSING",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bin_scramble_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
