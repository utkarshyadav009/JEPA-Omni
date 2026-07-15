"""scripts/verify_gradcache.py — STEP-0 correctness gate for GradCache
composed with the existing differentiable cross-rank all_gather
(gathered_info_nce) before scaling negatives past what fits in one forward.

Extends scripts/verify_grad_sync.py's pattern (same shared-batch technique,
same fp32-isolation technique, same l2-relative-deviation metric) to cover
the NEW gradcache_contrastive_step() path instead of the plain one-shot
pool_and_project + gathered_info_nce + backward().

Reference test (single process, world_size=1):
  - Build predictor + vision_proj/ambient_proj with a fixed seed, save init.
  - Collate ONE N_TOTAL-sample batch (fixed clip_ids) via av_collate_fn.
  - Compute the TRUE reference gradient directly: cache embeddings from
    N_TOTAL/MICRO chunks under no_grad, backward the resulting (N_TOTAL, dim)
    info_nce loss ONCE to get the target gradient w.r.t. those cached
    embeddings, replay each chunk WITH grad + surrogate backward. This is
    itself a (single-process) GradCache computation -- required because a
    single N_TOTAL-wide forward OOMs one GPU -- but it is NOT the code path
    under test, so it stands as an independent reference.

Distributed test (torchrun --nproc_per_node=4):
  - Load the SAME init weights.
  - Each rank slices its own [rank*N_PER_GPU:(rank+1)*N_PER_GPU] rows out of
    the SAME padded N_TOTAL batch (byte-identical inputs to the reference,
    since encode_source_tokens() is padding-amount-sensitive).
  - Rank further chunks its N_PER_GPU rows into MICRO-sized microbatches and
    calls gradcache_contrastive_step() (the actual production function) --
    this exercises chunking WITHIN a rank composed with the differentiable
    all_gather ACROSS ranks.
  - sync_grads() called EXACTLY ONCE after gradcache_contrastive_step()
    returns (never inside it).
  - Rank 0 compares its post-sync grads against the reference's.

Usage:
    /home/utkarsh/miniconda3/envs/jepa-omni/bin/python scripts/verify_gradcache.py --mode ref [--fp32]
    /home/utkarsh/miniconda3/envs/jepa-omni/bin/torchrun --nproc_per_node=4 --standalone \
        scripts/verify_gradcache.py --mode dist [--fp32]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.losses import info_nce
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import (
    pool_and_project, gradcache_contrastive_step, sync_grads,
    setup_distributed, cleanup_distributed,
)
from utils import cfg_get, get_rank, get_world_size, is_distributed, is_main_process, load_config

SCRATCH = "/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad"
INIT_CKPT = os.path.join(SCRATCH, "verify_gradcache_init.pt")
REF_GRADS = os.path.join(SCRATCH, "verify_gradcache_ref_grads.pt")

N_TOTAL   = 384   # 96/GPU across 4 GPUs -- big enough to force >1 microbatch/rank
N_PER_GPU = 96
MICRO     = 48    # matches the known-safe per-GPU batch size from probing
SEED      = 12345
TEMP      = 0.05
CONTRAST_DIM = 256


def build_model(cfg, device) -> tuple:
    predictor_cfg = AVJepaConfig(
        d_model      = int(cfg_get(cfg, "model.d_model",       default=1024)),
        depth        = int(cfg_get(cfg, "model.depth",         default=8)),
        heads        = int(cfg_get(cfg, "model.heads",         default=8)),
        mlp_ratio    = float(cfg_get(cfg, "model.mlp_ratio",  default=4.0)),
        max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
        dropout      = float(cfg_get(cfg, "model.dropout",    default=0.0)),
    )
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    d_model = predictor_cfg.d_model
    vision_proj = nn.Linear(d_model, CONTRAST_DIM).to(device)
    ambient_proj = nn.Linear(d_model, CONTRAST_DIM).to(device)
    return predictor, vision_proj, ambient_proj


def build_batch(cfg, device, n_total: int, fp32: bool = False):
    cache_dir    = str(cfg_get(cfg, "data.av_cache_dir", default="/dev/shm/jepa_m2_cache"))
    audio_mode   = str(cfg_get(cfg, "model.audio_mode",   default="mean"))
    max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512))

    probe = AVCachedDataset(cache_dir=cache_dir, max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
    clip_ids = sorted(probe.clip_ids)[:n_total]
    assert len(clip_ids) == n_total, f"only {len(clip_ids)} clips available, need {n_total}"

    dataset = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                               max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
    batch = av_collate_fn([dataset[i] for i in range(n_total)])
    cast = (lambda v: v.float()) if fp32 else (lambda v: v)
    feats = {k: cast(v.to(device)) for k, v in batch["feats"].items()}
    tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
    return feats, tbins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ref", "dist"], required=True)
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable bf16 autocast, to isolate whether a deviation is "
                             "precision noise vs a real algorithmic bug.")
    args = parser.parse_args()

    device = setup_distributed()
    cfg = load_config(args.config)
    amp_enabled = (device.type == "cuda") and not args.fp32

    if args.mode == "ref":
        assert not is_distributed() or get_world_size() == 1, \
            "--mode ref must run as a single process (no torchrun)"

        torch.manual_seed(SEED)
        predictor, vision_proj, ambient_proj = build_model(cfg, device)
        os.makedirs(SCRATCH, exist_ok=True)
        torch.save({
            "predictor": predictor.state_dict(),
            "vision_proj": vision_proj.state_dict(),
            "ambient_proj": ambient_proj.state_dict(),
        }, INIT_CKPT)

        feats_full, tbins_full = build_batch(cfg, device, N_TOTAL, fp32=args.fp32)

        # Independent reference computation (NOT the code path under test):
        # cache embeddings from N_TOTAL/MICRO chunks under no_grad, backward
        # the resulting (N_TOTAL, dim) info_nce loss ONCE, replay each chunk
        # WITH grad + surrogate backward. Mathematically identical to a
        # single N_TOTAL-wide backward.
        zv_chunks, za_chunks = [], []
        with torch.no_grad():
            for i in range(0, N_TOTAL, MICRO):
                f = {k: v[i:i + MICRO] for k, v in feats_full.items()}
                t = {k: v[i:i + MICRO] for k, v in tbins_full.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    zv_c, za_c = pool_and_project(predictor, vision_proj, ambient_proj, f, t)
                zv_chunks.append(zv_c)
                za_chunks.append(za_c)

        z_v = torch.cat(zv_chunks, 0).detach().requires_grad_(True)
        z_a = torch.cat(za_chunks, 0).detach().requires_grad_(True)
        loss, metrics = info_nce(z_v, z_a, temperature=TEMP)
        loss.backward()
        g_zv, g_za = z_v.grad, z_a.grad

        for i in range(0, N_TOTAL, MICRO):
            f = {k: v[i:i + MICRO] for k, v in feats_full.items()}
            t = {k: v[i:i + MICRO] for k, v in tbins_full.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled):
                zv_c, za_c = pool_and_project(predictor, vision_proj, ambient_proj, f, t)
                surrogate = (zv_c * g_zv[i:i + MICRO]).sum() + (za_c * g_za[i:i + MICRO]).sum()
            surrogate.backward()

        target_predictor_1 = dict(predictor.named_parameters())["blocks.0.mlp.2.weight"]
        target_predictor_2 = dict(predictor.named_parameters())["in_proj.vision.weight"]
        target_proj = dict(vision_proj.named_parameters())["weight"]

        torch.save({
            "loss": float(loss.detach()),
            "blocks.0.mlp.2.weight": target_predictor_1.grad.detach().clone(),
            "in_proj.vision.weight": target_predictor_2.grad.detach().clone(),
            "vision_proj.weight": target_proj.grad.detach().clone(),
        }, REF_GRADS)

        print(f"[verify_gradcache][ref] loss={float(loss.detach()):.6f} acc_v2t={metrics['acc_v2t']:.3f}")
        print(f"[verify_gradcache][ref] saved init -> {INIT_CKPT}")
        print(f"[verify_gradcache][ref] saved grads -> {REF_GRADS}")
        for name, g in [("blocks.0.mlp.2.weight", target_predictor_1.grad),
                         ("in_proj.vision.weight", target_predictor_2.grad),
                         ("vision_proj.weight", target_proj.grad)]:
            print(f"[verify_gradcache][ref]   {name}: norm={g.norm().item():.6f} shape={tuple(g.shape)}")

    else:  # dist
        assert is_distributed() and get_world_size() == 4, \
            f"--mode dist must run under torchrun --nproc_per_node=4 (got world_size={get_world_size()})"
        assert os.path.isfile(INIT_CKPT), "run --mode ref first to produce the init checkpoint"

        rank = get_rank()
        ckpt = torch.load(INIT_CKPT, map_location=device, weights_only=False)
        predictor, vision_proj, ambient_proj = build_model(cfg, device)
        predictor.load_state_dict(ckpt["predictor"])
        vision_proj.load_state_dict(ckpt["vision_proj"])
        ambient_proj.load_state_dict(ckpt["ambient_proj"])

        feats_full, tbins_full = build_batch(cfg, device, N_TOTAL, fp32=args.fp32)
        lo, hi = rank * N_PER_GPU, (rank + 1) * N_PER_GPU
        feats_rank = {k: v[lo:hi] for k, v in feats_full.items()}
        tbins_rank = {k: v[lo:hi] for k, v in tbins_full.items()}

        micro_batches = []
        for i in range(0, N_PER_GPU, MICRO):
            f = {k: v[i:i + MICRO] for k, v in feats_rank.items()}
            t = {k: v[i:i + MICRO] for k, v in tbins_rank.items()}
            micro_batches.append((f, t))

        loss, metrics = gradcache_contrastive_step(
            predictor, vision_proj, ambient_proj, micro_batches,
            temperature=TEMP, amp_enabled=amp_enabled,
        )
        # sync_grads() called EXACTLY ONCE here, after gradcache_contrastive_
        # step() has returned (i.e. after ALL microbatches' surrogate
        # backward calls) -- never inside it.
        sync_grads(predictor)
        sync_grads(vision_proj)
        sync_grads(ambient_proj)

        if is_main_process():
            ref = torch.load(REF_GRADS, map_location=device, weights_only=False)
            params = dict(predictor.named_parameters())
            params["vision_proj.weight"] = dict(vision_proj.named_parameters())["weight"]

            print(f"[verify_gradcache][dist] loss={loss:.6f} "
                  f"acc_v2t={metrics['acc_v2t']:.3f} global_B={metrics['global_B']}")
            worst_l2_rel = 0.0
            for name in ["blocks.0.mlp.2.weight", "in_proj.vision.weight", "vision_proj.weight"]:
                g_dist = params[name].grad.detach().float()
                g_ref  = ref[name].float().to(device)
                abs_diff = (g_dist - g_ref).abs()
                max_abs = abs_diff.max().item()
                ref_norm = g_ref.norm().item()
                dist_norm = g_dist.norm().item()
                l2_rel = (g_dist - g_ref).norm().item() / max(ref_norm, 1e-12)
                cos_sim = torch.nn.functional.cosine_similarity(
                    g_dist.flatten().unsqueeze(0), g_ref.flatten().unsqueeze(0)).item()
                print(f"[verify_gradcache][dist]   {name}: "
                      f"max_abs_dev={max_abs:.3e}  l2_rel_dev={l2_rel:.3e}  cos_sim={cos_sim:.8f}  "
                      f"ref_norm={ref_norm:.6f}  dist_norm={dist_norm:.6f}")
                worst_l2_rel = max(worst_l2_rel, l2_rel)

            print(f"[verify_gradcache][dist] WORST relative L2-norm deviation across all 3 params: {worst_l2_rel:.3e}")
            if worst_l2_rel > 1e-4:
                print("[verify_gradcache][dist] FAIL: exceeds 1e-4 relative (fp32 gate) -- "
                      "GradCache composition is miscomputing. DO NOT train on this path.")
            else:
                print("[verify_gradcache][dist] PASS: GradCache+gather matches the single-process "
                      f"{N_TOTAL}-wide reference within tolerance.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
