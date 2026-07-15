"""scripts/verify_grad_sync.py — pre-launch correctness check for the manual
gradient sync that replaced DDP in train_m2.py (world_state()/encode_source_
tokens() are called outside the tracked forward(), which is incompatible
with DDP's find_unused_parameters/static_graph hook mechanisms -- see
train_m2.py's "distributed sync" comment).

Reference test (single process, world_size=1):
  - Build predictor + vision_proj/ambient_proj with a fixed seed, save init
    weights to disk.
  - Collate ONE 192-sample batch (fixed clip_ids, so ordering/padding is
    reproducible) via av_collate_fn.
  - pool_and_project() -> z_v, z_a (192, dim); info_nce() (plain, in-batch,
    the same formula gathered_info_nce reduces to when not distributed);
    backward(); save target params' .grad to disk.

Distributed test (torchrun --nproc_per_node=4):
  - Load the SAME init weights (must match on every rank -- loaded directly
    from the reference's saved checkpoint, not re-broadcast, to keep this
    script independent of train()'s own broadcast step).
  - Load the SAME 192-sample collated batch, each rank slices ITS OWN row
    range [rank*48:(rank+1)*48] from the SAME padded tensors (not a
    separately-collated 48-sample batch) -- this is deliberate: encode_
    source_tokens() doesn't mask padding, so a sample's padding amount (and
    hence its embedding) can depend on what else was in its batch; slicing
    rows out of the SAME 192-batch guarantees byte-identical inputs to the
    reference run's per-sample computation.
  - pool_and_project() -> z_v_local, z_a_local; gathered_info_nce() (the
    actual production call); backward(); sync_grads() (the actual
    production function, imported, not reimplemented) on predictor,
    vision_proj, ambient_proj.
  - Rank 0 loads the reference's saved grads and reports max abs / max
    relative deviation for each of the 3 target parameters.

SIGReg (lam_sigreg) and the masked-prediction path (lam_pred) are OFF for
this test: SIGReg's loss is a genuine function of full-batch statistics, so
a 192-batch vs 4x48-batch SIGReg computation SHOULD differ by design (that's
correct behavior, not a bug) -- it would contaminate a test that's
specifically about the contrastive gather+sync path. Isolate it.

Usage:
    # 1) reference (single process, run FIRST):
    conda run -n jepa-omni python scripts/verify_grad_sync.py --mode ref

    # 2) distributed (compares against the reference's saved grads):
    conda run -n jepa-omni torchrun --nproc_per_node=4 --standalone \
        scripts/verify_grad_sync.py --mode dist
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
from train_m2 import pool_and_project, gathered_info_nce, sync_grads, setup_distributed, cleanup_distributed
from utils import cfg_get, get_rank, get_world_size, is_distributed, is_main_process, load_config

SCRATCH = "/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad"
INIT_CKPT  = os.path.join(SCRATCH, "verify_grad_sync_init.pt")
REF_GRADS  = os.path.join(SCRATCH, "verify_grad_sync_ref_grads.pt")

N_TOTAL   = 192
N_PER_GPU = 48
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


def build_192_batch(cfg, device, fp32: bool = False):
    cache_dir    = str(cfg_get(cfg, "data.av_cache_dir", default="/dev/shm/jepa_m2_cache"))
    audio_mode   = str(cfg_get(cfg, "model.audio_mode",   default="mean"))
    max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512))

    probe = AVCachedDataset(cache_dir=cache_dir, max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
    clip_ids = sorted(probe.clip_ids)[:N_TOTAL]
    assert len(clip_ids) == N_TOTAL, f"only {len(clip_ids)} clips available, need {N_TOTAL}"

    dataset = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                               max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
    batch = av_collate_fn([dataset[i] for i in range(N_TOTAL)])
    # cached feats are stored bf16; cast to fp32 explicitly when isolating
    # precision noise, since autocast=False alone won't convert the inputs.
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
                             "precision noise vs a real sync bug.")
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

        feats_192, tbins_192 = build_192_batch(cfg, device, fp32=args.fp32)

        # A single 192-batch forward OOMs one GPU (pool_and_project does 2
        # full backbone passes). Per-sample embeddings don't depend on batch
        # composition (attention is per-row, no cross-sample mixing) once
        # padding is pinned -- which it is here, since every chunk is sliced
        # from the SAME pre-collated 192-batch tensors. So: cache embeddings
        # from 4 chunks of 48 under no_grad (cheap), backward the resulting
        # small (192, dim) loss to get the target gradient w.r.t. those
        # cached embeddings, then redo each chunk's forward WITH grad one at
        # a time and backward a surrogate dot-product against that target
        # gradient (the GradCache trick) -- mathematically identical to a
        # single 192-batch backward, only one chunk's activations resident
        # in memory at a time.
        zv_chunks, za_chunks = [], []
        with torch.no_grad():
            for i in range(0, N_TOTAL, N_PER_GPU):
                f = {k: v[i:i + N_PER_GPU] for k, v in feats_192.items()}
                t = {k: v[i:i + N_PER_GPU] for k, v in tbins_192.items()}
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

        for i in range(0, N_TOTAL, N_PER_GPU):
            f = {k: v[i:i + N_PER_GPU] for k, v in feats_192.items()}
            t = {k: v[i:i + N_PER_GPU] for k, v in tbins_192.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled):
                zv_c, za_c = pool_and_project(predictor, vision_proj, ambient_proj, f, t)
                surrogate = (zv_c * g_zv[i:i + N_PER_GPU]).sum() + (za_c * g_za[i:i + N_PER_GPU]).sum()
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

        print(f"[verify_grad_sync][ref] loss={float(loss.detach()):.6f} acc_v2t={metrics['acc_v2t']:.3f}")
        print(f"[verify_grad_sync][ref] saved init -> {INIT_CKPT}")
        print(f"[verify_grad_sync][ref] saved grads -> {REF_GRADS}")
        for name, g in [("blocks.0.mlp.2.weight", target_predictor_1.grad),
                         ("in_proj.vision.weight", target_predictor_2.grad),
                         ("vision_proj.weight", target_proj.grad)]:
            print(f"[verify_grad_sync][ref]   {name}: norm={g.norm().item():.6f} shape={tuple(g.shape)}")

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

        feats_192, tbins_192 = build_192_batch(cfg, device, fp32=args.fp32)
        lo, hi = rank * N_PER_GPU, (rank + 1) * N_PER_GPU
        feats = {k: v[lo:hi] for k, v in feats_192.items()}
        tbins = {k: v[lo:hi] for k, v in tbins_192.items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
            loss, metrics = gathered_info_nce(z_v, z_a, temperature=TEMP)

        loss.backward()
        sync_grads(predictor)
        sync_grads(vision_proj)
        sync_grads(ambient_proj)

        if is_main_process():
            ref = torch.load(REF_GRADS, map_location=device, weights_only=False)
            params = dict(predictor.named_parameters())
            params["vision_proj.weight"] = dict(vision_proj.named_parameters())["weight"]

            print(f"[verify_grad_sync][dist] loss={float(loss.detach()):.6f} "
                  f"acc_v2t={metrics['acc_v2t']:.3f} global_B={metrics['global_B']}")
            worst_l2_rel = 0.0
            for name in ["blocks.0.mlp.2.weight", "in_proj.vision.weight", "vision_proj.weight"]:
                g_dist = params[name].grad.detach().float()
                g_ref  = ref[name].float().to(device)
                abs_diff = (g_dist - g_ref).abs()
                max_abs = abs_diff.max().item()
                ref_norm = g_ref.norm().item()
                dist_norm = g_dist.norm().item()
                # Relative L2-NORM error over the whole tensor -- the standard
                # numerical-agreement metric. A per-element ratio (abs_diff /
                # |g_ref_i|) blows up wherever any single element of g_ref is
                # near zero (common in a large weight-grad tensor), even when
                # the two tensors are otherwise near-identical; it produced
                # spurious ~1e4 "relative deviations" on the first run here
                # despite ref_norm/dist_norm agreeing to within ~0.1%.
                l2_rel = (g_dist - g_ref).norm().item() / max(ref_norm, 1e-12)
                cos_sim = torch.nn.functional.cosine_similarity(
                    g_dist.flatten().unsqueeze(0), g_ref.flatten().unsqueeze(0)).item()
                print(f"[verify_grad_sync][dist]   {name}: "
                      f"max_abs_dev={max_abs:.3e}  l2_rel_dev={l2_rel:.3e}  cos_sim={cos_sim:.8f}  "
                      f"ref_norm={ref_norm:.6f}  dist_norm={dist_norm:.6f}")
                worst_l2_rel = max(worst_l2_rel, l2_rel)

            print(f"[verify_grad_sync][dist] WORST relative L2-norm deviation across all 3 params: {worst_l2_rel:.3e}")
            if worst_l2_rel > 1e-3:
                print("[verify_grad_sync][dist] FAIL: exceeds 1e-3 relative -- manual sync is "
                      "miscomputing, DO NOT launch the week run.")
            else:
                print("[verify_grad_sync][dist] PASS: manual sync matches the single-process "
                      "192-wide reference within tolerance.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
