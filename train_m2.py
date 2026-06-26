"""train_m2.py — M2 joint audio-visual predictor training.

Modelled on train_m1.py's DDP/AMP/cosine structure.

AUTHORITATIVE imports (use as-is, do not rewrite):
    from models.av_jepa_predictor import AVJepaPredictor, AVJepaConfig, effective_rank
    from models.sigreg import sigreg

Loss = AVJepaPredictor(feats, tbins, mask) smooth-L1
     + lambda_sigreg * sigreg(world_state, global_step)

world_state = predictor.encode_world_state(feats, tbins)
  → UN-NORMALISED (never L2-norm it; sigreg shapes it toward N(0,I))

Masking: cross-modal.  For each batch, sample one modality to mask over a
random ~50% time window; predict its tokens from the other modality.

Logging every N steps:
  - prediction loss (smooth-L1 per modality)
  - sigreg loss
  - effective_rank(world_state) — read against batch ceiling
    min(batch_size-1, 1024), NOT hard-coded 1024

Usage:
    python train_m2.py --config configs/m2.yaml
    torchrun --nproc_per_node=2 train_m2.py --config configs/m2.yaml

    # 200-step smoke test:
    python train_m2.py --config configs/m2.yaml --max-steps 200
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ── project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── AUTHORITATIVE imports (do not modify) ───────────────────────────────────
from models.av_jepa_predictor import (
    AVJepaConfig,
    AVJepaPredictor,
    effective_rank,
)
from models.sigreg import sigreg

# ── other imports ────────────────────────────────────────────────────────────
from data.av_cached_dataset import AVCachedDataset, av_collate_fn, validate_av_manifest
from utils import AttrDict, cfg_get, get_local_rank, get_rank, get_world_size, \
    is_distributed, is_main_process, load_config


# ── Distributed helpers ────────────────────────────────────────────────────
def setup_distributed() -> torch.device:
    use_cuda = torch.cuda.is_available()
    if is_distributed():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(get_local_rank())
            return torch.device("cuda", get_local_rank())
        return torch.device("cpu")
    return torch.device("cuda" if use_cuda else "cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(t: Tensor) -> Tensor:
    if is_distributed() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return t


# ── Cross-modal masking ────────────────────────────────────────────────────
def sample_cross_modal_mask(
    tbins: Dict[str, Tensor],
    min_frac: float = 0.3,
    max_frac: float = 0.7,
    rng: Optional[random.Random] = None,
) -> Dict[str, Tensor]:
    """Sample a cross-modal mask: hide one modality over a random time window.

    Returns Dict[str, (B, T_m) bool] where True = MASKED (to predict).
    One modality is partially masked; the other is fully visible.
    """
    rng      = rng or random
    B        = next(iter(tbins.values())).shape[0]
    modalities = list(tbins.keys())

    # Randomly choose which modality to mask
    masked_mod = rng.choice(modalities)

    mask: Dict[str, Tensor] = {}
    for m in modalities:
        bins = tbins[m]   # (B, T_m)
        if m != masked_mod:
            # Fully visible
            mask[m] = torch.zeros_like(bins, dtype=torch.bool)
        else:
            # Mask a contiguous time window of random size
            m_tensor = torch.zeros_like(bins, dtype=torch.bool)
            for b in range(B):
                T     = bins.shape[1]
                frac  = rng.uniform(min_frac, max_frac)
                n_mask = max(1, int(T * frac))
                start = rng.randint(0, max(0, T - n_mask))
                m_tensor[b, start:start + n_mask] = True
            mask[m] = m_tensor

    return mask


# ── Optimiser / scheduler ──────────────────────────────────────────────────
def build_optimizer(
    model: AVJepaPredictor, cfg: AttrDict,
) -> torch.optim.Optimizer:
    lr     = float(cfg_get(cfg, "optim.lr",           default=1e-4))
    wd     = float(cfg_get(cfg, "optim.weight_decay", default=0.05))
    betas  = tuple(float(b) for b in
                   cfg_get(cfg, "optim.betas", default=(0.9, 0.98)))
    eps    = float(cfg_get(cfg, "optim.eps",           default=1e-8))

    decay_params    = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]

    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas, eps=eps,
    )


def build_scheduler(
    opt: torch.optim.Optimizer, warmup: int, total: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ── Dataset / DataLoader ────────────────────────────────────────────────────
def build_dataloader(
    cfg: AttrDict,
    clip_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    cache_dir    = str(cfg_get(cfg, "data.av_cache_dir",
                               default="/dev/shm/jepa_m2_cache"))
    audio_mode   = str(cfg_get(cfg, "model.audio_mode",   default="mean"))
    max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512))
    batch_size   = int(cfg_get(cfg, "train.batch_size",   default=32))
    num_workers  = int(cfg_get(cfg, "train.num_workers",  default=4))

    dataset = AVCachedDataset(
        cache_dir=cache_dir,
        clip_ids=clip_ids,
        max_tdm_bins=max_tdm_bins,
        audio_mode=audio_mode,
    )
    if limit is not None:
        dataset.clip_ids = dataset.clip_ids[:limit]

    drop_last = len(dataset) >= batch_size
    sampler: Optional[DistributedSampler] = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=av_collate_fn,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def infinite_batches(
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
) -> Iterator:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


# ── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(
    path: str,
    raw_model: AVJepaPredictor,
    step: int,
    best_loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    loss_ema: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step":      step,
        "best_loss": best_loss,
        "model":     raw_model.state_dict(),
    }
    if optimizer  is not None: payload["optimizer"]  = optimizer.state_dict()
    if scheduler  is not None: payload["scheduler"]  = scheduler.state_dict()
    if loss_ema   is not None: payload["loss_ema"]   = loss_ema
    torch.save(payload, path)


# ── Training ────────────────────────────────────────────────────────────────
def train(cfg: AttrDict, max_steps: Optional[int] = None,
          limit: Optional[int] = None) -> None:
    device = setup_distributed()
    rank   = get_rank()
    torch.manual_seed(int(cfg_get(cfg, "seed", default=0)) + rank)
    rng    = random.Random(int(cfg_get(cfg, "seed", default=0)) + rank)

    # ── validate manifest ─────────────────────────────────────────────
    cache_dir = str(cfg_get(cfg, "data.av_cache_dir",
                             default="/dev/shm/jepa_m2_cache"))
    if is_main_process():
        manifest = validate_av_manifest(cache_dir)
        print(
            f"[train_m2] Cache validated. "
            f"base_rate={manifest.get('wavjepa_base_token_rate_hz')} Hz  "
            f"nat_rate={manifest.get('wavjepa_nat_token_rate_hz')} Hz  "
            f"vision_pool={manifest.get('vision_spatial_pool')}",
            flush=True,
        )

    # ── build predictor ───────────────────────────────────────────────
    predictor_cfg = AVJepaConfig(
        d_model      = int(cfg_get(cfg, "model.d_model",        default=1024)),
        depth        = int(cfg_get(cfg, "model.depth",          default=8)),
        heads        = int(cfg_get(cfg, "model.heads",          default=8)),
        mlp_ratio    = float(cfg_get(cfg, "model.mlp_ratio",   default=4.0)),
        max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins",  default=512)),
        dropout      = float(cfg_get(cfg, "model.dropout",     default=0.0)),
    )
    predictor = AVJepaPredictor(predictor_cfg).to(device)

    if is_main_process():
        n_params = sum(p.numel() for p in predictor.parameters())
        print(f"[train_m2] AVJepaPredictor params={n_params:,}", flush=True)

    # ── SIGReg lambda ─────────────────────────────────────────────────
    lam_sigreg = float(cfg_get(cfg, "model.sigreg_lambda", default=0.0))
    num_slices = int(cfg_get(cfg, "model.sigreg_num_slices", default=256))

    # ── optimiser / scheduler ─────────────────────────────────────────
    optimizer    = build_optimizer(predictor, cfg)
    total_steps  = max_steps or int(cfg_get(cfg, "optim.total_steps", default=10000))
    warmup_steps = int(cfg_get(cfg, "optim.warmup_steps", default=500))
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)
    grad_clip    = float(cfg_get(cfg, "optim.grad_clip", default=1.0))

    # ── auto-resume ───────────────────────────────────────────────────
    ckpt_dir    = str(cfg_get(cfg, "train.ckpt_dir",
                               default="checkpoints/m2"))
    resume_path = os.path.join(ckpt_dir, "last.pt")
    start_step  = 0
    best_loss   = float("inf")
    loss_ema    = None

    if os.path.isfile(resume_path):
        if is_main_process():
            print(f"[train_m2] Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", -1) + 1
        best_loss  = ckpt.get("best_loss", float("inf"))
        loss_ema   = ckpt.get("loss_ema", None)
        del ckpt

    # ── DDP wrap ─────────────────────────────────────────────────────
    model: nn.Module = predictor
    if is_distributed():
        find_unused = bool(cfg_get(cfg, "train.ddp_find_unused_parameters",
                                   default=False))
        kwargs = {"find_unused_parameters": find_unused}
        if device.type == "cuda":
            kwargs["device_ids"] = [get_local_rank()]
        model = DDP(predictor, **kwargs)

    # ── data ──────────────────────────────────────────────────────────
    loader, sampler = build_dataloader(cfg, limit=limit)
    batches         = infinite_batches(loader, sampler)

    # ── masking config ────────────────────────────────────────────────
    mask_min = float(cfg_get(cfg, "train.mask_min_frac", default=0.3))
    mask_max = float(cfg_get(cfg, "train.mask_max_frac", default=0.7))

    # ── rank ceiling per spec ─────────────────────────────────────────
    batch_size_cfg = int(cfg_get(cfg, "train.batch_size", default=32))
    rank_ceil_ovr  = cfg_get(cfg, "train.rank_ceil_override", default=None)

    # ── logging ───────────────────────────────────────────────────────
    log_every  = int(cfg_get(cfg, "train.log_every",  default=20))
    save_every = int(cfg_get(cfg, "train.save_every", default=500))

    amp_enabled = device.type in ("cuda", "cpu")

    if is_main_process():
        print(
            f"[train_m2] device={device} world_size={get_world_size()} "
            f"total_steps={total_steps} lam_sigreg={lam_sigreg}",
            flush=True,
        )

    optimizer.zero_grad()
    model.train()

    for step in range(start_step, total_steps):
        batch = next(batches)

        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}

        # Sample cross-modal mask
        mask = sample_cross_modal_mask(tbins, mask_min, mask_max, rng=rng)
        mask = {k: v.to(device) for k, v in mask.items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            # ── prediction loss ────────────────────────────────────
            pred_loss, metrics = model(feats, tbins, mask)

            # ── world state + sigreg ───────────────────────────────
            # encode_world_state is @torch.no_grad() (authoritative — do not alter).
            # When lam_sigreg=0 (config default), sr_loss is for logging only.
            raw = predictor if isinstance(model, DDP) else model
            ws = raw.encode_world_state(feats, tbins)  # (B, d) UN-NORMALISED, no grad

            sr_loss = sigreg(ws.float(), global_step=step,
                             num_slices=num_slices)

            # lam=0 → total_loss = pred_loss only.  When lam>0, add differentiable
            # world-state path (bypass encode_world_state's no_grad decoration).
            total_loss = pred_loss + lam_sigreg * sr_loss

        total_loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_val = reduce_mean(pred_loss.detach()).item()
        loss_ema = loss_val if loss_ema is None \
            else 0.9 * loss_ema + 0.1 * loss_val

        if is_main_process() and (step % log_every == 0
                                   or step == total_steps - 1):
            lr_now = scheduler.get_last_lr()[0]

            # Effective rank against batch ceiling (per spec)
            B_actual = ws.shape[0]
            if rank_ceil_ovr is not None:
                rank_ceil = int(rank_ceil_ovr)
            else:
                rank_ceil = min(B_actual - 1, 1024)

            # Subsample world_state to rank_ceil rows for rank estimate
            ws_sub = ws[:rank_ceil + 1]
            eff_rk = effective_rank(ws_sub)

            sr_val   = sr_loss.item()
            pl_val   = pred_loss.item()
            nan_flag = "NaN!" if (
                not math.isfinite(pl_val) or not math.isfinite(sr_val)
            ) else ""

            log_parts = [
                f"step {step:6d}/{total_steps}",
                f"pred={pl_val:.4f}",
            ]
            for k, v in metrics.items():
                if k != "loss":
                    log_parts.append(f"{k}={v:.4f}")
            log_parts += [
                f"sigreg={sr_val:.4f}",
                f"eff_rank={eff_rk:.1f}/{rank_ceil}",
                f"lr={lr_now:.2e}",
                f"loss_ema={loss_ema:.4f}",
            ]
            if nan_flag:
                log_parts.append(nan_flag)

            print("[m2] " + "  ".join(log_parts), flush=True)

            if loss_ema < best_loss:
                best_loss = loss_ema
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    predictor, step, best_loss,
                )

        if is_main_process() and save_every > 0 \
                and (step + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                predictor, step, best_loss,
                optimizer=optimizer, scheduler=scheduler,
                loss_ema=loss_ema,
            )

    # ── final checkpoint ──────────────────────────────────────────────
    if is_main_process():
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"),
            predictor, total_steps - 1, best_loss,
            optimizer=optimizer, scheduler=scheduler,
            loss_ema=loss_ema,
        )
        print(
            f"[train_m2] done. best_loss={best_loss:.4f} "
            f"(ckpts in {ckpt_dir})",
            flush=True,
        )

    cleanup_distributed()


# ── entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train M2 AV JEPA predictor.")
    parser.add_argument("--config",    default="configs/m2.yaml")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total training steps (e.g. 200 for smoke).")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Limit dataset to N clips.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, max_steps=args.max_steps, limit=args.limit)


if __name__ == "__main__":
    main()
