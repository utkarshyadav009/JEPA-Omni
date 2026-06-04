"""Milestone M1 training: align a video/text spine over frozen SigLIP2.

Plain-PyTorch training loop (no trainer framework). Supports single-process
and ``torchrun`` DDP (the sbatch launches with ``--nproc_per_node=2``).

Highlights
----------
* Optimises only ``spine.trainable_parameters()`` with AdamW.
* When ``model.unfreeze_text`` is set, the text base is placed in its own
  param group at ``lr * optim.text_lr_mult``.
* Linear warmup followed by cosine decay.
* ``bfloat16`` autocast.
* Logs ``loss``, ``acc_v2t``, ``alignment`` and ``uniformity`` every
  ``train.log_every`` steps.
* Checkpoints the best model (by smoothed ``acc_v2t``) to ``train.ckpt_dir``.
* Reports the **M0 gate**: whether loss decreases over the first few hundred
  steps (a clear PASS/FAIL line).

Usage
-----
    python train_m1.py --config configs/m1.yaml
    torchrun --nproc_per_node=2 train_m1.py --config configs/m1.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.video_text_dataset import build_dataset, collate_fn
from models import SpineConfig, SpineM1
from utils import (
    AttrDict,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    load_config,
)


# --------------------------------------------------------------------------- #
# Distributed / device setup
# --------------------------------------------------------------------------- #
def setup_distributed() -> torch.device:
    """Initialise the process group (if launched with torchrun) and pick a device."""
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


def reduce_mean(value: Tensor) -> Tensor:
    """All-reduce-mean a scalar tensor across ranks (no-op if single process)."""
    if is_distributed() and dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= get_world_size()
    return value


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(spine: SpineM1, cfg: AttrDict) -> torch.optim.Optimizer:
    """AdamW with a separate low-lr group for the (optionally unfrozen) text base.

    Within each logical group, 1-D parameters (biases, norms, logit scale/bias)
    are excluded from weight decay.
    """
    optim_cfg = cfg["optim"]
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg["weight_decay"])
    text_lr_mult = float(optim_cfg.get("text_lr_mult", 1.0))
    unfreeze_text = bool(cfg["model"].get("unfreeze_text", False))

    text_base_ids = (
        {id(p) for p in spine.text_base_parameters()} if unfreeze_text else set()
    )

    groups: Dict[str, Dict[str, object]] = {
        "main_decay": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "main_no_decay": {"params": [], "lr": lr, "weight_decay": 0.0},
    }
    if unfreeze_text:
        text_lr = lr * text_lr_mult
        groups["text_base_decay"] = {"params": [], "lr": text_lr, "weight_decay": weight_decay}
        groups["text_base_no_decay"] = {"params": [], "lr": text_lr, "weight_decay": 0.0}

    for _name, param in spine.named_parameters():
        if not param.requires_grad:
            continue
        is_text_base = id(param) in text_base_ids
        no_decay = param.ndim <= 1
        if is_text_base:
            key = "text_base_no_decay" if no_decay else "text_base_decay"
        else:
            key = "main_no_decay" if no_decay else "main_decay"
        groups[key]["params"].append(param)  # type: ignore[attr-defined]

    param_groups = [g for g in groups.values() if g["params"]]
    betas = tuple(float(b) for b in optim_cfg.get("betas", (0.9, 0.98)))
    eps = float(optim_cfg.get("eps", 1e-8))
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_dataloader(
    cfg: AttrDict, limit: Optional[int]
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = build_dataset(cfg, "train", limit=limit, decode_device="cpu")
    sampler: Optional[DistributedSampler] = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["train"].get("num_workers", 4)) > 0,
    )
    return loader, sampler


def infinite_batches(
    loader: DataLoader, sampler: Optional[DistributedSampler]
) -> Iterator[Tuple[List[Tensor], List[str]]]:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    raw_spine: SpineM1,
    spine_cfg: SpineConfig,
    step: int,
    best_acc_v2t: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_acc_v2t": best_acc_v2t,
            "model": raw_spine.state_dict(),
            "spine_config": asdict(spine_cfg),
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: AttrDict, limit: Optional[int]) -> None:
    device = setup_distributed()
    rank = get_rank()
    torch.manual_seed(int(cfg.get("seed", 0)) + rank)

    spine_cfg = SpineConfig(**cfg["model"])
    spine = SpineM1(spine_cfg).to(device)

    optimizer = build_optimizer(spine, cfg)
    total_steps = int(cfg["optim"]["total_steps"])
    warmup_steps = int(cfg["optim"]["warmup_steps"])
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    model: nn.Module = spine
    if is_distributed():
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [get_local_rank()]
        model = DDP(spine, find_unused_parameters=False, **ddp_kwargs)

    loader, sampler = build_dataloader(cfg, limit)
    batches = infinite_batches(loader, sampler)

    grad_clip = float(cfg["optim"].get("grad_clip", 0.0))
    log_every = int(cfg["train"].get("log_every", 20))
    save_every = int(cfg["train"].get("save_every", 250))
    ckpt_dir = str(cfg["train"]["ckpt_dir"])
    m0_gate_steps = min(int(cfg["train"].get("m0_gate_steps", 300)), total_steps)

    amp_enabled = device.type in ("cuda", "cpu")
    if is_main_process():
        n_trainable = sum(p.numel() for p in spine.trainable_parameters())
        print(
            f"[train] device={device} world_size={get_world_size()} "
            f"trainable_params={n_trainable:,} total_steps={total_steps}",
            flush=True,
        )

    best_acc_v2t = -1.0
    acc_ema: Optional[float] = None
    m0_losses: List[float] = []
    m0_reported = False

    model.train()
    for step in range(total_steps):
        clips, captions = next(batches)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            out: Dict[str, Tensor] = model(clips, captions)
            loss = out["loss"]

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(spine.trainable_parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # Gather scalar metrics across ranks for logging / gating.
        loss_val = reduce_mean(out["loss"].detach()).item()
        acc_val = reduce_mean(out["acc_v2t"].detach()).item()
        align_val = reduce_mean(out["alignment"].detach()).item()
        unif_val = reduce_mean(out["uniformity"].detach()).item()

        acc_ema = acc_val if acc_ema is None else 0.9 * acc_ema + 0.1 * acc_val
        m0_losses.append(loss_val)

        if is_main_process() and (step % log_every == 0 or step == total_steps - 1):
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[train] step {step:6d}/{total_steps} "
                f"loss={loss_val:.4f} acc_v2t={acc_val:.4f} "
                f"alignment={align_val:.4f} uniformity={unif_val:.4f} "
                f"lr={lr_now:.2e}",
                flush=True,
            )

        # M0 gate: did the loss decrease over the first few hundred steps?
        if not m0_reported and (step + 1) >= m0_gate_steps:
            _report_m0_gate(m0_losses)
            m0_reported = True

        # Best-by-acc_v2t checkpointing (rank 0 only).
        if is_main_process() and acc_ema is not None and acc_ema > best_acc_v2t:
            best_acc_v2t = acc_ema
            save_checkpoint(
                os.path.join(ckpt_dir, "best.pt"), spine, spine_cfg, step, best_acc_v2t
            )
        if is_main_process() and save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"), spine, spine_cfg, step, best_acc_v2t
            )

    if is_main_process():
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"), spine, spine_cfg, total_steps - 1, best_acc_v2t
        )
        print(
            f"[train] done. best smoothed acc_v2t={best_acc_v2t:.4f} "
            f"(checkpoints in {ckpt_dir})",
            flush=True,
        )

    cleanup_distributed()


def _report_m0_gate(losses: List[float]) -> None:
    """Print a clear PASS/FAIL line: did training loss decrease early on?"""
    if not is_main_process() or len(losses) < 4:
        return
    window = max(1, min(50, len(losses) // 3))
    early = sum(losses[:window]) / window
    late = sum(losses[-window:]) / window
    passed = late < early  # strictly lower => loss is decreasing
    status = "PASS" if passed else "FAIL"
    print(
        f"[M0 GATE] {status}: loss over first {len(losses)} steps "
        f"early({window})={early:.4f} -> late({window})={late:.4f} "
        f"(delta={late - early:+.4f}); expected decrease.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M1 video/text spine.")
    parser.add_argument("--config", default="configs/m1.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Tiny debug subset of the train data."
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.limit)


if __name__ == "__main__":
    main()
