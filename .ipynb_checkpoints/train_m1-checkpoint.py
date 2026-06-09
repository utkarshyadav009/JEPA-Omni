"""Milestone M1 training: align the video/text spine.

Plain-PyTorch training loop (no trainer framework). Supports single-process
and ``torchrun`` DDP (the sbatch launches with ``--nproc_per_node=2``).

Imports the authoritative model package: ``SpineM1`` / ``SpineConfig`` and the
``info_nce`` loss live in ``models`` and are used as-is.

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
from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.video_text_dataset import build_dataset, collate_fn
from models import SpineConfig, SpineM1, info_nce
from utils import (
    AttrDict,
    cfg_get,
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
# Forward-output handling + diagnostics
# --------------------------------------------------------------------------- #
_VIDEO_KEYS = ("zv", "video_embeds", "video", "pred", "prediction", "predicted")
_TEXT_KEYS = ("zt", "text_embeds", "text", "target", "text_target", "caption_embeds")


def _first(mapping, keys) -> Optional[Tensor]:
    for key in keys:
        val = mapping.get(key) if hasattr(mapping, "get") else getattr(mapping, key, None)
        if isinstance(val, Tensor):
            return val
    return None


def resolve_outputs(out, loss_fn) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Extract ``(loss, video_embeds, text_embeds)`` from ``SpineM1.forward``.

    Tolerates the model returning the loss tensor directly, a mapping/dataclass
    that carries the loss (and optionally the embeddings), or a pair of
    embeddings (in which case ``loss_fn`` -- ``info_nce`` -- is applied).
    """
    if isinstance(out, Tensor):
        return out, None, None
    if isinstance(out, Mapping) or hasattr(out, "loss"):
        loss = out.get("loss") if isinstance(out, Mapping) else getattr(out, "loss", None)
        zv = _first(out, _VIDEO_KEYS)
        zt = _first(out, _TEXT_KEYS)
        if loss is None:
            if zv is None or zt is None:
                raise KeyError(
                    "SpineM1.forward output has no 'loss' and no recognisable "
                    "video/text embeddings to apply info_nce to."
                )
            loss = loss_fn(zv, zt)
        return loss, zv, zt
    if isinstance(out, (tuple, list)) and len(out) >= 1 and isinstance(out[0], Tensor):
        # SpineM1.forward returns (loss_tensor, metrics_dict). Also tolerate (loss,) and (zv, zt).
        if len(out) == 1 or isinstance(out[1], Mapping):
            return out[0], None, None          # (loss, metrics): loss is out[0]; re-embed for diagnostics on log steps
        return loss_fn(out[0], out[1]), out[0], out[1]   # (zv, zt) embeddings
    raise TypeError(f"Unsupported SpineM1.forward output type: {type(out)!r}")


def _uniformity(x: Tensor, t: float = 2.0) -> Tensor:
    if x.shape[0] < 2:
        return x.new_zeros(())
    sq_pdist = torch.pdist(x.float(), p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def diagnostics(zv: Tensor, zt: Tensor) -> Dict[str, Tensor]:
    """Compute ``acc_v2t`` / ``alignment`` / ``uniformity`` from paired embeds."""
    zv = F.normalize(zv.float(), dim=-1)
    zt = F.normalize(zt.float(), dim=-1)
    n = zv.shape[0]
    sims = zv @ zt.t()
    targets = torch.arange(n, device=zv.device)
    return {
        "acc_v2t": (sims.argmax(dim=1) == targets).float().mean(),
        "alignment": (zv - zt).pow(2).sum(dim=-1).mean(),
        "uniformity": 0.5 * (_uniformity(zv) + _uniformity(zt)),
    }


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def text_base_parameters(spine: nn.Module) -> List[nn.Parameter]:
    """Best-effort access to the trainable text-base parameters."""
    if hasattr(spine, "text_base_parameters"):
        return [p for p in spine.text_base_parameters() if p.requires_grad]
    for attr in ("text_target", "text_encoder", "text_model", "text_tower", "text"):
        mod = getattr(spine, attr, None)
        if isinstance(mod, nn.Module):
            return [p for p in mod.parameters() if p.requires_grad]
    return [
        p for name, p in spine.named_parameters()
        if p.requires_grad and "text" in name.lower()
    ]


def build_optimizer(spine: SpineM1, cfg: AttrDict) -> torch.optim.Optimizer:
    """AdamW with a separate low-lr group for the (optionally unfrozen) text base.

    Within each logical group, 1-D parameters (biases, norms, scalars) are
    excluded from weight decay.
    """
    lr = float(cfg_get(cfg, "optim.lr", "lr", default=1e-4))
    weight_decay = float(cfg_get(cfg, "optim.weight_decay", "weight_decay", default=0.0))
    text_lr_mult = float(cfg_get(cfg, "optim.text_lr_mult", "text_lr_mult", default=1.0))
    unfreeze_text = bool(cfg_get(cfg, "model.unfreeze_text", default=False))

    text_base_ids = (
        {id(p) for p in text_base_parameters(spine)} if unfreeze_text else set()
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
    betas = tuple(float(b) for b in cfg_get(cfg, "optim.betas", "betas", default=(0.9, 0.98)))
    eps = float(cfg_get(cfg, "optim.eps", "eps", default=1e-8))
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
    num_workers = int(cfg_get(cfg, "train.num_workers", "num_workers", default=4))
    batch_size = int(cfg_get(cfg, "train.batch_size", "batch_size", default=16))
    
    # --- Guard against zero-batch deadlock on small subsets ---
    # Only drop the last batch if we have enough samples to fill at least one batch.
    # Otherwise, force drop_last to False so the loader doesn't yield an empty epoch.
    drop_last_validated = True if len(dataset) >= batch_size else False

    sampler: Optional[DistributedSampler] = None
    if is_distributed():
        # Apply the same drop_last logic to the distributed sampler layout
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last_validated)
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last_validated,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        # --- DEFENSIVE HYGIENE: Use "spawn" context if workers are utilized ---
        multiprocessing_context="spawn" if num_workers > 0 else None,
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
    model_config: Dict[str, object],
    step: int,
    best_acc_v2t: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_acc_v2t": best_acc_v2t,
            "model": raw_spine.state_dict(),
            "model_config": dict(model_config),
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: AttrDict, limit: Optional[int]) -> None:
    device = setup_distributed()
    rank = get_rank()
    torch.manual_seed(int(cfg_get(cfg, "seed", default=0)) + rank)

    model_config = dict(cfg["model"])
    spine = SpineM1(SpineConfig(**model_config)).to(device)

    optimizer = build_optimizer(spine, cfg)
    total_steps = int(cfg_get(cfg, "optim.total_steps", "train.total_steps", "total_steps", default=5000))
    warmup_steps = int(cfg_get(cfg, "optim.warmup_steps", "train.warmup_steps", "warmup_steps", default=0))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    model: nn.Module = spine
    if is_distributed():
        ddp_kwargs = {
            "find_unused_parameters": bool(
                cfg_get(cfg, "train.ddp_find_unused_parameters", default=False)
            )
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [get_local_rank()]
        model = DDP(spine, **ddp_kwargs)

    loader, sampler = build_dataloader(cfg, limit)
    batches = infinite_batches(loader, sampler)

    grad_clip = float(cfg_get(cfg, "optim.grad_clip", "train.grad_clip", "grad_clip", default=0.0))
    log_every = int(cfg_get(cfg, "train.log_every", "log_every", default=20))
    save_every = int(cfg_get(cfg, "train.save_every", "save_every", default=0))
    ckpt_dir = str(cfg_get(cfg, "train.ckpt_dir", "ckpt_dir", default="checkpoints/m1"))
    m0_gate_steps = min(
        int(cfg_get(cfg, "train.m0_gate_steps", "m0_gate_steps", default=min(300, total_steps))),
        total_steps,
    )

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
            out = model(clips, captions)
            loss, zv, zt = resolve_outputs(out, info_nce)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(spine.trainable_parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = reduce_mean(loss.detach()).item()
        m0_losses.append(loss_val)

        is_log_step = step % log_every == 0 or step == total_steps - 1
        if is_log_step:
            # Diagnostics use the embeddings from the forward pass when present,
            # otherwise a cheap no-grad re-embed (only on log steps).
            if zv is None or zt is None:
                with torch.no_grad():
                    zv = spine.embed_video(clips)
                    zt = spine.embed_text(captions)
            diag = diagnostics(zv, zt)
            acc_val = reduce_mean(diag["acc_v2t"].detach()).item()
            align_val = reduce_mean(diag["alignment"].detach()).item()
            unif_val = reduce_mean(diag["uniformity"].detach()).item()
            acc_ema = acc_val if acc_ema is None else 0.9 * acc_ema + 0.1 * acc_val

            if is_main_process():
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[train] step {step:6d}/{total_steps} "
                    f"loss={loss_val:.4f} acc_v2t={acc_val:.4f} "
                    f"alignment={align_val:.4f} uniformity={unif_val:.4f} "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )
                if acc_ema is not None and acc_ema > best_acc_v2t:
                    best_acc_v2t = acc_ema
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        spine, model_config, step, best_acc_v2t,
                    )

        # M0 gate: did the loss decrease over the first few hundred steps?
        if not m0_reported and (step + 1) >= m0_gate_steps:
            _report_m0_gate(m0_losses)
            m0_reported = True

        if is_main_process() and save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                spine, model_config, step, best_acc_v2t,
            )

    if is_main_process():
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"),
            spine, model_config, total_steps - 1, best_acc_v2t,
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