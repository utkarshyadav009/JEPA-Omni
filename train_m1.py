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
from data.cached_feature_dataset import (
    CachedFeatureDataset,
    cached_collate_fn,
    validate_manifest,
)
from models import SpineConfig, SpineM1, info_nce, compute_siglip_loss
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
    cfg: AttrDict, limit: Optional[int], *, feature_cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    num_workers = int(cfg_get(cfg, "train.num_workers", "num_workers", default=4))
    batch_size = int(cfg_get(cfg, "train.batch_size", "batch_size", default=16))

    if feature_cache_dir:
        dataset = CachedFeatureDataset.from_config(cfg, "train", limit=limit)
        active_collate = cached_collate_fn
    else:
        dataset = build_dataset(cfg, "train", limit=limit, decode_device="cpu")
        active_collate = collate_fn

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
        collate_fn=active_collate,
        drop_last=drop_last_validated,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        # Use default "fork" context (Linux). Workers share the parent's
        # memory via copy-on-write — far lighter than "spawn" which re-imports
        # torch in every worker (~500MB each). Safe because workers only do
        # CPU video decoding and never touch CUDA.
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
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    loss_ema: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "best_acc_v2t": best_acc_v2t,
        "model": raw_spine.state_dict(),
        "model_config": dict(model_config),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if loss_ema is not None:
        payload["loss_ema"] = loss_ema
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: AttrDict, limit: Optional[int]) -> None:
    device = setup_distributed()
    rank = get_rank()
    torch.manual_seed(int(cfg_get(cfg, "seed", default=0)) + rank)

    # ---- Feature cache setup ----
    feature_cache_dir = str(cfg_get(cfg, "data.feature_cache_dir", default="")) or None
    use_cached = feature_cache_dir is not None

    if use_cached:
        manifest = validate_manifest(feature_cache_dir, cfg)
        if is_main_process():
            print(
                f"[cache] Using pre-computed features from {feature_cache_dir}"
                f" (hidden_size={manifest['hidden_size']}, dtype={manifest['dtype']})",
                flush=True,
            )

    model_config = dict(cfg["model"])
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(SpineConfig)}
    filtered_config = {k: v for k, v in model_config.items() if k in valid_fields}

    # When using cached features, skip the ~26GB VisionEncoder entirely.
    if use_cached:
        filtered_config["skip_encoder"] = True
        filtered_config["encoder_out_dim"] = manifest["hidden_size"]

    spine = SpineM1(SpineConfig(**filtered_config)).to(device)

    # Use the loss function from the model's config if not specified
    loss_type = cfg_get(cfg, "train.loss_type", "model.loss_type", default=spine.cfg.loss_type)
    spine.cfg.loss_type = loss_type

    # fallback loss_fn for diagnostics if resolve_outputs needs it
    from models import compute_siglip_loss, sigreg_jepa_loss
    loss_fn = info_nce
    if loss_type == "siglip":
        loss_fn = compute_siglip_loss
    elif loss_type == "sigreg":
        loss_fn = sigreg_jepa_loss

    accum_steps = int(cfg_get(cfg, "train.gradient_accumulation_steps", "optim.gradient_accumulation_steps", default=1))


    optimizer = build_optimizer(spine, cfg)
    total_steps = int(cfg_get(cfg, "optim.total_steps", "train.total_steps", "total_steps", default=5000))
    warmup_steps = int(cfg_get(cfg, "optim.warmup_steps", "train.warmup_steps", "warmup_steps", default=0))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # ---- Auto-resume from last checkpoint ----
    start_step = 0
    ckpt_dir = str(cfg_get(cfg, "train.ckpt_dir", "ckpt_dir", default="checkpoints/m1"))
    resume_path = os.path.join(ckpt_dir, "last.pt")
    if os.path.isfile(resume_path):
        if is_main_process():
            print(f"[resume] Loading checkpoint from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = spine.load_state_dict(ckpt["model"], strict=False)
        if is_main_process() and (missing or unexpected):
            print(
                f"[resume] state_dict: {len(missing)} missing, "
                f"{len(unexpected)} unexpected (expected if skip_encoder=True)",
                flush=True,
            )
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        has_scheduler_state = "scheduler" in ckpt
        if has_scheduler_state:
            scheduler.load_state_dict(ckpt["scheduler"])
        # Resume from the step *after* the saved step
        start_step = ckpt.get("step", -1) + 1
        # If the checkpoint predates the scheduler state, fast-forward the
        # LR schedule so the learning rate matches the resumed step.
        if not has_scheduler_state and start_step > 0:
            if is_main_process():
                print(f"[resume] Scheduler state not in checkpoint, fast-forwarding LR {start_step} steps...", flush=True)
            for _ in range(start_step):
                scheduler.step()
        if is_main_process():
            print(f"[resume] Resuming from step {start_step} (lr={scheduler.get_last_lr()[0]:.2e})", flush=True)
        del ckpt  # free memory

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

    loader, sampler = build_dataloader(cfg, limit, feature_cache_dir=feature_cache_dir)
    batches = infinite_batches(loader, sampler)

    grad_clip = float(cfg_get(cfg, "optim.grad_clip", "train.grad_clip", "grad_clip", default=0.0))
    log_every = int(cfg_get(cfg, "train.log_every", "log_every", default=20))
    save_every = int(cfg_get(cfg, "train.save_every", "save_every", default=0))
    # ckpt_dir already set above during resume logic
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

    best_loss = float("inf")
    loss_ema: Optional[float] = None
    m0_losses: List[float] = []
    m0_reported = start_step > 0   # skip M0 gate if resuming
    m0_accs: List[float] = []           
    m0_uniformities: List[float] = []  
  

    optimizer.zero_grad()
    model.train()


    for step in range(start_step, total_steps):
        batch = next(batches)

        if use_cached:
            # Cached path: features already [B, N, D] from DataLoader
            feats, captions = batch
            feats = feats.to(device)
        else:
            # Live path: decode video -> micro-chunk through frozen encoder
            clips, captions = batch
            micro_chunk = cfg.train.get("micro_chunk", 8)
            raw_spine = model.module if isinstance(model, DDP) else model

            with torch.no_grad():
                chunked_feats = []
                for i in range(0, len(clips), micro_chunk):
                    chunk = clips[i : i + micro_chunk]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        chunk_feats = raw_spine.encoder(chunk)
                    chunked_feats.append(chunk_feats)
                feats = torch.cat(chunked_feats, dim=0)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(feats, captions)
            loss, zv, zt = resolve_outputs(out, loss_fn)

        loss_accum = loss / accum_steps
        loss_accum.backward()

        if (step + 1) % accum_steps == 0 or step == total_steps - 1:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(spine.trainable_parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_val = reduce_mean(loss.detach()).item()
        m0_losses.append(loss_val)
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        is_log_step = step % log_every == 0 or step == total_steps - 1
        if is_log_step:
            # Diagnostics use the embeddings from the forward pass when present,
            # otherwise a cheap no-grad re-embed (only on log steps) using the precomputed features.
            if zv is None or zt is None:
                with torch.no_grad():
                    zv = spine.predictor(feats)
                    zt = spine.embed_text(captions)
            diag = diagnostics(zv, zt)
            acc_val = reduce_mean(diag["acc_v2t"].detach()).item()
            align_val = reduce_mean(diag["alignment"].detach()).item()
            unif_val = reduce_mean(diag["uniformity"].detach()).item()
            
            if not m0_reported: 
                m0_accs.append(acc_val)
                m0_uniformities.append(unif_val)
            
            if is_main_process():
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[train] step {step:6d}/{total_steps} "
                    f"loss={loss_val:.4f} acc_v2t={acc_val:.4f} "
                    f"alignment={align_val:.4f} uniformity={unif_val:.4f} "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )
                if loss_ema is not None and loss_ema < best_loss:
                    best_loss = loss_ema
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best.pt"),
                        spine, model_config, step, best_loss,
                    )

        # M0 gate: did the loss decrease over the first few hundred steps?
        if not m0_reported and (step + 1) >= m0_gate_steps:
            _report_m0_gate(m0_losses, m0_accs, m0_uniformities)
            m0_reported = True

        if is_main_process() and save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                spine, model_config, step, best_loss,
                optimizer=optimizer, scheduler=scheduler, loss_ema=loss_ema,
            )

    if is_main_process():
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"),
            spine, model_config, total_steps - 1, best_loss,
            optimizer=optimizer, scheduler=scheduler, loss_ema=loss_ema,
        )
        print(
            f"[train] done. best smoothed loss={best_loss:.4f} "
            f"(checkpoints in {ckpt_dir})",
            flush=True,
        )

    cleanup_distributed()


def _report_m0_gate(
    losses: List[float], 
    accs_v2t: List[float], 
    uniformities: List[float],
    chance_acc: float = 0.1250
) -> bool:
    if not is_main_process() or len(losses) < 4:
        return True
        
    # Loss window uses the full step count (e.g., 50)
    loss_window = max(1, min(50, len(losses) // 3))
    
    # Acc/Unif window uses the smaller logged count (e.g., 50 / 20 = ~3)
    metric_window = max(1, len(accs_v2t) // 3) 
    
    recent_acc = sum(accs_v2t[-metric_window:]) / metric_window
    recent_uniformity = sum(uniformities[-metric_window:]) / metric_window
    
    early_loss = sum(losses[:loss_window]) / loss_window
    late_loss = sum(losses[-loss_window:]) / loss_window
    
    loss_decreased = late_loss < early_loss
    acc_passed = recent_acc > (chance_acc + 0.05) 
    uniformity_passed = recent_uniformity < -0.20 
    
    overall_passed = loss_decreased and acc_passed and uniformity_passed
    status = "PASS" if overall_passed else "FAIL"
    
    print("\n--- [M0 GATE CHECK] ---", flush=True)
    print(f"STATUS: {status}", flush=True)
    print(f"  1. Loss Trend:   {'PASS' if loss_decreased else 'FAIL'} | early({loss_window})={early_loss:.4f} -> late({loss_window})={late_loss:.4f} (delta={late_loss - early_loss:+.4f})", flush=True)
    print(f"  2. Accuracy V2T: {'PASS' if acc_passed else 'FAIL'} | recent_avg({metric_window})={recent_acc:.4f}", flush=True)
    print(f"  3. Uniformity:   {'PASS' if uniformity_passed else 'FAIL'} | recent_avg({metric_window})={recent_uniformity:.4f}", flush=True)
    
    return overall_passed

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