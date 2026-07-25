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
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
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
from models.pooled_head import PooledXModalHeads, pooled_retrieval_eval
from models.losses import info_nce
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
    mask_mode: str = "windowed",
    mask_frac: Optional[float] = None,
    step: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Sample a cross-modal mask: hide one modality.

    mask_mode:
      'windowed'  — contiguous window covering min_frac..max_frac of tokens.
                    Within-modality context (visible prefix/suffix) leaks some
                    info; acts as a soft cross-modal task.
      'whole'     — mask ALL tokens of the chosen modality (zero visible tokens
                    for that stream). Prediction must come entirely from the
                    OTHER modality — no within-modality interpolation possible.
                    NOTE: with 100% of one modality masked every step, the
                    model can fall back on temporal-position priors instead of
                    genuine cross-modal content (no visible remainder pins
                    clip identity).
      'high_frac' — contiguous window covering a FIXED `mask_frac` (e.g.
                    0.75-0.95) of the chosen modality's tokens; the visible
                    remainder stays visible. The remainder pins clip identity
                    (defeats position-prior shortcuts) while the large masked
                    block still defeats within-modality interpolation.
                    Requires `mask_frac` to be set.
      'asym_curriculum' — same per-token windowing as 'high_frac' (requires
                    `mask_frac`, e.g. 0.9), but WHICH modality gets masked is
                    a deterministic function of `step` (alternating strictly
                    by parity: modalities sorted, step % 2 picks the modality
                    to mask) instead of rng.choice. Masking 'vision' forces
                    reliance on ambient ("audio-heavy" phase); masking
                    'ambient' forces reliance on vision ("vision-heavy"
                    phase). Strict parity guarantees exactly balanced exposure
                    to both phases across training, rather than relying on
                    i.i.d. randomness that can streak short-run.
                    Requires `step` and `mask_frac` to be set.

    Returns Dict[str, (B, T_m) bool] where True = MASKED (to predict).
    """
    rng        = rng or random
    B          = next(iter(tbins.values())).shape[0]
    modalities = list(tbins.keys())

    if mask_mode == "asym_curriculum":
        assert step is not None, "asym_curriculum mode requires step"
        modalities_sorted = sorted(modalities)
        masked_mod = modalities_sorted[step % len(modalities_sorted)]
    else:
        # Randomly choose which modality to mask this step
        masked_mod = rng.choice(modalities)

    mask: Dict[str, Tensor] = {}
    for m in modalities:
        bins = tbins[m]   # (B, T_m)
        if m != masked_mod:
            # Context modality: fully visible
            mask[m] = torch.zeros_like(bins, dtype=torch.bool)
        else:
            if mask_mode == "whole":
                # Mask every token of this modality
                mask[m] = torch.ones_like(bins, dtype=torch.bool)
            elif mask_mode in ("high_frac", "asym_curriculum"):
                assert mask_frac is not None, f"{mask_mode} mode requires mask_frac"
                m_tensor = torch.zeros_like(bins, dtype=torch.bool)
                for b in range(B):
                    T      = bins.shape[1]
                    n_mask = max(1, int(T * mask_frac))
                    start  = rng.randint(0, max(0, T - n_mask))
                    m_tensor[b, start:start + n_mask] = True
                mask[m] = m_tensor
            else:
                # windowed: contiguous random window
                m_tensor = torch.zeros_like(bins, dtype=torch.bool)
                for b in range(B):
                    T      = bins.shape[1]
                    frac   = rng.uniform(min_frac, max_frac)
                    n_mask = max(1, int(T * frac))
                    start  = rng.randint(0, max(0, T - n_mask))
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


# ── Memory safety cap ────────────────────────────────────────────────────────
# Ambient (audio) token count varies per batch (collate pads to that batch's
# own max T; VGGSound clips cluster near ~998 tokens but a rare batch can
# draw an outlier up to ~1150+). Self-attention cost is O(T^2), so a 15%
# longer T can cost noticeably more memory -- observed as multi-GB step-to-
# step swings (not a leak) once the base footprint is already near the GPU
# ceiling. MAX_AMBIENT_T bounds the worst case; at 1024 it only truncates
# the rare outlier tail (p99.9 ~= 1000 tokens on the full 199k-clip cache),
# so it has no material effect on what the model sees for the vast majority
# of clips. This is a memory-engineering cap, independent of negatives/temp/
# lam_sigreg/lam_fusion -- it does not touch the isolated experimental variable.
MAX_AMBIENT_T = 1024


def _cap_ambient_len(feats: Dict[str, Tensor], tbins: Dict[str, Tensor],
                      max_t: int = MAX_AMBIENT_T) -> None:
    """In-place: truncate the 'ambient' entries to at most max_t tokens."""
    if "ambient" in feats and feats["ambient"].shape[1] > max_t:
        feats["ambient"] = feats["ambient"][:, :max_t]
        tbins["ambient"] = tbins["ambient"][:, :max_t]


# ── Dataset / DataLoader ────────────────────────────────────────────────────
def build_dataloader(
    cfg: AttrDict,
    clip_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    exclude_ids: Optional[set] = None,
    batch_size_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    distributed_sampler: bool = True,
    drop_last_override: Optional[bool] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    cache_dir    = str(cfg_get(cfg, "data.av_cache_dir",
                               default="/dev/shm/jepa_m2_cache"))
    audio_mode   = str(cfg_get(cfg, "model.audio_mode",   default="mean"))
    max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512))
    batch_size   = batch_size_override or int(cfg_get(cfg, "train.batch_size", default=32))
    num_workers  = (num_workers_override if num_workers_override is not None
                    else int(cfg_get(cfg, "train.num_workers", default=4)))

    dataset = AVCachedDataset(
        cache_dir=cache_dir,
        clip_ids=clip_ids,
        max_tdm_bins=max_tdm_bins,
        audio_mode=audio_mode,
    )
    if limit is not None:
        dataset.clip_ids = dataset.clip_ids[:limit]
    if exclude_ids is not None and clip_ids is None:
        dataset.clip_ids = [c for c in dataset.clip_ids if c not in exclude_ids]

    drop_last = (drop_last_override if drop_last_override is not None
                 else len(dataset) >= batch_size)
    sampler: Optional[DistributedSampler] = None
    if is_distributed() and distributed_sampler:
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


# ── Contrastive (pooled, instance-discrimination) path ─────────────────────
@torch.no_grad()
def check_source_token_invariance(
    predictor: AVJepaPredictor, feats: Dict[str, Tensor], tbins: Dict[str, Tensor],
) -> float:
    """Re-verify the STEP-3 leak fix on THIS batch: modality m's source tokens
    (from encode_source_tokens) must be exactly invariant to the other
    modality's content, since the contrastive pooling below depends on it for
    an unleaked retrieval signal. Returns the max abs diff (should be 0.0)."""
    device = next(iter(feats.values())).device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        out1 = predictor.encode_source_tokens(feats, tbins)
        feats2 = dict(feats)
        mods = list(feats)
        feats2[mods[1]] = feats[mods[1]][torch.randperm(feats[mods[1]].shape[0], device=feats[mods[1]].device)]
        out2 = predictor.encode_source_tokens(feats2, tbins)
    # vision (mods[0]) tokens must be unaffected by permuting ambient (mods[1])
    return float((out1[mods[0]] - out2[mods[0]]).abs().max())


def pool_and_project(
    predictor: AVJepaPredictor,
    vision_proj: nn.Module,
    ambient_proj: nn.Module,
    feats: Dict[str, Tensor],
    tbins: Dict[str, Tensor],
    tokens: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Mean-pool the leak-fixed per-modality source tokens, project into the
    shared contrastive space, L2-normalise. This head is SEPARATE from and
    normalised unlike the world-state head (which stays un-normalised for
    SIGReg, per the dual-head design).

    tokens: pass a precomputed encode_source_tokens(feats, tbins) result to
    skip recomputing it (e.g. when another branch this same step, like
    CrossAttnFusionBridge, already needs the identical call -- avoids a
    redundant extra pair of backbone forward passes)."""
    src_by_mod = tokens if tokens is not None else predictor.encode_source_tokens(feats, tbins)
    z_v = F.normalize(vision_proj(src_by_mod["vision"].mean(1)).float(), dim=-1)
    z_a = F.normalize(ambient_proj(src_by_mod["ambient"].mean(1)).float(), dim=-1)
    return z_v, z_a


class CrossAttnFusionLayer(nn.Module):
    """One bidirectional co-attention block: vision and ambient tokens each
    attend to the OTHER modality's tokens (in parallel, both reading the
    pre-layer state -- not sequentially, to keep the block symmetric), then
    each side runs its own FFN. Mirrors the co-attention design used by
    CAV-MAE's joint fusion encoder / ViLBERT-style cross-modal blocks."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.v2a = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.a2v = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_v1 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

    def forward(self, v: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        v_in, a_in = v, a
        v2, _ = self.v2a(v_in, a_in, a_in, need_weights=False)
        a2, _ = self.a2v(a_in, v_in, v_in, need_weights=False)
        v = self.norm_v1(v_in + v2)
        a = self.norm_a1(a_in + a2)
        v = self.norm_v2(v + self.ffn_v(v))
        a = self.norm_a2(a + self.ffn_a(a))
        return v, a


class CrossAttnFusionBridge(nn.Module):
    """AUXILIARY-ONLY fusion branch (STEP 2). Takes the leak-fixed per-modality
    source tokens from encode_source_tokens (each modality's OWN masked-pass
    tokens -- see check_source_token_invariance), runs a few bidirectional
    cross-attention layers letting vision and ambient tokens attend to each
    other for a GIVEN pairing, pools each side, and scores whether the pairing
    is real via a small matching head. Trained with a real-pair vs shuffled-
    pair BCE loss.

    This NEVER feeds the normalised contrastive retrieval head
    (pool_and_project stays untouched, byte-for-byte) -- full-gallery R@1
    stays a validly independently-embedded number. Gradients reach the shared
    predictor trunk only through this auxiliary loss, which is the (indirect)
    mechanism by which fusion could improve the trunk's representations that
    the retrieval head also reads from.
    """

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttnFusionLayer(d_model, n_heads) for _ in range(n_layers)])
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.match_head = nn.Linear(d_model * 2, 1)

    def _pool(self, x: Tensor) -> Tensor:
        q = self.pool_query.expand(x.shape[0], 1, -1)
        attn = torch.softmax((q @ x.transpose(1, 2)) / (x.shape[-1] ** 0.5), dim=-1)
        return (attn @ x).squeeze(1)

    def match_logit(self, v_tokens: Tensor, a_tokens: Tensor) -> Tensor:
        v, a = v_tokens, a_tokens
        for layer in self.layers:
            v, a = layer(v, a)
        pooled_v, pooled_a = self._pool(v), self._pool(a)
        return self.match_head(torch.cat([pooled_v, pooled_a], dim=-1)).squeeze(-1)


def fusion_matching_loss(
    fusion_bridge: CrossAttnFusionBridge, v_tokens: Tensor, a_tokens: Tensor,
) -> Tuple[Tensor, float]:
    """Real-pair vs shuffled-pair BCE. Positive: (v_i, a_i). Negative:
    (v_i, a_perm(i)) for a derangement perm (perm(i) != i for all i, so every
    negative is a genuine mismatch, not an accidental self-pair). Returns
    (loss, accuracy) for logging."""
    B = v_tokens.shape[0]
    device = v_tokens.device
    if B < 2:
        # can't form a derangement with 1 sample; skip (no-op, 0 grad contribution)
        return v_tokens.sum() * 0.0, 0.0
    perm = (torch.arange(B, device=device) + 1 + torch.randint(0, max(1, B - 1), (1,), device=device)) % B
    pos_logit = fusion_bridge.match_logit(v_tokens, a_tokens)
    neg_logit = fusion_bridge.match_logit(v_tokens, a_tokens[perm])
    logits = torch.cat([pos_logit, neg_logit], dim=0)
    labels = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    with torch.no_grad():
        acc = ((logits > 0).float() == labels).float().mean().item()
    return loss, acc


def gathered_info_nce(
    z_v_local: Tensor, z_a_local: Tensor, temperature: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """Symmetric InfoNCE with negatives gathered across ALL DDP ranks (the
    standard CLIP/CAV-MAE trick), same formula as models.losses.info_nce but
    adapted for local-queries-vs-global-keys since gather makes the two
    sides different sizes. Uses torch.distributed.nn.functional.all_gather,
    which is GRAD-PRESERVING (backward correctly sums gradient contributions
    from every rank back to the originating rank's local tensor) -- unlike
    raw torch.distributed.all_gather, which detaches. Falls back to plain
    in-batch info_nce when not running distributed."""
    if not (is_distributed() and get_world_size() > 1):
        return info_nce(z_v_local, z_a_local, temperature=temperature)

    import torch.distributed.nn as dist_nn
    rank = get_rank()
    B_local = z_v_local.shape[0]

    z_v_global = torch.cat(dist_nn.functional.all_gather(z_v_local), dim=0)
    z_a_global = torch.cat(dist_nn.functional.all_gather(z_a_local), dim=0)
    global_B = z_v_global.shape[0]

    # Hard guard: every rank must contribute exactly B_local rows, or the
    # gathered "global" matrix is silently ragged (a dropped/short rank
    # would corrupt every other rank's negative pool without erroring
    # otherwise). Fail loudly rather than train on a wrong-shaped batch.
    expected_global_B = B_local * get_world_size()
    assert global_B == expected_global_B, (
        f"ragged gather: global_B={global_B} != {B_local}*{get_world_size()}="
        f"{expected_global_B} -- a rank dropped or sent a different local batch size"
    )
    assert z_v_global.shape[1] == z_a_global.shape[1] == z_v_local.shape[1], \
        "gathered embedding dim mismatch across ranks"

    labels = torch.arange(B_local, device=z_v_local.device) + rank * B_local
    logits_v2a = (z_v_local @ z_a_global.t()) / temperature   # (B_local, global_B)
    logits_a2v = (z_a_local @ z_v_global.t()) / temperature
    loss_v2a = F.cross_entropy(logits_v2a, labels)
    loss_a2v = F.cross_entropy(logits_a2v, labels)
    loss = 0.5 * (loss_v2a + loss_a2v)

    with torch.no_grad():
        acc_v2t = (logits_v2a.argmax(dim=1) == labels).float().mean().item()
        acc_t2v = (logits_a2v.argmax(dim=1) == labels).float().mean().item()

    return loss, {
        "loss": loss.item(), "loss_v2t": loss_v2a.item(), "loss_t2v": loss_a2v.item(),
        "acc_v2t": acc_v2t, "acc_t2v": acc_t2v, "global_B": global_B,
    }


def gradcache_contrastive_step(
    predictor: AVJepaPredictor,
    vision_proj: nn.Module,
    ambient_proj: nn.Module,
    micro_batches: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]],
    temperature: float,
    amp_enabled: bool,
    loss_weight: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """GradCache composed with the differentiable cross-rank all_gather (see
    gathered_info_nce): lets the EFFECTIVE per-rank negative pool exceed what
    fits in one forward pass, by chunking this rank's logical batch into
    micro_batches.

    Phase 1 (no grad): forward each microbatch, cache its pooled/projected
    embeddings.
    Phase 2: concatenate this rank's cached embeddings into ONE leaf tensor,
    run the actual gathered_info_nce() (differentiable all_gather -> true
    global loss across ALL ranks' full logical batches), backward ONCE to
    get the target gradient w.r.t. this rank's local embeddings.
    Phase 3 (with grad): replay each microbatch's forward again, backward a
    surrogate dot-product against its slice of the target gradient -- this
    routes the exact same gradient signal into the model's parameters as one
    giant all-ranks backward would, without ever holding more than one
    microbatch's activations at a time.

    ``loss_weight`` is applied to the phase-2 loss before target gradients
    are extracted, so GradCache obeys the same contrastive coefficient as the
    one-shot path.  Do not also scale the phase-3 surrogate.

    Does NOT call sync_grads() -- the caller must do that EXACTLY ONCE,
    after this returns (and after any other losses' backward calls this
    step). Calling sync_grads() per-microbatch, or anywhere inside this
    function, would partially-average and silently corrupt the gradient.
    """
    device = next(predictor.parameters()).device

    zv_chunks: List[Tensor] = []
    za_chunks: List[Tensor] = []
    with torch.no_grad():
        for feats, tbins in micro_batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled):
                z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
            zv_chunks.append(z_v)
            za_chunks.append(z_a)

    z_v_local = torch.cat(zv_chunks, 0).detach().requires_grad_(True)
    z_a_local = torch.cat(za_chunks, 0).detach().requires_grad_(True)

    loss, metrics = gathered_info_nce(z_v_local, z_a_local, temperature=temperature)
    (loss_weight * loss).backward()
    target_grad_v = z_v_local.grad.detach()
    target_grad_a = z_a_local.grad.detach()

    offset = 0
    for feats, tbins in micro_batches:
        b = next(iter(feats.values())).shape[0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            z_v_i, z_a_i = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
            surrogate = (z_v_i.float() * target_grad_v[offset:offset + b]).sum() \
                      + (z_a_i.float() * target_grad_a[offset:offset + b]).sum()
        surrogate.backward()
        offset += b

    return float(loss.detach()), metrics


def sync_grads(module: nn.Module) -> None:
    """Manual grad all-reduce replacing DDP (see train()'s "distributed sync"
    note): module.parameters() must already be identical across ranks
    (broadcast from rank 0 at construction) for this to converge them."""
    if not (is_distributed() and get_world_size() > 1):
        return
    # A None grad here just means "this rank contributed zero to this
    # parameter this step" (e.g. the masked modality's out_head, chosen
    # independently per rank) -- substitute zero so every rank has a
    # real tensor for every parameter, in module.parameters()'s fixed
    # order (same on every rank, since weights were broadcast from
    # rank 0 at construction).
    params = list(module.parameters())
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    # Flatten into ONE buffer for a SINGLE all_reduce call instead of
    # one per parameter tensor: with ~300 tensors in the predictor,
    # per-tensor all_reduce (kernel-launch/sync-latency dominated for
    # small tensors) measured ~2.7s/step -- 68 min for 1500 steps.
    # Flattening is the standard fix (what pre-bucketing DDP did).
    flat = _flatten_dense_tensors([p.grad for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= get_world_size()
    for p, synced in zip(params, _unflatten_dense_tensors(flat, [p.grad for p in params])):
        p.grad.copy_(synced)


@torch.no_grad()
def contrastive_retrieval_eval(
    predictor: AVJepaPredictor,
    vision_proj: nn.Module,
    ambient_proj: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_clips: int = 1545,
) -> Dict[str, float]:
    """Rank gallery by cosine similarity of the TRAINED contrastive
    embeddings (NOT the regression pooled head). Returns R@1/5/10 both
    directions, plus a temporal-shuffle sanity gap (mean matched cosine sim
    vs mean shuffled-pair cosine sim -- contrastive should separate these
    trivially if it learned real correspondence)."""
    predictor.eval(); vision_proj.eval(); ambient_proj.eval()
    zv_all, za_all = [], []
    n_clips = 0
    for batch in loader:
        if n_clips >= max_clips:
            break
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            z_v, z_a = pool_and_project(predictor, vision_proj, ambient_proj, feats, tbins)
        zv_all.append(z_v.cpu()); za_all.append(z_a.cpu())
        n_clips += z_v.shape[0]

    z_v = torch.cat(zv_all, 0)[:max_clips]
    z_a = torch.cat(za_all, 0)[:max_clips]
    N = z_v.shape[0]
    gt = torch.arange(N)

    sim = z_v @ z_a.T                      # (N, N) ambient in columns
    results: Dict[str, float] = {}
    for name, ranked in [("vision→ambient", (-sim).argsort(1)),
                          ("ambient→vision", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
            results[f"{name}_R@{k}"] = round(hits * 100, 2)

    # temporal-shuffle sanity: matched (diagonal) vs shuffled-pair cosine sim
    matched_sim = sim.diagonal().mean().item()
    perm = torch.randperm(N)
    for i in range(N):
        if perm[i] == i:
            perm[i], perm[(i + 1) % N] = perm[(i + 1) % N], perm[i]
    shuffled_sim = sim[torch.arange(N), perm].mean().item()
    results["matched_cos_sim"] = round(matched_sim, 4)
    results["shuffled_cos_sim"] = round(shuffled_sim, 4)
    results["shuffle_sanity_gap"] = round(matched_sim - shuffled_sim, 4)
    results["n_clips"] = float(N)

    predictor.train(); vision_proj.train(); ambient_proj.train()
    return results


# ── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(
    path: str,
    raw_model: AVJepaPredictor,
    step: int,
    best_loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    loss_ema: Optional[float] = None,
    pooled_heads: Optional[PooledXModalHeads] = None,
    vision_proj: Optional[nn.Module] = None,
    ambient_proj: Optional[nn.Module] = None,
    fusion_bridge: Optional[nn.Module] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step":      step,
        "best_loss": best_loss,
        "model":     raw_model.state_dict(),
    }
    if optimizer     is not None: payload["optimizer"]     = optimizer.state_dict()
    if scheduler     is not None: payload["scheduler"]     = scheduler.state_dict()
    if loss_ema      is not None: payload["loss_ema"]      = loss_ema
    if pooled_heads  is not None: payload["pooled_heads"]  = pooled_heads.state_dict()
    if vision_proj   is not None: payload["vision_proj"]   = vision_proj.state_dict()
    if ambient_proj  is not None: payload["ambient_proj"]  = ambient_proj.state_dict()
    if fusion_bridge is not None: payload["fusion_bridge"] = fusion_bridge.state_dict()
    torch.save(payload, path)


# ── Training ────────────────────────────────────────────────────────────────
def train(cfg: AttrDict, max_steps: Optional[int] = None,
          limit: Optional[int] = None,
          mask_mode: Optional[str] = None,
          mask_frac: Optional[float] = None,
          ckpt_dir_override: Optional[str] = None,
          lam_pooled: float = 0.0,
          eval_every: int = 2000,
          save_every_override: Optional[int] = None,
          eval_subset_path: Optional[str] = None,
          p_neg: float = 0.0,
          w_neg: float = 1.0,
          margin: float = 0.03,
          lam_sigreg_override: Optional[float] = None,
          lam_pred: float = 1.0,
          lam_contrastive: float = 0.0,
          contrast_dim: int = 256,
          contrast_temp: float = 0.05,
          batch_size_override: Optional[int] = None,
          tag_ckpts: bool = False,
          gradcache_micro_steps: int = 1,
          lam_fusion: float = 0.0,
          fusion_layers: int = 2) -> None:
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
    lam_sigreg = (lam_sigreg_override if lam_sigreg_override is not None
                  else float(cfg_get(cfg, "model.sigreg_lambda", default=0.0)))
    num_slices = int(cfg_get(cfg, "model.sigreg_num_slices", default=256))

    # ── Pooled cross-modal head (STEP 3 — MJEPA Sec 4.3) ─────────────
    modality_dims = {"vision": 1024, "ambient": 768}
    pooled_heads: Optional[PooledXModalHeads] = None
    if lam_pooled > 0.0:
        pooled_heads = PooledXModalHeads(
            modality_dims=modality_dims,
            d_model=int(cfg_get(cfg, "model.d_model", default=1024)),
        ).to(device)
        if is_main_process():
            n_ph = sum(p.numel() for p in pooled_heads.parameters())
            print(f"[train_m2] PooledXModalHeads params={n_ph:,}", flush=True)

    # ── Contrastive pooled heads (PIVOT: instance discrimination) ────
    # Separate, L2-NORMALISED projection heads -- distinct from the
    # un-normalised world-state head used by SIGReg/M3. Pools the
    # leak-fixed encode_source_tokens() output (see check_source_token_
    # invariance below), so a->v / v->a scoring can't cheat via joint
    # self-attention leakage.
    vision_proj: Optional[nn.Module] = None
    ambient_proj: Optional[nn.Module] = None
    if lam_contrastive > 0.0:
        d_model = int(cfg_get(cfg, "model.d_model", default=1024))
        vision_proj = nn.Linear(d_model, contrast_dim).to(device)
        ambient_proj = nn.Linear(d_model, contrast_dim).to(device)
        if is_main_process():
            n_cp = sum(p.numel() for p in vision_proj.parameters()) \
                 + sum(p.numel() for p in ambient_proj.parameters())
            print(f"[train_m2] contrastive proj heads params={n_cp:,} "
                  f"dim={contrast_dim} temp={contrast_temp}", flush=True)

    # ── Cross-attention fusion bridge (STEP 2, AUXILIARY ONLY) ────────
    # See CrossAttnFusionBridge docstring: never feeds the retrieval head
    # above, only trains via fusion_matching_loss added into total_loss.
    fusion_bridge: Optional[CrossAttnFusionBridge] = None
    if lam_fusion > 0.0:
        d_model = int(cfg_get(cfg, "model.d_model", default=1024))
        fusion_bridge = CrossAttnFusionBridge(
            d_model=d_model, n_layers=fusion_layers,
            n_heads=int(cfg_get(cfg, "model.heads", default=8)),
        ).to(device)
        if is_main_process():
            n_fb = sum(p.numel() for p in fusion_bridge.parameters())
            print(f"[train_m2] CrossAttnFusionBridge params={n_fb:,} "
                  f"layers={fusion_layers} lam_fusion={lam_fusion} "
                  f"(AUXILIARY ONLY -- does not feed the retrieval head)", flush=True)

    # ── optimiser / scheduler ─────────────────────────────────────────
    optimizer    = build_optimizer(predictor, cfg)
    if pooled_heads is not None:
        # add pooled head params to the same optimizer
        wd = float(cfg_get(cfg, "optim.weight_decay", default=0.05))
        optimizer.add_param_group({
            "params": [p for p in pooled_heads.parameters() if p.ndim > 1],
            "weight_decay": wd,
        })
        optimizer.add_param_group({
            "params": [p for p in pooled_heads.parameters() if p.ndim <= 1],
            "weight_decay": 0.0,
        })
    if vision_proj is not None:
        wd = float(cfg_get(cfg, "optim.weight_decay", default=0.05))
        optimizer.add_param_group({
            "params": list(vision_proj.parameters()) + list(ambient_proj.parameters()),
            "weight_decay": wd,
        })
    if fusion_bridge is not None:
        wd = float(cfg_get(cfg, "optim.weight_decay", default=0.05))
        optimizer.add_param_group({
            "params": [p for p in fusion_bridge.parameters() if p.ndim > 1],
            "weight_decay": wd,
        })
        optimizer.add_param_group({
            "params": [p for p in fusion_bridge.parameters() if p.ndim <= 1],
            "weight_decay": 0.0,
        })
    total_steps  = max_steps or int(cfg_get(cfg, "optim.total_steps", default=10000))
    warmup_steps = int(cfg_get(cfg, "optim.warmup_steps", default=500))
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)
    grad_clip    = float(cfg_get(cfg, "optim.grad_clip", default=1.0))

    # ── auto-resume ───────────────────────────────────────────────────
    ckpt_dir    = ckpt_dir_override or str(cfg_get(cfg, "train.ckpt_dir",
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
        if pooled_heads is not None and "pooled_heads" in ckpt:
            pooled_heads.load_state_dict(ckpt["pooled_heads"], strict=False)
        if vision_proj is not None and "vision_proj" in ckpt:
            vision_proj.load_state_dict(ckpt["vision_proj"], strict=False)
            ambient_proj.load_state_dict(ckpt["ambient_proj"], strict=False)
        if fusion_bridge is not None and "fusion_bridge" in ckpt:
            fusion_bridge.load_state_dict(ckpt["fusion_bridge"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", -1) + 1
        best_loss  = ckpt.get("best_loss", float("inf"))
        loss_ema   = ckpt.get("loss_ema", None)
        del ckpt

    # ── distributed sync (manual, NOT DDP-wrapped) ─────────────────────
    # predictor's methods are routinely called OUTSIDE a single tracked
    # forward() (world_state(), encode_source_tokens() -- both bypass
    # whatever wrapper would be tracking forward() for autograd hooks), and
    # which out_head branch fires varies per step (sample_cross_modal_mask
    # masks exactly one modality, chosen independently per rank). Both
    # DDP(find_unused_parameters=True) alone ("marked ready twice" on
    # pool_query, since it's only reachable via the bypassed calls) and
    # DDP(static_graph=True) (graph shape genuinely changes step to step)
    # fail on this combination. Sidestep DDP's hook/bucket machinery
    # entirely: broadcast rank 0's weights once, then manually all-reduce
    # every gradient after backward() -- simple, correct, and small enough
    # (105M params) that skipping DDP's overlapped bucketing is a fine
    # trade for an exploratory/config-finding run.
    model: nn.Module = predictor
    if is_distributed():
        # All manually-synchronised modules must begin from the SAME state.
        # sync_grads() only averages updates; it cannot correct different
        # random initialisations of the contrastive/pooled heads.
        for module in (predictor, pooled_heads, vision_proj, ambient_proj, fusion_bridge):
            if module is not None:
                for t in module.state_dict().values():
                    dist.broadcast(t, src=0)

    # ── eval subset for STEP-3 retrieval metric ───────────────────────
    # Only activated when explicitly passed: prevents accidental exclusion during
    # STEP 1 go/no-go runs where we want all cached clips for training.
    eval_clip_ids: Optional[List[str]] = None
    eval_loader  = None
    if eval_subset_path and os.path.isfile(eval_subset_path):
        with open(eval_subset_path) as _f:
            eval_clip_ids = [l.strip() for l in _f if l.strip()]
    eval_ids_set = set(eval_clip_ids) if eval_clip_ids else set()

    # ── data ──────────────────────────────────────────────────────────
    # num_workers=0 under torchrun: worker processes get forked AFTER
    # CUDA/NCCL are already initialized in this process, a known hazard
    # that reproduced as multi-run intermittent indefinite stalls (some
    # launches fine, others hung 15-25min with zero output, no error) when
    # num_workers>0 -- non-deterministic, consistent with a fork/CUDA race.
    # AVCachedDataset reads pre-cached RAM-resident tensors (no decode/
    # augmentation), so a single process easily keeps up with ~1.4s/step
    # GPU compute; this trades hypothetical loader parallelism for
    # eliminating the whole hazard class.
    loader, sampler = build_dataloader(cfg, limit=limit, exclude_ids=eval_ids_set,
                                       batch_size_override=batch_size_override,
                                       num_workers_override=(0 if is_distributed() else None))
    batches         = infinite_batches(loader, sampler)

    # ── eval loader (only built if eval clips are cached) ─────────────
    if eval_clip_ids and (pooled_heads is not None or vision_proj is not None):
        # Check how many eval clips are actually cached
        from data.av_cached_dataset import AVCachedDataset as _DS
        _probe = _DS(
            cache_dir=str(cfg_get(cfg, "data.av_cache_dir",
                                  default="/dev/shm/jepa_m2_cache")),
            clip_ids=eval_clip_ids,
            max_tdm_bins=int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
            audio_mode=str(cfg_get(cfg, "model.audio_mode", default="mean")),
        )
        n_eval_avail = len(_probe.clip_ids)
        if n_eval_avail >= 64:   # enough for meaningful retrieval
            eval_loader, _ = build_dataloader(
                cfg, clip_ids=_probe.clip_ids,
                batch_size_override=int(cfg_get(cfg, "eval.batch_size", default=64)),
                # num_workers=0: this loader is constructed AFTER CUDA/NCCL
                # init under torchrun; forking worker processes at that
                # point is a known hazard and is the reproducible cause of
                # the multi-minute stalls seen when --eval-subset was set
                # (runs without it were consistently fast). Eval is small
                # (1545 clips, run only on rank 0, infrequently) so the
                # lack of worker parallelism here is not a real cost.
                num_workers_override=0,
                # distributed_sampler=False: only rank 0 ever iterates this
                # loader (see is_main_process() guard below and at the
                # retrieval-eval call site). A DistributedSampler here would
                # silently hand rank 0 only its 1/world_size shard of the
                # gallery (e.g. 384 of 1545 on 4 GPUs) instead of the full
                # gallery -- this was a real, confirmed bug that inflated
                # every multi-GPU retrieval R@1 this project has reported.
                distributed_sampler=False,
                # drop_last_override=False: eval must see every gallery
                # clip. The default drop_last (len>=batch_size) silently
                # dropped the tail partial batch (1536/1545 at batch=64).
                drop_last_override=False,
            )
            if is_main_process():
                print(f"[train_m2] eval_loader: {n_eval_avail} eval clips cached "
                      f"(of {len(eval_clip_ids)} requested)", flush=True)
        else:
            if is_main_process():
                print(f"[train_m2] eval_loader DISABLED: only {n_eval_avail} eval clips "
                      f"in cache (need ≥64). Retrieval eval will be skipped.", flush=True)
        del _probe

    # ── masking config ────────────────────────────────────────────────
    mask_min  = float(cfg_get(cfg, "train.mask_min_frac", default=0.3))
    mask_max  = float(cfg_get(cfg, "train.mask_max_frac", default=0.7))
    mask_mode = mask_mode or str(cfg_get(cfg, "train.mask_mode", default="windowed"))

    # ── rank ceiling per spec ─────────────────────────────────────────
    batch_size_cfg = int(cfg_get(cfg, "train.batch_size", default=32))
    rank_ceil_ovr  = cfg_get(cfg, "train.rank_ceil_override", default=None)

    # ── logging ───────────────────────────────────────────────────────
    log_every  = int(cfg_get(cfg, "train.log_every",  default=20))
    save_every = save_every_override or int(cfg_get(cfg, "train.save_every", default=500))

    amp_enabled = device.type in ("cuda", "cpu")

    if is_main_process():
        print(
            f"[train_m2] device={device} world_size={get_world_size()} "
            f"total_steps={total_steps} lam_sigreg={lam_sigreg} lam_pred={lam_pred} "
            f"lam_pooled={lam_pooled} lam_contrastive={lam_contrastive} "
            f"contrast_dim={contrast_dim} contrast_temp={contrast_temp} "
            f"eval_every={eval_every} "
            f"save_every={save_every} mask_mode={mask_mode} mask_frac={mask_frac} "
            f"p_neg={p_neg} w_neg={w_neg} margin={margin} "
            f"ckpt_dir={ckpt_dir}",
            flush=True,
        )
        if lam_contrastive > 0.0:
            _probe_batch = next(iter(loader))
            _feats0 = {k: v.to(device) for k, v in _probe_batch["feats"].items()}
            _tbins0 = {k: v.to(device) for k, v in _probe_batch["tbins"].items()}
            _cap_ambient_len(_feats0, _tbins0)
            _micro_b = next(iter(_feats0.values())).shape[0]
            if gradcache_micro_steps > 1:
                print(f"[train_m2] GradCache: micro_batch={_micro_b} x "
                      f"gradcache_micro_steps={gradcache_micro_steps} = "
                      f"per_rank_contrastive_batch={_micro_b * gradcache_micro_steps}  "
                      f"effective_gathered_negatives={_micro_b * gradcache_micro_steps * get_world_size()}",
                      flush=True)
            else:
                print(f"[train_m2] contrastive batch size = {_micro_b}", flush=True)
            inv_diff = check_source_token_invariance(predictor, _feats0, _tbins0)
            print(f"[train_m2] source-token invariance check (max abs diff, "
                  f"should be 0.0): {inv_diff:.2e}", flush=True)
            assert inv_diff < 1e-5, "STEP-3 leak fix regressed -- source tokens are not invariant!"

    optimizer.zero_grad()
    if pooled_heads is not None:
        pooled_heads.train()
    if vision_proj is not None:
        vision_proj.train(); ambient_proj.train()
    model.train()

    for step in range(start_step, total_steps):
        batch = next(batches)

        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        _cap_ambient_len(feats, tbins)

        # GradCache: pull the REST of this step's microbatches now (feats/
        # tbins above stays the "primary" microbatch -- used for pred_loss/
        # sigreg/hinge exactly as before, unchanged scale). The contrastive
        # term below is what actually consumes all gradcache_micro_steps
        # microbatches, via gradcache_contrastive_step().
        gc_micro_batches: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]] = []
        if lam_contrastive > 0.0 and gradcache_micro_steps > 1:
            gc_micro_batches.append((feats, tbins))
            for _ in range(gradcache_micro_steps - 1):
                _b = next(batches)
                _f = {k: v.to(device) for k, v in _b["feats"].items()}
                _t = {k: v.to(device) for k, v in _b["tbins"].items()}
                _cap_ambient_len(_f, _t)
                gc_micro_batches.append((_f, _t))

        # Sample cross-modal mask
        mask = sample_cross_modal_mask(tbins, mask_min, mask_max, rng=rng,
                                       mask_mode=mask_mode, mask_frac=mask_frac,
                                       step=step)
        mask = {k: v.to(device) for k, v in mask.items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            # ── prediction loss ────────────────────────────────────
            # Always run with grad, even when lam_pred==0 (e.g. the
            # contrastive-primary pivot): under DDP every registered
            # parameter (incl. out_head, only used here) must receive a
            # real .grad tensor each step, or the reducer errors/hangs on
            # "unused parameters". lam_pred==0 just zeroes its contribution.
            pred_loss, metrics = model(feats, tbins, mask)

            # ── world state + sigreg ───────────────────────────────
            # encode_world_state() is @no_grad (authoritative — do not alter).
            # world_state() is the grad-enabled twin added to av_jepa_predictor.py.
            raw = predictor
            if lam_sigreg > 0:
                # grad-enabled path: SIGReg flows through pool_query + blocks
                ws = raw.world_state(feats, tbins)       # (B, d) with grad
                sr_loss = sigreg(ws.float(), global_step=step, num_slices=num_slices)
                total_loss = lam_pred * pred_loss + lam_sigreg * sr_loss
            else:
                # lam==0: compute sigreg for logging only (no effect on loss)
                ws = raw.encode_world_state(feats, tbins)  # (B, d) no grad
                with torch.no_grad():
                    sr_loss = sigreg(ws.float(), global_step=step, num_slices=num_slices)
                total_loss = lam_pred * pred_loss

            # ── pooled cross-modal auxiliary (STEP 3, optional) ───
            pooled_loss_val = 0.0
            if pooled_heads is not None and lam_pooled > 0.0:
                src_by_mod = raw.encode_source_tokens(feats, tbins)
                pl = pooled_heads.combined_loss(src_by_mod, feats)
                total_loss = total_loss + lam_pooled * pl
                pooled_loss_val = float(pl.detach())

            # ── shared source tokens for fusion + plain-path retrieval ──
            # One encode_source_tokens() call (2 masked backbone passes) here
            # instead of two separate ones -- fusion and pool_and_project both
            # consume the SAME leak-fixed tokens below. Not built under
            # GradCache (that path caches/replays its own microbatches).
            use_gradcache = lam_contrastive > 0.0 and gradcache_micro_steps > 1
            need_shared_tokens = (fusion_bridge is not None and lam_fusion > 0.0) or \
                                  (lam_contrastive > 0.0 and not use_gradcache)
            shared_src_tokens: Optional[Dict[str, Tensor]] = None
            if need_shared_tokens:
                shared_src_tokens = raw.encode_source_tokens(feats, tbins)

            # ── cross-attention fusion bridge (STEP 2, AUXILIARY ONLY) ──
            # See CrossAttnFusionBridge docstring: real-pair vs shuffled-pair
            # matching loss on the leak-fixed source tokens. Gradients reach
            # the shared trunk through THIS loss only -- pool_and_project's
            # retrieval embeddings above/below are never touched by
            # fusion_bridge, so full-gallery R@1 stays validly comparable.
            # Subsampled to a fixed FUSION_BATCH independent of the
            # contrastive batch size -- this auxiliary branch does not
            # participate in the InfoNCE negatives count at all, so bounding
            # its own memory footprint doesn't touch the 192-negative
            # isolation this run is testing.
            fusion_loss_val = 0.0
            fusion_acc = 0.0
            if fusion_bridge is not None and lam_fusion > 0.0:
                FUSION_BATCH = 8
                B_full = shared_src_tokens["vision"].shape[0]
                fb = min(FUSION_BATCH, B_full)
                f_loss, f_acc = fusion_matching_loss(
                    fusion_bridge,
                    shared_src_tokens["vision"][:fb],
                    shared_src_tokens["ambient"][:fb],
                )
                total_loss = total_loss + lam_fusion * f_loss
                fusion_loss_val = float(f_loss.detach())
                fusion_acc = f_acc

            # ── instance-discrimination hinge (temporal-shuffle as a TRAINING
            # signal, not just an eval control): for a p_neg fraction of the
            # batch, ALSO predict this clip's masked target modality from a
            # DIFFERENT clip's context, and require the matched prediction
            # loss to beat the mismatched one by a margin. This is additive —
            # it reuses forward() unmodified via a second call on a mismatched
            # subset; the matched-prediction path above is unchanged.
            mismatch_loss_val = 0.0
            hinge_val = 0.0
            if p_neg > 0.0:
                B = next(iter(feats.values())).shape[0]
                n_neg = max(1, int(round(p_neg * B)))
                # context modality this step = the one with an all-visible (all-False) mask
                ctx_mod = next(m for m in mask if not mask[m].any())

                sel = torch.randperm(B, device=device)[:n_neg]
                donor = torch.randint(0, B, (n_neg,), device=device)
                same = donor == sel
                donor[same] = (donor[same] + 1) % B  # no self-pairing

                feats_mm = {m: v[sel] for m, v in feats.items()}
                tbins_mm = {m: v[sel] for m, v in tbins.items()}
                mask_mm  = {m: v[sel] for m, v in mask.items()}
                feats_mm[ctx_mod] = feats[ctx_mod][donor]   # mismatched context only

                mismatch_loss, _ = model(feats_mm, tbins_mm, mask_mm)
                hinge = torch.relu(margin - (mismatch_loss - pred_loss))
                total_loss = total_loss + w_neg * hinge
                mismatch_loss_val = float(mismatch_loss.detach())
                hinge_val = float(hinge.detach())

            # ── PIVOT: pooled cross-modal contrastive (instance discrimination) ──
            # Symmetric InfoNCE, gathered across all DDP ranks when distributed
            # (models.losses.info_nce reused as the single-process fallback,
            # unmodified) between L2-normalised, mean-pooled, LEAK-FIXED source
            # tokens. Separate head from world-state; world-state stays
            # un-normalised for SIGReg/M3.
            contrastive_loss_val = 0.0
            contrastive_acc = 0.0
            global_negatives = 0
            if lam_contrastive > 0.0 and not use_gradcache:
                assert amp_enabled, "week run requires bf16 autocast active for the contrastive path"
                z_v, z_a = pool_and_project(raw, vision_proj, ambient_proj, feats, tbins,
                                            tokens=shared_src_tokens)
                c_loss, c_metrics = gathered_info_nce(z_v, z_a, temperature=contrast_temp)
                total_loss = total_loss + lam_contrastive * c_loss
                contrastive_loss_val = float(c_loss.detach())
                contrastive_acc = 0.5 * (c_metrics["acc_v2t"] + c_metrics["acc_t2v"])
                global_negatives = c_metrics.get("global_B", z_v.shape[0])

        total_loss.backward()

        # ── GradCache contrastive path (composes with the differentiable
        # all_gather in gathered_info_nce -- see gradcache_contrastive_step's
        # docstring). Deliberately OUTSIDE the autocast block above: it
        # manages its own autocast per microbatch internally. Its own
        # backward calls (phase-2 target-grad backward + phase-3 per-
        # microbatch surrogate backwards) accumulate into predictor/
        # vision_proj/ambient_proj's .grad tensors ON TOP OF total_loss's
        # backward() above -- sync_grads() below still fires EXACTLY ONCE,
        # after ALL of this step's backward calls (never inside the
        # microbatch loop, which would partially-average and corrupt it).
        if use_gradcache:
            assert amp_enabled, "gradcache contrastive path requires bf16 autocast active"
            c_loss_val, c_metrics = gradcache_contrastive_step(
                raw, vision_proj, ambient_proj, gc_micro_batches,
                temperature=contrast_temp, amp_enabled=amp_enabled,
                loss_weight=lam_contrastive,
            )
            contrastive_loss_val = c_loss_val
            contrastive_acc = 0.5 * (c_metrics["acc_v2t"] + c_metrics["acc_t2v"])
            global_negatives = c_metrics.get("global_B", 0)

        # ── manual grad sync (see "distributed sync" note above) ────────────
        # predictor's grad (from ALL its call sites this step -- forward(),
        # world_state(), encode_source_tokens(), and (if gradcache is
        # active) every microbatch's surrogate backward) and the proj/
        # pooled heads' grad are averaged identically here; the gathered_
        # info_nce collective already routes cross-rank contributions
        # correctly for the GLOBAL-KEY role, but this all-reduce is what
        # makes every replica's parameters converge on the same value
        # overall (also covers the LOCAL-QUERY role for the proj heads,
        # which the gather alone doesn't sync).
        sync_grads(predictor)
        if vision_proj is not None:
            sync_grads(vision_proj)
            sync_grads(ambient_proj)
        if pooled_heads is not None:
            sync_grads(pooled_heads)
        if fusion_bridge is not None:
            sync_grads(fusion_bridge)

        if grad_clip > 0:
            params_to_clip = list(predictor.parameters())
            if pooled_heads is not None:
                params_to_clip += list(pooled_heads.parameters())
            if vision_proj is not None:
                params_to_clip += list(vision_proj.parameters()) + list(ambient_proj.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
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
            if pooled_heads is not None:
                log_parts.append(f"pooled={pooled_loss_val:.4f}")
            if fusion_bridge is not None:
                log_parts.append(f"fusion={fusion_loss_val:.4f}")
                log_parts.append(f"fusion_acc={fusion_acc:.3f}")
            if p_neg > 0.0:
                log_parts.append(f"mismatch={mismatch_loss_val:.4f}")
                log_parts.append(f"hinge={hinge_val:.4f}")
            if lam_contrastive > 0.0:
                log_parts.append(f"contrastive={contrastive_loss_val:.4f}")
                log_parts.append(f"c_acc={contrastive_acc:.3f}")
                log_parts.append(f"negatives={global_negatives}x{global_negatives}")
                log_parts.append(f"dtype={'bf16' if amp_enabled else 'fp32'}")
            if nan_flag:
                log_parts.append(nan_flag)

            print("[m2] " + "  ".join(log_parts), flush=True)

            if loss_ema < best_loss:
                best_loss = loss_ema
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    predictor, step, best_loss,
                    pooled_heads=pooled_heads,
                    vision_proj=vision_proj, ambient_proj=ambient_proj,
                    fusion_bridge=fusion_bridge,
                )

        # ── STEP-3 / contrastive retrieval eval ─────────────────────────
        if is_main_process() and eval_loader is not None \
                and eval_every > 0 and (step + 1) % eval_every == 0:
            if pooled_heads is not None:
                print(f"[m2] === RETRIEVAL EVAL (regression head) @ step {step+1} ===", flush=True)
                ret = pooled_retrieval_eval(
                    pooled_heads, raw, eval_loader, device, modality_dims,
                )
                for k, v in sorted(ret.items()):
                    print(f"[m2]   {k}={v:.2f}%", flush=True)
            if vision_proj is not None:
                print(f"[m2] === RETRIEVAL EVAL (contrastive head) @ step {step+1} ===", flush=True)
                cret = contrastive_retrieval_eval(
                    raw, vision_proj, ambient_proj, eval_loader, device,
                )
                n_clips_seen = int(cret.pop("n_clips"))
                # Full-gallery guard: eval_loader must NOT be sharded by a
                # DistributedSampler (only rank 0 ever iterates it -- a
                # DistributedSampler here would silently truncate rank 0 to
                # its 1/world_size shard instead of the full gallery, as
                # actually happened before this assertion was added).
                assert n_clips_seen == n_eval_avail, (
                    f"eval_loader yielded {n_clips_seen} clips, expected the "
                    f"full gallery ({n_eval_avail}) -- did a DistributedSampler "
                    f"get attached to the eval loader again?"
                )
                print(f"[m2]   dataset_len={n_eval_avail}  clips_seen={n_clips_seen}  (full-gallery OK)",
                      flush=True)
                for k, v in sorted(cret.items()):
                    if "R@" in k:
                        print(f"[m2]   {k}={v:.2f}%", flush=True)
                    else:
                        print(f"[m2]   {k}={v:.4f}", flush=True)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                     enabled=amp_enabled):
                    ws_eval = raw.encode_world_state(feats, tbins)
                # effective_rank() calls torch.linalg.eigvalsh, which has no
                # bf16 CUDA kernel -- must run outside autocast (same pattern
                # as the per-step log block above, which is why that one
                # doesn't crash).
                rank_ceil_eval = min(ws_eval.shape[0] - 1, ws_eval.shape[1])
                eff_rk_eval = effective_rank(ws_eval[:rank_ceil_eval + 1])
                print(f"[m2]   world_state_eff_rank={eff_rk_eval:.2f}/{rank_ceil_eval}", flush=True)

        if is_main_process() and save_every > 0 \
                and (step + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                predictor, step, best_loss,
                optimizer=optimizer, scheduler=scheduler,
                loss_ema=loss_ema,
                pooled_heads=pooled_heads,
                vision_proj=vision_proj, ambient_proj=ambient_proj,
                fusion_bridge=fusion_bridge,
            )
            if tag_ckpts:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"step{step + 1}.pt"),
                    predictor, step, best_loss,
                    loss_ema=loss_ema,
                    pooled_heads=pooled_heads,
                    vision_proj=vision_proj, ambient_proj=ambient_proj,
                    fusion_bridge=fusion_bridge,
                )

    # ── final checkpoint ──────────────────────────────────────────────
    if is_main_process():
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pt"),
            predictor, total_steps - 1, best_loss,
            optimizer=optimizer, scheduler=scheduler,
            loss_ema=loss_ema,
            pooled_heads=pooled_heads,
            vision_proj=vision_proj, ambient_proj=ambient_proj,
            fusion_bridge=fusion_bridge,
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
    parser.add_argument("--mask-mode", default=None,
                        choices=["windowed", "whole", "high_frac", "asym_curriculum"],
                        help="Masking mode: windowed (default), whole, high_frac, or "
                             "asym_curriculum.")
    parser.add_argument("--mask-frac", type=float, default=None,
                        help="Fixed contiguous mask fraction for --mask-mode high_frac "
                             "(e.g. 0.75/0.85/0.95). Required when mask-mode=high_frac.")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Override checkpoint directory (for parallel runs).")
    parser.add_argument("--lam-pooled", type=float, default=0.0,
                        help="Weight for pooled cross-modal auxiliary loss (STEP 3). "
                             "0 disables it (default). Week run uses 0.1.")
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="Retrieval eval interval in steps (default 2000).")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Disk checkpoint interval override (default: from config). "
                             "Week run uses 2000.")
    parser.add_argument("--eval-subset", default=None,
                        help="Path to eval clip-id list (default: data/vggsound_eval_1545.txt).")
    parser.add_argument("--p-neg", type=float, default=0.0,
                        help="Fraction of the batch per step used for the mismatched-context "
                             "instance-discrimination hinge (0 disables it, default).")
    parser.add_argument("--w-neg", type=float, default=1.0,
                        help="Weight on the hinge term (default 1.0).")
    parser.add_argument("--margin", type=float, default=0.03,
                        help="Hinge margin: matched loss must beat mismatched loss by this "
                             "much (default 0.03, calibrated to matched-loss scale ~0.28).")
    parser.add_argument("--lam-sigreg", type=float, default=None,
                        help="Override model.sigreg_lambda from the config (default: use config).")
    parser.add_argument("--lam-pred", type=float, default=1.0,
                        help="Weight on the masked-prediction (smooth-L1) loss (default 1.0). "
                             "Set to 0 to make the contrastive loss the sole objective.")
    parser.add_argument("--lam-contrastive", type=float, default=0.0,
                        help="Weight for the pooled cross-modal contrastive (InfoNCE) loss "
                             "(0 disables it, default). This is the PIVOT objective.")
    parser.add_argument("--contrast-dim", type=int, default=256,
                        help="Shared contrastive embedding dim (default 256).")
    parser.add_argument("--contrast-temp", type=float, default=0.05,
                        help="InfoNCE temperature (default 0.05).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override train.batch_size from the config (main loader only).")
    parser.add_argument("--tag-ckpts", action="store_true",
                        help="Also write a step-numbered checkpoint copy (stepN.pt) at every "
                             "save_every interval, in addition to the rolling last.pt.")
    parser.add_argument("--gradcache-micro-steps", type=int, default=1,
                        help="GradCache: number of --batch-size microbatches to compose per "
                             "rank per training step for the contrastive loss ONLY (pred/sigreg/"
                             "hinge stay at --batch-size). 1 (default) = off, unchanged behavior. "
                             "Effective per-rank contrastive batch = batch_size * this; effective "
                             "gathered negatives = that * world_size.")
    parser.add_argument("--cache-dir", default=None,
                        help="Override data.av_cache_dir from the config (in-memory only, "
                             "does not touch the config file) -- e.g. to point at a RAID-backed "
                             "copy of the feature cache instead of /dev/shm.")
    parser.add_argument("--lam-fusion", type=float, default=0.0,
                        help="STEP 2: weight for the CrossAttnFusionBridge auxiliary real-pair "
                             "vs shuffled-pair matching loss. 0.0 (default) = off. This branch "
                             "NEVER feeds the contrastive retrieval head -- it only trains the "
                             "shared predictor trunk via this loss.")
    parser.add_argument("--fusion-layers", type=int, default=2,
                        help="Number of bidirectional cross-attention layers in the "
                             "CrossAttnFusionBridge (only used when --lam-fusion > 0).")
    args = parser.parse_args()

    if args.mask_mode in ("high_frac", "asym_curriculum") and args.mask_frac is None:
        parser.error(f"--mask-mode {args.mask_mode} requires --mask-frac")

    cfg = load_config(args.config)
    if args.cache_dir is not None:
        cfg.setdefault("data", AttrDict())["av_cache_dir"] = args.cache_dir
    train(cfg, max_steps=args.max_steps, limit=args.limit,
          mask_mode=args.mask_mode, mask_frac=args.mask_frac,
          ckpt_dir_override=args.ckpt_dir,
          lam_pooled=args.lam_pooled, eval_every=args.eval_every,
          save_every_override=args.save_every, eval_subset_path=args.eval_subset,
          p_neg=args.p_neg, w_neg=args.w_neg, margin=args.margin,
          lam_sigreg_override=args.lam_sigreg,
          lam_pred=args.lam_pred, lam_contrastive=args.lam_contrastive,
          contrast_dim=args.contrast_dim, contrast_temp=args.contrast_temp,
          batch_size_override=args.batch_size, tag_ckpts=args.tag_ckpts,
          gradcache_micro_steps=args.gradcache_micro_steps,
          lam_fusion=args.lam_fusion, fusion_layers=args.fusion_layers)


if __name__ == "__main__":
    main()
