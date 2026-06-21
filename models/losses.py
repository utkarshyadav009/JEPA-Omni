"""
models/losses.py

Bi-directional InfoNCE (the VL-JEPA / CLIP objective), decomposed into an alignment
term (pull matched pairs together) and a uniformity term (spread on the hypersphere;
anti-collapse). The InfoNCE denominator (negatives) IS the uniformity pressure.

Two entry points:
  * info_nce(z_v, z_t)                     -> in-batch negatives only (B-1 per sample)
  * info_nce_with_queue(z_v, z_t, neg_*)   -> in-batch + a MoCo-style negative queue,
                                              so the effective #negatives is B-1 + K.

Why the queue matters here: on one H100 the physical batch caps at ~8, so plain
InfoNCE only ever contrasts against 7 negatives. The model then trains on an 8-way
problem but is *evaluated* on 1000-way retrieval -- far too easy at train time, which
caps the learned representation. The queue decouples the contrastive difficulty from
the GPU batch size.

Inputs are assumed L2-normalized. Queue negatives are DETACHED (no gradient flows into
stale embeddings); gradients flow only through the current batch's z_v / z_t.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _diagnostics(z_v: Tensor, z_t: Tensor) -> Tuple[float, float]:
    """Wang & Isola alignment / uniformity on the current batch (monitoring only)."""
    alignment = (z_v - z_t).pow(2).sum(-1).mean().item()
    sq = torch.pdist(torch.cat([z_v, z_t], 0), p=2).pow(2)
    uniformity = sq.mul(-2).exp().mean().clamp_min(1e-12).log().item()
    return alignment, uniformity


def _ddp_avg(t: torch.Tensor) -> torch.Tensor:
    """all_reduce(AVG) with complex + single-GPU support."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    ws = dist.get_world_size()
    if torch.is_complex(t):
        tr = torch.view_as_real(t).contiguous()
        dist.all_reduce(tr, op=dist.ReduceOp.SUM)
        tr /= ws
        return torch.view_as_complex(tr)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / ws
    return t


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Args
        x            : (N, K) embeddings to regularise (pooled World-State, or in
                       M1 the pooled video embedding). Do NOT pre-standardise:
                       SIGReg enforces unit variance per projection by design.
        global_step  : seeds the projection sampler so directions are identical
                       across DDP ranks and resampled every step (resampling
                       beats fixed directions — paper Fig. 7).
        num_slices   : M = |A|, number of random projection directions
                       (paper default = 256).
        reduce       : 'mean' -> scalar loss (Def. 2 average over directions);
                       'none' -> per-direction statistic, shape (num_slices,).

    Returns
        Scalar SIGReg loss, or (num_slices,) tensor if reduce='none'.
    """
    assert x.dim() == 2, f"expected (N, K), got {tuple(x.shape)}"
    dev = dict(device=x.device)

    # --- slice sampling: synced across devices via global_step seed ---
    g = torch.Generator(**dev)
    g.manual_seed(int(global_step))
    A = torch.randn((x.size(1), num_slices), generator=g, **dev)  # (K, M)
    A = A / A.norm(p=2, dim=0)                                    # unit-norm columns

    # --- Epps-Pulley statistic toward N(0,1) ---
    t = torch.linspace(-5.0, 5.0, 17, **dev)        # quadrature grid (Algorithm 1)
    target_cf = torch.exp(-0.5 * t**2)              # CF of N(0,1) (real) + Gauss. window

    x_t = (x.to(A.dtype) @ A).unsqueeze(2) * t      # (N, M, T)
    ecf = (1j * x_t).exp().mean(dim=0)              # (M, T) empirical CF, batch-mean
    # ecf = _ddp_avg(ecf)                           # Deliberately keep SIGReg per-rank on local batch (no cross-GPU gather)

    # weighted-L2 between empirical and target CF, weighted by the Gaussian window
    err = (ecf - target_cf).abs().square().mul(target_cf)   # (M, T)
    N = x.size(0)                                   # Deliberately use local batch size per rank
    per_dir = torch.trapz(err, t, dim=1) * N               # (M,)

    if reduce == "mean":
        return per_dir.mean()
    if reduce == "none":
        return per_dir
    raise ValueError(f"reduce must be 'mean' or 'none', got {reduce!r}")


def sigreg_loss(z: Tensor, global_step: int = 0, sketch_dim: int = 256, **kwargs) -> Tensor:
    return sigreg(z, global_step=global_step, num_slices=sketch_dim)


def sigreg_jepa_loss(
    z_v: Tensor,
    z_t: Tensor,
    global_step: int = 0,
    lamb: float = 10.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    LeJEPA total loss: Prediction Error (MSE) + lambda * SigReg Regularization.
    Inputs are assumed to be embeddings from the predictor and text target.
    """
    # Prediction loss (MSE)
    loss_mse = F.mse_loss(z_v, z_t)

    # Regularization loss
    reg_v = sigreg(z_v, global_step=global_step, num_slices=256)
    reg_t = sigreg(z_t, global_step=global_step, num_slices=256)
    loss_reg = 0.5 * (reg_v + reg_t)

    loss = loss_mse + lamb * loss_reg

    with torch.no_grad():
        alignment, uniformity = _diagnostics(z_v, z_t)
        # acc_v2t as a monitoring metric (even though not used for loss)
        sims = z_v @ z_t.t()
        targets = torch.arange(z_v.shape[0], device=z_v.device)
        acc_v2t = (sims.argmax(dim=1) == targets).float().mean().item()

    return loss, {
        "loss": loss.item(), "loss_mse": loss_mse.item(), "loss_reg": loss_reg.item(),
        "acc_v2t": acc_v2t, "alignment": alignment, "uniformity": uniformity,
    }


class TiledContrastiveLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_v, z_t, temperature=0.07, tile_size=128):
        ctx.save_for_backward(z_v, z_t)
        ctx.temperature = temperature
        ctx.tile_size = tile_size
        
        B, D = z_v.shape
        device = z_v.device
        dtype = z_v.dtype
        
        # Track running max (m) and running sum of exp (d) in float32 for stability
        m_row = torch.full((B,), float('-inf'), device=device, dtype=torch.float32)
        d_row = torch.zeros((B,), device=device, dtype=torch.float32)
        
        m_col = torch.full((B,), float('-inf'), device=device, dtype=torch.float32)
        d_col = torch.zeros((B,), device=device, dtype=torch.float32)
        
        # Pass 1: Online Log-Sum-Exp computation
        for i in range(0, B, tile_size):
            z_v_chunk = z_v[i:i+tile_size].float()
            for j in range(0, B, tile_size):
                z_t_chunk = z_t[j:j+tile_size].float()
                
                # Compute local tile similarity
                S_tile = (z_v_chunk @ z_t_chunk.t()) / temperature
                
                # Update row statistics (v2t)
                m_tile_row, _ = S_tile.max(dim=1)
                m_row_new = torch.maximum(m_row[i:i+tile_size], m_tile_row)
                d_row[i:i+tile_size] = (
                    d_row[i:i+tile_size] * torch.exp(m_row[i:i+tile_size] - m_row_new) +
                    torch.exp(S_tile - m_row_new.unsqueeze(1)).sum(dim=1)
                )
                m_row[i:i+tile_size] = m_row_new
                
                # Update col statistics (t2v)
                m_tile_col, _ = S_tile.max(dim=0)
                m_col_new = torch.maximum(m_col[j:j+tile_size], m_tile_col)
                d_col[j:j+tile_size] = (
                    d_col[j:j+tile_size] * torch.exp(m_col[j:j+tile_size] - m_col_new) +
                    torch.exp(S_tile - m_col_new.unsqueeze(0)).sum(dim=0)
                )
                m_col[j:j+tile_size] = m_col_new
        
        # Final loss value computation
        diag_sims = torch.sum(z_v.float() * z_t.float(), dim=1) / temperature
        
        lse_row = m_row + torch.log(d_row)
        loss_v2t = (-diag_sims + lse_row).mean()
        
        lse_col = m_col + torch.log(d_col)
        loss_t2v = (-diag_sims + lse_col).mean()
        
        loss = 0.5 * (loss_v2t + loss_t2v)
        
        ctx.m_row = m_row
        ctx.d_row = d_row
        ctx.m_col = m_col
        ctx.d_col = d_col
        
        return loss.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z_v, z_t = ctx.saved_tensors
        temperature = ctx.temperature
        tile_size = ctx.tile_size
        m_row = ctx.m_row
        d_row = ctx.d_row
        m_col = ctx.m_col
        d_col = ctx.d_col
        
        B, D = z_v.shape
        device = z_v.device
        dtype = z_v.dtype
        
        grad_zv = torch.zeros_like(z_v, dtype=torch.float32)
        grad_zt = torch.zeros_like(z_t, dtype=torch.float32)
        
        # Pass 2: Accumulate gradients block-by-block
        for i in range(0, B, tile_size):
            z_v_chunk = z_v[i:i+tile_size].float()
            m_r = m_row[i:i+tile_size].unsqueeze(1)
            d_r = d_row[i:i+tile_size].unsqueeze(1)
            
            for j in range(0, B, tile_size):
                z_t_chunk = z_t[j:j+tile_size].float()
                m_c = m_col[j:j+tile_size].unsqueeze(0)
                d_c = d_col[j:j+tile_size].unsqueeze(0)
                
                S_tile = (z_v_chunk @ z_t_chunk.t()) / temperature
                
                # Local softmax probability calculation
                P_tile = torch.exp(S_tile - m_r) / d_r
                Q_tile = torch.exp(S_tile - m_c) / d_c
                
                d_logits = (P_tile + Q_tile) / (2.0 * B)
                
                grad_zv[i:i+tile_size] += (d_logits @ z_t_chunk) / temperature
                grad_zt[j:j+tile_size] += (d_logits.t() @ z_v_chunk) / temperature
                
        # Diagonal subtraction corrections
        grad_zv -= z_t.float() / (B * temperature)
        grad_zt -= z_v.float() / (B * temperature)
        
        return grad_zv.to(dtype) * grad_output, grad_zt.to(dtype) * grad_output, None, None


def info_nce(
    z_v: Tensor,              # (B, D) normalized video embeddings
    z_t: Tensor,              # (B, D) normalized text embeddings
    temperature: float = 0.07,
    tile_size: int = 128,
) -> Tuple[Tensor, Dict[str, float]]:
    # Compute the loss using the memory-efficient tiled autograd function
    loss = TiledContrastiveLoss.apply(z_v, z_t, temperature, tile_size)
    
    with torch.no_grad():
        # Compute diagnostics (row/col softmax metrics)
        # Done under torch.no_grad() so it doesn't store gradients or consume backprop memory
        logits = (z_v @ z_t.t()) / temperature
        targets = torch.arange(z_v.shape[0], device=z_v.device)
        loss_v2t = F.cross_entropy(logits, targets)
        loss_t2v = F.cross_entropy(logits.t(), targets)
        acc_v2t = (logits.argmax(dim=1) == targets).float().mean().item()
        acc_t2v = (logits.t().argmax(dim=1) == targets).float().mean().item()
        alignment, uniformity = _diagnostics(z_v, z_t)
        
    return loss, {
        "loss": loss.item(), "loss_v2t": loss_v2t.item(), "loss_t2v": loss_t2v.item(),
        "acc_v2t": acc_v2t, "acc_t2v": acc_t2v,
        "alignment": alignment, "uniformity": uniformity, "queue_negatives": 0,
    }


def info_nce_with_queue(
    z_v: Tensor,                       # (B, D) current-batch video embeds (grad)
    z_t: Tensor,                       # (B, D) current-batch text embeds (grad)
    neg_t: Optional[Tensor] = None,    # (Kt, D) queued TEXT negatives for v->t (detached)
    neg_v: Optional[Tensor] = None,    # (Kv, D) queued VIDEO negatives for t->v (detached)
    temperature: float = 0.07,
) -> Tuple[Tensor, Dict[str, float]]:
    """InfoNCE where the positive for row i is the in-batch diagonal (column i), and the
    queue contributes EXTRA negative columns. The positive is always a fresh in-batch
    pair, never a queued (stale) one."""
    B = z_v.shape[0]
    targets = torch.arange(B, device=z_v.device)

    # v -> t : [ in-batch (B) | queued text (Kt) ]   positive at column i
    logits_v = z_v @ z_t.t()                                  # (B, B)
    if neg_t is not None and neg_t.numel() > 0:
        logits_v = torch.cat([logits_v, z_v @ neg_t.t()], dim=1)   # (B, B+Kt)
    logits_v = logits_v / temperature
    loss_v2t = F.cross_entropy(logits_v, targets)

    # t -> v : [ in-batch (B) | queued video (Kv) ]
    logits_t = z_t @ z_v.t()                                  # (B, B)
    if neg_v is not None and neg_v.numel() > 0:
        logits_t = torch.cat([logits_t, z_t @ neg_v.t()], dim=1)   # (B, B+Kv)
    logits_t = logits_t / temperature
    loss_t2v = F.cross_entropy(logits_t, targets)

    loss = 0.5 * (loss_v2t + loss_t2v)

    with torch.no_grad():
        # acc is now over the FULL (B + K)-way problem -> a real proxy for retrieval,
        # not the easy 8-way number that used to saturate at 1.0.
        acc_v2t = (logits_v.argmax(dim=1) == targets).float().mean().item()
        acc_t2v = (logits_t.argmax(dim=1) == targets).float().mean().item()
        alignment, uniformity = _diagnostics(z_v, z_t)
        n_neg = int(neg_t.shape[0]) if (neg_t is not None) else 0

    return loss, {
        "loss": loss.item(), "loss_v2t": loss_v2t.item(), "loss_t2v": loss_t2v.item(),
        "acc_v2t": acc_v2t, "acc_t2v": acc_t2v,
        "alignment": alignment, "uniformity": uniformity, "queue_negatives": n_neg,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    v = F.normalize(torch.randn(8, 1536), dim=-1)
    t = F.normalize(torch.randn(8, 1536), dim=-1)
    qv = F.normalize(torch.randn(2048, 1536), dim=-1)
    qt = F.normalize(torch.randn(2048, 1536), dim=-1)
    l0, m0 = info_nce(v, t)
    l1, m1 = info_nce_with_queue(v, t, neg_t=qt, neg_v=qv)
    print(f"[losses] in-batch only : loss={m0['loss']:.3f} acc_v2t={m0['acc_v2t']:.2f} negs={m0['queue_negatives']}")
    print(f"[losses] with queue    : loss={m1['loss']:.3f} acc_v2t={m1['acc_v2t']:.2f} negs={m1['queue_negatives']}")
    # aligned sanity: identical embeds -> ~0 loss even against a big queue
    la, ma = info_nce_with_queue(v, v, neg_t=qt, neg_v=qv)
    print(f"[losses] aligned+queue : loss={ma['loss']:.3f} acc_v2t={ma['acc_v2t']:.2f}")


def compute_siglip_loss(z_v, z_t, temp=14.0, bias=-10.0):
    """
    Pairwise Sigmoid Loss (SigLIP style)
    z_v: (B, D) Normalized video features
    z_t: (B, D) Normalized text features
    """
    B = z_v.size(0)
    sim_matrix = torch.matmul(z_v, z_t.t())
    labels = 2 * torch.eye(B, device=z_v.device) - 1
    logits = sim_matrix * temp + bias
    loss = -F.logsigmoid(labels * logits).sum() / B
    return loss

