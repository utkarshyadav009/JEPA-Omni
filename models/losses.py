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
import torch.nn.functional as F
from torch import Tensor


def _diagnostics(z_v: Tensor, z_t: Tensor) -> Tuple[float, float]:
    """Wang & Isola alignment / uniformity on the current batch (monitoring only)."""
    alignment = (z_v - z_t).pow(2).sum(-1).mean().item()
    sq = torch.pdist(torch.cat([z_v, z_t], 0), p=2).pow(2)
    uniformity = sq.mul(-2).exp().mean().clamp_min(1e-12).log().item()
    return alignment, uniformity


def sigreg_loss(z: Tensor, sketch_dim: int = 64, t_range: float = 5.0, num_t: int = 17) -> Tensor:
    """
    SigReg (Sketched Isotropic Gaussian Regularization) from LeJEPA (2511.08544).
    Forces the distribution of embeddings to match an Isotropic Gaussian N(0, I).
    z: [Batch, Dimension]
    """
    N, D = z.shape
    if N <= 1:
        return z.new_zeros(())

    # 1. Random 1D Projections (The Sketch)
    # Ideally A should be fixed or a registered buffer, but for M1 we generate it.
    # We use a fixed seed for A to ensure stability within a step if called multiple times.
    generator = torch.Generator(device=z.device).manual_seed(42)
    A = torch.randn(D, sketch_dim, device=z.device, dtype=z.dtype, generator=generator)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
    proj = z @ A  # [N, sketch_dim]

    # 2. Integration points for the Characteristic Function (CF)
    t = torch.linspace(-t_range, t_range, num_t, device=z.device, dtype=z.dtype)
    phi_gauss = torch.exp(-0.5 * t**2)  # Theoretical Gaussian CF

    # 3. Empirical Characteristic Function (ECF)
    # ECF(t) = 1/N * sum(exp(i * t * proj))
    args = proj.unsqueeze(2) * t.view(1, 1, -1)  # [N, sketch_dim, num_t]
    ecf_real = torch.cos(args).mean(dim=0)       # [sketch_dim, num_t]
    ecf_imag = torch.sin(args).mean(dim=0)       # [sketch_dim, num_t]

    # 4. Weighted L2 Distance (Epps-Pulley style)
    # |ecf - phi_gauss|^2 = (ecf_real - phi_gauss)^2 + ecf_imag^2
    diff_sq = (ecf_real - phi_gauss.unsqueeze(0)).pow(2) + ecf_imag.pow(2)
    weighted_diff = diff_sq * phi_gauss.unsqueeze(0)

    # 5. Integrate over t (trapezoidal rule)
    loss = torch.trapezoid(weighted_diff, t, dim=1)
    return loss.mean()


def sigreg_jepa_loss(
    z_v: Tensor,
    z_t: Tensor,
    lamb: float = 10.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    LeJEPA total loss: Prediction Error (MSE) + lambda * SigReg Regularization.
    Inputs are assumed to be embeddings from the predictor and text target.
    """
    # Prediction loss (MSE)
    loss_mse = F.mse_loss(z_v, z_t)

    # Regularization loss
    reg_v = sigreg_loss(z_v)
    reg_t = sigreg_loss(z_t)
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


def info_nce(
    z_v: Tensor,              # (B, D) normalized video embeddings
    z_t: Tensor,              # (B, D) normalized text embeddings
    temperature: float = 0.07,
) -> Tuple[Tensor, Dict[str, float]]:
    logits = (z_v @ z_t.t()) / temperature      # (B, B)
    targets = torch.arange(z_v.shape[0], device=z_v.device)
    loss_v2t = F.cross_entropy(logits, targets)
    loss_t2v = F.cross_entropy(logits.t(), targets)
    loss = 0.5 * (loss_v2t + loss_t2v)

    with torch.no_grad():
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

