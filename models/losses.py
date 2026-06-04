"""
models/losses.py

Bi-directional InfoNCE (the VL-JEPA / CLIP objective), which the VL-JEPA paper
(2512.10942) decomposes into:
  1) an alignment term  -> pulls matched (video, text) embeddings together, and
  2) a uniformity term  -> spreads embeddings on the hypersphere (anti-collapse).
The InfoNCE denominator (negatives) IS the uniformity pressure; that is why InfoNCE
needs no separate EMA target to avoid collapse. We additionally log the explicit
alignment/uniformity diagnostics of Wang & Isola for monitoring.

Inputs are assumed L2-normalized (the encoders normalize their outputs).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def info_nce(
    z_v: torch.Tensor,        # (B, D) normalized video embeddings
    z_t: torch.Tensor,        # (B, D) normalized text embeddings
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = (z_v @ z_t.t()) / temperature      # (B, B)
    targets = torch.arange(z_v.shape[0], device=z_v.device)
    loss_v2t = F.cross_entropy(logits, targets)
    loss_t2v = F.cross_entropy(logits.t(), targets)
    loss = 0.5 * (loss_v2t + loss_t2v)

    with torch.no_grad():
        # in-batch retrieval accuracy (proxy for the M1 gate)
        acc_v2t = (logits.argmax(dim=1) == targets).float().mean().item()
        acc_t2v = (logits.t().argmax(dim=1) == targets).float().mean().item()
        # Wang & Isola diagnostics
        alignment = (z_v - z_t).pow(2).sum(-1).mean().item()
        sq = torch.pdist(torch.cat([z_v, z_t], 0), p=2).pow(2)
        uniformity = sq.mul(-2).exp().mean().clamp_min(1e-12).log().item()

    metrics = {
        "loss": loss.item(),
        "loss_v2t": loss_v2t.item(),
        "loss_t2v": loss_t2v.item(),
        "acc_v2t": acc_v2t,
        "acc_t2v": acc_t2v,
        "alignment": alignment,
        "uniformity": uniformity,
    }
    return loss, metrics


if __name__ == "__main__":
    torch.manual_seed(0)
    v = F.normalize(torch.randn(8, 1536), dim=-1)
    t = F.normalize(torch.randn(8, 1536), dim=-1)
    # perfectly-aligned case should give near-0 loss and acc ~1.0
    loss_rand, m_rand = info_nce(v, t)
    loss_aligned, m_aligned = info_nce(v, v)
    print(f"[losses] random:  loss={m_rand['loss']:.3f} acc_v2t={m_rand['acc_v2t']:.2f}")
    print(f"[losses] aligned: loss={m_aligned['loss']:.3f} acc_v2t={m_aligned['acc_v2t']:.2f}")
