"""models/jepa_identity_head.py — Phase 2 of the JEPA-memory track.

The trainable head that turns FROZEN perceptual features into an identity
embedding. This is the module the memory stores and looks up.

WHY IT EXISTS (all measured in Phase 1, see JEPA_MEMORY_PLAN.md):
  * Identity IS present in the frozen features -- within-session vision top-1
    0.860 vs 0.446 chance; cross-session voice 250-way top-1 0.250 vs 0.004.
  * But frozen features alone are ~4x short of deployment grade (a dedicated
    speaker model reaches ~99% where frozen WavJEPA reaches 25%). So a trained
    head is REQUIRED, not an optimization.

TWO ARCHITECTURE DECISIONS, BOTH MEASURED RATHER THAN ASSUMED:
  1. **Vision taps raw ViT-L, NOT the M2 world-state.** M2 loses ~16pp of
     identity (0.860 -> 0.703) because its cross-modal congruence objective
     treats "who this is" as nuisance variation. Fusing M2 back in does NOT
     recover it (0.846, CI overlaps ViT-L alone) -- M2 is a strictly lossy view
     of its own input for this purpose. So the head reads a tap point that
     production already computes and currently discards. No M2 retrain.
  2. **Voice uses WavJEPA, NOT Moonshine** (0.250 vs 0.166 cross-session).
     Moonshine's encoder is ASR-trained, i.e. deliberately speaker-INVARIANT,
     so its "already resident on the Jetson" advantage does not survive contact
     with the measurement.

POOLING: statistics pooling (concat of mean and std over time/tokens). This is
the x-vector / ECAPA-TDNN standard for speaker embeddings precisely because the
*variability* of a representation over an utterance carries speaker information
that the mean alone throws away. Phase 1 used mean-only, so any gain here is
attributable and reported separately from the head itself.

LOSS: AAM-softmax (ArcFace). Standard for open-set identity because it trains an
angular margin on a hypersphere, which is exactly the geometry a cosine-similarity
memory lookup uses at inference -- unlike plain softmax, which optimizes a
decision boundary the memory never sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def stats_pool(tokens: Tensor, dim: int = 1) -> Tensor:
    """(B, T, D) -> (B, 2D) as concat(mean, std). The std half is the part that
    mean-pooling discards and that speaker-ID literature relies on."""
    mu = tokens.mean(dim)
    sd = tokens.std(dim, unbiased=False).clamp_min(1e-5)
    return torch.cat([mu, sd], dim=-1)


@dataclass
class IdentityHeadConfig:
    # input dims AFTER stats pooling (2x the encoder dim), per modality
    in_dims: Dict[str, int] = field(default_factory=lambda: {"vision": 2048, "audio": 1536})
    hidden: int = 1024
    emb_dim: int = 256          # the stored identity embedding
    dropout: float = 0.1
    n_layers: int = 2


class IdentityHead(nn.Module):
    """Per-modality trunk -> fused -> L2-normalised identity embedding.

    Works with any SUBSET of the configured modalities: pass only the ones you
    have. Missing modalities are zero-filled AND flagged via a learned
    presence embedding, so "no audio this tick" is a state the model has seen
    rather than an out-of-distribution zero vector. That matters for deployment:
    BMO often sees a face before the person speaks."""

    def __init__(self, cfg: IdentityHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.trunks = nn.ModuleDict()
        for m, d in cfg.in_dims.items():
            layers: List[nn.Module] = [nn.LayerNorm(d)]
            prev = d
            for _ in range(cfg.n_layers):
                layers += [nn.Linear(prev, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
                prev = cfg.hidden
            self.trunks[m] = nn.Sequential(*layers)
        self.present = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, cfg.hidden)) for m in cfg.in_dims}
        )
        self.absent = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, cfg.hidden)) for m in cfg.in_dims}
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden * len(cfg.in_dims)),
            nn.Linear(cfg.hidden * len(cfg.in_dims), cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.emb_dim),
        )
        for p in list(self.present.values()) + list(self.absent.values()):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, feats: Dict[str, Optional[Tensor]]) -> Tensor:
        """feats: {modality: (B, in_dim) or None} -> (B, emb_dim), L2-normalised."""
        B = next(x.shape[0] for x in feats.values() if x is not None)
        dev = next(x.device for x in feats.values() if x is not None)
        parts = []
        for m in self.cfg.in_dims:
            x = feats.get(m)
            if x is None:
                h = self.absent[m].expand(B, -1)
            else:
                h = self.trunks[m](x.float()) + self.present[m]
            parts.append(h.to(dev))
        return F.normalize(self.fuse(torch.cat(parts, -1)), dim=-1)


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax (ArcFace). Trains the SAME cosine geometry
    the memory uses at lookup time."""

    def __init__(self, emb_dim: int, n_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.margin, self.scale = margin, scale

    def forward(self, emb: Tensor, labels: Tensor) -> Tensor:
        cos = F.linear(F.normalize(emb, dim=-1), F.normalize(self.W, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.margin)
        oh = F.one_hot(labels, self.W.shape[0]).to(cos.dtype)
        logits = self.scale * (oh * target + (1 - oh) * cos)
        return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = IdentityHeadConfig()
    head = IdentityHead(cfg)
    crit = AAMSoftmax(cfg.emb_dim, 100)
    B = 8
    both = {"vision": torch.randn(B, 2048), "audio": torch.randn(B, 1536)}
    z = head(both)
    print(f"params={sum(p.numel() for p in head.parameters()):,}  z={tuple(z.shape)}  "
          f"norm={z.norm(dim=-1).mean():.4f}")
    loss = crit(z, torch.randint(0, 100, (B,)))
    loss.backward()
    print(f"AAM loss={loss.item():.4f}  backward OK")
    z_audio_only = head({"vision": None, "audio": both["audio"]})
    print(f"audio-only path OK: {tuple(z_audio_only.shape)}  "
          f"differs from joint: {not torch.allclose(z, z_audio_only)}")
    print(f"stats_pool: {tuple(stats_pool(torch.randn(4, 996, 768)).shape)} (expect (4,1536))")
