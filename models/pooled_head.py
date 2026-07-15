"""models/pooled_head.py — STEP 3 READABLE GATE.

Cross-modal pooled prediction head following MJEPA Sec 4.3:
  "Cross-modal predictors are simple 3-layer MLPs that operate only on the
   mean-pooled, last-layer features of the encoder."

Architecture (per direction, e.g. audio→vision):
  mean-pool(source backbone tokens) → 3-layer MLP → predicted target pool repr
  L1 loss against mean-pool(frozen target latents)

Training: added as an OPTIONAL auxiliary (weight ~0.1) on top of the main
  token-level smooth-L1 prediction loss.

Retrieval (PRIMARY M2 metric per spec):
  For each query clip: predict target pool repr from source.
  Rank gallery by L1(predicted_tgt_pool, actual_tgt_pool).
  Report R@1, R@5, R@10 both directions (a→v and v→a).

Secondary metric: cosine similarity in shared space (CAV-MAE-comparable).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossModalPooledHead(nn.Module):
    """3-layer MLP: mean-pool(source tokens) → predicted target pool repr.

    Args:
        d_model:    predictor hidden dim (input dim = mean-pooled source tokens)
        target_dim: frozen encoder output dim for the target modality
        hidden_ratio: MLP hidden multiplier (MJEPA uses same width as encoder)
    """

    def __init__(self, d_model: int, target_dim: int, hidden_ratio: float = 4.0):
        super().__init__()
        h = int(d_model * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, h),  nn.GELU(),
            nn.Linear(h, h),        nn.GELU(),
            nn.Linear(h, target_dim),
        )
        nn.init.trunc_normal_(self.mlp[0].weight, std=0.02)
        nn.init.trunc_normal_(self.mlp[2].weight, std=0.02)
        nn.init.trunc_normal_(self.mlp[4].weight, std=0.02)

    def forward(self, src_tokens: Tensor) -> Tensor:
        """(B, T_src, d_model) → (B, target_dim) predicted target pool repr."""
        return self.mlp(src_tokens.mean(1))

    def loss(self, src_tokens: Tensor, tgt_feats: Tensor) -> Tensor:
        """L1 between predicted pool and mean-pooled frozen target features."""
        pred = self.forward(src_tokens)              # (B, target_dim)
        tgt  = tgt_feats.mean(1).detach().float()   # (B, target_dim) frozen
        return F.l1_loss(pred.float(), tgt)


class PooledXModalHeads(nn.Module):
    """Pair of CrossModalPooledHeads — one per cross-modal direction.

    Keys are (src_modality, tgt_modality). Typical: {"ambient→vision", "vision→ambient"}.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int,
        hidden_ratio: float = 4.0,
    ):
        super().__init__()
        mods = list(modality_dims.keys())
        assert len(mods) == 2, "PooledXModalHeads expects exactly 2 modalities"
        m0, m1 = mods[0], mods[1]
        self.heads = nn.ModuleDict({
            f"{m0}→{m1}": CrossModalPooledHead(d_model, modality_dims[m1], hidden_ratio),
            f"{m1}→{m0}": CrossModalPooledHead(d_model, modality_dims[m0], hidden_ratio),
        })
        self.modality_dims = modality_dims
        self.d_model = d_model

    def combined_loss(
        self,
        src_tokens_by_mod: Dict[str, Tensor],
        feats: Dict[str, Tensor],
    ) -> Tensor:
        """Average L1 loss over both cross-modal directions.

        Args:
            src_tokens_by_mod: {modality → (B, T_m, d)} backbone tokens per modality
            feats: {modality → (B, T_m, dim)} frozen latents (for pooled targets)
        """
        mods = list(self.modality_dims.keys())
        m0, m1 = mods[0], mods[1]
        l01 = self.heads[f"{m0}→{m1}"].loss(src_tokens_by_mod[m0], feats[m1])
        l10 = self.heads[f"{m1}→{m0}"].loss(src_tokens_by_mod[m1], feats[m0])
        return (l01 + l10) * 0.5


# ── Retrieval eval ─────────────────────────────────────────────────────────────

@torch.no_grad()
def pooled_retrieval_eval(
    heads: PooledXModalHeads,
    predictor,               # AVJepaPredictor (no type import to avoid circular)
    loader,
    device: torch.device,
    modality_dims: Dict[str, int],
    max_clips: int = 1545,
) -> Dict[str, float]:
    """Rank-based retrieval eval using pooled cross-modal prediction error.

    PRIMARY metric: rank gallery by L1(predicted_tgt_pool, actual_tgt_pool).
    SECONDARY metric: cosine similarity of mean-pooled backbone tokens.

    Returns dict with R@1/5/10 for both directions and both metric types.
    """
    mods = list(modality_dims.keys())
    m0, m1 = mods[0], mods[1]

    heads.eval()
    predictor.eval()

    # Build gallery
    pools: Dict[str, List[Tensor]] = {m: [] for m in mods}
    cos_pools: Dict[str, List[Tensor]] = {m: [] for m in mods}  # backbone mean-pools

    n_clips = 0
    for batch in loader:
        if n_clips >= max_clips:
            break
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            src_by_mod = predictor.encode_source_tokens(feats, tbins)

        for m in mods:
            pools[m].append(feats[m].mean(1).cpu().float())         # frozen pool
            cos_pools[m].append(src_by_mod[m].mean(1).cpu().float())  # backbone pool
        n_clips += next(iter(feats.values())).shape[0]

    for m in mods:
        pools[m]     = torch.cat(pools[m], 0)[:max_clips]    # (N, dim_m)
        cos_pools[m] = torch.cat(cos_pools[m], 0)[:max_clips]

    N = pools[m0].shape[0]

    # Build prediction-error similarity (negative L1 as score) for both directions
    results: Dict[str, float] = {}

    for src_m, tgt_m in [(m0, m1), (m1, m0)]:
        key = f"{src_m}→{tgt_m}"
        head = heads.heads[key]

        # Predict target pool for each query
        src_pool = cos_pools[src_m].to(device)      # (N, d_model)
        # head.forward expects (B, T, d) → expand T dim
        pred_tgt = head.mlp(src_pool).cpu().float()  # (N, target_dim)

        tgt_pool = pools[tgt_m]                      # (N, target_dim)

        # L1 distance matrix (N, N)
        l1_mat = torch.cdist(pred_tgt, tgt_pool, p=1)   # (N, N)
        # rank by ascending L1 (smaller = predicted closer)
        ranked_l1 = l1_mat.argsort(1)

        # Cosine secondary metric
        src_cos = F.normalize(cos_pools[src_m], dim=-1)  # (N, d)
        tgt_cos = F.normalize(cos_pools[tgt_m], dim=-1)  # (N, d)
        cos_mat = src_cos @ tgt_cos.T                     # (N, N)
        ranked_cos = (-cos_mat).argsort(1)

        gt = torch.arange(N)
        for k, ranked, suffix in [(1, ranked_l1, "_l1"), (5, ranked_l1, "_l1"),
                                   (10, ranked_l1, "_l1"),
                                   (1, ranked_cos, "_cos"), (5, ranked_cos, "_cos"),
                                   (10, ranked_cos, "_cos")]:
            pass  # compute below

        for ranked, suffix in [(ranked_l1, "_l1"), (ranked_cos, "_cos")]:
            for k in (1, 5, 10):
                hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
                results[f"{key}_R@{k}{suffix}"] = round(hits * 100, 2)

    heads.train()
    predictor.train()
    return results
