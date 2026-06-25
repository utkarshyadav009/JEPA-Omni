"""
av_jepa_predictor.py — M2 joint audio-visual predictor (the congruency core).

Design (grounded; see notes per line):
  * Frozen perception encoders feed CACHED latents per modality on a shared temporal
    (TDM) axis: vision = V-JEPA2 ViT-L (1024-d), ambient = WavJEPA (768-d), [speech later].
  * TRAINABLE = this predictor only. Targets are the FROZEN cached latents of the masked
    region -> collapse is prevented by construction (fixed non-degenerate teacher), so
    NO EMA target encoder is needed (unlike vanilla I-JEPA/V-JEPA/WavJEPA, whose encoders
    are trainable). This is the simplification the frozen-encoder regime buys us.
  * Objective: cross-modal masked latent prediction. Mask a modality's tokens over a time
    window; predict their frozen latents from the OTHER modality (+ own visible tokens).
    Loss = smooth-L1 (V-JEPA v1 found L1 more stable than L2 for latent regression).
  * World-State vector = attention-pooled predictor output over the FULL (unmasked) seq.
    It is LEFT UN-NORMALISED on purpose: SIGReg (added beside this loss in training) targets
    N(0,I), which is geometrically incompatible with an L2-normalised sphere — the lesson
    paid for across the M1 SIGReg sweeps. Watch effective_rank(WSV); collapse = var-up/rank-down.

What this module owns: projection, fusion, masking, prediction, World-State, prediction loss,
rank diagnostics. What it does NOT own: the frozen encoders and the cached-feature I/O
(that's the extraction pipeline) — it consumes their cached outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AVJepaConfig:
    # frozen encoder output dims (defaults: V-JEPA2 ViT-L = 1024, WavJEPA = 768)
    modality_dims: Dict[str, int] = field(
        default_factory=lambda: {"vision": 1024, "ambient": 768}
    )
    d_model: int = 1024          # predictor width + World-State dim (default locked earlier)
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    max_tdm_bins: int = 512      # shared temporal axis resolution (timestamps -> bins)
    dropout: float = 0.0


# --------------------------------------------------------------------------- #
# Diagnostics (the falsifier we learned to trust over loss/variance)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def effective_rank(x: Tensor) -> float:
    """Participation-ratio effective rank of (N, D) embeddings: (sum eig)^2 / sum(eig^2).
    Computed in fp32 (covariance + eig are precision-sensitive)."""
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    s = eig.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / (eig.pow(2).sum() + 1e-12))


# --------------------------------------------------------------------------- #
# Transformer block (non-causal; vision+audio co-attend, per VL-JEPA)
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Joint predictor
# --------------------------------------------------------------------------- #
class AVJepaPredictor(nn.Module):
    def __init__(self, cfg: AVJepaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # per-modality input projection (frozen dim -> d_model) and a per-modality
        # output head that predicts back INTO that modality's frozen latent space
        self.in_proj = nn.ModuleDict(
            {m: nn.Linear(dim, d) for m, dim in cfg.modality_dims.items()}
        )
        self.out_head = nn.ModuleDict(
            {m: nn.Linear(d, dim) for m, dim in cfg.modality_dims.items()}
        )
        self.modality_emb = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, d)) for m in cfg.modality_dims}
        )
        # learnable query token for masked positions (shared; position carried by temporal emb)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.temporal_emb = nn.Parameter(torch.zeros(1, cfg.max_tdm_bins, d))

        self.blocks = nn.ModuleList(
            [Block(d, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d))  # attentive pool -> World-State

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_emb, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        for p in self.modality_emb.values():
            nn.init.trunc_normal_(p, std=0.02)

    # --- shared embedding step: project + add modality + temporal(TDM) positional ---
    def _embed(self, feats: Dict[str, Tensor], tbins: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns concatenated tokens (B, S, d), a modality-id per token (B, S),
        and the per-token temporal bin (B, S). tbins[m] are integer TDM bins in [0, max_tdm_bins)."""
        toks, mod_ids, bins = [], [], []
        for idx, (m, x) in enumerate(feats.items()):
            h = self.in_proj[m](x) + self.modality_emb[m]
            h = h + self.temporal_emb[:, tbins[m], :].squeeze(0) if False else h  # see note
            # temporal emb gathered per-token below (kept explicit for correctness)
            te = self.temporal_emb[0, tbins[m]]               # (B, T_m, d)
            h = h + te
            toks.append(h)
            mod_ids.append(torch.full(x.shape[:2], idx, device=x.device, dtype=torch.long))
            bins.append(tbins[m])
        return torch.cat(toks, 1), torch.cat(mod_ids, 1), torch.cat(bins, 1)

    def _backbone(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)

    def forward(
        self,
        feats: Dict[str, Tensor],     # {modality: (B, T_m, dim_m)} frozen cached latents
        tbins: Dict[str, Tensor],     # {modality: (B, T_m)} integer TDM bins
        mask: Dict[str, Tensor],      # {modality: (B, T_m) bool} True = MASKED (to predict)
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Cross-modal masked latent prediction to frozen targets. Returns (loss, metrics)."""
        tokens, mod_ids, _ = self._embed(feats, tbins)         # (B, S, d)
        B, S, d = tokens.shape

        # frozen targets at masked positions (detached: fixed teacher, no grad to encoders)
        flat_mask = torch.cat([mask[m] for m in feats], 1)     # (B, S) bool

        # replace masked tokens with the query token (position retained via temporal emb,
        # which was already added in _embed) -> predictor sees context + "predict here" slots
        q = self.mask_token.expand(B, S, d)
        tokens = torch.where(flat_mask.unsqueeze(-1), q + self._embed_pos_only(feats, tbins), tokens)

        h = self._backbone(tokens)                              # (B, S, d)

        # per-modality output heads -> predict into each modality's frozen latent space
        loss, n = tokens.new_zeros(()), 0
        offset = 0
        metrics: Dict[str, float] = {}
        for idx, (m, x) in enumerate(feats.items()):
            T_m = x.shape[1]
            seg = h[:, offset:offset + T_m]                     # (B, T_m, d)
            mm = mask[m]                                        # (B, T_m)
            if mm.any():
                pred = self.out_head[m](seg[mm])                # (n_masked, dim_m)
                tgt = x[mm].detach()                            # FROZEN target, stop-grad
                lm = F.smooth_l1_loss(pred, tgt)
                loss = loss + lm
                n += 1
                metrics[f"loss_{m}"] = float(lm.detach())
            offset += T_m
        loss = loss / max(1, n)
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def _embed_pos_only(self, feats: Dict[str, Tensor], tbins: Dict[str, Tensor]) -> Tensor:
        """Modality + temporal positional content for masked slots (no input features)."""
        parts = []
        for idx, (m, x) in enumerate(feats.items()):
            te = self.temporal_emb[0, tbins[m]]                # (B, T_m, d)
            parts.append(te + self.modality_emb[m])
        return torch.cat(parts, 1)

    @torch.no_grad()
    def encode_world_state(self, feats: Dict[str, Tensor], tbins: Dict[str, Tensor]) -> Tensor:
        """Full (unmasked) sequence -> attentive-pool -> UN-NORMALISED World-State (B, d).
        Do NOT L2-normalise this; SIGReg in training shapes it toward N(0,I)."""
        tokens, _, _ = self._embed(feats, tbins)
        h = self._backbone(tokens)
        q = self.pool_query.expand(h.shape[0], 1, -1)
        # single-query attentive pool
        attn = torch.softmax((q @ h.transpose(1, 2)) / (h.shape[-1] ** 0.5), dim=-1)  # (B,1,S)
        return (attn @ h).squeeze(1)                            # (B, d) un-normalised


if __name__ == "__main__":
    # CPU smoke test: forward + loss + World-State + effective rank
    torch.manual_seed(0)
    cfg = AVJepaConfig()
    model = AVJepaPredictor(cfg)
    B, Tv, Ta = 4, 32, 50               # toy TDM-aligned token counts
    feats = {"vision": torch.randn(B, Tv, 1024), "ambient": torch.randn(B, Ta, 768)}
    tbins = {  # map each modality's tokens onto the shared TDM axis (integer bins)
        "vision": torch.linspace(0, 99, Tv).long().unsqueeze(0).repeat(B, 1),
        "ambient": torch.linspace(0, 99, Ta).long().unsqueeze(0).repeat(B, 1),
    }
    # cross-modal mask: hide a time-window of ambient, predict from vision
    mask = {"vision": torch.zeros(B, Tv, dtype=torch.bool),
            "ambient": torch.zeros(B, Ta, dtype=torch.bool)}
    mask["ambient"][:, 20:35] = True
    loss, m = model(feats, tbins, mask)
    ws = model.encode_world_state(feats, tbins)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"loss={loss.item():.4f}  metrics={m}")
    print(f"world_state shape={tuple(ws.shape)}  eff_rank={effective_rank(ws):.2f} (of {cfg.d_model})")
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"backward OK, total grad norm={gnorm:.3f}")
