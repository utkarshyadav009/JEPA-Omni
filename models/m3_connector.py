"""models/m3_connector.py — M3 connector: Perceiver-style aggregator that maps
the frozen M2 predictor's pre-pool token sequence (encode_pre_pool_tokens(),
variable length, d_model=1024) down to a fixed set of soft-prompt embeddings
in the frozen LLM's hidden space.

This is the ONLY trainable component of M3 stage 1. M2 (vision/ambient
encoders + predictor trunk) and the LLM are both frozen.

Design: `n_latents` learned query vectors cross-attend the input token
sequence at every layer (classic Perceiver-IO pattern: latents get refined
across layers, input tokens stay the fixed K/V source throughout -- this
lets the query count stay small and constant regardless of how long the
input sequence is, which matters here since vision+ambient token counts vary
per clip). A final projection maps the latents into the LLM's hidden size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class M3ConnectorConfig:
    d_model: int = 1024        # must match AVJepaConfig.d_model (predictor output dim)
    n_latents: int = 32
    n_layers: int = 3
    n_heads: int = 8
    mlp_ratio: float = 4.0
    llm_hidden: int = 1536     # Qwen2.5-1.5B-Instruct hidden_size -- read from its
                                # AutoConfig at build time, never hardcode at the call site
    dropout: float = 0.0


class PerceiverLayer(nn.Module):
    """One round of: latents cross-attend the (fixed) input tokens, then a
    per-latent FFN. Pre-norm, residual, matching the style of Block in
    av_jepa_predictor.py."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, latents: Tensor, tokens: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        q = self.norm_q(latents)
        kv = self.norm_kv(tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.mlp(self.norm_mlp(latents))
        return latents


class M3Connector(nn.Module):
    def __init__(self, cfg: M3ConnectorConfig):
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(torch.randn(1, cfg.n_latents, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverLayer(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.llm_hidden)

    def forward(self, pre_pool_tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """pre_pool_tokens: (B, S, d_model) from AVJepaPredictor.encode_pre_pool_tokens().
        key_padding_mask: (B, S) bool, True at PADDED positions (for batches with
        variable sequence length -- MultiheadAttention convention).
        Returns (B, n_latents, llm_hidden) soft-prompt embeddings."""
        B = pre_pool_tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, pre_pool_tokens, key_padding_mask)
        latents = self.norm_out(latents)
        return self.proj(latents)


if __name__ == "__main__":
    # CPU smoke test
    cfg = M3ConnectorConfig()
    model = M3Connector(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"M3Connector params={n_params:,}  ({n_params/1e6:.1f}M)")
    B, S = 4, 1510
    tokens = torch.randn(B, S, cfg.d_model)
    out = model(tokens)
    print(f"output shape={tuple(out.shape)}  (expect ({B}, {cfg.n_latents}, {cfg.llm_hidden}))")
    loss = out.sum()
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"backward OK, total grad norm={gnorm:.3f}")
