"""models/perception_prefix.py — perception as an EMBEDDING PREFIX for the LLM.

THE POINT: make perception OPEN-VOCABULARY again. The retrieval path can only ever say what
its bank contains — pointed at a real bedroom it answered "a tuning fork being tapped",
because a VGGSound sound-event bank has no caption for that room. This module replaces the
bank with generation: the scene becomes k soft tokens in the LLM's own embedding space, and
the LLM writes the description in its own words.

WHY THIS IS NOT JUST RE-RUNNING M3 (which was dropped for latency):
    M3 era: Qwen2.5-1.5B @ 203 ms/token -> a 30-token description cost ~6 s.
    Now:    Qwen3-0.6B thinker @ 18.2 ms/token -> the same description costs ~550 ms,
            and LFM2.5-350M @ 9.4 ms/token would cost ~280 ms.
    Plus `llama_batch.embd` is PROVEN to accept a raw embedding prefix byte-identically
    (scripts/prototype_llama_embd_input.py), so no soft-prompt plumbing or C++ fork.
The idea was right; the decoder was too slow. The decoder got 11-21x faster for unrelated
reasons, so the idea is now affordable.

DESIGN CHOICE THAT MATTERS — the query is TEXT, not a conditioning vector.
    prompt = [k soft perception tokens] + "Task: {question}\\n" + {answer}
The projector encodes the SCENE; the QUESTION arrives as ordinary text the LLM already
understands. This is the LLaVA/Ultravox pattern and it is deliberately chosen over
conditioning the projector on a query embedding, because it means **an unseen question type
works with no retraining** — which is the whole point of getting genericity back. A
query-conditioned projector would re-create the closed-set problem one level up.

WHERE IDENTITY AND THE QUERY PATH FIT (asked directly, 2026-08-14):
  * **Query** — text in the prompt, as above. Free-form, no retraining, any phrasing.
  * **Identity** — the identity head outputs a 256-d embedding that is matched against the
    memory to yield a NAME. A name is symbolic, and LLMs consume names natively, so it is
    injected as TEXT ("You recognise Alice.") rather than trained into the projector. That
    keeps enrolment instant (no retraining when a new person is added) and keeps this module
    responsible for one thing: turning perception into describable content.
  * **Retrieval** — kept as the fast in-distribution path; a score-gated router picks
    retrieval (~8 ms) when confident and generation (~550 ms) when not. See
    PERCEPTION_GENERALIZATION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PerceptionPrefixConfig:
    source_dims: Dict[str, int] = field(
        default_factory=lambda: {"m2": 1024, "vision": 1024, "ambient": 768}
    )
    llm_hidden: int = 1024        # Qwen3-0.6B hidden size; read from the model, not assumed
    n_prefix: int = 16            # soft tokens handed to the LLM
    d_model: int = 768
    depth: int = 3
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class _Block(nn.Module):
    """Cross-attend the perception tokens, then self-attend among the latents."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n_q, self.n_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n_s = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n_f = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, lat: Tensor, tok: Tensor, pad: Optional[Tensor]) -> Tensor:
        q, kv = self.n_q(lat), self.n_kv(tok)
        lat = lat + self.cross(q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        s = self.n_s(lat)
        lat = lat + self.self_attn(s, s, s, need_weights=False)[0]
        return lat + self.ff(self.n_f(lat))


class PerceptionPrefix(nn.Module):
    """{stream: (B,S,D)} -> (B, n_prefix, llm_hidden) soft tokens for the LLM."""

    def __init__(self, cfg: PerceptionPrefixConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_proj = nn.ModuleDict({s: nn.Linear(dim, d) for s, dim in cfg.source_dims.items()})
        self.src_emb = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(1, 1, d)) for s in cfg.source_dims})
        self.latents = nn.Parameter(torch.zeros(1, cfg.n_prefix, d))
        self.blocks = nn.ModuleList(
            [_Block(d, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.llm_hidden)
        nn.init.trunc_normal_(self.latents, std=0.02)
        for p in self.src_emb.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, sources: Dict[str, Tensor],
                padding_masks: Optional[Dict[str, Tensor]] = None) -> Tensor:
        toks, pads = [], []
        B = next(iter(sources.values())).shape[0]
        for s, x in sources.items():
            toks.append(self.tok_proj[s](x.float()) + self.src_emb[s])
            pads.append(padding_masks[s] if (padding_masks and s in padding_masks)
                        else torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device))
        tok = torch.cat(toks, 1)
        pad = torch.cat(pads, 1)
        if not pad.any():
            pad = None
        lat = self.latents.expand(B, -1, -1)
        for blk in self.blocks:
            lat = blk(lat, tok, pad)
        return self.out(self.norm(lat))          # (B, n_prefix, llm_hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PerceptionPrefixConfig()
    m = PerceptionPrefix(cfg)
    src = {"m2": torch.randn(2, 812, 1024), "vision": torch.randn(2, 512, 1024),
           "ambient": torch.randn(2, 300, 768)}
    p = m(src)
    print(f"params={sum(x.numel() for x in m.parameters())/1e6:.1f}M  prefix={tuple(p.shape)}")
    a, b = m({k: v[:1] for k, v in src.items()}), m({k: v[1:] for k, v in src.items()})
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"different scenes -> different prefix (cos={cos:+.4f}, must not be ~1.0)")
    p.sum().backward(); print("backward OK")
