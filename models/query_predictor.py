"""models/query_predictor.py — the query-conditioned perception predictor.

THE CAPABILITY: the thinker asks perception a QUESTION and gets an answer grounded
in what is actually being seen right now. "It's a room with a TV" -> user asks
"describe the room in detail" -> the thinker queries perception with that question
and gets a detailed, scene-grounded answer, instead of being stuck with whatever
single fixed summary the predictor happened to emit.

HOW THIS DIFFERS FROM WHAT ALREADY EXISTS
  * `train_m2_embed_predictor.py`'s Predictor produces ONE vector per clip, matched
    against caption embeddings. One clip -> one summary, fixed granularity. There is
    no way to ask it for more.
  * `train_m3.py` already got half way here without framing it that way: its
    GRANULARITY_TAGS ("Task: describe the sequence of actions in detail.") select
    WHICH KIND of description the connector emits. That is a query -- just a
    hardcoded, 5-valued one, consumed by a slow autoregressive LLM.
  This module generalises that to free-form natural-language queries in the FAST
  embedding space: query text -> cross-attention over the perceptual tokens ->
  one query-conditioned embedding. Non-autoregressive, so it keeps the 5ms
  predictor / 3ms lookup budget measured on the Jetson rather than the 1-6s
  generation the M3 connector cost.

ARCHITECTURE (Perceiver-style cross-attention, deliberately not a big model)
  tokens (B,S,1024) --proj--> (B,S,d)
  query  (B,Dq)     --proj--> (B,L,d)  L learned latents, each seeded by the query
  L x [ cross-attn(latents <- tokens) ; self-attn(latents) ; FFN ]
  pool latents -> head -> (B, shared_dim), L2-normalised

  The query conditions the LATENTS, not the tokens, so a different question
  re-reads the SAME frozen perception rather than needing a re-encode. That is
  what makes "ask a follow-up" cheap: perception runs once per tick, queries are
  ~5ms each on top.

WHY CROSS-ATTENTION AND NOT A CONCATENATED MLP: a fixed pool (what the current
predictor does) decides what to keep BEFORE knowing the question. The whole point
here is to let the question decide which tokens to read. See the plan's §3d
finding that spatial pooling to 16 tokens/bin is the prime suspect for missing
fine detail -- a query-conditioned read is only as good as the tokens it reads,
so this module should be fed the LEAST-pooled tokens affordable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class QueryPredictorConfig:
    """`source_dims` is the UNIFIED part: the predictor cross-attends over the
    concatenation of several token streams, each with its own input projection and
    a learned source embedding so it knows which stream a token came from.

    Available streams, all free from the existing AV cache (no re-extraction):
      m2      (1024) AVJepaPredictor.encode_pre_pool_tokens -- the CROSS-MODAL FUSED
                     view. This is what carries audio-visual CONGRUENCE, which is M2's
                     whole trained purpose and is not reproducible from either raw
                     stream alone. Keeping it is the point of the unified design.
      vision  (1024) the cached spatially-pooled V-JEPA2 ViT-L tokens themselves,
                     i.e. M2's own INPUT, read directly. Phase 1C measured that M2
                     discards information present in this stream (identity 0.860 ->
                     0.703), so giving the query a direct path to it adds detail M2
                     cannot supply.
      ambient ( 768) the cached WavJEPA base+nat tokens, read directly.
    Setting source_dims to a single entry reproduces the original single-stream model,
    which is how the ablation isolates what each stream contributes."""
    source_dims: Dict[str, int] = field(
        default_factory=lambda: {"m2": 1024, "vision": 1024}
    )
    query_dim: int = 768        # TextTarget native (EmbeddingGemma 768 / MiniLM 384)
    d_model: int = 768
    n_latents: int = 8
    depth: int = 3
    heads: int = 8
    mlp_ratio: float = 4.0
    shared_dim: int = 1536      # matches TextTarget's shared space
    dropout: float = 0.0


class _CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n_q1 = nn.LayerNorm(d); self.n_kv = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n_q2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n_ff = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, lat: Tensor, tok: Tensor, tok_pad: Optional[Tensor]) -> Tensor:
        q, kv = self.n_q1(lat), self.n_kv(tok)
        lat = lat + self.cross(q, kv, kv, key_padding_mask=tok_pad, need_weights=False)[0]
        q2 = self.n_q2(lat)
        lat = lat + self.self_attn(q2, q2, q2, need_weights=False)[0]
        return lat + self.ff(self.n_ff(lat))


class QueryPredictor(nn.Module):
    def __init__(self, cfg: QueryPredictorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_proj = nn.ModuleDict(
            {s: nn.Linear(dim, d) for s, dim in cfg.source_dims.items()}
        )
        self.source_emb = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(1, 1, d)) for s in cfg.source_dims}
        )
        self.q_proj = nn.Linear(cfg.query_dim, d)
        self.latents = nn.Parameter(torch.zeros(1, cfg.n_latents, d))
        self.blocks = nn.ModuleList(
            [_CrossBlock(d, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.shared_dim)
        nn.init.trunc_normal_(self.latents, std=0.02)
        for p in self.source_emb.values():
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, sources: Dict[str, Tensor], query_emb: Tensor,
                padding_masks: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """sources {name: (B,S_i,D_i)}; query_emb (B,query_dim) -> (B,shared_dim) L2-normed.

        All streams are projected to d_model, tagged with a learned source embedding,
        and concatenated into ONE token sequence the latents cross-attend. The query
        conditions the latents rather than the tokens, so asking a follow-up re-reads
        the same perception instead of re-encoding it."""
        toks, pads = [], []
        B = next(x.shape[0] for x in sources.values())
        for s, x in sources.items():
            toks.append(self.tok_proj[s](x.float()) + self.source_emb[s])
            if padding_masks is not None and s in padding_masks:
                pads.append(padding_masks[s])
            else:
                pads.append(torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device))
        tok = torch.cat(toks, 1)
        pad = torch.cat(pads, 1)
        if not pad.any():
            pad = None
        q = self.q_proj(query_emb.float()).unsqueeze(1)        # (B,1,d)
        lat = self.latents.expand(B, -1, -1) + q               # every latent is query-seeded
        for blk in self.blocks:
            lat = blk(lat, tok, pad)
        return F.normalize(self.head(self.norm(lat).mean(1)), dim=-1)


# ── Query phrasings ───────────────────────────────────────────────────────────
# Multiple paraphrases per caption field, so the model learns the INTENT rather
# than memorising 5 fixed strings -- the thinker will ask in its own words. The
# eval holds out the LAST phrasing of each field to test exactly that.
QUERY_BANK = {
    "gpt_action_brief": [
        "What is happening?",
        "What is the main action?",
        "In one line, what is going on?",
        "Briefly: what is being done here?",
        "Sum up the action in a few words.",
    ],
    "gpt_action_detailed": [
        "Describe the sequence of actions in detail.",
        "Walk me through step by step what happens.",
        "Give me a detailed account of what is being done.",
        "Explain everything that happens, in order.",
        "Tell me in detail what the person is doing.",
    ],
    "gpt_summary_brief": [
        "Summarize the scene in one sentence.",
        "What is this, briefly?",
        "Give a one-line summary of the whole scene.",
        "In short, what is this a scene of?",
        "Briefly, what am I looking at?",
    ],
    "gpt_summary_detailed": [
        "Describe the room and setting in detail.",
        "What does this place look like? Describe it fully.",
        "Give me a detailed description of the scene, setting and context.",
        "Describe everything you can see about where this is.",
        "Tell me in detail what the surroundings look like.",
    ],
    "gpt_sound_acoustic": [
        "What do you hear?",
        "What does this sound like?",
        "Describe the audio briefly.",
        "In a few words, what is the sound?",
        "Quickly: what can you hear?",
    ],
    "gpt_sound_acoustic_v1_original": [
        "Describe the sound in detail and what is making it.",
        "Tell me everything about what you can hear.",
        "Give a detailed description of the audio and its source.",
        "Describe the acoustic character of this in detail.",
        "In detail, what is the sound and where is it coming from?",
    ],
}

# The VGGSound caption set is really a 3x2 grid, and naming it that way is the
# point of the whole module: a query has to resolve BOTH axes.
#   aspect      x  granularity
#   action      x  {brief, detailed}
#   summary     x  {brief, detailed}
#   sound       x  {brief, detailed}
# Measured mean caption lengths (40k-row sample, JEPA_MEMORY_PLAN.md):
#   action  4.8 / 37.0 words | summary 8.9 / 58.1 | sound 12.2 / 18.0
# Action100M supplies only the action row (3.2 / 27.1 words) but ~400k clips of
# it -- which is exactly the brief<->detailed axis the "describe it in more
# detail" use case lives on, so it is worth mixing in despite the narrower grid.
VGGSOUND_FIELDS = [
    "gpt_action_brief", "gpt_action_detailed",
    "gpt_summary_brief", "gpt_summary_detailed",
    "gpt_sound_acoustic", "gpt_sound_acoustic_v1_original",
]
ACTION100M_FIELDS = ["gpt_action_brief", "gpt_action_detailed"]
HELD_OUT_PHRASING_IDX = -1   # last phrasing of each field is never trained on


def get_query(field: str, rng, train: bool = True) -> str:
    opts = QUERY_BANK[field]
    pool = opts[:HELD_OUT_PHRASING_IDX] if train else opts[HELD_OUT_PHRASING_IDX:]
    return pool[int(rng.integers(0, len(pool)))]


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    for sd in ({"m2": 1024}, {"vision": 1024}, {"m2": 1024, "vision": 1024},
               {"m2": 1024, "vision": 1024, "ambient": 768}):
        cfg = QueryPredictorConfig(source_dims=dict(sd))
        m = QueryPredictor(cfg)
        src = {s: torch.randn(B, 512 if s != "ambient" else 300, d) for s, d in sd.items()}
        qa, qb = torch.randn(B, cfg.query_dim), torch.randn(B, cfg.query_dim)
        za, zb = m(src, qa), m(src, qb)
        cos = F.cosine_similarity(za, zb).mean().item()
        print(f"sources={str(list(sd)):<40} params={sum(p.numel() for p in m.parameters())/1e6:5.1f}M "
              f"z={tuple(za.shape)} same-tok/diff-query cos={cos:+.4f}")
    za.sum().backward()
    print("backward OK")
