"""models/m5_perception_query.py — the bridge that lets the THINKER ask PERCEPTION
for detail, at runtime, about what BMO is looking at right now.

THE LOOP THIS CLOSES
    user:    "what does the room look like?"
    thinker: <tool_call name=look query="describe the room and setting in detail"/>
    -> PerceptionQueryEngine.ask() runs the trained QueryPredictor against the CURRENT
       perception tokens and returns a scene-grounded answer
    thinker: folds that answer into what it says out loud

WHY A TOOL, NOT A NEW PROTOCOL
`models/m5_tools.py` already parses `<tool_call name=... />` out of LLM output, executes a
handler, and folds the result back into the spoken line -- built, tested, and already
carrying time/date/weather/timer. Perception is just another tool. That means the thinker
needs NO new output format, no retraining to use it, and the existing `assets/tool_call.gbnf`
grammar keeps the syntax valid. Registering a handler is the whole integration.

WHY RETRIEVAL, NOT AN EMBEDDING PREFIX (for v1)
`scripts/prototype_llama_embd_input.py` proved llama.cpp accepts a raw embedding prefix, so
feeding z_q straight into the thinker is mechanically possible. It is deliberately NOT what
this does yet, for a measured reason: the thinker has never been trained to INTERPRET
JEPA-space vectors, and this project already paid for that lesson twice -- the M3 soft-prompt
connector was dropped for being slow and confidently wrong, and the M4b/Ultravox projector
plateaued at WER 0.94. Retrieval is the path that was measured to work on-device
(5ms predictor + 3ms nearest-neighbour, `scripts/jetson_embed_predictor_latency.py`).

**The honest limitation: the answer vocabulary IS the candidate bank.** The engine can only
say things the bank contains. That is why the bank is built from the full caption corpora
(188k VGGSound x 6 granularities + 400k Action100M) rather than a hand-written list --
breadth of bank is breadth of what BMO can report. `ask()` returns the retrieval score
alongside the text precisely so a caller can refuse to speak on a weak match.

GROUNDING SAFETY
`set_perception()` stamps a timestamp. `ask()` refuses to answer against perception older
than `max_age_s` and returns `None` instead -- BMO saying nothing is strictly better than
BMO describing a room it saw a minute ago as if it were live.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.query_predictor import QueryPredictor, QueryPredictorConfig


@dataclass
class PerceptionAnswer:
    text: str
    score: float                 # cosine of the predicted embedding to the retrieved caption
    query: str
    age_s: float                 # how old the perception it answered from was


class PerceptionQueryEngine:
    """Holds the trained QueryPredictor + a candidate bank, and answers questions
    about the CURRENT perception tokens."""

    def __init__(
        self,
        query_predictor: QueryPredictor,
        text_target,                       # models.text_target.TextTarget (trained proj loaded)
        bank_emb: Tensor,                  # (N, shared_dim) L2-normalised candidate embeddings
        bank_text: List[str],
        device: torch.device,
        max_age_s: float = 15.0,
        min_score: float = -1.0,           # refuse below this cosine. MUST default to -1.0,
                                           # not 0.0: cosine is legitimately negative, so a
                                           # 0.0 floor silently refuses valid weak-but-real
                                           # matches. -1.0 is the true "never refuse".
    ):
        assert bank_emb.shape[0] == len(bank_text), "bank embeddings/texts length mismatch"
        self.qp = query_predictor.eval()
        self.tt = text_target
        # `build_bank` already L2-normalises, so re-normalising here is redundant -- and
        # doing it via .float() materialises a 2x fp32 copy of the WHOLE bank at load time
        # (measured: a 355 MiB fp16 bank spiked ~710 MiB and OOMed the Jetson at exactly the
        # wrong moment). Check cheaply on a sample and only normalise if actually needed,
        # in chunks, never as one giant fp32 temporary.
        n_chk = min(256, bank_emb.shape[0])
        norms = bank_emb[:n_chk].float().norm(dim=-1)
        if torch.allclose(norms, torch.ones_like(norms), atol=1e-2):
            self.bank_emb = bank_emb.to(device)                  # already unit-norm
        else:
            out = torch.empty_like(bank_emb)
            for i in range(0, bank_emb.shape[0], 8192):          # bounded temporaries
                out[i:i + 8192] = F.normalize(bank_emb[i:i + 8192].float(), dim=-1).to(bank_emb.dtype)
            self.bank_emb = out.to(device)
        self.bank_text = bank_text
        self.device = device
        self.max_age_s = max_age_s
        self.min_score = min_score
        self._sources: Optional[Dict[str, Tensor]] = None
        self._masks: Optional[Dict[str, Tensor]] = None
        self._stamp: float = 0.0

    # ── write path: the streaming loop calls this every perception tick ──
    def set_perception(self, sources: Dict[str, Tensor],
                       masks: Optional[Dict[str, Tensor]] = None) -> None:
        """sources: {stream_name: (1, S, D)} exactly as build_sources() produces."""
        self._sources = {k: v.to(self.device) for k, v in sources.items()}
        self._masks = {k: v.to(self.device) for k, v in masks.items()} if masks else None
        self._stamp = time.time()

    @property
    def age_s(self) -> float:
        return float("inf") if self._stamp == 0.0 else time.time() - self._stamp

    @property
    def source_names(self) -> List[str]:
        return list(self.qp.cfg.source_dims)

    @torch.no_grad()
    def update_from_features(self, feats: Dict[str, Tensor], tbins: Dict[str, Tensor],
                             duplex_loop) -> None:
        """LIVE equivalent of train_query_predictor.build_sources(), for the streaming
        loop: takes the SAME `feats`/`tbins` that world_state_builder produced this tick
        and assembles exactly the token streams this checkpoint was trained on.

        The engine reads its own `source_dims`, so the caller never has to know which
        streams the checkpoint uses -- swapping in a differently-configured checkpoint
        needs no change at the call site.

        `m2` tokens come from `duplex_loop.compute_pre_pool()`, the SAME method M3 used,
        rather than a reimplementation -- this repo's item-0 bug was three independent
        reimplementations of feature construction that each diverged from training.

        **Measured cost note:** the loop already calls `compute_world_state()`, and
        `encode_pre_pool_tokens` shares everything with it except the final attentive
        pool, so requesting the `m2` stream costs ONE EXTRA M2 backbone pass (~160ms on
        Jetson, measured). That is paid on the vision-refresh thread, which is already
        off the response critical path. It is not deduplicated because deriving the
        world-state from the pre-pool tokens here would mean reimplementing the attentive
        pool -- exactly the class of divergence this docstring warns about. Configure a
        checkpoint without the `m2` stream if the cost is not wanted.

        No padding masks: live windows are single, full-length sequences with nothing to
        pad, so every token is real (unlike the batched training path)."""
        names = self.source_names
        src: Dict[str, Tensor] = {}
        if "m2" in names:
            src["m2"] = duplex_loop.compute_pre_pool(feats, tbins).float()
        if "vision" in names:
            src["vision"] = feats["vision"].float()
        if "ambient" in names:
            src["ambient"] = feats["ambient"].float()
        missing = [n for n in names if n not in src]
        if missing:
            raise KeyError(f"perception engine needs stream(s) {missing} that this tick "
                           f"did not provide (have: {sorted(feats)})")
        self.set_perception(src, None)

    # ── read path: the thinker calls this through the tool dispatcher ──
    @torch.no_grad()
    def ask(self, query: str, top_k: int = 1) -> Optional[PerceptionAnswer]:  # noqa: D401
        """Answer `query` from the current perception. Returns None when there is no
        fresh perception to answer from, or when the best match is too weak -- a refusal
        is a valid, and often correct, answer."""
        if self._sources is None:
            return None
        age = self.age_s
        if age > self.max_age_s:
            return None
        q_emb = self.tt.encode_text_frozen_raw([query]).to(self.device)   # native space, as trained
        z_q = self.qp(self._sources, q_emb, self._masks)                  # (1, shared_dim)
        sims = (z_q.to(self.bank_emb.dtype) @ self.bank_emb.T)[0].float()
        score, idx = torch.max(sims, dim=0)
        score = float(score)
        if score < self.min_score:
            return None
        return PerceptionAnswer(text=self.bank_text[int(idx)], score=score,
                                query=query, age_s=age)

    @torch.no_grad()
    def ask_topk(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Diagnostic: the k best candidates and their scores. Useful for judging whether
        a weak top-1 is a near-miss or genuinely off."""
        if self._sources is None:
            return []
        q_emb = self.tt.encode_text_frozen_raw([query]).to(self.device)
        z_q = self.qp(self._sources, q_emb, self._masks)
        sims = (z_q.to(self.bank_emb.dtype) @ self.bank_emb.T)[0].float()
        vals, idxs = torch.topk(sims, min(k, sims.shape[0]))
        return [(self.bank_text[int(i)], float(v)) for v, i in zip(vals, idxs)]

    # ── tool integration ──────────────────────────────────────────────────
    def as_tool_handler(self):
        """Handler for models/m5_tools.py::ToolRegistry.register('look', ...).

        The thinker emits e.g. <tool_call name=look query="describe the room in detail"/>.
        Accepts `query`/`q`/`about` as the parameter name so a small model's paraphrasing
        does not silently produce a no-op, and falls back to a general scene question when
        no parameter is given at all."""
        def handler(params: Dict[str, str]) -> str:
            q = (params.get("query") or params.get("q") or params.get("about")
                 or "Describe the scene, setting and context in detail.")
            ans = self.ask(q)
            if ans is not None:
                return ans.text
            # Report the ACTUAL reason -- an earlier version said "stale" for every
            # refusal, including fresh-but-weak matches, which would have had BMO
            # blaming its eyes for what was really low retrieval confidence.
            if self._sources is None:
                return "I can't see anything right now."
            if self.age_s > self.max_age_s:
                return f"my view is {self.age_s:.0f}s stale, so I'd rather not guess"
            return "I can see, but I'm not confident enough to describe it"
        return handler


# ── bank construction ─────────────────────────────────────────────────────
@torch.no_grad()
def build_bank(text_target, captions: List[str], device: torch.device,
               batch: int = 256, verbose: bool = True) -> Tensor:
    """Encode candidate captions with the SAME trained path training used for targets
    (`encode_text`, i.e. through the trained .proj), so retrieval geometry matches."""
    out = []
    for i in range(0, len(captions), batch):
        out.append(text_target.encode_text(captions[i:i + batch]).float().cpu())
        if verbose and (i // batch) % 20 == 0:
            print(f"[bank] {i}/{len(captions)}", flush=True)
    return F.normalize(torch.cat(out, 0), dim=-1)


def load_perception_query_engine(
    query_ckpt: str,
    text_target,
    device: torch.device,
    bank_emb_path: Optional[str] = None,
    bank_captions: Optional[List[str]] = None,
    max_age_s: float = 15.0,
    min_score: float = 0.0,
    bank_emb: Optional[Tensor] = None,
    bank_text: Optional[List[str]] = None,
) -> "PerceptionQueryEngine":
    """Build a ready-to-use engine from a trained checkpoint.

    Exists so a caller (e.g. scripts/bmo_jetson_startup.py's build_bmo_stack) does not
    have to duplicate checkpoint-loading logic -- including the easy-to-miss step of
    restoring the CO-TRAINED `text_target.proj`, without which retrieval geometry silently
    mismatches training and every answer degrades for no visible reason.

    Pass `bank_emb_path` to load a precomputed bank (a dict with `emb` + `text`), which is
    what deployment should do -- encoding tens of thousands of captions at boot would cost
    real seconds on the Jetson. `bank_captions` encodes on the fly instead, for tests."""
    ck = torch.load(query_ckpt, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768, "scene": 768}
    # Read the geometry OFF the checkpoint instead of hardcoding 1536. Two target spaces
    # now exist -- EmbeddingGemma's learned 1536-d and SigLIP2's frozen 768-d -- and a
    # silent mismatch here degrades every answer with no error, which has already happened
    # once in this project. head.weight is (shared_dim, d); q_proj.weight is (d, query_dim).
    sd = ck["query_predictor"]
    shared_dim = int(sd["head.weight"].shape[0])
    query_dim = int(sd["q_proj.weight"].shape[1])
    if query_dim != text_target.native_dim:
        raise ValueError(
            f"checkpoint expects query_dim={query_dim} but text_target.native_dim="
            f"{text_target.native_dim}. The query encoder must match the one trained "
            f"against (cfg.text_backbone={ck['cfg'].get('text_backbone')}).")
    qp = QueryPredictor(QueryPredictorConfig(
        source_dims={s: SRC[s] for s in names},
        query_dim=query_dim, shared_dim=shared_dim)).to(device)
    qp.load_state_dict(sd)
    qp.eval()
    # SigLIP2 targets have no trainable projection at all ({} state dict, Identity module),
    # which is the point: the target space stays the pretrained joint space, so banks are
    # checkpoint-independent. EmbeddingGemma targets DO carry a co-trained proj that must be
    # restored or retrieval geometry silently mismatches training.
    tp = ck.get("text_target_proj") or {}
    if tp:
        if isinstance(text_target.proj, torch.nn.Identity):
            # PreEncodedTextSpace path. This is CORRECT, not a fallback: at inference the
            # query side only ever calls encode_text_frozen_raw (raw space), and the
            # candidates must already have been projected OFFLINE -- that is precisely what
            # lets the device ship no text encoder. Refuse if the caller did not do it,
            # because retrieving raw candidates against a projected predictor output is
            # exactly the silent geometry mismatch this loader exists to prevent.
            if bank_emb is None:
                raise ValueError(
                    "checkpoint carries a trained text_target_proj but text_target has an "
                    "Identity proj and no pre-projected bank_emb was supplied. Build the "
                    "candidates with scripts/build_bank_siglip2.py --ckpt, or pass a "
                    "TextTarget/SigLIP2TextTarget that owns the projection.")
            print("[pq] proj applied offline to candidates; query side uses raw space "
                  "(no text encoder resident)", flush=True)
        else:
            text_target.proj.load_state_dict(tp)

    if bank_emb is not None and bank_text is not None:
        # already loaded AND already in the checkpoint's space -- used when the caller has
        # to apply the trained projection itself (candidates are stored RAW so that one
        # candidate file stays valid across retrains)
        emb, texts = bank_emb, bank_text
    elif bank_emb_path:
        b = torch.load(bank_emb_path, map_location="cpu", weights_only=False)
        emb, texts = b["emb"], b["text"]
    elif bank_captions:
        emb, texts = build_bank(text_target, bank_captions, device), bank_captions
    else:
        raise ValueError("need bank_emb/bank_text, bank_emb_path, or bank_captions")
    if emb.shape[1] != shared_dim:
        raise ValueError(f"candidate embeddings are {emb.shape[1]}-d but this checkpoint's "
                         f"space is {shared_dim}-d -- wrong bank, or the trained projection "
                         f"was not applied")
    return PerceptionQueryEngine(qp, text_target, emb, texts, device,
                                 max_age_s=max_age_s, min_score=min_score)


if __name__ == "__main__":
    # Mechanism-only smoke test (no checkpoints): random predictor + tiny fake bank.
    torch.manual_seed(0)
    dev = torch.device("cpu")

    class _FakeTT:
        native_dim = 768
        def encode_text_frozen_raw(self, xs):
            g = torch.Generator().manual_seed(abs(hash(xs[0])) % (2**31))
            return torch.randn(len(xs), 768, generator=g)

    cfg = QueryPredictorConfig(source_dims={"m2": 1024, "vision": 1024}, query_dim=768)
    qp = QueryPredictor(cfg)
    bank_text = ["a dog barking in a yard", "a person cooking pasta",
                 "a quiet room with a television"]
    bank = F.normalize(torch.randn(len(bank_text), 1536), dim=-1)
    eng = PerceptionQueryEngine(qp, _FakeTT(), bank, bank_text, dev, max_age_s=5.0)

    print("no perception yet ->", eng.ask("what do you see?"))
    eng.set_perception({"m2": torch.randn(1, 512, 1024), "vision": torch.randn(1, 512, 1024)})
    a = eng.ask("describe the room in detail")
    b = eng.ask("what do you hear?")
    print(f"query A -> {a.text!r} score={a.score:+.4f} age={a.age_s:.2f}s")
    print(f"query B -> {b.text!r} score={b.score:+.4f}")
    print("tool handler ->", eng.as_tool_handler()({"query": "what is happening?"}))
    eng._stamp = time.time() - 60          # simulate stale perception
    print("stale perception ->", eng.ask("what do you see?"),
          "| handler:", eng.as_tool_handler()({}))
