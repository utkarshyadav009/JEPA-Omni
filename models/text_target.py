"""
models/text_target.py

Y-Encoder: maps the text target into the shared embedding space.

VL-JEPA recipe (2512.10942): Y-Encoder = EmbeddingGemma-300M (native 768-dim,
Matryoshka), projected (with the predictor) into a shared 1,536-dim space; trained
with bi-directional InfoNCE. The paper's ablation shows a *frozen* text encoder with a
trainable linear projection on top is a valid configuration, so that is the M1 default
(simplest, no gated-model gradients). Set `unfreeze_base=True` to train it jointly with
a small LR multiplier (paper uses ~0.05) later.

Defaults to an UNGATED encoder so the smoke test runs with no HF license:
  - "minilm"          -> sentence-transformers/all-MiniLM-L6-v2 (384-dim, ungated)  [default]
  - "embeddinggemma"  -> google/embeddinggemma-300m (768-dim, GATED: accept license + HF token)

Native dim is read from config; never hardcoded.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


_PRESETS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "embeddinggemma": "google/embeddinggemma-300m",
}


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class TextTarget(nn.Module):
    def __init__(
        self,
        backbone: str = "minilm",
        shared_dim: int = 1536,
        max_length: int = 512,
        unfreeze_base: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        repo = _PRESETS.get(backbone, backbone)
        self.device_str = device
        self.dtype = dtype
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.base = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device)
        self.unfreeze_base = unfreeze_base
        self.base.train(unfreeze_base)
        for p in self.base.parameters():
            p.requires_grad_(unfreeze_base)

        native = int(self.base.config.hidden_size)
        # Trainable projection into the shared space (kept in fp32 for stability).
        self.proj = nn.Linear(native, shared_dim).to(device)
        self.shared_dim = shared_dim
        self.native_dim = native

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """texts -> (B, shared_dim), L2-normalized."""
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device_str)

        ctx = torch.enable_grad() if self.unfreeze_base else torch.no_grad()
        with ctx:
            out = self.base(**tok).last_hidden_state
            pooled = _mean_pool(out, tok["attention_mask"])

        z = self.proj(pooled.float())
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode_text_frozen_raw(self, texts: List[str]) -> torch.Tensor:
        """texts -> (B, native_dim), L2-normalized, PRE-proj (native EmbeddingGemma/
        MiniLM space, never passes through self.proj). Always no_grad regardless of
        unfreeze_base -- this is a deliberately-stable side-channel similarity
        signal, not part of the main InfoNCE forward pass.

        Added for Soft-InfoNCE's caption-similarity target (train_m2_embed_predictor.py):
        the reference implementation (github.com/Alex-HaochenLi/Soft-InfoNCE,
        "Rethinking Negative Pairs in Code Search", EMNLP 2023) deliberately computes
        soft weights from an EXTERNAL, frozen source (BM25 / a separately-trained
        SimCSE model / a fixed pretrained checkpoint) rather than the model being
        trained, to avoid circular dependency -- a model could otherwise learn to
        game its own soft targets instead of genuinely improving retrieval.
        encode_text()'s output isn't a clean fit for that: self.proj is one of this
        project's trainable modules, so those embeddings shift step-to-step as
        training progresses. self.base, however, IS already frozen by default
        (unfreeze_base=False) and is a real pretrained encoder (EmbeddingGemma-300M)
        with genuinely meaningful semantic structure from step 0 -- no periodic
        snapshotting or extra model needed, just skip self.proj and read the
        pre-projection pooled embedding directly."""
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device_str)
        out = self.base(**tok).last_hidden_state
        pooled = _mean_pool(out, tok["attention_mask"])
        return F.normalize(pooled.float(), dim=-1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)


class SigLIP2TextTarget(nn.Module):
    """Y-Encoder = **SigLIP2's own frozen text tower**, used as the target space directly.

    WHY THIS EXISTS. The EmbeddingGemma TextTarget above defines a target space that only
    the trained `self.proj` knows about. Three measured consequences pushed us off it:

      1. **EmbeddingGemma must be resident on the Jetson** (578 MiB) purely to encode the
         question and the bank -- on top of SigLIP2, which already has a text tower. The
         core-pipeline fit test OOM'd at 659 MiB free carrying both.
      2. **Every bank is coupled to one checkpoint.** `perception_bank_max_fp16.pt` was
         encoded with `query_predictor_ddp_lw0.3`'s proj and silently produced mismatched
         retrieval geometry for the ablation arms. That is a structural hazard of having a
         *trainable* target projection, not a one-off mistake.
      3. **The bank had to be corpus-shaped** (121k VGGSound/Action100M captions, 355 MiB),
         because a learned space is only meaningful near its training distribution -- the
         "live scenarios won't match the corpus" failure.

    SigLIP2 as the BASE fixes 1 outright and shrinks 3. What it does NOT fix is 2, and an
    earlier version of this class got that wrong -- see the correction below.

      * Candidates are **pre-encoded once offline** and only the vectors ship, so the
        538 MiB text tower never loads on-device (measured split: vision 92.9M/177 MiB vs
        text 282.3M/538 MiB, the text side 3x larger because of a 256k-token Gemma vocab
        embedding). **This is what buys the memory, and it is independent of whether
        `self.proj` is Identity or a Linear** -- a projection is a matmul that can be
        applied at pre-encode time.

    **CORRECTION (2026-08-15).** This docstring previously said `self.proj` must stay
    `nn.Identity` and warned against "improving" it into a Linear, on the theory that
    targets had to remain in the image tower's space. That was trained head-to-head and
    **falsified**: with everything else identical, Identity lost to EmbeddingGemma on
    VGGSound within-clip 0.654 vs 0.883 and R@1 0.489 vs 0.681, with the within-clip curve
    FLAT from step 249. The projection is not overhead, it is where the representation
    learning happens (measured: it moves caption geometry from within/cross cos 0.7306 /
    0.6075 to 0.4340 / 0.1533). SigLIP2's raw space was never the problem -- it is no worse
    than EmbeddingGemma's raw space, 0.7556 vs 0.7306. Pass `shared_dim` to restore the
    projection; `shared_dim=None` keeps the falsified frozen configuration and exists only
    to reproduce that result.

    The real cost of the projection is hazard 2 returning: banks become checkpoint-coupled
    and must be rebuilt on retrain. That is a build-time chore, not a memory cost. Pure
    zero-shot scoring of novel tag phrases against the IMAGE tower still works -- but it
    must read `encode_text_frozen_raw` (pre-proj), not `encode_text`.

    PADDING IS NOT OPTIONAL. SigLIP/SigLIP2 were trained with `padding="max_length"` at a
    fixed 64 tokens; using dynamic padding silently shifts the embeddings. `max_length=64`
    also means long narrated captions (this corpus averages 132 chars) sit at the edge of
    what the tower represents well -- an independent argument for short, tag-like candidates.

    `encode_text_frozen_raw` is always the raw pretrained space (the QUERY side, so
    `query_dim == native_dim`); `encode_text` is the TARGET and passes through `self.proj`.
    """

    def __init__(self, repo: str = "google/siglip2-base-patch16-224",
                 max_length: int = 64, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 shared_dim: Optional[int] = None) -> None:
        super().__init__()
        from transformers import AutoProcessor
        self.device_str = device
        self.dtype = dtype
        self.max_length = max_length
        self.repo = repo

        base = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.processor = AutoProcessor.from_pretrained(repo)

        self.native_dim = int(base.config.text_config.hidden_size)
        self.unfreeze_base = False

        # MEASURED 2026-08-15 -- read this before setting shared_dim=None.
        # `shared_dim=None` gives proj = Identity, i.e. the pure frozen joint space. That
        # configuration was trained head-to-head against EmbeddingGemma with everything else
        # identical (run sig_runA_matched3stream vs query_predictor_ddp_lw0.3) and LOST
        # badly: VGGSound within-clip 0.654 vs 0.883, R@1 0.489 vs 0.681. The within-clip
        # curve was FLAT from step 249 -- a ceiling, not slow convergence.
        # Root cause, measured on 400 held-out clips' 6 caption fields:
        #     EmbeddingGemma raw          within-clip cos 0.7306 | cross-clip 0.6075
        #     EmbeddingGemma+trained proj within-clip cos 0.4340 | cross-clip 0.1533
        # The trainable projection is doing the representation learning -- it spreads a
        # space crammed between 0.61-0.73 out to 0.15-0.43. SigLIP2's RAW space is no worse
        # than EmbeddingGemma's raw space (0.7556 vs 0.7306); what was missing was the proj.
        # Setting shared_dim restores it and still keeps BOTH memory wins, because the
        # projection is a matmul that can be applied OFFLINE when pre-encoding: no text
        # tower and no EmbeddingGemma need ship. The only thing given up is that the bank
        # becomes checkpoint-coupled again (rebuild on retrain -- cheap).
        if shared_dim is None:
            self.shared_dim = self.native_dim
            self.proj = nn.Identity()
        else:
            self.shared_dim = int(shared_dim)
            self.proj = nn.Linear(self.native_dim, self.shared_dim).to(device)

        # Optional pre-encoded lookup: text -> row of self._cache_vecs. Lets training skip
        # the text tower entirely, since it is frozen and the captions repeat every epoch.
        self._cache_idx: dict = {}
        self._cache_vecs: torch.Tensor | None = None

    def load_cache(self, texts: List[str], vecs: torch.Tensor) -> None:
        """Attach pre-encoded caption vectors (see scripts/encode_captions_siglip2.py)."""
        assert vecs.shape[0] == len(texts), f"{vecs.shape[0]} vecs vs {len(texts)} texts"
        self._cache_idx = {t: i for i, t in enumerate(texts)}
        self._cache_vecs = F.normalize(vecs.float(), dim=-1).to(self.device_str)

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tok = self.processor(text=texts, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device_str)
        out = self.base.get_text_features(**tok)
        out = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(out.float(), dim=-1)

    @torch.no_grad()
    def _raw(self, texts: List[str], bs: int = 512) -> torch.Tensor:
        """texts -> (B, native_dim) in SigLIP2's frozen joint space, L2-normalized.

        Serves cache hits from the pre-encoded table and only runs the tower on misses, so
        a fully-cached training run never touches the 282M-param text tower. The cache
        stores RAW (pre-proj) vectors, which is what keeps one cache valid across every
        `shared_dim` choice and every retrain."""
        if self._cache_vecs is not None:
            rows = [self._cache_idx.get(t, -1) for t in texts]
            if all(r >= 0 for r in rows):
                return self._cache_vecs[torch.as_tensor(rows, device=self.device_str)]
            miss = [i for i, r in enumerate(rows) if r < 0]
            out = torch.empty(len(texts), self.native_dim, device=self.device_str)
            hit = [i for i, r in enumerate(rows) if r >= 0]
            if hit:
                out[hit] = self._cache_vecs[torch.as_tensor([rows[i] for i in hit],
                                                            device=self.device_str)]
            out[miss] = self._encode([texts[i] for i in miss])
            return out
        if len(texts) <= bs:
            return self._encode(texts)
        return torch.cat([self._encode(texts[i:i + bs]) for i in range(0, len(texts), bs)], 0)

    def encode_text(self, texts: List[str], bs: int = 512) -> torch.Tensor:
        """The TARGET. Raw SigLIP2 space, then the (optionally trainable) projection.

        Mirrors TextTarget.encode_text exactly: the base is always frozen and runs under
        no_grad, while `self.proj` carries gradient when it is a Linear."""
        z = self._raw(texts, bs=bs)
        return F.normalize(self.proj(z.float()), dim=-1)

    @torch.no_grad()
    def encode_text_frozen_raw(self, texts: List[str]) -> torch.Tensor:
        """The QUERY side: raw pretrained space, never through self.proj -- same contract
        as TextTarget.encode_text_frozen_raw, so query_dim == native_dim either way."""
        return self._raw(texts)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)


class PreEncodedTextSpace:
    """A text ENCODER stand-in that holds only pre-encoded vectors -- no model, no weights.

    This is what makes the Jetson text-encoder-free. Once the target space is frozen
    (SigLIP2), both things the engine needs to encode at runtime are drawn from small fixed
    sets: the QUESTION (a handful of phrasings per field) and the CANDIDATES (bank or tags).
    Encoding them offline removes SigLIP2's 538 MiB text tower AND EmbeddingGemma's 578 MiB
    from the device entirely.

    MISS BEHAVIOUR IS THE HONEST PART. The thinker emits free text, so an exact-match table
    will miss. On a miss this falls back to the nearest known phrasing by word overlap -- a
    cheap, model-free router that is defensible ONLY because the query predictor was trained
    on a closed set of ~6 field intents, so a question outside those phrasings was never a
    supported capability regardless of how it is encoded. `last_fallbacks` records every
    miss so a caller can see when routing is guessing instead of matching, and
    `strict=True` refuses instead of guessing.

    The cleaner long-term fix is to have the thinker emit the FIELD, not a sentence -- the
    tool-call grammar (`assets/tool_call.gbnf`) can constrain that exactly, making the
    lookup total and the fallback dead code.
    """

    def __init__(self, texts: List[str], vecs: torch.Tensor, device: str = "cuda",
                 strict: bool = False) -> None:
        assert vecs.shape[0] == len(texts), f"{vecs.shape[0]} vecs vs {len(texts)} texts"
        self.texts = list(texts)
        self.vecs = F.normalize(vecs.float(), dim=-1).to(device)
        self.native_dim = int(self.vecs.shape[1])
        self.shared_dim = self.native_dim
        self.device_str = device
        self.strict = strict
        self.proj = nn.Identity()
        self.unfreeze_base = False
        self._idx = {t.strip().lower(): i for i, t in enumerate(self.texts)}
        self._toks = [set(t.lower().split()) for t in self.texts]
        self.last_fallbacks: List[tuple] = []

    def _row(self, q: str) -> int:
        k = q.strip().lower()
        if k in self._idx:
            return self._idx[k]
        if self.strict:
            raise KeyError(f"no pre-encoded vector for {q!r} (strict mode)")
        qt = set(k.split())
        best, bi = -1.0, 0
        for i, tt in enumerate(self._toks):
            u = len(qt | tt)
            j = (len(qt & tt) / u) if u else 0.0
            if j > best:
                best, bi = j, i
        self.last_fallbacks.append((q, self.texts[bi], round(best, 3)))
        return bi

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        rows = [self._row(t) for t in texts]
        return self.vecs[torch.as_tensor(rows, device=self.device_str)]

    def encode_text_frozen_raw(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)

    def __call__(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)


def build_text_target(backbone: str, shared_dim: int = 1536, device: str = "cuda",
                      siglip_shared_dim: Optional[int] = None, **kw):
    """Factory so trainers can switch target space from one flag.

    `siglip_shared_dim=None` reproduces the pure-frozen configuration that MEASURABLY LOST
    to EmbeddingGemma (see SigLIP2TextTarget's docstring); pass a dim to restore the
    trainable projection that was doing the work."""
    if backbone.startswith("siglip"):
        repo = backbone if "/" in backbone else "google/siglip2-base-patch16-224"
        return SigLIP2TextTarget(repo=repo, device=device, shared_dim=siglip_shared_dim)
    return TextTarget(backbone=backbone, shared_dim=shared_dim, device=device, **kw)


if __name__ == "__main__":
    yt = TextTarget(backbone="minilm")
    z = yt.encode_text(["a dog catching a frisbee", "glass shattering on the floor"])
    print(f"[text_target] z={tuple(z.shape)} native={yt.native_dim} shared={yt.shared_dim} dtype={z.dtype}")
