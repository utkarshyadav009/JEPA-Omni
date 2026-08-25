"""models/m4_cognitive_core.py — two-tier cognitive core for the duplex loop.

Design (agreed 2026-08-04, following MiniCPM-o 4.5 / DuplexOmni research):
  Fast tier   -- small LLM, plain-text prompted (ASR transcript + a
                 homeostatic-state prefix + short history), runs
                 SYNCHRONOUSLY in the tick loop. Handles short, bounded
                 exchanges (greetings, state check-ins, backchannel-
                 adjacent replies). Cannot reason about anything outside
                 what it was tuned on.
  Thinker     -- the EXISTING DuplexLoop.generate_interruptible path
                 (M3/M4b soft-prompt + frozen LLM), now run ASYNCHRONOUSLY
                 off the tick loop via AsyncThinker, so a slow deliberate
                 generation never blocks the fast tier's ability to keep
                 responding. Re-integration is explicit (poll each tick),
                 not forced -- mirrors DuplexOmni's [WAIT]-gated pattern:
                 nothing is discarded, the Thinker's result is picked up
                 whenever it's ready, not synchronously awaited.
  Routing     -- always try the fast tier first. The escalation signal to
                 the Thinker comes from INSIDE the fast tier's own
                 generation (mean per-token negative log-likelihood of its
                 own greedy output = a real confidence proxy), not a
                 separate trained classifier -- matching what the MiniCPM-o
                 4.5 (LS control-token-first) and DuplexOmni ([THINK]
                 trigger token) research found: the "whether" decision is
                 architecturally separated from "what to say", but neither
                 published system uses a bolted-on gate model for it.

KNOWN GAP, not papered over: homeostatic-state variables (energy/mood/
etc.) do not exist anywhere else in this codebase yet -- FastTier.generate
accepts an open `state: Dict[str, float]` so it's ready to consume real
values once the state system is defined, but nothing here invents what
those variables mean. Passing state={} is valid and just omits the prefix.

REQUIRED LOAD ORDER ON JETSON, real and verified 2026-08-07 -- not
optional: construct GGUFFastTier/GGUFReasoningTier and any other
llama.cpp-backed model (including TTS) BEFORE loading the torch/int8
perception stack (VisionEncoder, AudioEncoder/WavJEPA, AVJepaPredictor,
Whisper). Real, measured finding: perception-stack-first left the largest
contiguous free GPU memory block on Jetson's unified 7.6GB memory at
single-digit MiB by the time llama.cpp tried to allocate its own models --
reliable "Failed to create llama_context" / NvMap ENOMEM failures.
Reversing the order (llama.cpp models first, perception last) let the
IDENTICAL full stack (perception + STT projector + both LLM tiers + TTS,
real generation through all of them) load and run with 1003MiB headroom,
zero errors, verified via scripts/jetson_full_stack_v4_reversed_order.py.
Root cause: PyTorch's caching allocator and llama.cpp's own allocator
compete for the same physical unified memory without coordinating: torch's
many smaller incremental allocations (especially during int8 quantization,
which round-trips tensors CPU<->GPU) fragment the address space in a way
llama.cpp's later large contiguous allocations can't route around; loading
the large contiguous blocks FIRST, while the space is still clean, avoids
the problem entirely. This is a real operational constraint, not a one-off
bug -- any future change to the Jetson startup sequence must preserve
"llama.cpp models before torch perception stack" or re-verify against the
same test.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch


def _state_prefix(state: Dict[str, float]) -> str:
    """Renders an open-ended homeostatic-state dict as a literal
    conditioning prefix, e.g. {"energy": 0.7} -> "[energy=0.70]". Order is
    preserved from the dict (caller's responsibility to pass a stable
    ordering if consistency across turns matters for the fine-tuned
    model's learned associations)."""
    if not state:
        return ""
    parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in state.items()]
    return "[" + " ".join(parts) + "] "


@dataclass
class FastTierResult:
    text: str
    n_tokens: int
    mean_neg_logprob: float     # confidence proxy: LOWER = more confident
    elapsed_ms: float
    per_token_ms: float
    mood: Optional[str] = None  # homeostatic mood at generation time; the live
                                # loop passes it to StreamingVoice.speak() as the
                                # <|MOOD|> voice control token (one state -> words
                                # AND voice delivery). Default None = neutral.
    reasoning: Optional[str] = None
    # The `<think>...</think>` block, KEPT rather than discarded. Added 2026-08-16
    # after a real on-device finding: GGUFReasoningTier stripped the CoT and returned
    # only the answer, so the thinker paid the full autoregressive cost of
    # deliberating and then threw the deliberation away. The speaker downstream only
    # ever saw the thinker's ANSWER -- itself already a finished BMO line -- which is
    # why the speaker visibly echoed the thinker instead of acting on its reasoning.
    # The answer is still what gets spoken; this field is what the speaker should be
    # CONDITIONED on. None means the model emitted no think block.


class FastTier:
    """Small-LLM synchronous responder. Model-agnostic (works with
    MiniCPM5-1B, Gemma-3-270M-it, or Qwen2.5-1.5B -- whichever real
    benchmark ends up winning); this class only depends on the standard
    HF chat-template + generate() contract, no model-specific code."""

    def __init__(self, model, tokenizer, device: torch.device, max_new_tokens: int = 24):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _build_prompt_ids(self, transcript: str, state: Dict[str, float],
                           history: Optional[List[Dict[str, str]]] = None) -> torch.Tensor:
        prefix = _state_prefix(state)
        msgs = list(history or [])
        msgs.append({"role": "user", "content": prefix + transcript})
        # enable_thinking=False: same finding as scripts/jetson_minicpm5_latency.py
        # (2026-08-01) -- default thinking mode burns the token budget on a
        # <think>...</think> preamble and never reaches an actual answer within
        # a short max_new_tokens budget. Missing this the first time here made
        # every NLL number meaningless (confident preamble text, not content).
        out = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                   enable_thinking=False, return_tensors="pt")
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return ids.to(self.device)

    @torch.no_grad()
    def generate(self, transcript: str, state: Dict[str, float],
                 history: Optional[List[Dict[str, str]]] = None) -> FastTierResult:
        """Greedy decode with per-token logprob tracking -- the confidence
        signal routing depends on comes directly out of this generation,
        not a second forward pass or separate model."""
        ids = self._build_prompt_ids(transcript, state, history)
        prompt_len = ids.shape[1]

        t0 = time.perf_counter()
        out = self.model.generate(
            ids, max_new_tokens=self.max_new_tokens, do_sample=False,
            repetition_penalty=1.15, pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        seq = out.sequences[0, prompt_len:]
        n_new = seq.shape[0]
        # mean NLL of the tokens the model actually chose (greedy argmax),
        # i.e. how "sure" it was of its own output -- real confidence proxy,
        # not a heuristic string-match.
        neg_logprobs = []
        for step_scores, tok_id in zip(out.scores, seq):
            logp = torch.log_softmax(step_scores[0].float(), dim=-1)
            neg_logprobs.append(-logp[tok_id].item())
        mean_nll = sum(neg_logprobs) / max(1, len(neg_logprobs))

        text = self.tokenizer.decode(seq, skip_special_tokens=True)
        return FastTierResult(
            text=text, n_tokens=n_new, mean_neg_logprob=mean_nll,
            elapsed_ms=elapsed_ms, per_token_ms=elapsed_ms / max(1, n_new),
            mood=state.get("mood") if isinstance(state, dict) else None,
        )


class GGUFFastTier:
    """Drop-in FastTier replacement backed by llama.cpp/GGUF. Real, measured
    win on Jetson: 23.08ms/token vs transformers' 119.3ms/token on the exact
    same BMO-fine-tuned checkpoint (5.2x faster) -- see
    scripts/jetson_embed_predictor_latency.py-style benchmark run and
    checkpoints/falsifier_tracking.md-adjacent notes. Returns the same
    FastTierResult dataclass, so CognitiveCoreRouter and BmoDuplexTick work
    completely unchanged -- this is a backend swap, not an interface change.

    Uses the HF tokenizer ONLY for apply_chat_template (cheap, CPU-only, no
    generation cost) so prompt formatting stays byte-identical to the
    transformers-backed FastTier; llama.cpp handles actual generation +
    per-token logprobs for the confidence-routing signal. `logits_all=True`
    must be passed at construction, or llama.cpp raises "logprobs is not
    supported for models created with logits_all=False" -- not documented
    prominently, found by hitting it.

    Real bug found and fixed (2026-08-06): the obvious way to get logprobs
    -- the high-level `Llama.__call__(..., logprobs=1)` completion API --
    was measured at ~165-174ms/token on Jetson Orin and ~24ms/token on a
    Blackwell desktop GPU, both a genuine ~13-17x tax vs. plain generation
    with no logprobs (~13ms/token Jetson, ~1.4ms/token Blackwell). Verified
    NOT a CUDA/model-architecture cost (decode batches are n_tokens=1
    either way, and a controlled test forcing real 30/100/250-token
    generations showed a FLAT per-token cost in both configurations, ruling
    out a one-time prefill tax). Root cause, found by reading
    llama_cpp/llama.py directly: `_create_completion` calls
    `sorted(zip(logprobs, range(vocab_size)), reverse=True)` -- a pure
    -Python sort of the ENTIRE vocabulary (65536 tokens for LFM2) on EVERY
    generated token, only to build a `top_logprobs` field this class never
    reads (only `token_logprobs`, the sampled token's own probability, is
    used). Timed the sort alone: ~14ms/call, matching most of the gap.
    Fix: bypass the high-level completion API entirely -- use the low
    -level `Llama.generate()` token iterator plus direct `self.llm.scores`
    access and `Llama.logits_to_logprobs()` (vectorized numpy, no sort) to
    compute just the one probability needed per token. Measured after the
    fix: ~1.5-3ms/token, matching the no-logprobs baseline -- same exact
    confidence signal, ~16x faster.

    n_ctx default lowered from 2048 to 512 (2026-08-07): hit a real
    "Failed to create llama_context" / NvMap ENOMEM error on Jetson when
    loading this tier alongside GGUFReasoningTier, even with ~4.9GB
    reported "available" memory -- tegrastats showed the largest
    contiguous free block was only ~84MB (severe fragmentation on Jetson's
    unified memory, not a real total-memory shortage). Since BMO's actual
    exchanges are short single-turn Q&A, 512 tokens of KV-cache headroom is
    plenty and reliably fits within the fragmented free space where 2048
    intermittently did not."""

    def __init__(self, gguf_path: str, tokenizer, max_new_tokens: int = 24,
                 n_gpu_layers: int = -1, n_ctx: int = 512):
        from llama_cpp import Llama
        # flash_attn=True: MEASURED -228 MiB on this device (253.0 -> 25.0 MiB for the same
        # model, prompt and n_ctx; jetson_kv_logits_probe.py + nctx sweep). The boot log
        # named the cost itself: "the V embeddings have different sizes across layers and FA
        # is not enabled - padding V cache to 512".
        #
        # logits_all STAYS True. Turning it off looks like a bigger win (253 -> 4 MiB) and
        # SILENTLY BREAKS confidence routing: mean_neg_logprob pins to 11.0904 in every
        # configuration, which is exactly ln(65536) = ln(vocab), i.e. a uniform distribution
        # over an all-zero buffer. llama-cpp-python 0.3.34 does not populate `scores` in the
        # generate() path without it, and every accessor was tried (scores[n_tokens-1],
        # scores[0], scores[-1], eval_logits[-1] all return zeros; _ctx.get_logits_ith(-1)
        # raises a ctypes buffer-format error). A broken router is not a saving.
        self.llm = Llama(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                          logits_all=True, flash_attn=True, verbose=False)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._eos = self.llm.token_eos()

    def _build_prompt_text(self, transcript: str, state: Dict[str, float],
                            history: Optional[List[Dict[str, str]]] = None) -> str:
        prefix = _state_prefix(state)
        msgs = list(history or [])
        msgs.append({"role": "user", "content": prefix + transcript})
        return self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    def generate(self, transcript: str, state: Dict[str, float],
                 history: Optional[List[Dict[str, str]]] = None) -> FastTierResult:
        from llama_cpp import Llama
        prompt = self._build_prompt_text(transcript, state, history)
        # special=True is required so chat-template control tokens
        # (<|im_start|>, <|im_end|>, etc.) are parsed as actual special
        # tokens rather than literal sub-word text -- the default
        # special=False silently garbled every generation (echoed template
        # tokens back as output) until caught by inspecting real output.
        # add_bos=False: apply_chat_template's rendered string already
        # starts with the BOS marker, so add_bos=True double-counts it.
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

        t0 = time.perf_counter()
        gen_tokens: list[int] = []
        neg_logprobs: list[float] = []
        for token in self.llm.generate(tokens, temp=0.0, repeat_penalty=1.15):
            if token == self._eos:
                break
            logits = self.llm.scores[self.llm.n_tokens - 1, :]
            lp = Llama.logits_to_logprobs(logits)[token]
            neg_logprobs.append(-float(lp))
            gen_tokens.append(token)
            if len(gen_tokens) >= self.max_new_tokens:
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        text = self.llm.detokenize(gen_tokens).decode("utf-8", errors="ignore").strip()
        n_new = len(gen_tokens)
        mean_nll = sum(neg_logprobs) / max(1, n_new)

        return FastTierResult(
            text=text, n_tokens=n_new, mean_neg_logprob=mean_nll,
            elapsed_ms=elapsed_ms, per_token_ms=elapsed_ms / max(1, n_new),
            mood=state.get("mood") if isinstance(state, dict) else None,
        )


class GGUFReasoningTier(GGUFFastTier):
    """Second tier, same BMO voice as the fast tier but a bigger model
    (MiniCPM5-1B vs LFM2-700M) and a longer token budget -- used only on
    escalation, not every tick. Design per user direction 2026-08-06: keep
    LFM2 as the always-on fast tier (confirmed fast after the logits_all
    fix), add MiniCPM5 as a reasoning tier for cases the fast tier itself
    flags as low-confidence, rather than replacing one model with the
    other. Distinct from AsyncThinker/the M3/M4b Thinker path -- this tier
    has no JEPA world-state/vision grounding, it's purely a bigger
    same-character LLM; the M3/M4b Thinker remains the path for anything
    that needs real environmental grounding. Identical generate() contract
    to GGUFFastTier (same FastTierResult, same fixed low-level logprob
    path) -- only the constructor defaults differ (more tokens, since the
    point of this tier is a more complete answer, not instant response)."""

    def __init__(self, gguf_path: str, tokenizer, max_new_tokens: int = 60,
                 n_gpu_layers: int = -1, n_ctx: int = 512):
        super().__init__(gguf_path, tokenizer, max_new_tokens=max_new_tokens,
                          n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

    def _build_prompt_text(self, transcript, state, history=None):
        """enable_thinking=**True**. THE THINKER WAS NEVER THINKING.

        Measured on-device 2026-08-16: across 6 live rounds the thinker emitted no
        `<think>` block at ALL -- every `.reasoning` came back None. Root cause is one
        inherited keyword. `GGUFFastTier._build_prompt_text` passes
        `enable_thinking=False`, which is correct there (the fast tier must answer
        instantly, and a CoT preamble burns its whole token budget -- see the note at
        line 109). GGUFReasoningTier inherited that method untouched, and Qwen3's chat
        template implements `enable_thinking=False` by writing an EMPTY
        `<think>\\n\\n</think>` pair into the prompt, which the model then treats as its
        already-completed deliberation and skips straight to answering.

        So the entire reason this tier exists -- native Qwen3 CoT -- was disabled by a
        default, the 320-token budget provisioned for `CoT + answer` was spent on answer
        alone, and every escalation was just a second conversational model. That is also
        why the thinker's output looked like a spoken line rather than deliberation.

        Only this subclass flips it; the fast tier keeps enable_thinking=False."""
        prefix = _state_prefix(state)
        msgs = list(history or [])
        msgs.append({"role": "user", "content": prefix + transcript})
        return self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=True, tokenize=False)

    def generate(self, transcript, state, history=None):
        """Qwen3-0.6B thinker (2026-08-08): the model emits
        `<think>reasoning</think>answer` (+ optional <tool_call ...>). Only the
        ANSWER is spoken, so strip the internal chain-of-thought before it
        leaves this tier. Needs a generous max_new_tokens (~300) so the CoT +
        answer fit -- the old 60-token budget would cut off mid-reasoning.

        CHANGED 2026-08-16: the CoT is still stripped from `.text` (it must never be
        spoken) but is now RETAINED on `.reasoning`. Stripping and discarding it was
        a measured defect: the thinker spent its whole token budget deliberating, the
        deliberation was deleted, and the only thing reaching the speaker was an
        answer that was already a complete BMO utterance. Two models then produced the
        same kind of sentence and the speaker looked like it was echoing -- because it
        was. Downstream consumers should condition on `.reasoning`."""
        import re
        r = super().generate(transcript, state, history)
        m = re.search(r"<think>(.*?)</think>", r.text, flags=re.DOTALL)
        r.reasoning = m.group(1).strip() if m else None
        t = re.sub(r"<think>.*?</think>\s*", "", r.text, flags=re.DOTALL).strip()
        if "<think>" in t:  # unclosed reasoning ran past the token budget
            head, _, tail = t.partition("<think>")
            r.reasoning = (tail.strip() or r.reasoning)   # truncated CoT is still signal
            t = head.strip()
        r.text = t or r.text
        return r


class LazyGGUFReasoningTier:
    """Same generate() contract as GGUFReasoningTier, but does NOT keep the
    underlying llama.cpp model resident in memory -- loads it fresh on each
    call and frees it immediately after. Real finding, 2026-08-07: on
    Jetson, running the full perception stack (vision+WavJEPA+M2) + STT
    projector + the always-on fast tier ALREADY collapses the largest
    contiguous free GPU memory block to single-digit MiB (measured: 2-8MiB)
    -- not enough headroom to also keep a ~1.1GB reasoning-tier model
    permanently resident. Since escalation is the exception path (not every
    tick), lazy-loading trades a real, measured load latency (~1-2s on
    Jetson, GGUF weights read from local disk) for not needing that memory
    the other 80-90% of the time. Verified deletion actually releases GPU
    memory back (not just Python-side): `del llm; gc.collect()` dropped
    tegrastats usage from 2335MiB to 1997MiB in a real Jetson test (close
    to, though not exactly, the 1870MiB pre-load baseline -- some residual
    allocator overhead is normal and expected).

    CORRECTION, 2026-08-07, same day: using this class instead of eager
    GGUFReasoningTier in the real full-stack startup (scripts/
    bmo_jetson_startup.py) made things WORSE, not better -- perception
    loading crashed reproducibly (NVML/CUDACachingAllocator assertion at
    the second WavJEPA encoder) with lazy-loading, but succeeded reliably
    with EAGER loading of all three llama.cpp models (fast tier, reasoning
    tier, TTS) before perception. Best-guess explanation: llama.cpp
    claiming one large contiguous block upfront leaves the remaining
    address space in a cleaner state for torch's later many-small
    -allocations quantization pattern than leaving that space nominally
    "free" but unclaimed. The REAL fix for the full-stack OOM turned out to
    be restoring `malloc_trim()` calls in the perception stack's int8
    quantization helper (CPU-side allocator fragmentation matters on
    Jetson's unified memory, where CPU and GPU share the same physical
    RAM) -- once that was in place, eager-loading everything worked with
    real headroom to spare. This class is kept as a documented option (the
    memory-release mechanism it relies on is real and verified), but is
    NOT used in the default startup path -- do not switch to it there
    without re-running the full-stack test."""

    def __init__(self, gguf_path: str, tokenizer, max_new_tokens: int = 60,
                 n_gpu_layers: int = -1, n_ctx: int = 512):
        self.gguf_path = gguf_path
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx

    def generate(self, transcript: str, state: Dict[str, float],
                 history: Optional[List[Dict[str, str]]] = None) -> FastTierResult:
        import gc
        tier = GGUFReasoningTier(self.gguf_path, self.tokenizer, self.max_new_tokens,
                                  self.n_gpu_layers, self.n_ctx)
        try:
            return tier.generate(transcript, state, history)
        finally:
            del tier.llm
            del tier
            gc.collect()


@dataclass
class ThinkerResult:
    text: str
    ready: bool
    started_at: float
    finished_at: Optional[float] = None


class AsyncThinker:
    """Wraps the EXISTING DuplexLoop.generate_interruptible path (M3/M4b
    soft-prompt + frozen LLM) to run off the tick loop. Mirrors
    DuplexOmni's async-reasoning + explicit-re-integration pattern: start()
    kicks off a background thread immediately and returns; poll() is what
    the tick loop calls each cycle to check readiness (non-blocking) --
    there is no code path where the tick loop blocks waiting on this."""

    def __init__(self, duplex_loop) -> None:
        self.duplex_loop = duplex_loop
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[ThinkerResult] = None
        self._lock = threading.Lock()

    def start(self, soft_prompt: torch.Tensor, attention_mask: torch.Tensor,
              max_new_tokens: int = 60) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("AsyncThinker already running -- poll()/is_running() before starting a new task")
        started_at = time.perf_counter()
        with self._lock:
            self._result = ThinkerResult(text="", ready=False, started_at=started_at)

        def _run():
            gen = self.duplex_loop.generate_interruptible(
                soft_prompt, attention_mask, max_new_tokens=max_new_tokens)
            with self._lock:
                self._result = ThinkerResult(
                    text=gen.text, ready=True,
                    started_at=started_at, finished_at=time.perf_counter(),
                )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def poll(self) -> Optional[ThinkerResult]:
        """Non-blocking. Returns the ThinkerResult (ready=True) once done,
        None if never started, or a ready=False placeholder while running."""
        with self._lock:
            return self._result


# Bounded-exchange heuristic: cheap, explicit patterns that should NEVER
# need the Thinker regardless of confidence score (greetings, thanks,
# goodbyes) -- a floor under the confidence signal, not a replacement for
# it. Per the design note: routing signal comes from generation itself
# first; this only short-circuits the unambiguous cases.
_BOUNDED_PATTERNS = (
    "hello", "hi ", "hey", "how are you", "good morning", "good night",
    "thank", "thanks", "bye", "goodbye", "what's up", "whats up",
)


def looks_bounded(transcript: str) -> bool:
    t = transcript.lower().strip()
    return any(p in t for p in _BOUNDED_PATTERNS)


@dataclass
class RoutingDecision:
    use_fast_tier_response: bool
    escalate_to_thinker: bool
    reason: str
    fast_result: FastTierResult
    reasoning_result: Optional[FastTierResult] = None


class CognitiveCoreRouter:
    """Ties FastTier + AsyncThinker together per the agreed policy:
    always run the fast tier first (cheap, robot never goes silent);
    escalate to the Thinker based on the fast tier's OWN confidence
    (mean_neg_logprob) unless the bounded-pattern heuristic already
    settles it. No separate trained gate model.

    Optional `reasoning_tier` (GGUFReasoningTier, added 2026-08-06): when
    provided, escalation calls this SYNCHRONOUSLY and returns its result in
    `reasoning_result` -- viable now that the logits_all fix makes even the
    bigger MiniCPM5 tier fast (tens of ms/token, not hundreds). Without it,
    escalation still just sets escalate_to_thinker=True as before, for
    callers still using the async M3/M4b Thinker path unchanged."""

    def __init__(self, fast_tier: FastTier, nll_escalate_threshold: float = 3.0,
                 reasoning_tier: Optional["GGUFReasoningTier"] = None) -> None:
        self.fast_tier = fast_tier
        self.nll_escalate_threshold = nll_escalate_threshold
        self.reasoning_tier = reasoning_tier

    def route(self, transcript: str, state: Dict[str, float],
              history: Optional[List[Dict[str, str]]] = None) -> RoutingDecision:
        fast_result = self.fast_tier.generate(transcript, state, history)

        if looks_bounded(transcript):
            return RoutingDecision(
                use_fast_tier_response=True, escalate_to_thinker=False,
                reason=f"bounded-pattern match, fast-tier NLL={fast_result.mean_neg_logprob:.2f}",
                fast_result=fast_result,
            )

        escalate = fast_result.mean_neg_logprob > self.nll_escalate_threshold
        reasoning_result = None
        if escalate and self.reasoning_tier is not None:
            reasoning_result = self.reasoning_tier.generate(transcript, state, history)

        return RoutingDecision(
            use_fast_tier_response=not escalate, escalate_to_thinker=escalate,
            reason=f"confidence-based: NLL={fast_result.mean_neg_logprob:.2f} "
                   f"{'>' if escalate else '<='} threshold={self.nll_escalate_threshold}",
            fast_result=fast_result, reasoning_result=reasoning_result,
        )
