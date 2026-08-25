"""models/m5_speculative.py -- Tier 1 speculative turn-taking ("predict the
user, run ahead"), the Jetson-feasible realization of the Moshi dual-stream
intuition (full design in SPECULATIVE_TURNTAKING.md).

Idea: people finish a sentence a beat before the endpointer fires. So during
the user's turn, on each partial transcript, we SPECULATIVELY run the fast-tier
LLM (and optionally pre-decode the first audio chunk) in the background -- the
GPU is idle then anyway. When the real end-of-turn arrives, if the final
transcript matches a speculation we already have, we COMMIT it instantly
(perceived latency ~= 0). On a miss we discard silently and fall back to the
normal path -- so a wrong guess is NEVER spoken. Strictly a latency win.

This is a pure mechanism: it takes a `generate_fn(transcript, state) -> result`
(the fast tier) and an optional `tts` (StreamingVoice) for first-chunk
pre-decode. No dependency on the loop internals, so it's unit-testable and
benchmarkable standalone, then wired into m5_streaming_loop behind a flag.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(s: str) -> List[str]:
    return _WORD.findall(s.lower())


def transcript_match(a: str, b: str) -> float:
    """Symmetric token-overlap (Jaccard) of two transcripts, 0..1. Robust to
    trailing words / minor ASR wobble -- what we need to decide 'is the final
    close enough to what we speculated on'."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class _Speculation:
    partial: str
    state: Dict
    future_done: threading.Event
    result: object = None          # generate_fn result (has .text)
    first_audio: object = None     # optional pre-decoded first chunk (np.ndarray)
    gen_ms: float = 0.0
    audio_ms: float = 0.0
    error: Optional[BaseException] = None
    t_started: float = field(default_factory=time.perf_counter)


class SpeculativePrefetcher:
    def __init__(self, generate_fn: Callable[[str, Dict], object], *,
                 tts=None, match_threshold: float = 0.80,
                 min_partial_words: int = 3, prewarm_audio: bool = True,
                 commit_wait_ms: float = 250.0):
        """
        generate_fn: (transcript, state) -> result with a `.text` attribute
                     (the fast tier's generate).
        tts:         optional StreamingVoice; if given and prewarm_audio, the
                     speculation also pre-decodes the FIRST audio chunk so the
                     reply can start playing the instant we commit.
        match_threshold: min token-overlap between final and speculated
                     transcript to accept a speculation (else fall back).
        min_partial_words: don't speculate on very short partials (low value,
                     high churn).
        commit_wait_ms: at commit time, if the matching speculation is still
                     in flight, wait at most this long for it (it usually
                     finished during the user's remaining speech).
        """
        self.generate_fn = generate_fn
        self.tts = tts
        self.match_threshold = match_threshold
        self.min_partial_words = min_partial_words
        self.prewarm_audio = prewarm_audio and tts is not None
        self.commit_wait_ms = commit_wait_ms

        self._lock = threading.Lock()
        self._specs: List[_Speculation] = []
        self._inflight: Dict[str, _Speculation] = {}
        # metrics (read by the benchmark / logs)
        self.n_speculations = 0
        self.n_hits = 0
        self.n_misses = 0
        self.saved_ms_total = 0.0

    # ---- speculation (called on each partial transcript during the turn) ----
    def speculate(self, partial: str, state: Dict) -> None:
        partial = (partial or "").strip()
        if len(_tokens(partial)) < self.min_partial_words:
            return
        with self._lock:
            if partial in self._inflight:
                return  # already speculating this exact partial
            spec = _Speculation(partial=partial, state=dict(state or {}),
                                future_done=threading.Event())
            self._inflight[partial] = spec
            self._specs.append(spec)
            self.n_speculations += 1
        threading.Thread(target=self._run_spec, args=(spec,), daemon=True).start()

    def _run_spec(self, spec: _Speculation) -> None:
        try:
            t0 = time.perf_counter()
            spec.result = self.generate_fn(spec.partial, spec.state)
            spec.gen_ms = (time.perf_counter() - t0) * 1000.0
            if self.prewarm_audio and getattr(spec.result, "text", ""):
                ta = time.perf_counter()
                mood = getattr(spec.result, "mood", None)
                # first stream() yield = first decoded audio chunk (time-to-first-audio)
                gen = self.tts.stream(spec.result.text, emotion=mood)
                spec.first_audio = next(gen, None)
                spec.audio_ms = (time.perf_counter() - ta) * 1000.0
        except BaseException as e:  # noqa - a speculative failure must never crash the turn
            spec.error = e
        finally:
            spec.future_done.set()

    # ---- commit (called when the real end-of-turn transcript is known) ----
    def commit(self, final_transcript: str, state: Dict):
        """Return (result, first_audio, saved_ms) if a speculation matches the
        final transcript, else (None, None, 0.0) -> caller runs the normal path.
        `saved_ms` is the wall time that speculation ran DURING the user's turn
        (i.e. latency removed from the response path on this hit)."""
        final = (final_transcript or "").strip()
        with self._lock:
            candidates = list(self._specs)
        best, best_score = None, 0.0
        for spec in candidates:
            score = transcript_match(final, spec.partial)
            if score > best_score:
                best, best_score = spec, score

        if best is None or best_score < self.match_threshold:
            self._reset()
            self.n_misses += 1
            return None, None, 0.0

        # matching speculation -- wait briefly if it's still finishing
        if not best.future_done.is_set():
            best.future_done.wait(timeout=self.commit_wait_ms / 1000.0)
        if not best.future_done.is_set() or best.error is not None or best.result is None:
            self._reset()
            self.n_misses += 1
            return None, None, 0.0

        saved = best.gen_ms + best.audio_ms
        self.n_hits += 1
        self.saved_ms_total += saved
        self._reset()
        return best.result, best.first_audio, saved

    def _reset(self) -> None:
        with self._lock:
            self._specs.clear()
            self._inflight.clear()

    def stats(self) -> Dict:
        hit_rate = self.n_hits / max(1, self.n_hits + self.n_misses)
        return {"speculations": self.n_speculations, "hits": self.n_hits,
                "misses": self.n_misses, "hit_rate": round(hit_rate, 3),
                "saved_ms_total": round(self.saved_ms_total, 1),
                "avg_saved_per_hit_ms": round(self.saved_ms_total / max(1, self.n_hits), 1)}
