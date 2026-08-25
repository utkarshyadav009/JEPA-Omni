# Speculative turn-taking — the "predict-the-user" latency idea (Moshi-flavored)

Captured 2026-08-07 from the user's idea: *"predict what the user might say, so we can
prompt and get a response from the LLM ahead of time — similar to how Moshi predicts both
the user and the model."*

## What Moshi actually does (and why we can't copy it directly)
Moshi (Kyutai) is ONE end-to-end full-duplex speech↔speech model. It models the user
stream AND its own stream as parallel token streams at a fixed 12.5 Hz frame rate
(RQ-Transformer + an "Inner Monologue" text stream that runs a few steps ahead of the
acoustic tokens). Because it jointly predicts both streams, it can start forming a reply
before the user finishes and handle overlap/barge-in natively.

**BMO is a pipeline, not a unified duplex model** (Moonshine STT → 3-class decision head →
two-tier LLM → NeuTTS streaming voice). We cannot literally run Moshi's joint model — it
wouldn't fit 7.6 GB alongside the rest, and it throws away the whole invested stack. But
the *intent* — "run ahead of the user so the reply feels instant" — is achievable as
**speculative execution** layered on the existing pipeline. Speculation is a pure win when
two conditions hold, and both do here: (1) spare compute while the user talks (the Jetson
GPU is idle then), and (2) the speculative result is verified before it's committed (so a
wrong guess is silently discarded, never spoken).

## Tiered design (build cheap → ambitious)

### Tier 0 — backchannel masking (already deployed)
`thinking_filler` clip fires at SPEAK decision. Hides latency; not prediction. Baseline.

### Tier 1 — partial-transcript prefetch + first-chunk pre-decode  ← RECOMMENDED NEXT
No user-*word* prediction needed; just start early on words already spoken. People usually
finish a sentence a beat before they stop talking / before the endpointer fires.
1. During the user's turn, on each VAD micro-pause (or every ~300 ms), run Moonshine
   (37 ms, cheap) on audio-so-far → partial transcript.
2. When the partial looks sentence-complete, **speculatively** launch fast-tier LLM(partial
   transcript, current mood) → candidate reply, and pre-run the FIRST ~30 speech tokens
   through `StreamingVoice` so the first audio chunk is decoded and ready.
3. On real end-of-turn: compare final transcript to the speculated one. If ≥ ~0.9
   token-overlap / semantic match → **commit** the precomputed reply + pre-decoded first
   chunk instantly (perceived latency ≈ 0). Else → discard, run the normal path (Tier 0
   backchannel masks it).
4. Cap concurrent speculations at 1; cancel + re-speculate if the transcript grows
   materially while a speculation is in flight.
- **Cost**: one extra fast-tier run (337 ms) that may be discarded — but the GPU is
  otherwise idle during the user's turn, so it's ~free in latency terms (only energy).
- **Risk**: GPU contention with the async perception thread — already handled by Track D
  (opportunistic vision scheduling: refresh vision only during mic-gated TTS playback).
- **Safety**: verified-commit means a wrong guess is never spoken. Strictly a win.

### Tier 2 — true next-utterance / intent prediction (the literal Moshi flavor)
Predict the user's likely full utterance/intent BEFORE they finish, from partial transcript
+ dialogue history + BMO's last turn (dialogue acts are very predictable: "how are you" →
"I'm good, you?"). Options: a tiny predictor head, or reuse the reasoning tier to generate
likely continuations. Pre-run the fast tier on the top-1 predicted utterance; commit only
on a match, else discard.
- **Gate on measurement first**: instrument Tier 1's hit rate = P(final transcript ==
  last partial at the pre-endpoint moment). If that's already high (likely for a companion's
  patterned interactions), Tier 1 captures most of the gain and Tier 2's extra
  complexity/misfire-risk isn't worth it. Only build Tier 2 if the measured gap is real.

### Tier 3 — full duplex model (replace the pipeline with a Moshi-style joint model)
Rejected for now: doesn't fit the memory budget, discards the whole stack. Revisit only if
the companion goal demands true barge-in/overlap that Tiers 1–2 can't fake.

## Recommendation
Build **Tier 1** after the emotion voice lands. It is the highest-ROI latency win remaining
now that TTS streams (TTFA 625 ms), it's Jetson-safe (idle-time compute, verified commit),
and it reuses every existing component (Moonshine, fast tier, `StreamingVoice`). Treat
Tier 2 as measurement-gated. This delivers most of Moshi's "feels instant" without a
model rewrite.

## BUILT + BENCHMARKED (2026-08-08)
`models/m5_speculative.py::SpeculativePrefetcher` implements Tier 1 as a pure
mechanism (takes a `generate_fn` + optional `tts`, so it's unit-testable and
wireable without loop internals). `scripts/bench_tier1_speculative.py` runs it
on the REAL Jetson models (LFM2 fast tier + StreamingVoice, no perception so no
M3/preflight issue). Measured:
- **Baseline response path (Jetson): fast-tier 430ms + TTS-first-audio 463ms = 893ms.**
- **On a HIT: perceived response latency ~0ms** — the reply (and first audio chunk)
  were already computed during the user's remaining speech. ~811ms avg removed per hit.
- On a MISS: silent fallback to the normal path (never speaks a wrong guess) — verified
  with a non-matching pair (`"tell me a story"` → `"what time is it"` → correct miss).
- Match metric: symmetric token-overlap (Jaccard), threshold 0.80 — conservative on
  purpose (a miss just forgoes the speedup; a false hit would speak the wrong reply).
  Tunable; a containment metric (partial ⊆ final) would catch more trailing-word
  elaborations if we want a higher hit rate at some safety cost.

Remaining for live use (NOT yet done): wire into `m5_streaming_loop` — the loop must feed
PARTIAL transcripts during the turn (needs mid-turn Moonshine decodes, which the
feature-based decision path doesn't currently produce) via `prefetcher.speculate(partial,
state)`, and call `prefetcher.commit(final, state)` right before `generate_fn()` in the
speak path (use the returned result + first_audio instead of regenerating on a hit).
End-to-end verification needs mic hardware (the current Jetson has none), so this
integration is deferred; the component + benchmark stand on their own as the proof.

## Where it hooks into the code
- Endpointing / decision: `models/m5_streaming_loop.py` (decision path, backchannel hook)
  and the 3-class decision head (`m4_decision_head_3class_speechonly_moonshine`).
- Speculative LLM run: `models/m4_cognitive_core.py` fast tier (`_state_prefix` already
  injects mood, so the speculative run is mood-correct for free).
- Pre-decode first chunk: `models/m5_streaming_voice.py::StreamingVoice.stream()` (already
  yields incrementally; buffer the first yield).
