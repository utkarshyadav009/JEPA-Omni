# Streaming/causal STT: scoping proposal (not implemented)

Written 2026-08-07. This is a scoping document, not a build -- explicitly deferred to a future
session per user direction ("that's a different call for me to make later").

## Current state (verified by reading the code, not assumed)

- `models/m4d_stt_projector.py`'s `AudioEncoderProjector` (trained tonight, `scripts/
  train_stt_projector.py`) processes a whole pre-segmented utterance through Whisper's encoder in
  one forward pass. It has no notion of partial/incremental audio.
- `models/m4_duplex_loop.py`'s `compute_speech_activity()` does the same thing for a different
  purpose (turn-taking decision, not transcription): takes a full `(waveform, duration_sec)` and
  runs it through Whisper once per call. The M4/M5 tick loop calls this on a *sliding window* of
  recently-accumulated audio each tick -- that's chunked reprocessing, not true incremental/causal
  streaming (each tick re-encodes the whole window from scratch, doesn't carry state forward).
- Whisper's own encoder is non-causal (bidirectional self-attention over the full input) --
  it fundamentally cannot produce a "partial" transcription of audio that hasn't finished arriving
  without some workaround.

## What "real" streaming STT would need

Two workable approaches, not mutually exclusive:

1. **VAD-gated chunking (cheaper, reuses what exists)**: use the existing
   `compute_speech_activity` machinery as a voice-activity detector to decide *when* a chunk of
   speech has ended (a pause), then run the full (now-complete) chunk through the STT projector
   once. This is NOT true low-latency streaming (you wait for a pause before transcribing), but it
   is a real, buildable improvement over the current design where nothing chunks by speech
   boundaries at all. Lowest engineering cost of the two options.

2. **True incremental/causal STT**: replace or augment Whisper with an architecture that can
   process audio in small causal chunks and update a running transcription -- e.g. a
   streaming-friendly encoder (chunked/causal attention) or an approach like Simul-Whisper
   (truncation-aware decoding on partial Whisper windows, referenced in earlier session research
   but never implemented here). Real engineering lift: new model integration, new training/eval
   data for partial-utterance transcription quality, and a redesign of `AudioEncoderProjector`'s
   single-forward-pass training objective to work incrementally.

## Recommendation

Do (1) first if this gets picked up -- it's a real, scoped, low-risk win (transcribe on actual
speech boundaries instead of a fixed tick-driven window) that reuses `compute_speech_activity`
verbatim. Treat (2) as a separate, larger project with its own falsifier-style validation (does
partial-window transcription quality hold up against Whisper's non-causal full-utterance
baseline?) before committing engineering time.

## Explicitly not done tonight

No code changes for either approach. This file exists so the scoping work (what's missing, why,
and the two real options) isn't lost, not to claim partial implementation.
