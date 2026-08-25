# LIMITATIONS.md

Known, disclosed scope boundaries of the system as of the submission
freeze (`freeze-submission-v1`). These are not failed gates (see
`NEGATIVE_RESULTS.md` for those) — they are design/scope limits accepted
to ship a working demo within the 18-day window, stated explicitly rather
than left implicit.

**Ego4D representation quality gate never met.** The within-modality
cosine gate (target ≤0.25) has not passed on any M2 checkpoint in this
project, including the locked one (0.4358 vision / 0.3893 ambient). M2's
retrieval numbers (VGGSound, Ego4D R@1) are strong; its embedding-space
spread is not yet where the pre-registered target says it should be.

**Mic gating cannot distinguish self-echo from a real interruption.**
`MicGate` mutes the mic entirely while the system is speaking — measured
0% false-interruption from self-echo, but also 0% detection of a genuine
simultaneous user interruption during that window (`scripts/
m4_echo_test.py`'s Case B). This is a deliberate, disclosed tradeoff
(AEC alone was measured insufficient), not an oversight — the alternative
(adaptive echo cancellation) needs a convergence warm-up period this
project did not have time to validate as production-safe.

**No genuinely-paired AV data for simultaneous decision + generation
training.** The decision head and the M3/M4b connectors were never
trained on a single dataset where vision, ambient audio, AND speech are
all real and correctly paired for the same moment. Each was trained/gated
on the paired data it had (EasyCom for speech/turn-taking, VGGSound/Ego4D
for AV). The A1 six-condition falsifier's finding (World-State doesn't
help turn-taking) mitigates the sharpest edge of this gap for the decision
head specifically, but does not close it for the system as a whole.

**Turn-taking is speech-only by design, not vision-informed.** The locked
decision head (`m4_decision_head_3class_speechonly_v2`) takes no
World-State input. This was a measured, not assumed, choice — but it
means the system cannot use visual cues (gaze, gesture, someone visibly
about to speak) to inform when to yield the floor. Vision still grounds
what is generated, just not when.

**Jetson memory headroom is tight and long-session stability is
unverified.** 854MiB headroom against the full stack, with
`NvMapMemAllocInternalTagged` warnings observed (non-fatal, but indicating
the allocator is near its contiguity limit) during the measurement that
produced that figure. Whether these warnings escalate to failures over a
long (15-minute) conversation is exactly what Phase D2 was designed to
test — not yet run (see the phase-boundary report for the access blocker).

**No live end-to-end hardware validation at freeze time.** Every
Jetson number in `RESULTS_TABLE.md` comes from an isolated component or
sub-stack test (memory footprint, TTS latency, tick latency in isolation).
No run has yet exercised live camera + live mic + real TTS output + real
mic-gating + real generation together, continuously, on the Jetson. Phases
C and D of the freeze plan exist specifically to close this gap and are
prepared (code written, checkpoints locked, scripts pointed at the right
paths) but not yet executed.

**TTS is CPU-only and a fixed small backchannel set.** Piper
(`en_US-lessac-medium`) was chosen for its zero CUDA footprint, not
benchmarked against GPU-resident alternatives for naturalness/quality.
Backchannels come from a fixed 5-phrase pre-synthesized inventory
(round-robin) — not adaptive to conversational content, a deliberate
demo-robustness simplification over the earlier LLM-generated approach.

**Corpus scaling beyond VGGSound+Ego4D is an open, confounded question.**
RUN-3 (AudioSet addition) failed both retrieval gates, but the result is
confounded by two simultaneous changes (negatives, ambient cap — see
`NEGATIVE_RESULTS.md`) and cannot be read as evidence against AudioSet
itself. Action100M was never extracted or trained on at all. Both are
explicitly future work, not built on by the locked checkpoints.

**English-only, single-speaker-oriented.** Whisper, Piper, and Qwen2.5 are
all used in their English configurations here; no multilingual claim is
made. EasyCom (the turn-taking training source) contains multi-speaker
sessions, but the decision head's training/eval framing is per-segment
speech-activity, not multi-speaker diarization — the system has not been
validated against more than one live human speaker at a time.
