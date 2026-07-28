# Falsifier tracking — M3 / M4b standalone grounding, by stage

Standing rule (added 2026-07-24, see CLAUDE.md): after EVERY stage that
touches the M3 connector or the M4b projector (joint training, LoRA, M4c),
re-run BOTH standalone falsifiers and append a row here. The risk being
tracked is COMPOUNDING degradation across stages, not any single stage's
hit in isolation.

Methodology (fixed across rows for comparability): `scripts/m4_joint_eval.py`,
n=100 VGGSound clips / 100 EasyCom segments, seed=5, `gpt_sound_acoustic`
field, `sentence-transformers/all-MiniLM-L6-v2` for semantic cosine.

## M3 standalone (word-overlap F1, no speech stream present)

| stage | ckpt | F1 normal | F1 swapped | F1 zeroed | cos normal |
|---|---|---|---|---|---|
| frozen-LLM baseline (multi-granularity M3, pre-M4) | `m3_multigran_best/connector.pt` | 0.471 | 0.268 | 0.274 | 0.724 |
| joint-exposure fine-tune (step 700) | `m4_joint/best.pt` (`m3_connector`) | 0.430 | 0.270 | 0.249 | 0.714 |
| **Phase 1a LoRA (step 1200), connector unchanged from joint stage** | `m4a_lora/best.pt` (LoRA+delta on top) | **0.377** | 0.294 | 0.262 | **0.651** |
| Fix 1 (r=8, silence_weight=6x, 8000 steps, step 2400) | `m4a_lora_fix1/best.pt` | 0.365 | 0.290 | 0.290 | 0.623 |
| Fix 2a (r=4, silence_weight=6x, 8000 steps, step 3600) | `m4a_lora_fix2a/best.pt` | **0.391** | 0.283 | 0.294 | 0.667 |
| Fix 2b (r=4, attn-only, silence_weight=6x, 8000 steps, step 4600) | `m4a_lora_fix2b/best.pt` | 0.318 | 0.236 | 0.268 | 0.615 |
| **REVERTED (LoRA dropped, M4c base)** | `m4_joint/best.pt`, plain frozen LLM | **0.430** | 0.270 | 0.249 | **0.714** |

Deltas vs baseline: F1 normal −8.7% (joint), −12.3% (joint→lora), **−20.0%
cumulative**. cos −1.4% (joint), −8.8% (joint→lora), **−10.1% cumulative**.
Gap (normal − swapped): baseline=0.203, post-joint=0.160,
**post-LoRA=0.083** — still positive/discriminating but now less than half
the original baseline gap. This is the compounding-drift pattern the
standing rule exists to catch: neither single stage looked catastrophic,
but two stages in, M3's F1 has dropped a fifth from where it started.

## M4b standalone (semantic cosine, no M3 stream present)

| stage | ckpt | cos normal | cos swapped-vs-target | cos swapped-vs-donor |
|---|---|---|---|---|
| frozen-LLM baseline (M4b stage-1) | `m4b/best.pt` | 0.517 | 0.152 | 0.517 |
| joint-exposure fine-tune (step 700) | `m4_joint/best.pt` (`m4b_projector`) | 0.419 | 0.133 | 0.419 |
| **Phase 1a LoRA (step 1200), projector unchanged from joint stage** | `m4a_lora/best.pt` (LoRA+delta on top) | **0.455** | 0.124 | 0.455 |
| Fix 1 (r=8, silence_weight=6x, 8000 steps) | `m4a_lora_fix1/best.pt` | 0.519 | 0.163 | 0.519 |
| Fix 2a (r=4, silence_weight=6x, 8000 steps) | `m4a_lora_fix2a/best.pt` | **0.594** | 0.152 | 0.594 |
| Fix 2b (r=4, attn-only, silence_weight=6x, 8000 steps) | `m4a_lora_fix2b/best.pt` | **0.652** | 0.173 | 0.652 |
| **REVERTED (LoRA dropped, M4c base)** | `m4_joint/best.pt`, plain frozen LLM | **0.419** | 0.133 | 0.419 |

Delta vs baseline: cos normal −19.0% (joint), **+8.6% (joint→lora, i.e.
LoRA partially RECOVERED M4b's cost from the joint stage)**, −12.0%
cumulative. `swapped_vs_donor == cos_normal` exactly preserved across all
three rows (swap-control mechanism intact throughout). M4b fared better
under LoRA than M3 did — asymmetric, same pattern as the composition checks.

## Composition checks (c)/(d), for context (not part of the standing per-stage rule, but tracked here since they're the reason joint training happened)

| check | condition | normal | swapped/zeroed |
|---|---|---|---|
| (c) M3 + fixed real speech | pre-joint (frozen M3 + frozen M4b) | 0.000 | 0.000 / 0.001 |
| (c) M3 + fixed real speech | post-joint | 0.391 | 0.299 / 0.244 |
| (d) M4b + fixed M3 latents | pre-joint | 0.244 | vs-target 0.120 |
| (d) M4b + fixed M3 latents | post-joint | 0.291 | vs-target 0.115 |

## Phase 1a silence-rate gate — FAILED (original run and both fix attempts)

`checkpoints/m4a_lora/silence_rate_report.json`, n=150 per source, held-out:

| source | gt_silence_rate | silence_recall | speak_recall |
|---|---|---|---|
| VGGSound pseudo-timeline | 0.813 | **0.000** | 1.000 |
| EasyCom turn-taking | 0.520 | **0.000** | 1.000 |

Model collapsed to always predicting "speak," never `<silence>`, on both
sources. Verified not a scoring bug: direct next-token inspection on
silence-labeled examples shows P(`<silence>`) ≈ 0.0004, barely above the
~0.0000066 uniform-chance floor, while the model's actual top candidate
sits at only 1.35% probability (a flat, unconverged distribution, not a
confident wrong answer).

**Fix 1 attempt (loss reweighting 6x on silence targets + 8000 steps, 4x
longer than the original run) — FAILED, no movement at all.** Full
trajectory logged every 200 steps across all 8000 steps
(`checkpoints/m4a_lora_fix1/train_log.jsonl`): `silence_recall=0.000` at
EVERY single one of the 40 checkpoints, start to finish. This is not slow
convergence -- it's flat at zero the entire run, on both the r=8 config
(Fix 1) and the r=4 config (Fix 2a, same weighting/duration). Loss
reweighting and 4x more training did not move this metric by any
measurable amount.

Per the pre-agreed decision tree, this result points toward Fix 1c
(curriculum: learn the speak/silence decision alone before mixing in
content generation) as the next thing to try -- not yet attempted, flagging
for review rather than starting unilaterally.

## Phase 1a grounding-drift gate (hard floor: M3 normal-vs-swapped gap ≥ 0.12)

| config | M3 gap (normal−swapped) | vs floor |
|---|---|---|
| original Phase 1a (r=8, no reweighting, 2000 steps) | 0.083 | FAIL |
| Fix 1 (r=8, weighted, 8000 steps) | 0.075 | FAIL (slightly worse) |
| **Fix 2a (r=4, weighted, 8000 steps)** | **0.108** | **FAIL, but closest yet** |

Lowering LoRA rank 8→4 moved the gap the right direction (0.083→0.108) and
clearly helped M4b too (cos_normal 0.455→0.594, well above even the
pre-LoRA post-joint baseline of 0.419) — but still short of the 0.12 floor.
Per the pre-agreed decision tree, next step is Fix 2b (attention-only LoRA,
dropping FFN targets, likely combined with r=4 since that's the improvement
already found) — not yet attempted.

## PIVOT (2026-07-24): decision moved OUT of the LLM — decision head, not LoRA

Per direction: Fix 1c (curriculum) not attempted — flat-zero across 40
checkpoints, 2 ranks, 6x weighting, 4x duration is not a convergence-speed
problem, and SIGReg doesn't apply (continuous-embedding-collapse
regularizer; this was a discrete token-emission failure with no embedding
distribution to shape). Replaced the LLM/LoRA route entirely with a small
dedicated `SpeakSilenceHead` (models/m4_decision_head.py, 1.32M params) that
reads [World-State (M2, real for VGGSound / zero for EasyCom); speech-
activity feature (mean-pooled frozen Whisper hidden state, real for EasyCom
/ zero for VGGSound)] and outputs a binary speak/silence logit. Plain BCE +
pos_weight class balancing (pos_weight=2.10, computed from the actual
25,949-example combined training pool: 8,370 speak / 17,579 silence). The
LLM is NOT touched by this at all.

### Decision-head gate — PASSED, both classes, both sources

`checkpoints/m4_decision_head/gate_results.json`, held-out test (n=4000
VGGSound ticks from 1000 clips, n=1952 EasyCom ticks):

| source | gt_speak_rate | speak_recall | silence_recall | accuracy |
|---|---|---|---|---|
| VGGSound pseudo-timeline | 0.250 | **0.808** | **0.924** | 0.895 |
| EasyCom turn-taking (real gaps) | 0.570 | **0.994** | **0.976** | 0.986 |

Both recalls clearly non-zero on both sources — contrast with the LLM/LoRA
route's flat 0.000/1.000 collapse. Decision threshold = 0.5 (sigmoid).

### Decision-head swap-control — PASSED

`checkpoints/m4_decision_head/swap_control_results.json`, n=4000 VGGSound
test ticks, World-State swapped in from a different (rolled) example:

| | accuracy |
|---|---|
| normal (own World-State vs own label) | 0.895 |
| swapped vs TARGET's label (want ~chance) | 0.493 |
| swapped vs DONOR's label (want ≈ normal) | **0.895** (exact match) |

40.4% of decisions changed when the World-State was swapped — the head is
reading its input, not memorizing a fixed prior; swapped decisions track
whichever World-State they were actually given, same pattern as every other
swap-control run in this project.

## LoRA recommendation: DROP IT

LoRA's only stated purpose throughout M4 was teaching control-token
emission. That task no longer exists in the LLM (the decision head owns it
now). M3/M4b grounding never needed LoRA — both worked fine frozen. With
the reason for LoRA gone, recommend **reverting to the frozen LLM +
joint-trained connectors** (checkpoints/m4_joint/best.pt), restoring M3 to
0.430 F1 / 0.714 cos and M4b to 0.419 cos — the last state before any LoRA
cost was paid, and the correct floor to build M4c on.

One attention-only r=4 LoRA run was run for the record per instruction:
M3 gap = 0.082 (0.318−0.236) — actually slightly WORSE than full-target
r=4's 0.108, still well below the 0.12 floor, and silence_recall still flat
zero (expected, irrelevant now). M4b continued improving (cos_normal=0.652,
the best M4b number recorded across every stage). Confirms the
recommendation: dropping FFN targets doesn't reliably fix M3's grounding
drift either, and the underlying reason for LoRA is gone regardless of
this result.

## M4c step 1: reversion confirmed, not assumed

Re-ran `scripts/m4_joint_eval.py` (same script, same seed, same n=100 pools)
against the plain frozen LLM + `checkpoints/m4_joint/best.pt` connectors,
zero LoRA in the loop at all. Result is an EXACT bit-for-bit match to the
originally-recorded post-joint row (both M3 and M4b) — deterministic
greedy decoding confirms no residual state/drift survived dropping LoRA.
Full LoRA-stage cost recovered. This is the confirmed base for M4c.

## M4c duplex loop — built and gated (2026-07-25)

`models/m4_duplex_loop.py` (tick grid, interruptible per-token generation)
+ `scripts/m4c_gate_eval.py`. Base: frozen LLM (no LoRA) + `m4_joint`
connectors (grounding confirmed restored above) + decision head + external
VAD-driven interruption controller (no `<stop_interruption>` token).

**Shift-trigger metric — selected on real data, no longer deferred.**
`scripts/m4_shift_metric_easycom.py`, 4,168 consecutive EasyCom segment
pairs, boundary = real speaker change: euclidean AUC=0.614 (Cohen's
d=0.400), cosine AUC=0.612 (d=0.363), **mahalanobis AUC=0.497 (d=-0.004,
chance level)**. The previously-default Mahalanobis (justified by M2's
measured World-State anisotropy) fails on the task that actually matters;
default flipped to euclidean in `models/m4_shift_trigger.py`. Honest
caveat: AUC=0.614 is real but modest — mean-pooled Whisper features discard
most speaker-identity detail; a dedicated speaker embedding would likely
do better, out of scope here.

**Turn-taking P/R — EasyCom held-out SESSIONS (headline)**:

| session | n | speak_P | speak_R | speak_F1 | silence_R |
|---|---|---|---|---|---|
| 10 | 477 | 0.985 | 0.989 | 0.987 | 0.981 |
| 11 | 751 | 0.989 | 0.995 | 0.992 | 0.984 |
| 12 | 724 | 0.974 | 0.995 | 0.984 | 0.965 |
| **overall** | 1952 | **0.982** | **0.994** | **0.988** | **0.976** |

**VGGSound pseudo-timeline — MECHANISM CHECK ONLY, not a turn-taking
result** (synthetic scene-continuity labels, not real conversational
turn-taking): speak_P=0.779, speak_R=0.808, speak_F1=0.793, silence_R=0.924.

**Interruption latency**: per-token forward-pass mean=6.98ms, p95=7.09ms,
max=64.07ms (n=30 generations, interrupt simulated at step 5). This IS the
worst-case interruption latency in this design (checked between tokens,
not mid-token) — **sub-second target PASSED with a >10x margin** (worst
case ~64ms, not ~1000ms).

**End-to-end tick latency** (perception → decision → first token, n=120
ticks): overall mean=5.53ms, max=12.55ms. Split by branch: silence-only
(decision, no generation) mean=2.50ms (n=82); speak (decision + first-token
generation) mean=12.08ms (n=38). **Hardware: NVIDIA RTX PRO 6000 Blackwell
Server Edition — NOT a consumer RTX 4090.** Flagging explicitly since this
number feeds M5 capacity planning: a real 4090 would likely be meaningfully
slower, especially on the generation branch; these numbers should not be
assumed to transfer directly to consumer-GPU deployment without a separate
benchmark on that actual hardware.

**Falsifier regression**: unchanged from the reversion row above (LLM
frozen again, connectors untouched by anything in this stage) — verified,
not assumed, via the same M3/M4b standalone re-run.

## KNOWN LIMITATION — flagged, not yet addressed

All EasyCom turn-taking data (training AND the held-out session evaluation
above) comes from **7 distinct speakers across 12 sessions**, all recorded
in the same small set of rooms/setups. Checked directly, not assumed: the
5 speaker IDs appearing in the held-out test sessions (10/11/12) are
`{3,4,5,6,7}` — **the exact same 5 IDs that appear in the train sessions
(1-9)**. Zero speakers in the test set are unseen during training. "Held-
out session" here means held-out *recording*, not held-out *voice* — the
0.988 session-level F1 is real and honestly measured on sessions never
touched during training, but it says nothing about generalization to a
genuinely new speaker, accent, or room. This is the single biggest
external-validity gap in the current M4 turn-taking result and should be
weighed accordingly before treating 0.988 as a realistic deployment-scale
number.

## Backchannel handling — 3-class decision head (2026-07-25)

Full detail: checkpoints/m4_decision_head_3class/PROVENANCE.txt. No M3/M4b
connector or LLM weights touched (new head only + generation-budget gating
at the call site) — outside the standing rule's trigger condition, logged
here anyway per the gate's own tracking language.

Mined 899/4482 (20.1%) EasyCom segments as backchannel (lexicon-based,
conservative: single/double acknowledgment-only words, or bare disfluency
markers). Extended SpeakSilenceHead -> ThreeClassHead (silence/speak/
backchannel), retrained on cached VGGSound + re-extracted EasyCom features.

**Gate — PASSED, all 3 EasyCom classes clearly non-zero**: silence
recall=0.980, speak recall=0.962, backchannel recall=0.871 (n=1952 held-out
EasyCom ticks). VGGSound mechanism check (no backchannel concept):
silence=0.922, speak=0.801.

**Swap-control — PASSED**: normal acc=0.892, swapped-vs-target≈chance
(0.492), swapped-vs-donor=0.892 (exact match to normal) — head reads its
actual input, doesn't memorize a prior.

**DETECTION side (false-halt rate on backchannel segments)**: OLD 2-class
policy (any speech -> halt) = 0.972 false-halt rate on backchannel;
**NEW 3-class policy (only SPEAK halts) = 0.101** — a ~10x reduction. Honest
cost: true-halt rate on genuine user speech drops 0.999 -> 0.962 (3.7pp),
a direct, measured consequence of backchannel recall being 0.871 not 1.0.

**PRODUCTION side**: 3-class decision gates generation budget (SPEAK->40
tokens, BACKCHANNEL->8 tokens) on the unmodified frozen-LLM+M4b pipeline —
no new capability trained into the LLM. Routing accuracy 0.875 (n=16
sample), mean generated length 15.4 tokens (backchannel GT) vs 21.9 tokens
(speak GT). Honest limitation: budget-capped LLM continuations are
truncated transcript-like text, not fluent backchannel language (e.g.
"[L] [H] [_] [_") — a canned short-response fallback bypassing the LLM
entirely on BACKCHANNEL decisions would likely read better for a live demo;
flagged as follow-up, not implemented.

## Interruption policy — post-halt state machine (2026-07-25)

Full detail: checkpoints/m4_interruption_policy/PROVENANCE.txt. No M3/M4b
connector or LLM weights touched — outside the standing rule's trigger
condition, logged here for continuity with the rest of M4's tracking.

Mined 1,796 genuine (non-backchannel) interruption events from all 12
EasyCom sessions (scripts/m4_interruption_mining.py): RESUME 4.7% (median
interrupter duration 1.40s), RE-PLAN 61.1% (1.90s), ABANDON 34.1% (2.15s).
Timing-based ABANDON split is ground truth; RESUME-vs-RE-PLAN split is a
stated text-similarity PROXY, not a verified intent label — EasyCom has no
ground truth for "same utterance continued" vs "new utterance".

Built InterruptionPolicy (models/m4_interruption_policy.py): reuses the
already-validated euclidean shift-trigger metric (tau=18.65, midpoint of
the earlier turn-boundary study's real within/boundary means) + the mined
duration medians as a soft co-requirement for ABANDON. RE-PLAN is the
default/fallback, matching the real 61.1% prior. Smoke test confirms all 3
transitions reachable — PASSED. Explicitly flagged as a grounded heuristic,
not a trained classifier (mined label set too small/proxy-based to fit one
honestly).

Built and VERIFIED the RESUME mechanism: generate_interruptible now
captures {past_key_values, inputs_embeds, attn} on halt; resume_interrupted
continues the same KV-cache. Directly verified: a run halted at step 5 and
resumed for 15 more tokens produced a token-id sequence that is an EXACT
prefix match + byte-identical final text to an uninterrupted 20-token run
of the same total length — zero drift across the halt/resume boundary.

RE-PLAN and ABANDON's branches reuse existing, already-verified machinery
(fresh SPEAK-path generation; default return-to-IDLE) with no new code
needed. Live end-to-end tick-loop integration not built — scope was
"define and implement the state machine," delivered at the
verified-mechanism level, not a full live-demo wiring.

## Speaker generalization — leave-one-speaker-out (2026-07-25)

Full detail: checkpoints/m4_decision_head/LOSO_PROVENANCE.txt. Answers the
KNOWN LIMITATION flagged above (session-split held out recordings, not
voices — all 5 usable-audio speakers appeared in both train and test).

Feasible with 5 speakers (only 5/7 EasyCom participants have their own
close-mic audio — a long-standing dataset constraint, not new). Ran a real
5-fold LOSO (scripts/m4_loso_eval.py): fresh SpeakSilenceHead per fold,
trained on VGGSound + the other 4 speakers' EasyCom ticks, evaluated on the
held-out speaker across all their sessions.

**Result: macro-average speak_recall=0.986, silence_recall=0.987,
accuracy=0.986 — statistically indistinguishable from the session-split
headline (0.994/0.976/0.986)**. Per-speaker range: accuracy 0.968 (speaker
5, smallest fold, n=493) to 0.994 (speakers 3 and 7). This is a genuinely
reassuring, non-obvious result: the flagged limitation predicted possible
degradation on unseen voices and the measured degradation is essentially
zero. Plausible explanation: mean-pooled Whisper features mostly discard
speaker identity, so turn-taking (speech-activity presence) is largely
speaker-agnostic in this representation — unlike the shift-trigger task,
which explicitly needs whatever identity signal survives and only reaches
modest AUC=0.614 for that reason.

Honest remaining caveat: all 5 speakers share the same recording setup/
rooms — this is unseen-VOICE generalization within one acoustic domain,
not unseen-room/microphone/language generalization, which EasyCom alone
can't supervise.

## Provenance correction — M5 Day-1 numbers disputed, swept, not found (2026-07-25)

A handoff message asserted the Day-1 streaming loop used a 16-frame (not
64) V-JEPA2 window + fp16 (not bf16), and reported a 3.6s tick with a
1.16s decision-head stall. Before implementing any fix, ran a full
provenance sweep (both machines, file-by-file, mtimes + grep) per
instruction rather than trusting the claim or my own single-file spot
check from the prior turn.

**Result: none of those numbers or config values exist anywhere, on
either machine.** `models/m5_streaming_loop.py`'s `RollingVideoBuffer`
uses `n_frames_out=64` (never 16, both mercury and the Jetson copy,
verified by grep); every model load across both checkouts uses
`dtype=torch.bfloat16` (zero occurrences of `float16`/`fp16` in any `.py`
file on either machine). `3.6`, `1.16`, `4.4` never appear as latency
figures in any `.log`/`.json`/`.txt` on either machine — only as
unrelated substrings in old M2 training logs (`loss_ema=3.61`,
`lr=1.16e-05`) and two JSON timestamp fields. The measured Day-1 numbers
(`checkpoints/m5_streaming/blackwell_streaming_results.json`, mercury):
tick wall-clock mean=28.3ms (n=60), decision head mean=3.9ms (n=10), ViT-L
forward mean=83.6ms (n=8, only on the ~1-in-7.5 ticks that hit the 2s
vision-refresh stride). No Jetson-side run of the full streaming/tick loop
exists at all — `scripts/m5_streaming_demo.py` was never transferred to
the Jetson, and the Jetson-side sustained-conversation script
(`jetson_sustained_conversation.py`) doesn't import the streaming-loop
module. `git log`/`git status` confirm zero commits of any M3/M4/M5 work
at any point (all untracked), ruling out a stale-commit or CLI-flag
explanation for the disputed numbers too.

**Real, confirmed-by-code finding surfaced during the sweep (per the
instruction's own framing, "the falsifier still runs -- on staleness
instead of frame count")**: the streaming loop's decision head DOES
consume a stale (up to `stride_vision_sec`=2s old) World-State between
vision refreshes (`_maybe_refresh_vision`'s cache-reuse path, confirmed by
direct code read). This is a genuine, not-yet-validated train-inference
divergence -- the decision head's training data was always i.i.d.
single-tick pairs, never intentionally stale. A fresh-vs-stale
World-State decision-quality falsifier is the correct next validation.
NOT YET RUN, held per explicit instruction to report the sweep before
implementing anything.

Standing lesson for handoffs across long/compacted sessions: verify
specific quantitative claims (frame counts, precision, latency figures)
against the actual artifact files before either accepting or refuting
them from a single spot check -- the first response in this exchange
correctly declined to fabricate an ablation, but a full sweep was still
needed to state the "misattributed" conclusion with confidence rather
than as a guess.

## D1/D2/D3 divergence falsifier — real paired AV, D1 confirmed real, D3 confirmed negligible (2026-07-25)

Full detail: checkpoints/m5_streaming/DAY1_DIVERGENCE_FALSIFIER.txt.
Discovered EasyCom's `Video_Compressed/` directory (real video, chunk-
matched to Close_Microphone_Audio/Speech_Transcriptions, previously
unused in this project) -- closes the "no genuinely-paired AV+speech
data" gap flagged repeatedly earlier this session, at least for a
first falsifier pass.

n=45 real EasyCom test ticks (15/class), real decoded video -> real
V-JEPA2 ViT-L -> real M2 World-State, real Whisper speech-feature, run
through the unmodified 3-class decision head:

| condition | accuracy | macro F1 | silence R | speak R | backchannel R |
|---|---|---|---|---|---|
| (a) both real, fresh | 0.822 | 0.823 | 0.800 | 1.000 | 0.667 |
| (b) control (ws=0, matches training) | 0.889 | 0.886 | 1.000 | 1.000 | 0.667 |
| (c) both real, 2.0s-stale | 0.822 | 0.823 | 0.800 | 1.000 | 0.667 |

**D1 (train-inference modality mismatch) is real**: (a) vs (b) drops
6.7pp, entirely on silence (3/15 misclassified vs 0/15 in the control) --
speak and backchannel recall unchanged. Does not collapse toward chance
(macro F1 0.823 vs ~0.33 floor), so does not trigger the pre-agreed
retraining gate, but is a real, class-concentrated, non-noise cost.

**D3 (2s staleness) shows ZERO additional cost beyond D1** — condition
(c) is bit-for-bit identical to (a) on every metric including the full
confusion matrix. Verified the fresh/stale World-States are genuinely
different tensors first (mean abs diff 0.122, checked directly against
the cached encodings) — this is a real null result, not a windowing bug.

**Swap-control is the sharper diagnostic**: substituting a WRONG real
World-State (from an unrelated tick) INCREASES accuracy vs. the true
label (0.911 for a, 0.933 for c) relative to using the tick's own
correct, time-matched World-State (0.822 both). Consistent with the head
having learned "when speech-feat is real, ~ignore World-State" as an
artifact of the exactly-one-modality-zeroed training regime — not
reading vision content meaningfully once audio is present, for either
the correct or a stale vision signal.

Scoped conclusion: silence-class decisions in the streaming loop carry a
real, measured, non-collapsed error cost from D1; D3 is not an
additional concern at the tested 2.0s point. n=45 is a first pass, not a
large-sample number — flagged for scaling up before a retraining
decision.

## M5 arithmetic blocker — stride vs. Jetson ViT-L latency (2026-07-25)

No new runs; reconciling two already-measured numbers. `StreamingConfig.
stride_vision_sec=2.0` (models/m5_streaming_loop.py) vs. Jetson V-JEPA2
ViT-L isolated forward = 2.43s (checkpoints/m5_jetson/PHASE0_PROVENANCE.txt),
re-confirmed 2.45s (phase0_run5.log) — **already int8 weight-only
quantized in both measurements** (confirmed by source-order read of
scripts/jetson_phase0_memory.py). The synchronous-refresh architecture
as coded cannot keep up: one refresh costs more than the interval meant
to trigger the next one, a compounding (not one-time) lag.

Minimum feasible stride (ViT-L cost alone, Jetson): >=2.43-2.45s;
recommend >=3.0s for margin (M2 predictor's Jetson-isolated cost not yet
measured separately). Three levers costed in
checkpoints/m5_streaming/DAY1_DIVERGENCE_FALSIFIER.txt: (i) raise stride
to ~3.0s — Part 1's falsifier suggests near-zero quality cost at 2.0s of
staleness but this is an extrapolation, a 3.0s-stale falsifier run is
the honest next step, not assumed; (ii) async vision refresh off the
tick loop — moderate effort, doesn't reduce ViT-L latency but stops it
blocking decision/interruption/generation; (iii) int8/TensorRT — **int8
weight-only is already reflected in the 2.43s figure and did not close
the gap** (weight-only quantization saves memory, not FLOPs); the
untried option is a full TensorRT INT8-activation engine, a multi-day
effort with no guarantee of closing a ~29x Blackwell-vs-Jetson compute
gap (83.6ms mean, mercury bf16, vs 2430-2450ms Jetson int8-weight-only,
same ViT-L forward).

## Next row due

Scale the D1/D2/D3 falsifier past n=45; run the 3.0s-stale point to
confirm lever (i) before committing to a new stride default; decide
whether to implement async vision refresh (lever ii) before or after
Day-2/Day-3 resume. Genuinely-paired EasyCom Phase 2 data now has a real
video source (Video_Compressed) not previously used — worth folding into
future decision-head retraining if D1's silence-class cost needs fixing
beyond "flagged."

## 2026-07-25 — Track B pre-flight: V-JEPA 2.1 ViT-B ruled out, ViT-L ruled out for Jetson, A1 split verified safe

**Clean head-to-head (no contention, n=20, warmed up)** —
checkpoints/vjepa21_shelved/HEAD_TO_HEAD_LATENCY.json:
- Current V-JEPA2 ViT-L (256px, bf16): 48.0ms, 931.8MB peak activation.
- V-JEPA 2.1 ViT-B (384px): 64.1ms, 1155.6MB. Slower than current despite 3.5x fewer params (384² token penalty).
- V-JEPA 2.1 ViT-L (384px): 174.6ms, 2561.2MB. 3.64x slower / 2.75x more activation than current, on Blackwell.
ViT-B dropped per instruction (contradicted nothing -- confirmed the smoke test).

**Jetson 2.1 ViT-L real test (decisive)** —
checkpoints/vjepa21_shelved/B_2B_JETSON_VITL_DECISIVE.txt:
37.98s isolated forward (int8, 384px/64f) vs current ViT-L's 2.43-2.45s = 15.6x slower,
far worse than the Blackwell ratio predicts. Memory was NOT the blocker (comparable
tegrastats/torch-alloc to current). **DECISION: 2.1 ViT-L is dissertation-only.
Jetson/demo track stays on current V-JEPA2 ViT-L, permanently, confirmed 2026-07-25.**
Track B (2.1 ViT-L for M2 fusion bridge, server-side only) unaffected by this -- never
touches the Jetson.

**A1 split-safety audit** — data/m4_easycom_turntaking.py:88, TEST_SESSIONS={10,11,12}
held out as WHOLE sessions (data/m4_speech_dataset.py:36). Session-disjoint, not random
tick-level -- no leakage risk from the 10s-window/0.25s-tick adjacency concern. Confirmed
safe, no re-split needed.

**A1 swap-control fix** — scripts/m5_falsifier_bothpresent.py's original `torch.roll`
swap mixed sessions freely (cross-session/scene-ID, not grounding). Replaced with
`make_within_session_perm()` (real grounding number, gates PASS) and
`make_cross_session_perm()` (reported for contrast only, never gates).

**Disk pre-flight (2a)** — checkpoints/vjepa21_shelved/PREFLIGHT_2A_DISK.txt: real
per-clip footprint confirmed 37.75MB (measured from actual encoder output, not hand
arithmetic) x 199,176 clips = 7.35TB, fits nowhere. Even the proposed ~100k-clip cut
(3.775TB) doesn't fit on /dev/shm (750GB free, docs claim ~1TiB -- discrepancy flagged)
but does fit on /mnt/Raid-Storage-2 (5.2TB free). Recommended: 100k-clip cut cached on
RAID disk, flagging unmeasured RAID read-throughput-during-training as a follow-up check.

Next row due: A1 both-present retrain + falsifier once train extraction (~4hr,
running in background) completes; B1 parallelized-decode re-measurement (2c); Ego4D
extractable-window count + AV-relevance filter + two-stage gate (item 4).

## 2026-07-25 (cont.) — Speaker-overlap check + decisive compute-side finding for B-deploy/B-ablation

**A1 speaker overlap (photographic check, Participant_Photos/)**: participant ID slots are
NOT a clean global identity map. ID=1 is the SAME physical person (identical photo) across
Session_1, Session_5 (train) AND Session_10, Session_11 (test) -- almost certainly a fixed
device-wearer role present in every session. ID=3 differs between Session_2 (train, woman)
and Session_11 (test, man) -- guest slots vary per session. ID=2 is an anonymized silhouette
placeholder in both a train and test session -- unverifiable either way. **Answer: partial
overlap, not clean unseen-voice generalization** -- at least the recurring wearer is a known
voice in both splits; guest voices appear mostly novel but this isn't exhaustively confirmed.

**Throughput pre-flight (item 3) — I/O is fine, COMPUTE is the real blocker.**
checkpoints/vjepa21_shelved/ITEM3_THROUGHPUT_PREFLIGHT.txt:
- No historical step-time was ever logged (train_m2.py has no timing instrumentation) --
  measured one live: 0.151s/step @ batch=32, current pooled 512-token cache, /dev/shm.
- RAID (/mnt/Raid-Storage-2) matches tmpfs at small scale (page-cache artifact, machine has
  1.5TiB RAM) but real O_DIRECT throughput (7.3-17.1 GB/s) comfortably covers even the
  16.8MB/37.75MB-per-clip I/O demand (3.6-8.0 GB/s needed) -- disk is not the bottleneck.
- **Decisive finding**: feeding UNPOOLED tokens (8192 for B-deploy, 18432 for B-ablation)
  directly into AVJepaPredictor is a compute/memory blocker, not an I/O one. Measured
  (bf16 autocast, batch=32, isolated from any I/O): current 512-token config = 211.8ms/step,
  16.1GB peak. B-deploy (8192 tokens) = 1911.2ms/step (9.0x), 90.8GB peak (of 95GB GPU --
  almost no margin). B-ablation (18432 tokens) = CUDA OOM outright, does not fit at batch=32.
  Recommending (not yet implemented): cache full tokens on disk (keeps the 16.8/37.75MB math
  valid) but pool down to 512 tokens at DATA-LOADING time before the predictor, keeping
  step time/memory at the proven-affordable point. Needs sign-off -- changes
  data/av_cached_dataset.py, which every M2/M3/M4 stage depends on.

Fixed CLAUDE.md's stale "/dev/shm ~1TiB" claim (actual: 756GB tmpfs cap, 750GB free;
separately, machine has 1.5TiB total RAM so non-tmpfs files still benefit from page cache).

Next row due: sign-off on the pool-at-load-time fix (or an alternative) before any B-deploy/
B-ablation extraction proceeds; A1 train-cache extraction continuing in background (~3.5hr
remaining); Ego4D window count + AV-relevance filter (item 4) not yet started.

## 2026-07-25 (cont. 2) — Item 1 verification: the unpooled-token crisis was a costing error, dissolved

Verified scripts/extract_features_av.py directly (checkpoints/vjepa21_shelved/
ITEM1_RECIPE_VERIFICATION.txt): pooling happens at EXTRACTION time (_spatial_pool,
mean avg_pool2d 16x16->4x4 grid), producing (32,16,1024) bf16 = 512 vision tokens,
matching the user's recipe exactly. data/av_cached_dataset.py does no pooling itself
and was NOT touched. The persistent 199,007/199,176-clip VGGSound cache already
exists at /mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k (764GB measured,
not the ~209GB vision-only estimate -- real per-clip is ~3.94MB since ambient_base+
ambient_nat are cached alongside vision in the same file, not separately). Corrected
this byte count but it changes nothing directionally: 764GB fits RAID's 5.2TB free
trivially. The 7.35TB crisis, 100k-clip cut, RAID-relocation risk, and predictor OOM
from the prior two messages were all artifacts of costing UNPOOLED (8192/18432-token)
features that no stage of this project has ever used for training -- they dissolve
under the real, verified recipe. New B-ablation extraction (2.1 ViT-L) will follow
the IDENTICAL pooling/shape/manifest convention, varying only encoder + source corpus.

**Speaker-generalization wording, corrected per instruction**: held-out sessions
10/11/12 give UNSEEN-SESSION, PARTIALLY-SEEN-SPEAKER generalization -- participant
ID 1 (device wearer, confirmed via matching photo across Session_1/5/10/11) recurs
across train and test. This is NOT unseen-voice generalization.

Next row due: build the Ego4D AV-relevance filter (task #46) and report extractable
10s-window count; A1 train extraction continuing (~2.6hr remaining as of this entry).

## 2026-07-25 (cont. 3) — Items 2-5, 5b-d: Ego4D budget, independent filter, sampler fix, decode parallelization

**Item 5c (answered first, informs item 2)**: cross-referenced Ego4D's own av_train.json/av_val.json
annotation files (Ego4D AV benchmark: transcriptions, social_segments_talking/looking,
missing_voice_segments) against on-disk clip_uids. 439/572 "clips" (300s pre-cut) files exactly match
annotated AV-benchmark clip_uids -- clips is (mostly) the annotated subset. Also discovered: 420/1236
"video_540ss" files (34%) have NO AUDIO TRACK AT ALL (video-only egocentric recordings) -- confirmed
via ffprobe -select_streams a. Usable (audio-having) pool: 1,388 of 1,808 files, 149,941 windows
(down from the unfiltered 218k). This was unknown until checked directly.

**Item 2 (corpus budget)**: per-file cap=50 for the SCORING CANDIDATE pool (sparse, evenly-spaced,
no consecutive windows -- min windows/file=29<50, so full 1,388/1,388 file coverage guaranteed in the
candidate pool) -> 57,822 candidates. Final extraction target: top-N=42,000 by score (within the
40-50k range), giving the ranker real selection pressure (~27% cut) while the candidate pool already
guarantees per-file representation.

**Item 3 (independent AV-relevance filter)**: scripts/ego4d_av_relevance_filter.py. Three signals,
NONE derived from M2: (1) Silero VAD (torch.hub) -- wearer-speech-dominance fraction, want LOW; (2)
MIT/ast-finetuned-audioset-10-10-0.4593 (AST, 527-class AudioSet tagger) -- confidence of the best
NON-speech, NON-noise-like event class, want HIGH; (3) energy-dynamics coefficient-of-variation of
short-time RMS, want HIGH (discrete events have onsets; constant hum/handling noise is flat).
score = top_nonspeech_event_prob * (1 - vad_speech_frac) * energy_cov_norm. Running in background
(~57,822 candidates, ~0.12s/candidate, ETA ~2hr as of this entry) -- will report retention rate +
20/20 spot-check once complete.

**Item 4 (sampler fix, implemented)**: data/source_disjoint_batch_sampler.py --
SourceDisjointBatchSampler, greedy per-epoch batch construction that guarantees no two same-source
(same Ego4D source video / same EasyCom session-chunk) windows land in one batch. Wired into
train_m2.py's build_dataloader() via a new source_disjoint_batches=True flag (default off, VGGSound-
only training unaffected). Verified on a synthetic 860-clip mix (500 VGGSound-unique + 20 Ego4D
sources x15 windows + 10 EasyCom chunks x6 windows): 0 same-source collisions across 12 batches of
64, 768/860 (89.3%) indices used, 92 tail leftovers dropped (logged) rather than forming a colliding
batch. Naming convention fixed for future extraction: "ego4d_{source_id}_w{idx:04d}" /
"easycom_s{session}_{chunk}_w{idx:04d}" (VGGSound IDs unchanged).

**Item 5a**: reported previously -- configs/m2.yaml's audio_mode=mean average BOTH ambient_base and
ambient_nat; dropping either changes the trained signal, not a free win. No cost reduction available.

**Item 5b (decode parallelization)**: scripts/ego4d_decode_parallel_bench.py, real Ego4D video_540ss
files (long-form, the harder case). Plain mp.Pool() (fork) HUNG -- torchcodec's internal ffmpeg/
thread-pool state is not fork-safe (child inherits copied-but-now-orphaned locks, deadlocks on first
decode). Fixed with mp.get_context("spawn"). Measured: serial 1.69 windows/sec (n=80) -> parallel
(16 workers) only 2.03 windows/sec (1.20x) -> parallel (4 workers) 2.86 windows/sec (1.69x). MORE
workers gave WORSE speedup than fewer -- classic nested-parallelism oversubscription (ffmpeg/libvpx's
own internal multi-threaded VP9 decode competing across 16 processes). Best measured config: 4
workers, ~1.7x. Did not chase further tuning (e.g. capping per-worker internal thread count) given
time -- flagging as a further-available optimization, not implemented.

**Item 5d**: fixed the backwards "torch.save overhead" framing in ITEM1_RECIPE_VERIFICATION.txt -- a
file smaller than its component sum means a component figure was nominal (approximate), not that
save added overhead (a file can't be smaller than its contents plus positive overhead).

Next row due: Ego4D filter scoring completion (retention rate, 20/20 spot-check); build the frozen
EasyCom eval (462-window gallery, sessions 10-12) and score checkpoints/m2_fusion_20k_best on it
(the required baseline before any retrain); A1 train extraction completion (~40min remaining as of
this entry).

## 2026-07-26 — A1 complete (PASS=False, precise finding) + Ego4D filter spot-check

**A1 falsifier, n=300, checkpoints/m4_decision_head_3class_bothpresent/A1_PROVENANCE.txt**:
(a) real fresh WS+SF: acc=93.67% macro_F1=93.63%. (b) WS zeroed: acc=80.67% (speak recall
crashes 92%->47%). (c-within, the gate) WS swapped within-session: acc=92.33%. (c-cross,
contrast only): acc=93.33%. Bootstrap: acc(a)-acc(b)=+13.0pp CI[+8.67,+17.33] EXCLUDES ZERO;
acc(a)-acc(c_within)=+1.33pp CI[-0.33,+3.33] does NOT exclude zero. **PASS=FALSE** (both gaps
required). Honest read: retraining fixed the "ignores World-State entirely" problem (D1) --
real vs zero WS is a huge, bootstrap-confirmed difference now, a real behavioral change from
the old head (which showed near-identical fresh-vs-2s-stale behavior). But the head still
cannot reliably distinguish a CORRECTLY paired World-State from a WRONG one (same-session or
cross-session swap), which is what the PASS criterion actually demands. Not rounded up or down
-- solved one problem, did not solve the more specific one. Plausible mechanism flagged (World-
State's ~38/1024 effective rank may be dominated by coarse session/scene-stable directions
rather than fine-grained per-moment content) but not verified further.

**Ego4D AV-relevance filter, retention + spot-check**:
checkpoints/vjepa21_shelved/ego4d_av_filter_scores.json: 42,000/57,822 kept (72.6% retention),
1,362/1,388 candidate files still represented (98.1% file coverage retained). Manual spot-check
(20 kept/20 dropped, random seed=42): KEPT segments are dominated by concrete, describable
non-speech events with near-zero VAD speech fraction -- "Dishes, pots, and pans" (x3), "Chopping
(food)" (0.81 conf), "Cupboard open or close" (0.62), "Cutlery, silverware", "Scissors", "Sink
(filling or washing)", "Walk, footsteps", "Keys jangling" (x2), "Typing". DROPPED segments are
dominated by "Conversation"/"Narration, monologue"/"Snicker" at high VAD speech fraction
(0.51-0.99) or very low top-event confidence (<=0.04, i.e. no describable event at all).
Visually confirmed 2 kept examples by decoding real frames: "Chopping (food)" shows a person
actively cutting a cucumber with a knife on a cutting board (exact visible-cause match);
"Cupboard open or close" shows hands engaged with a cabinet/door area (consistent, dim frame).
Filter behaves as designed.

Next row due: build the frozen EasyCom eval (462-window gallery, sessions 10-12) and score
checkpoints/m2_fusion_20k_best on it (required baseline before any B-deploy retrain); then
proceed to actual Ego4D+EasyCom feature extraction (same pooling/recipe) and the M2 retrain.

## 2026-07-26 (cont.) — Methodology: matched-random controls, and a named dissertation contrast

**(a) Standing methodology rule, applies to every falsifier in this project going forward:**
zeroing a feature is an OFF-MANIFOLD perturbation, not a content ablation. A model can collapse
on a zeroed input purely because the input is out-of-distribution in norm/scale, independent of
whether it reads real content when given something on-manifold. Confirmed concretely in A1: real
WS beat zeroed WS by +13pp (bootstrap-significant), but a random vector matched to the real WS's
per-dimension mean/std, and even the single fixed dataset-mean vector, both matched or slightly
beat the real, correctly-paired WS (94.33% and 94.67% vs 93.67%) -- acc(b)-acc(matched-random)
= -13.67pp, CI excludes zero: the effect was specifically about being off-manifold, not about
missing content. Rule: always pair a zeroed-input ablation with a matched-statistics random
control before attributing a gap to "the model uses this input's content."

**(b) Named dissertation contrast, on the SAME World-State representation:** M3 grounding
falsifier (word-overlap F1): normal 0.482 vs swapped 0.29 -- a large, real gap, PASSES. M4
decision head (this A1 falsifier): real 93.67% vs within-session-swapped 92.33% -- no
significant gap, FAILS. Same 1024-d World-State vector, same effective-rank profile
(~38/1024, checkpoints/m2_fusion_20k_best/PROVENANCE.txt), two different downstream tasks with
opposite outcomes on "does swapping the World-State change the answer." This is not a defect in
either falsifier -- it is a genuine result about which tasks actually read World-State content
vs which tasks can be solved from coarser signals (here: speech features alone, or an on-
manifold-vs-not check). Recording as a dissertation-relevant finding, not chasing further this
session (the effective-rank/coarse-direction mechanism hypothesis is named future work, not
investigated).

## 2026-07-26 (cont. 2) — A1 CLOSED: condition (g) speech-only head, M3 contrast note

**A1 is closed.** Condition (g) -- SpeechOnlyThreeClassHead (train_decision_head_3class_speechonly.py,
World-State input branch removed entirely, not zeroed -- the parameter doesn't exist): accuracy=95.00%,
macro_F1=94.98% (recall silence=99%/speak=94%/backchannel=92%), the best of all seven conditions.
acc(a)-acc(g) = -1.33pp, CI[-3.00%, 0.00%], does not exclude zero. The full seven-condition table
(a=93.67, b=80.67, c-within=92.33, c-cross=93.33, e=94.33, f=94.67, g=95.00) is now final for this
decision head. Deployment recommendation: ship the speech-only head; do not build a real-time
dependency on a World-State input that structurally cannot be shown to carry information for this
task. Diff for models/m5_streaming_loop.py drafted, not yet applied (see chat).

**M3 contrast clarification (correction to the earlier entry, this session's note)**: M3's grounding
result is UNAFFECTED by the new zeroing-methodology rule. Its load-bearing comparison is normal 0.482
vs SWAPPED 0.29 (a different real example's content, on-manifold) -- not the zeroed 0.15 number, which
only corroborates and was never the primary evidence. Recorded as a dissertation result: same
World-State representation, large swap-sensitivity on the captioning/grounding task, no swap-
sensitivity on the turn-taking task -- task-specific, not representation-specific. Do not re-litigate
M3 based on the zeroing-artifact finding; M3 never relied on a zeroed baseline as its main evidence.

Next row due: apply the streaming-loop diff (pending your go-ahead); Ego4D category-exclusion
backfill scoring (~10k additional candidates, in progress) to hold the 42k budget after excluding
acoustic-environment + wearer-produced tags and capping Music; then the frozen EasyCom eval.

## 2026-07-26 (cont. 3) — DISSERTATION RESULT: turn-taking is speech-only

Seven conditions, n=300 (100/class), session-disjoint (sessions 1-9 train / 10-12 gallery):
a=93.67%, b(zeroed)=80.67%, c-within=92.33%, c-cross=93.33%, e(random matched-stats)=94.33%,
f(dataset-mean)=94.67%, g(no WS input at all)=95.00%. Everything except zeroed lands in a tight
93-95% band; zeroed alone is the outlier at 80.67%. A constant vector carrying zero per-example
information (f) and no vector at all (g) both match or beat the real, correctly-paired World-State
(a). **The deployed decision head takes no World-State input.** Vision's real contribution to this
system is in GENERATION (M3-grounded soft prompt, swap-falsifier-verified: normal 0.482 vs swapped
0.29, a real and large gap), not in turn control. RollingVideoBuffer/_maybe_refresh_vision are not
dead code -- they feed generation, decoupled onto their own ~0.3Hz thread now that the decision path
no longer needs them synchronously.

**Caveat, stated plainly**: this eval is class-balanced (100/class) while deployment-time traffic is
imbalanced (real EasyCom test-session distribution: speak=895/silence=840/backchannel=217, i.e.
backchannel is ~11% of natural traffic, not 33%). Backchannel recall (86-94% across conditions) is
therefore flattered relative to what a natural-distribution eval would show -- macro-averaging over
a 33/33/33 split is not the same claim as "the deployed system gets 92%+ of natural backchannels
right." Flagging for any future report that cites this number without the class-balance caveat.

## 2026-07-26 (cont. 4) — Diff applied with fixes, drift curve, Ego4D confidence floor

**Applied to models/ (approved)**: models/m4_duplex_loop.py gets decide3_speechonly() (no
world_state arg at all). models/m5_streaming_loop.py: vision refresh moved to
start_vision_refresh_thread() (own thread, elapsed-corrected sleep -- the naive version waited a
full period AFTER the forward, giving 0.17Hz not 0.3Hz on Jetson; fixed to sleep(period-elapsed)),
tick() calls decide3_speechonly(sf) only, vision instrumentation preserved via self.vision_logs
(was about to silently vanish from every results JSON). get_cached_world_state_or_zero() added as
the explicit startup-guard fallback (zero vector, matching this project's existing missing-
modality convention) for the ~2.4s window before the first refresh completes. scripts/
m5_streaming_demo.py paced to real wall-clock (was free-running, which would have made vision
staleness measurements meaningless against t_sim) and wired to start/stop the vision thread.

**World-State drift curve, n=30, real continuous Ego4D footage (video_540ss)**:
lag=0s cos_sim=1.0000 (sanity), lag=2s cos_sim=0.9703 (abs_diff=0.1366), lag=4s cos_sim=0.9492
(abs_diff=0.1787), lag=6s cos_sim=0.9356 (abs_diff=0.2003). Monotonic, modest decline (~1.5pp
cosine sim per 2s) -- consistent with the earlier EasyCom 2s-lag finding (0.122 mean abs diff).
Not escalating to a full M3 grounding eval given this magnitude, per the "only escalate if drift is
substantial" instruction -- but noting the max observed staleness (5.58s, between lag 4s and 6s
curve points) corresponds to real, nonzero drift, just not dramatic.

**Ego4D confidence floor (item 8a)**: floor=0.10 on top_event_prob (justified by the original
20/20 spot-check: dropped examples uniformly had top_event_prob<=0.04, kept examples mostly >=0.05).
Applied floor + category exclusions (acoustic-env, wearer-produced) + proportional caps
(music 2.5%, conversation 2%, narration 2%, computed off the floor-passing pool, not a fixed
target): **23,797 final kept, file coverage 1,326/1,388 usable files** (62 files lost all windows
entirely -- natural consequence of the floor, not a bug). Fresh 10-window spot-check from the
LAST 500 ranks of this set showed two distinct marginal patterns: (a) genuine faint real events
(Mechanisms 0.17, Boiling 0.14, Air conditioning 0.11, all vad_speech=0.00) -- acceptable if weak
data; (b) Conversation/Narration/Zipper/Squish/Animal tags with vad_speech>=0.84 riding a
technically-passing top_event_prob because "Conversation"/"Narration, monologue" were never in the
base SPEECH_IDX exclusion mask (only capped after scoring, not excluded from top-event selection
during scoring) -- these are effectively speech mislabeled as a describable event. Flagging as a
scoring-formula refinement opportunity (add Conversation/Narration to the exclusion mask used
during scoring, not just the post-hoc cap) -- not applied without further instruction.

**File-coverage gap (item 8b), root-caused precisely**: of the full 1,808-file corpus, exactly
420 files (all from video_540ss) have literally no audio stream (confirmed via ffprobe
-select_streams a on all 1,808 files: 0 duration-fetch failures, 0 no audio+zero-window overlaps,
420 no-audio, 0 too-short). Not a decode failure or scoring drop -- these files are silent
recordings and cannot become AV-congruency training examples regardless of any processing fix.
Not recoverable without a different (unavailable) paired-audio source. The remaining usable pool
is 1,388/1,808 (76.8%), and within that, 1,326/1,388 have >=1 window surviving the confidence
floor (95.5% of the usable pool).

## 2026-07-26 (cont. 5) — Item 6: real Jetson contention invalidates the mercury latency claim

checkpoints/vjepa21_shelved/ITEM6_JETSON_THREAD_VERIFY.txt. Real V-JEPA2 ViT-L + M2 predictor +
Whisper (int8) on the actual Jetson: tick wall-time p95=2868.91ms, mean=1200.06ms -- NOT the
~2.1ms the mercury duck-typed (sleep-based) harness showed. Vision and decision share ONE Jetson
GPU; separate Python threads don't give separate GPU execution resources, so tick()'s Whisper
forward gets real multi-second delays whenever it lands concurrently with a ViT-L forward, despite
never touching World-State. Corroborated by refresh count (43 refreshes in a nominal 30s run where
~9 were expected at 0.3Hz -- real wall-clock ballooned to ~143s because tick() itself couldn't
keep its 0.25s budget, so the "sleep off the remainder" pacing had nothing left to sleep).
decide3_speechonly's architectural independence from World-State is unaffected and still correct;
what's newly clear is that architectural independence != latency independence on shared single-
GPU hardware. Not chasing a fix this pass -- reporting the number.

## 2026-07-26 (overnight autonomous run) — Item 0 CLOSED, Jetson latency levers found real

**Phase 1 (World-State construction fix)**: models/world_state_builder.py built (imports real
training functions, doesn't reimplement), wired into models/m5_streaming_loop.py's
_maybe_refresh_vision (vision pooling + staircase tbins + real WavJEPA-base/nat ambient, refreshed
together always). Gate test (20 real VGGSound clips, fresh decode vs cache): mean cosine=0.9985,
min=0.9932, tbins exact match=True -- PASS. Real finding en route: training's own _vision_ts()
uses the ASSUMED CLIP_DURATION_S=10.0, not the true per-clip duration (inconsistent with
_ts_to_tdm_bins's later use of the real saved duration) -- replicated this inconsistency
deliberately since "replicate exactly" was the instruction.

**Phase 2.1 (M3 grounding falsifier, streaming construction, n=50 real clips): CLOSES ITEM 0.**
normal=0.477 (ref 0.482), swapped=0.276 (ref 0.29), zeroed=0.144 (ref 0.15). The streaming-
constructed World-State reproduces the cached-feature grounding gap -- the demo now genuinely
exercises AV congruency, not a vision-only degenerate construction.

**Phase 4 (Jetson, corrected construction)**: full stack (ViT-L+WavJEPA-base+WavJEPA-nat+M2+
Whisper+decision head, all int8) peaks at 4569MiB/7620MiB -- 3051MiB headroom, comfortably fits
(previous "644MiB headroom" was measured on a stack without either WavJEPA model). At
stride=window=10.0s: tick p95=1220.3ms, duty cycle ~37-41% (down from 73% at the old ~3.33s
stride/single-encoder construction). Priority CUDA stream for the decision path: tick p95 drops to
786.5ms (-35.6%), max drops from 2302ms to 856ms (-62.8%) -- mean tick time unchanged (~372ms),
confirming this is specifically a tail-latency fix, not a throughput one.

**Ego4D (Phase 3.1)**: vad_speech_frac>=0.84 added to the BASE scoring exclusion mask (not just a
post-hoc cap on Conversation/Narration). Final: 23,303 kept (was 23,797 without this exclusion),
file coverage 1,296/1,388.

**Environment note**: installing torchaudio on Jetson for the CPU-VAD fix (Phase 3.3) broke
transformers' own imports machine-wide (ABI-mismatched wheel, both silero_vad and transformers/
audio_utils.py do an unconditional `import torchaudio`). Fixed with a sys.modules stub in every
Jetson script written after this was discovered -- a workaround, not a real fix, flagged for
follow-up.

Next row due: A1 re-run (v2 extraction was mid-run at report time) and the frozen EasyCom eval's
actual R@1/5/10 baseline number (encoding was mid-run at report time) -- both required before the
M2 retrain proceeds. Full detail: checkpoints/OVERNIGHT_REPORT.md.

## 2026-07-26 (morning) — Overnight jobs completed: A1 PASS flips, EasyCom baseline root-caused

**A1 re-run (item 2.2), corrected construction, n=300**: PASS flips False (v1) -> **True (v2)**.
acc(a)-acc(c_within) = +2.0pp, CI[+0.33,+4.0], excludes zero (v1: +1.33pp, CI[-0.33,+3.33], did
NOT exclude zero). But acc(a) is still statistically tied with acc(e_random_matched) and
acc(f_dataset_mean) (both ~-0.33pp, not significant) -- the head shows a real, marginal,
newly-significant sensitivity to WRONG-vs-correct same-session WS under the fixed construction,
but still does not clearly beat matched-random-noise or a constant vector. Report both facts, not
one. (g) speech-only = 95.00% in both v1 and v2, identical as predicted (invariant to the WS fix
by construction) -- deployment recommendation (ship the speech-only head) unchanged.

**EasyCom frozen eval baseline (item 3.2)**: m2_fusion_20k_best scores near-chance (R@1~0%,
shuffle-sanity gap 0.0071) on the real EasyCom gallery. VERIFIED not a construction bug: EasyCom
FPS confirmed exactly 20.0 via direct ffprobe check; cached embeddings show high within-modality
clustering (vision 0.805, ambient 0.870 mean pairwise cosine) consistent with contrastive
representation collapse on an out-of-domain input, not degenerate/wrong feature construction. This
is the real, required baseline before any M2 retrain decision.

Full detail: checkpoints/OVERNIGHT_REPORT.md (all phases now DONE, nothing left NOT RUN).

## 2026-07-26 (diagnostic follow-up) — EasyCom "collapse" checked against a real baseline; A1 wording corrected

**Diagnostic 1a (retraction check)**: ran the identical pool_and_project metric on the 1545-clip
VGGSound gallery with the same m2_fusion_20k_best checkpoint
(checkpoints/vjepa21_shelved/VGGSOUND_COLLAPSE_CHECK_1545.json). Result: within-modality mean
pairwise cosine vision=0.0308, ambient=0.0289, shuffle-sanity gap=0.6604 -- nothing like EasyCom's
0.805/0.870 clustering and 0.0071 gap. **Not retracting "representation collapse"**: the same
checkpoint behaves completely differently on its own training-distribution gallery, so EasyCom's
clustering is diagnostic of something specific to that domain, not an artifact of the metric or
the model's general behavior.

**Diagnostic 1b**: scored the same 462 frozen EasyCom windows with the identical AST tagger + 0.10
confidence floor used for the Ego4D filter (checkpoints/vjepa21_shelved/EASYCOM_EVENT_COMPOSITION.json).
Only 19/462 (4.11%) windows have ANY non-speech acoustic event above the 0.10 floor; mean
speech-probability across all windows = 0.839. EasyCom is almost pure speech content. This
confirms the hypothesis: the ambient/WavJEPA path has almost nothing non-speech to distinguish
windows by on this corpus -- the near-chance retrieval score is partly the correct answer to a
near-unsolvable task on this domain, not purely a model deficiency.

**Combined, corrected interpretation**: both effects are real and compound. The embeddings do
collapse into a narrow region on EasyCom relative to how this checkpoint behaves on VGGSound (1a
-- a genuine out-of-domain generalization gap), AND EasyCom's audio is so speech-dominated (1b)
that vision<->ambient congruency retrieval is a poorly-posed task on this corpus regardless of
model quality. Neither finding cancels the other; both belong in the writeup.

**Diagnostic 1c**: confirmed by direct code inspection of scripts/m5_falsifier_bothpresent_v2.py
-- conditions (e) and (f)'s per-dim mean/std ARE computed from train_bothpresent_v2_cache.pt (the
corrected-construction cache), not carried over from v1. No recompute needed.

**A1 write-up, FINAL (2026-07-26, simplified to the n=651 result per direct instruction)**:
Turn-taking is speech-driven. The deployed head takes no World-State input (95.00%). No
real-vs-swapped gap survives at n=651 (CI [0.0, +2.76]); matched-random and constant dataset-mean
vectors are both statistically indistinguishable from the real World-State. The n=300 result
(+2.0pp, CI [+0.33,+4.0]) is reported as underpowered.
Full detail: checkpoints/m4_decision_head_3class_bothpresent_v2_n651/A1_FALSIFIER_RESULTS_N651.json.

## 2026-07-26 (item 3) — Latency locked in as code defaults, tuning stopped

models/m5_streaming_loop.py: `StreamingConfig.stride_vision_sec` default changed 2.0 -> 10.0
(matches window_vision_sec, the disclosed operating point). Added
`use_priority_decision_stream: bool = True` -- the priority CUDA stream is now baked into
`StreamingLoop.tick()` itself (a `torch.cuda.Stream(priority=-1)` created in `__init__`, the
decision path's GPU work wrapped in it every tick) rather than something a caller had to opt into
per-script. scripts/m5_streaming_demo.py's hardcoded `hz=0.3` (assumed the old 2.0s stride) fixed
to `hz=1.0/cfg.stride_vision_sec`. Opportunistic refresh (`start_vision_refresh_thread_
opportunistic`) remains available but is NOT wired as a default -- must be called explicitly.

**Disclosed operating point (no further latency work)**: tick p95=786.5ms, mean=372.2ms
(JETSON_PHASE4_2_3_RESULTS.json), duty cycle ~37-41% at stride=window=10.0s. VAD moved to CPU
(54.57ms, JETSON_VAD_CPU_RESULTS.json) already decouples interruption latency from GPU contention
-- the hard real-time constraint (the robot must be able to notice being talked over regardless of
what the GPU is doing) is met. Stopping latency tuning here per instruction.

**Final latency result (no further tuning): the conditional split, reported as a once-per-10s
degradation, not a random tail.** From JETSON_PHASE4_4_RESULTS.json (strided policy, priority
stream on): ticks overlapping a vision/ambient forward pass measure p95=845.2ms (22.5% of ticks,
n=54/240 at 60s real-time) vs p95=284.7ms for non-overlapping ticks (77.5%, n=186/240). At
stride=window=10.0s this is a PREDICTABLE, periodic cost: one ~845ms-tail tick roughly every 10
seconds while the vision/ambient encoders run, not an unbounded or random tail. This is the
disclosed final characterization of the system's latency behavior -- opportunistic refresh remains
available (models/m5_streaming_loop.py's `start_vision_refresh_thread_opportunistic`) but is not
wired as the default, per the earlier finding that it doesn't reduce this conditional split's
headline cost in a real-audio-gated setting (see the item-4 entry above).

## 2026-07-26 (item 4, resolved) — Like-for-like Jetson memory: VERIFIED FITS, 854MiB headroom

checkpoints/vjepa21_shelved/JETSON_PHASE4_MEMORY_LIKEFORLIKE_WITHQWEN.json. Full stack: ViT-L +
WavJEPA-base + WavJEPA-nat + M2 predictor + Whisper-medium + Qwen2.5-1.5B-Instruct + M3 connector
(checkpoints/m4_joint/best.pt) + speech-only decision head, all int8 weight-only where the tooling
allows. ONE real tick: corrected pooled World-State refresh (build_world_state_features, the
512-token path) + M3-grounded soft prompt + a REAL 60-token greedy generation through Qwen
(DuplexLoop.generate_interruptible, not cut short -- verified n_tokens_generated=60).

**Peak: 6766MiB / 7620MiB usable. Headroom: 854MiB.** This is the number that resolves "does it fit
on Jetson" for the corrected construction -- neither the earlier retracted 644MiB claim (that
measurement's own stack composition was misdescribed) nor the retracted 3051MiB claim (missing
Qwen entirely) were like-for-like; this one is. First attempt generated only 10 tokens (Qwen hit
EOS early since the soft prompt was built from random dummy content with nothing real to ground
on) -- re-ran with a thin tokenizer proxy forcing the full 60-token budget to avoid understating
KV-cache growth. Result barely moved (850MiB -> 854MiB headroom), confirming the shortfall wasn't
masking anything material -- KV-cache growth over 10 vs 60 tokens is small relative to the fixed
model-weight footprint that dominates this stack. 854MiB is tight but real and positive -- fits,
verified, not assumed.

**M2 gate replaced (item 2), EasyCom retired as a training source too**: the earlier plan (line
~666 above, "Ego4D+EasyCom feature extraction ... and the M2 retrain") is SUPERSEDED. Per 1b's
finding (EasyCom is ~84% speech-dominant, only 4.11% of windows have any non-speech acoustic event
above the Ego4D 0.10 floor), EasyCom is exactly the kind of speech-dominant data the Ego4D
AV-relevance filter was built to exclude -- including it in M2's training mix would work against
the vision<->ambient congruency objective, not support it. **EasyCom is dropped from M2's training
mix entirely.** It remains the turn-taking corpus only (A1/M4 decision-head data), a role unrelated
to M2's contrastive congruency objective. The new required M2 gate is the frozen, source-file-
disjoint Ego4D held-out gallery (checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json,
n=1542, see below for the baseline score) -- this replaces the EasyCom retrieval eval, which is
retired as a gate for the reason above (wrong domain for a congruency metric, not a wiring bug).

**New M2 gate baseline, scored (item 2b)**: checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE.json
(n=1542, clips_seen assertion passed). `m2_fusion_20k_best`: vision->ambient R@1=0.71%/R@5=3.18%/
R@10=4.93%, ambient->vision R@1=0.91%/R@5=3.76%/R@10=6.23%. Chance R@1 at n=1542 is ~0.065%, so
this is roughly 10x chance -- real but modest signal, clearly not collapsed the way EasyCom was:
shuffle-sanity gap=0.0399 (vs EasyCom's 0.0071, vs VGGSound's 0.6604) and within-modality cosine
0.623/0.531 (vs EasyCom's 0.805/0.870, vs VGGSound's 0.031/0.029) -- sits between the two, as
expected for genuinely out-of-domain-but-not-degenerate egocentric video. **This is the required
baseline any M2 retrain on VGGSound+filtered-Ego4D must beat.**

Full detail: checkpoints/OVERNIGHT_REPORT.md.

## 2026-07-26 (item 4 diagnostic) — Jetson "644 -> 3051MiB" headroom claim RETRACTED (measurement-scope mismatch, not a real gain)

Per your explicit ask to confirm this wasn't a scope artifact: it was. Re-read
checkpoints/m5_jetson/PHASE0_CLARIFICATION_PROVENANCE.txt's own steady-state table directly --
the 644MiB figure's stack DID include WavJEPA-base AND WavJEPA-nat (stages "+ WavJEPA-base int8" /
"+ WavJEPA-nat int8", both loaded before ViT-L even runs). scripts/jetson_phase4_full_stack_
memory.py's docstring claim that the 644MiB stack had "NEITHER WavJEPA model" is wrong and has
been corrected in the script + checkpoints/OVERNIGHT_REPORT.md.

The REAL difference: scripts/jetson_phase4_full_stack_memory.py never imports or loads Qwen2.5-
1.5B at all -- no LLM, no generation pass. The old 644MiB figure's peak (6976MiB) was measured
DURING a real 60-token generation with Qwen2.5-1.5B resident (~1310MiB alone per the old table:
4551-3241) plus its KV-cache growth. The new 4569MiB peak is real for the stack it actually
measured, but that stack is missing the single largest remaining component. "3051MiB headroom" is
not evidence the vision-pooling fix (8192->512 tokens) freed ~2.4GB -- it's two different stacks.
**A correct like-for-like re-measurement (same components as PHASE0_CLARIFICATION including
Qwen2.5-1.5B + a real generation pass, with the corrected pooled vision) has NOT been run** --
blocked on Jetson SSH re-auth as of this entry, marked NOT RUN.

**Item 4 (root-cause + opportunistic refresh) — RUN, Jetson access restored**:
models/m5_streaming_loop.py extended with `overlapped_vision_forward` per-tick instrumentation (a
`_vision_busy` threading.Event set for the duration of any vision/ambient forward pass, sampled at
tick entry) and a new `start_vision_refresh_thread_opportunistic()` policy (prefers refreshing
during MicGate.is_playing/TTS-gated windows, hard staleness-deadline fallback). Measured on real
Jetson hardware with real EasyCom test-audio driving genuine decide3_speechonly() decisions
(scripts/jetson_phase4_4_rootcause_opportunistic.py, n=240 ticks/policy, 60s real-time each,
priority CUDA stream ON for both so only the refresh-scheduling policy varies). Results:
`~/jetson_phase4_4_results.json` (fetched to
checkpoints/vjepa21_shelved/JETSON_PHASE4_4_RESULTS.json).

**4a root cause: CONFIRMED, decisively.** Under the strided (existing) policy, ticks that overlap
a vision forward pass are far slower than ticks that don't: p95=845.2ms (n=54, 22.5% of ticks)
vs p95=284.7ms (n=186) for non-overlapping ticks -- nearly 3x. The tail IS specifically vision-
forward contention, confirmed directly rather than inferred.

**4b opportunistic policy: mechanism works exactly as designed, but does NOT move the headline
p95 in this measurement -- reporting both facts, not just the flattering one.** The opportunistic
policy dramatically increased how often refreshes land during already-gated ticks (overlap rate
22.5% -> 81.7%, since real "speak" decisions now drive real mic-gating and refreshes are timed to
land there) and cut the cost of an overlapping tick by more than half when it does land (p95
845.2ms -> 348.0ms). But overall all-tick p95 barely moved: 346.2ms (strided) vs 343.1ms
(opportunistic) -- essentially unchanged. Explanation, not spin: with real EasyCom audio driving
real decisions, 195/240 (81%) of ticks in BOTH runs were already "gated" (near-free returns,
~0.2ms) regardless of refresh policy -- the p95 percentile in a mostly-gated tick stream is
governed by that mix either way, so a cheaper-when-it-happens overlap doesn't show up as a big
aggregate win at the 95th percentile in this specific real-audio window.

**Important caveat on comparability**: these 343-346ms figures are NOT directly comparable to the
earlier-reported Phase 4.3 p95=786.5ms. That earlier measurement used a harness with
`generate_fn=None` -- mic-gating was NEVER engaged (every tick ran the full decision path, 0%
gated), a fundamentally different tick-composition regime from this one (81% gated, real audio-
driven decisions). Do not read "343ms < 786.5ms" as "opportunistic refresh + real audio beat the
old result" -- they measure different things. A true like-for-like re-run (same no-gating,
continuous-decision harness as Phase 4.3, strided vs opportunistic policy) has NOT been done and
would be needed to make that specific comparison honestly.

Bottom line for item 4: root cause is real and confirmed. The opportunistic policy is a genuine,
verified mechanism improvement (much higher overlap-with-cheap-ticks rate, much lower per-overlap
cost) but is not shown to reduce end-to-end tail latency in a real-audio conversational setting
where gating already dominates the tick mix -- worth keeping as an available policy, not yet
justified as a required deployment change on this evidence alone.

## 2026-07-26 (item 5) — A1 extended to n=651 balanced (217/class, backchannel ceiling): well-powered NULL

checkpoints/m4_decision_head_3class_bothpresent_v2_n651/A1_FALSIFIER_RESULTS_N651.json. Same v2
(corrected) construction, same trained head, extraction extended to use ALL 217 available
backchannel windows in the test sessions plus 217 speak / 217 silence (seed=11, same sampling
convention as v1/v2's n=300 run) -- n=651 vs n=300, roughly double the previous sample and using
the actual ceiling for the rarest class.

| condition | n=300 (v2) | n=651 (this run) |
|---|---|---|
| (a) real fresh WS | 94.00% | 94.16% |
| (c-within) swapped, same session | 92.00% | 92.78% |
| (c-cross) swapped, cross-session | 92.67% | 92.63% |
| (e) random, matched stats | 94.33% | 93.55% |
| (f) dataset-mean | 94.33% | 93.55% |

**acc(a)-acc(c_within): +2.0pp CI[+0.33,+4.0] at n=300 (excluded zero, drove the PASS flip) ->
+1.38pp CI[0.0,+2.76] at n=651 (CI's lower bound lands exactly on zero -- does NOT exclude it).
PASS = False at n=651.** This is the well-powered read: the n=300 real-vs-swapped gap was a
borderline effect that does not hold up with more data. acc(a) remains statistically tied with
(e) random-matched and (f) dataset-mean at n=651 too (both CIs include zero), consistent with n=300.
acc(a)-acc(b) [zeroed] remains large and clearly significant at n=651 (+16.4pp, CI excludes zero)
-- the one robust, reproduced finding across both sample sizes is that the head behaves very
differently when World-State is entirely absent (zeroed) vs present in ANY form (real, swapped,
random-matched, or constant-mean) -- but it cannot distinguish CORRECT from WRONG World-State once
one is present, at either n. **Final read: not a PASS, at any sample size tested.** This does not
change the deployment recommendation (ship the speech-only head, (g)=95.00%, invariant to
World-State by construction) -- it removes the ambiguity from the earlier framing.

## 2026-07-26 (gate fix + PRE-REGISTRATION) — Ego4D gate rescored sibling-excluded; thresholds locked BEFORE the retrain

**Two flaws in the original Ego4D held-out gate, both real**: (a) baseline R@1=0.71% is so close to
chance that "beat baseline" is a noise-clearable bar; (b) 1542 windows from only 81 source files
(~19/file, one file has 48) means each query's distractor pool contained ~18 near-duplicate
windows from the same continuous footage -- not like-for-like with VGGSound's 1545 gallery (1545
DISTINCT videos).

**Rescored with same-source-file windows excluded from each query's distractor pool** (no
re-extraction -- embeddings already cached), scripts/phase_ego4d_heldout_rescore_siblingexcl.py ->
checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_SIBLINGEXCL.json:
- Sibling-excluded R@1: vision->ambient 0.84% (was 0.71% unfiltered), ambient->vision 1.04% (was
  0.91%). **Barely moved** -- mean effective gallery size after exclusion is 1510.6/1542, so
  contamination was NOT the main driver of the low score; the weak signal is real, not noise
  hidden by near-duplicates.
- File-level R@1 (lenient secondary metric, full pool including siblings -- credit for getting the
  right FILE even with the wrong window): 7.2%/6.87%. Checked its chance baseline before treating
  this as informative (one file has 48/1542 windows, so chance is NOT the naive 1/81=1.2%): actual
  chance = 2.04-2.10%. So this is ~3.4x chance -- real but modest scene-identification signal,
  more modest than the raw 7% number suggests on its own.
- Within-modality cosine: 0.623 (vision) / 0.531 (ambient), shuffle-sanity gap 0.0399 -- both
  essentially unchanged from the unfiltered baseline (as expected; these don't depend on sibling
  exclusion).

**PRE-REGISTERED THRESHOLDS for the M2 retrain (VGGSound-60k + Ego4D-21.7k), locked BEFORE seeing
any retrain result**, referenced to this checkpoint's in-domain VGGSound behaviour (within-modality
cosine 0.0308/0.0289, shuffle gap 0.6604, checkpoints/vjepa21_shelved/VGGSOUND_COLLAPSE_CHECK_1545.json):
  - sibling-excluded Ego4D R@1 >= 10% (both directions) -- ~150x chance (0.065%) at n=1542, and
    ~12x the current sibling-excluded baseline (0.84%/1.04%)
  - Ego4D within-modality cosine <= 0.25 (both modalities) -- down from 0.623/0.531
  - Ego4D shuffle-sanity gap >= 0.20 -- >=30% of the in-domain 0.6604 reference
  - NO regression below 52% R@1 on the 1545 VGGSound gallery (the established M2 milestone gate)

Assessment before running the retrain: sibling exclusion confirming the low baseline is REAL (not
a contamination artifact) means these thresholds are appropriately stringent, not vacuous --
clearing them requires genuine representation improvement, not noise clearance. Given Ego4D is
only ~27% of the retrain mix, there is real risk of falling short of R@1>=10% specifically; that
risk is accepted rather than lowering the bar after the fact. No objection raised to any threshold.

## 2026-07-26 (gate REBUILT, item 1 resolved) — v1 gallery was ambiguity-bound; v2 is the authoritative baseline

Diagnostic (scripts/phase_ego4d_gate_diagnostic.py -> EGO4D_GATE_DIAGNOSTIC.json) confirmed the v1
gallery (1542 windows, only 81 files, mean 19/file, up to 48) was ambiguity-bound: file-level R@k
was consistently 1.7-3.8x chance (real signal) while instance-level R@1 was stuck at 0.84%
sibling-excluded, nowhere near the 10% pre-registered floor -- largely an artifact of each query
competing against dozens of near-duplicate siblings. Also checked (2a): 0/1542 v1 windows hit the
silent zero-audio fallback bug -- real code smell, but it did not corrupt v1's numbers. fps
verified 30/1 across all 81 v1 files (2d) -- no issue there either.

**Rebuilt (scripts/phase_ego4d_heldout_gallery_build_v2.py, seed=43): cap<=2 windows/file, 350
files, 674 windows** (checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json).
Train split correspondingly rebuilt to 17,140 windows / 946 files
(EGO4D_TRAIN_SPLIT_FILEDISJOINT_V2.json) -- still a strict file-level partition, same guarantee as
v1. Re-scored m2_fusion_20k_best fresh (scripts/phase_ego4d_heldout_gallery_score_v2.py, which also
applies item 2c's fix: decode_audio() now raises on zero-length audio and the window is recorded as
failed/excluded, never silently zeroed -- 0/674 windows hit this path here either).

**checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_V2.json -- THE authoritative baseline, v1's
number is superseded**:
| metric | v1 (1542w/81f, ambiguity-bound) | v2 (674w/350f, rebuilt) | pre-registered threshold |
|---|---|---|---|
| sibling-excl. R@1 (v->a / a->v) | 0.84% / 1.04% | **2.82% / 1.78%** | >=10% |
| file-level R@1 (v->a / a->v) | 7.2% / 6.87% (chance 1.9-2.1%) | 3.86% / 3.26% (chance 0.22%) | secondary, not gated |
| within-modality cosine (vis/amb) | 0.623 / 0.531 | **0.559 / 0.512** | <=0.25 |
| shuffle-sanity gap | 0.0399 | **0.0934** | >=0.20 |

The rebuild moved every primary number in the right direction (instance-level R@1 roughly 2-3x
higher, shuffle gap more than doubled, cosine down) -- confirming the ambiguity diagnosis was
real, not wishful. But it is STILL far short of every pre-registered threshold (R@1 needs another
~3.5-5.6x, shuffle gap needs to more than double again, cosine needs to drop by more than half).
The thresholds are not being loosened to match -- this is the honest pre-retrain baseline the M2
retrain must clear, referenced against EGO4D_HELDOUT_BASELINE_V2.json going forward, not the
retired v1 file.

## 2026-07-26 (mid-retrain) -- VGGSound-60k subsample premise questioned; data-scaling plan set

**Retraction of an unverified assumption**: the 60k VGGSound subsample (item 2, this run) was
justified by a claimed "documented saturation at 199k" -- checked and this is NOT actually
documented anywhere in this repo (grepped logs/tracking docs, found nothing supporting it). The
user's own firsthand recollection contradicts it directly: the original M2 training run started at
~51k VGGSound clips and only reached ~R@1=30%, then jumped to 52% after scaling to the full ~199k
corpus -- i.e. VGGSound scale itself was very plausibly the main driver of that 30%->52% jump, not
something that saturates well below 199k.

**Live tension this creates**: this run's 60k subsample was deliberately shrunk specifically to
make Ego4D ~22% of the batch mix (fixing the earlier "Ego4D only 9.9%, too diluted" problem). If
VGGSound scale is really what drives R@1 toward 52%, shrinking it back to 60k may cap this run
well below 52% regardless of anything else being correct -- Ego4D-proportion and VGGSound-scale
pull in opposite directions under a plain subsample-and-mix approach. Consistent with this: step
5000's R@1 (~31-33%) already sits suspiciously close to the recalled ~51k-scale ceiling (~30%),
though this is not yet conclusive (could still be early-training trajectory, not a plateau).

**Decision (explicit, from the user): do NOT stop the current run.** Let it finish as the FIRST
controlled datapoint in a deliberate data-scaling study, not a one-shot final answer. Then check
whether adding more data -- especially egocentric (Ego4D) data -- actually moves the eval numbers,
rather than assuming either "more VGGSound" or "more Ego4D" alone is the fix.

**Longer-term data plan (if/when a from-scratch M2 retrain is warranted), stated by the user**:
  - Pretrain on VGGSound + AudioSet-500k COMBINED (both already on disk: VGGSound persistent cache
    ~199k clips, AudioSet-500k at /mnt/Raid-Storage-2/utkarsh-data/audioset_500k, 152GB, confirmed
    present -- a 20k subset also exists at audioset_20k, 24GB).
  - Use a LARGER corpus of Ego4D data than the current 23,303-window kept pool. Raw footage
    headroom already confirmed on disk: 1,236 video_540ss files (461GB) + 572 clips files (27GB) =
    1,808 total Ego4D files, of which the AV-relevance filter's own candidate pool
    (ego4d_av_filter_scores.json) already scored 57,822 windows -- more than double the 23,303
    currently kept. The per-file cap (50, previously reverted down from a rejected 70) and the
    floor/exclusion thresholds are the actual limiting factors on Ego4D volume right now, not a
    lack of raw footage -- there is real headroom to draw on before needing to download anything
    new.
  - Note for later stages, not M2: the M3 connector was trained on a custom rich-caption dataset
    the user built via a Qwen-Omni model (action-labelling-style captions, richer than VGGSound's
    basic category labels) -- relevant context if a future M2-scale change ever needs to be
    threaded through to M3's training data too, not an M2 change itself.

Full detail on the running datapoint once it converges: checkpoints/m2_retrain_vggsound60k_ego4d17k/.

## 2026-07-27 -- M2 retrain (VGGSound-60k + Ego4D-17.1k) COMPLETE: first scaling-study datapoint

20000 steps, 4-GPU DDP, batch=48/GPU (negatives=192x192, matching the original recipe's real
all-gathered negative count -- see the rank-aware SourceDisjointBatchSampler fix below),
lam_sigreg=0.03/lam_pred=0.0/lam_pooled=0.0/lam_contrastive=1.0/lam_fusion=1.0/fusion_layers=2,
contrast_dim=256/temp=0.05 -- same recipe as the original m2_fusion_20k_best run in every respect
except which data it saw. Checkpoints saved every 1000 steps (step1000.pt..step20000/last.pt) plus
best.pt (lowest loss_ema) -- nothing overwritten silently, full history on disk in
checkpoints/m2_retrain_vggsound60k_ego4d17k/.

**Trajectory (VGGSound 1545-gallery R@1, vision->ambient / ambient->vision), read from the
per-1000-step console log at the time (exact per-step JSON not separately saved -- only the
final step-20000 numbers below come from the eval's own printed/parsed output, the intermediate
figures here are transcribed from the live monitoring log and are directional, not to be treated
as a re-derivable source of truth)**: rose steadily from ~15% at step 1000 to ~37-38% by step
8000-9000, then held in a 36-38% band for roughly 5 consecutive evals (steps 6000-10000) despite
train contrastive accuracy already at 100% by step 9000 -- the training objective itself had
saturated on the 192-negative task while held-out ranking hadn't, a real plateau, not noise.
Broke upward again as the LR annealed past step ~12000-13000, climbing to the low-40s by
step 14000 and holding there through the end. **step20000 (FINAL, from the actual printed eval):
vision->ambient R@1=42.27%, ambient->vision R@1=41.68%**, shuffle_sanity_gap=0.6049 (healthy,
close to the in-domain VGGSound reference of 0.6604), matched_cos_sim=0.6166.

**GATE RESULT, VGGSound side: FAILS the 52% no-regression floor by ~10pp (42.27% vs 52%).** Not
softened -- this is the real number. Directly confirms the mid-run concern: shrinking VGGSound
from ~199k to a 60k subsample (done specifically to give Ego4D a meaningful ~22% batch share)
cost real ceiling. The plateau-then-partial-recovery shape (flat 36-38% for 5 evals, then climbing
again as the LR annealed past step ~12000) suggests the model was fighting a genuine data/negative-
diversity limit for a long stretch, not just needing more steps at this same configuration.

**Retracted assumption, confirmed wrong**: item 2's original justification for the 60k subsample
("documented saturation at 199k") had no actual documentation backing it in this repo and is now
directly contradicted by this result plus the user's firsthand recollection (51k->~30% R@1,
199k->52%). VGGSound scale is a real, load-bearing factor, not something that saturates well
below the full corpus.

**Ego4D-side gate result: DRAMATIC, real improvement across every metric.**
checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RETRAINED_RESULT.json vs the pre-retrain baseline
(EGO4D_HELDOUT_BASELINE_V2.json), same 674-window frozen gallery, same m2_fusion architecture:

| metric | pre-retrain baseline | retrained (this run) | pre-registered threshold | met? |
|---|---|---|---|---|
| sibling-excl. R@1 (v->a / a->v) | 2.82% / 1.78% | **18.40% / 18.40%** | >=10% | **PASS** |
| shuffle-sanity gap | 0.0934 | **0.2453** | >=0.20 | **PASS** |
| within-modality cosine (vis/amb) | 0.559 / 0.512 | **0.383 / 0.356** | <=0.25 | not quite (much closer) |
| file-level R@1 (v->a / a->v) | 3.86% / 3.26% | **21.96% / 20.92%** | secondary, not gated | ~95-100x chance (0.22%) |

Sibling-excluded R@1 improved ~6.5-10x. Shuffle-sanity gap more than doubled and now clears its
pre-registered floor. Within-modality clustering dropped substantially (less collapsed) though
not quite below the 0.25 target. **2 of 3 pre-registered Ego4D thresholds are met outright; the
third (cosine) improved dramatically without fully clearing its bar.** This is unambiguous: the
Ego4D data DID teach the model something real about egocentric AV correspondence that VGGSound-
only training did not have.

**Combined verdict, stated plainly, not averaged away**: this retrain is a genuine two-sided
result. It clearly WORKS for Ego4D (2/3 thresholds passed outright, every metric dramatically
better) and clearly REGRESSES on VGGSound (42.27% vs the 52% floor, ~10pp short). Both are real,
both matter, and one does not cancel the other. The likely mechanism, consistent with the user's
own recollection: VGGSound scale (cut from ~199k to 60k) is what the in-domain ceiling depends on,
while Ego4D REPRESENTATION in the batch mix (raised from ~9.9% to ~22%) is what the egocentric
generalization depends on -- this run traded one for the other rather than solving both, exactly
the tension flagged mid-run. The follow-up scaling experiments (more VGGSound scale back, and/or
more Ego4D data -- both have real headroom, see above) are what will show whether both can be
had together, not this single datapoint.

**Status: first datapoint in a deliberate scaling study, per explicit user instruction -- NOT a
final verdict on whether Ego4D integration works.** The user's decision after seeing the mid-run
trend: do not stop this run; treat it as the starting point, then check whether adding more data
(VGGSound scale back up, and/or more Ego4D data -- real headroom already confirmed: 1,808 raw
Ego4D files on disk, 57,822 already-scored candidate windows vs 23,303 currently kept) actually
moves the eval numbers before concluding anything about whether this architecture/recipe can reach
the 52% floor with Ego4D included.

**Side fix made along the way, now permanent**: SourceDisjointBatchSampler (data/
source_disjoint_batch_sampler.py) was single-GPU-only before this run -- attempting a naive
torchrun would have silently given every rank the SAME batches (redundant, not complementary
shards), corrupting the "192 distinct negatives" assumption. Fixed to be rank-aware (every rank
computes the identical global batch list from the same seed+epoch, then slices round-robin by
rank) -- unit-tested for no cross-rank batch duplication and no source collisions within any
batch, then verified under a real 4-GPU torchrun smoke test before the full run. This is a
real, permanent capability add, not a one-off hack for this run.

## 2026-07-27 -- Scale hypothesis verified from logs; RUN-1 launched (scaling study, step 2)

**DIAGNOSIS, checked before building on it (not assumed)**: no single logged "~51k -> ~40% R@1"
data point exists -- both historical 51,508-clip runs (logs/m2_fusion_bridge.log, logs/
m2_diag192.log) were cut short at 6000/6000 steps, never annealed to full convergence, so neither
produced a genuine converged endpoint at that scale. The clean, apples-to-apples evidence that
DOES exist: at the IDENTICAL step count (6000), IDENTICAL lam_fusion=1.0 setup, IDENTICAL 192
negatives, on the SAME 1545-clip gallery (dataset_len=1545/clips_seen=1545 confirmed in both logs)
-- 51,508 clips (m2_fusion_bridge.log) gave R@1=33.46%/34.24%; 199,007 clips (m2_fusion_fullscale.
log) gave R@1=44.27%/43.95%. A genuine ~10pp gap from corpus scale alone, everything else held
constant. **The scale hypothesis holds up on real evidence, even though the specific "~40%"
recollection isn't independently pinned to one exact logged number.**

**RUN-1 launched (free, no extraction)**: full 199,007-clip VGGSound cache (minus only the 1545
eval gallery, matching the original production exclude_ids convention exactly -- no additional
60k-style subsampling) + the v2 Ego4D train split (17,140 windows, already extracted, file-
disjoint from the active v2 gate). Confirmed at startup: `197462 VGGSound + 17140 Ego4D = 214602
total clips (8.0% Ego4D)`. Same recipe as the completed run in every other respect: batch=48/GPU
x4 (negatives=192x192), lam_sigreg=0.03/lam_pred=0/lam_pooled=0/lam_contrastive=1.0/lam_fusion=1.0/
fusion_layers=2, contrast_dim=256/temp=0.05, 20000 steps, eval-every/save-every 1000,
--tag-ckpts. Checkpoints: checkpoints/m2_retrain_vggsound199k_ego4d17k/.

**Pre-registered gates for RUN-1 (before seeing any result)**:
  - VGGSound 1545 gallery: R@1 >= 52% (recovery check)
  - Ego4D held-out (v2 gate, EGO4D_HELDOUT_BASELINE_V2.json comparison): sibling-excl R@1 >= 18.40%,
    shuffle gap >= 0.2453 (hold the gains already achieved at 22.2% Ego4D batch share)

**Batch composition caveat, stated up front**: Ego4D is only 8.0% of batches in RUN-1 (vs 22.2% in
the completed run) -- this is the real, unmodified size of both corpora with no subsampling either
direction. If RUN-1 clears the VGGSound floor but LOSES the Ego4D gains, the honest read is "Ego4D
got diluted back down," not "full-scale VGGSound and Ego4D learning are incompatible" -- this
ambiguity is exactly what RUN-1 is designed to surface, so both gates must be read together, not
either one in isolation.

**Ego4D headroom (answering a standing question, not new information)**: 1,808 raw Ego4D files on
disk (1,236 video_540ss + 572 clips, 488GB total), of which the AV-relevance filter's own scored
candidate pool already covers 57,822 windows -- more than 3x the 17,140 currently used for
training. The per-file cap and floor/exclusion thresholds are the actual limiter, not a lack of
raw footage; no new download is needed to substantially grow the Ego4D training volume.

**AudioSet status, per direct user report (not independently re-verified here)**: the AudioSet-500k
download already confirmed audio-only (no video stream) by the user before this session started --
correctly ruled out as a source for AV binding. A replacement, "AudioSet Strong" (timestamped
annotations, real video), is downloading now (scripts/download_audioset_mp4.py --subset strong
--workers 64, ~103,463 candidate URLs, observed ~11-12% dead-link failure rate early on, ETA ~2
hours per the user's own estimate, consistent with the observed download rate). P1-P5 preflight
work is deliberately NOT being done against the old audio-only 500k set (would be wasted effort on
data already ruled out) -- will run once AudioSet Strong's download completes.

## 2026-07-27 -- RUN-1 COMPLETE: VGGSound gate PASSES, Ego4D gains partially lost (batch-share mechanism confirmed)

20000 steps, 4-GPU DDP, identical recipe to the completed run in every respect except corpus:
full 199,007-clip VGGSound cache (minus only the 1545 eval gallery) + the same 17,140-window Ego4D
v2 train split, giving `197462 VGGSound + 17140 Ego4D = 214602 total (8.0% Ego4D)`.

**VGGSound gate: PASSES.** Final eval (step 20000, dataset_len=1545/clips_seen=1545 confirmed):
R@1 = 55.15% (vision->ambient) / 55.53% (ambient->vision), clearing the 52% floor by ~3pp.
Trajectory tracked well ahead of the completed 60k-Ego4D run at every matched step (e.g. step 7000
already exceeded that run's FINAL step-20000 number) -- decisive, real-time confirmation of the
scale hypothesis. shuffle_sanity_gap=0.7081 (very healthy, close to the pure-VGGSound in-domain
reference of 0.6604 -- actually higher here, plausibly because Ego4D windows widen the negative
pool's diversity even at only 8% share).

**Ego4D gate: FAILS the "hold the gains" thresholds, but not back to baseline -- a real, partial
loss.** checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN1_RESULT.json vs both reference points:

| metric | pre-retrain baseline | prior run (22.2% Ego4D) | RUN-1 (8.0% Ego4D) | RUN-1 gate (hold >=prior) |
|---|---|---|---|---|
| sibling-excl R@1 (v->a/a->v) | 2.82%/1.78% | 18.40%/18.40% | **11.57%/10.68%** | FAIL |
| shuffle-sanity gap | 0.0934 | 0.2453 | **0.2140** | FAIL (close, not quite) |
| within-modality cosine | 0.559/0.512 | 0.383/0.356 | **0.483/0.429** | (not gated, but worse) |
| file-level R@1 (secondary) | 3.86%/3.26% | 21.96%/20.92% | 15.13%/13.35% | vs chance 0.22% |

RUN-1's Ego4D numbers sit roughly MIDWAY between the pre-retrain baseline and the 22.2%-Ego4D run
on every single metric -- not a full regression to baseline, but a clear, real, partial loss of
the gains. **This is decisive evidence for the batch-composition mechanism specifically, isolated
from Ego4D's absolute data volume**: the Ego4D windows themselves are IDENTICAL between this run
and the completed run (same 17,140-window cache, same file-disjoint-from-gate guarantee) -- the
ONLY thing that changed is how often the model sees them per batch (22.2% -> 8.0%). Diluting Ego4D's
batch presence cost roughly half of its measured gain on every metric, while its underlying
per-window information content obviously didn't change.

**Combined verdict**: neither a clean win nor a clean loss -- VGGSound scale and Ego4D batch-share
are BOTH real, independent levers, and RUN-1 confirms you cannot get full benefit of both by
simply restoring VGGSound to full scale while holding Ego4D's absolute volume fixed. This is
exactly the setup RUN-2 (more Ego4D data, restoring its batch share at full VGGSound scale) is
designed to test -- proceeding to the AudioSet-Strong preflight now to build the data for it.

## 2026-07-27 -- AudioSet-Strong preflight P1-P3

**P1 PASSES**: AudioSet-Strong download (scripts/download_audioset_mp4.py --subset strong
--workers 64) finished. Confirmed real, usable data: 35,247 mp4 files (40GB) at
/home/utkarsh/raid2-data/audioset_mp4/strong/, ffprobe on 3 sampled files confirms h264 video +
aac audio streams in every case (not audio-only, unlike the ruled-out 500k set). Real yield was
much lower than hoped: only 30,261/103,463 unique YouTube IDs succeeded (~29.2%), NOT the ~70k
the user estimated, because YouTube's bot-detection kicked in partway through the download (64
concurrent workers -- reproduced directly: `yt-dlp` on a known-good video returns "Sign in to
confirm you're not a bot"). Flagged to the user during the download; they chose to let it finish
rather than restart with fixes. Clip durations are highly variable (1-10s+ observed directly via
ffprobe), unlike VGGSound/Ego4D's fixed ~10s windows -- extraction must handle this (see P3 below).

**P2 PASSES, zero leakage**: extracted all YouTube IDs from AudioSet-20k's train+test webdataset
tar shards (confirmed itself audio-only, same family as the ruled-out 500k set -- fine, only used
here for ID comparison, not as training/eval data). Overlap between AudioSet-Strong's 30,261
unique IDs and AudioSet-20k's actual held-out `test` split (18,886 IDs): **0**. (Overlap with the
20k's own `train` split, 2,066 IDs, is irrelevant -- not our eval set.) Nothing removed from the
draw; no leakage to guard against.

**P3 PASSES, well under the ~36h trigger**: real decode+encode throughput measured on 1000 real
AudioSet-Strong clips (reusing scripts/extract_features_av.py's whole-clip _decode_video_raw/
_decode_audio_raw directly -- AudioSet-Strong clips are short whole-clips, not windows into a
longer video, so no new decode logic was needed unlike Ego4D).
  - BEFORE (single process): 986/1000 ok, 14 failed, **1.578 clips/s** (0.634s/clip -- actually
    faster than the ~0.744s/clip figure cited before this check).
  - AFTER (4-way GPU-sharded, 250 clips/shard, matching scripts/extract_features_ego4d_train.py's
    proven pattern): 986/1000 ok (same 14 failures, consistent), wall-clock = slowest shard's
    202.8s, **combined 4.931 clips/s -- a 3.12x speedup** (sub-linear vs the naive 4x, expected
    from per-process model-load overhead and shared-CPU contention across 4 concurrent decoders).
  - **Projected full-corpus (35,247 clips) wall-clock: 6.2h single-threaded -> 2.0h sharded.**
    Well under the ~36h report-before-proceeding trigger -- no need to pause here.
  - Failure breakdown (same 14/1000 in both runs, real data-quality artifacts, not decode-library
    bugs): (a) clips too short for uniform 64-frame sampling ("Requested next frame while there
    are no more frames left to decode"), (b) a smaller number of literal zero-duration clips
    where the download's start_sec == end_sec ("Clip has no frames, num_total=0") -- a real
    artifact in a fraction of the AudioSet-Strong segment timestamps, not a bug in this decode
    code. Both classes must be filtered/handled in the real extraction pass, not silently retried.

Next: P4 (AV-relevance filter, same methodology as Ego4D -- AST tagger, 0.10 confidence floor,
speech/acoustic-environment/wearer-produced exclusions) and P5 (storage budget), then RUN-2.

## 2026-07-27 -- AudioSet-Strong preflight P4-P5: audio-only filter is NOT sufficient, added a
## visual gate. Final corpus 8,588 clips. Both PASS after mitigation.

**P4 stage 1 (audio-only filter, scripts/audioset_av_relevance_filter.py, same methodology as
Ego4D -- Silero VAD + MIT AST tagger + energy-CoV, FLOOR=0.10, VAD_SPEECH_EXCLUDE=0.84)**:
33,620/35,247 files scored (1,627 failed to decode audio). 27,467 passed floor+VAD exclusion
(retention_rate_of_scored=44.6%). Naive top-N-by-score selection was badly unbalanced (Music
22.5%, Vehicle 8.6% of the floor-passed pool, 407 distinct tags) -- fixed with a post-hoc
per-tag-capped reselection directly on the already-scored data (no VAD/AST re-run needed),
PER_TAG_CAP=750 (5% of 15,000 target), yielding 15,000 kept / 383 distinct tags, no category
over 5%. This is `audioset_av_filter_scores_kept_CAPPED.json`.

**Manual visual spot-check (16 samples, decoded frames via ffmpeg, not just text/score review)
found the audio-only filter is NOT sufficient on its own.** ~9-10/16 "kept" clips had no genuine
visible sound-cause: a CAD software screenshot ("Typing"), a static stock photo of a guinea pig
("Whispering" -- also a straight label mismatch), a horror-movie digital-art image ("Music"), two
animated clips (stick-figure walk cycle, claymation car+traffic-light, both "Music"), a text-only
title card ("Vehicle"), an animated cartoon ("Fart"), and a person's face mislabeled "Animal".
Only ~6/16 (real dogs, a guitar effects pedal, a bird cage, real fireworks, a real train, a bird
in bushes) showed genuine real-world AV correspondence. **Root cause**: the filter scores AUDIO
ONLY -- confident, dynamic audio from a screen recording, a static photo with a music bed, or an
animated short ranks identically to genuine footage with a real visible sound source. This is
exactly the risk flagged before extraction: AudioSet-Strong is generic YouTube content, not
curated for visible sound sources like VGGSound, and has no structural protection against this
the way Ego4D's egocentric footage does (wearer is usually the sound source).

**Mitigation, not a re-run of the expensive stage**: built `scripts/audioset_visual_gate.py`,
a cheap visual-admissibility check applied on top of the already-scored 15,000-clip pool (audio
scores stay valid, no VAD/AST re-run). Two signals per clip, sampled at two frames (t=0.4s,
t=3.0s): (1) staticness -- mean frame-to-frame pixel diff < 0.015 flags static photos/screenshots/
title cards; (2) CLIP zero-shot (openai/clip-vit-base-patch32) 5-way classification ("real
photograph/video frame" vs "cartoon/animation/claymation" vs "screenshot of an app/document" vs
"slideshow/title card/text image" vs "stock photo on plain background") -- admissible requires
"real photograph" to be argmax AND >=0.35 (chance=0.20). Ran on all 15,000 clips in 2.9 min total
(24-thread ffmpeg frame extraction + batched GPU CLIP inference).

Result: **8,588/15,000 visually admissible (57.3%)**. 2,163 dropped as static, 4,220 dropped as
not-real-photo (cartoon/screenshot/slideshow/stock-photo), 29 frame-extraction failures.
`checkpoints/vjepa21_shelved/audioset_visual_gate_result.json` + `_kept.json` + `_dropped.json`.

**Second manual visual spot-check (8 samples from the post-gate kept set, decoded frames)
confirms the mitigation works**: 7/8 genuine real-world video (real dogs, a real camera lens, a
real ice-cream truck, a real sewing machine close-up, real tigers, real storm/tornado footage, a
person at a real window) vs 0/8 cartoons/screenshots/static-photos -- a clean reversal of the
pre-gate rate. One clip had a label/content mismatch (window-opening footage tagged "Cupboard
open or close") and one was ambiguous (forest b-roll tagged "Heart sounds, heartbeat") -- these
are AST audio-tagger label-precision issues, a pre-existing and separate concern from the
AV-correspondence problem the visual gate targets, and were already a known limitation of the
audio-only tagger before this check.

**P4 verdict: CONDITIONAL PASS.** The audio-only filter alone would have FAILED P4's actual intent
(AV-relevance, not just "confident audio") -- reporting that honestly rather than averaging it
into a clean pass. With the visual gate added, the final corpus (8,588 clips, 383+ distinct AST
tags, no category over 5%) is fit for purpose. This closes as a two-sided finding per standing
project convention: audio-only filtering is not portable to non-curated (generic YouTube) sources
even when it worked fine for Ego4D's egocentric footage; a visual admissibility check is now a
required stage for any future non-curated AV corpus, not just this one.

**P5 (storage budget): PASSES, trivially.** 8,588 clips x 4,130,621 bytes/clip (measured directly,
same real per-clip size as the VGGSound cache -- vision (32,16,1024 bf16) + ambient_base +
ambient_nat) = 35.5 GB (33.0 GiB) projected for the AudioSet-Strong feature cache. RAID
(`/mnt/Raid-Storage-2`, `/dev/md1`) has 5.1 TB free of 7.0 TB total; existing VGGSound (764GB) +
Ego4D-train-v1 (66GB) caches already occupy 830GB there. Adding AudioSet-Strong's 35.5GB leaves
~5.07 TB free -- no storage concern at all at this scale.

**P1-P5 status: P1 PASS, P2 PASS, P3 PASS, P4 CONDITIONAL PASS (visual gate required and applied),
P5 PASS. Ready to extract the 8,588-clip visually-gated corpus and proceed to RUN-2.**

## 2026-07-27 -- AudioSet-Strong extraction complete; RUN-2 launched

Extracted the 8,588 visually-gated clips via scripts/extract_features_audioset.py (new, whole-clip
variant of extract_features_ego4d_train.py -- reuses extract_features_av.py's _decode_video_raw
directly, but NOT its _decode_audio_raw, which silently zeros on failure; this script raises
instead, same item-2c convention as the Ego4D extractor). 4-GPU sharded, ~13 min wall-clock (far
under the P3 projection). **8,550/8,588 succeeded (38 failed, 0.44%)** -- same two failure classes
P3 already characterized (too-short-for-64-frame-sampling, zero-duration segments), nothing new.
Cache: `/mnt/Raid-Storage-2/utkarsh-data/feature_cache_audioset_strong_v1` (schema-identical to the
VGGSound/Ego4D caches, real per-clip duration used for timestamps since AudioSet-Strong durations
are genuinely variable, unlike VGGSound/Ego4D's fixed ~10s). clip_id list written to
`data/audioset_train_clip_ids.txt` (8,550 lines).

**Generalized the mixing infra from 2 to N sources** (was VGGSound+Ego4D-only): rewrote
`data/mixed_av_cached_dataset.py`'s `MixedAVCachedDataset` to take a list of `MixedSource(name,
cache_dir, clip_ids)` instead of two hardcoded positional args, and `train_m2.py`'s
`build_dataloader`/`train`/CLI to take a repeatable `--mixed-source name=cache_dir=clip_ids_path`
instead of the old single `--mixed-cache-dir`/`--mixed-clip-ids` pair. `source_key_from_clip_id`
(data/source_disjoint_batch_sampler.py) needed no change -- AudioSet-Strong's clip_id convention
(raw filename stem, e.g. "kM9QvdZela4_250_260") doesn't match the `ego4d_.../easycom_...` prefix
pattern, so it correctly falls into the "each clip is its own source" bucket (right behavior --
AudioSet-Strong clips are independent YouTube videos, no same-source-window risk like Ego4D).
30-step smoke test on all 3 sources passed cleanly before the real launch (verified batch
composition, negatives=192x192, eval pipeline, source-token invariance check).

**RUN-2 launched**: 20,000 steps, 4-GPU DDP, identical recipe to RUN-1 (same lam_fusion=1.0/
fusion_layers=2, same batch-size=48/GPU -> negatives=192x192, same source-disjoint sampler) plus
the new AudioSet-Strong source. Batch composition: `197,462 VGGSound + 17,140 Ego4D + 8,550
AudioSet = 223,152 total (88.5% VGGSound, 7.7% Ego4D, 3.8% AudioSet)` -- Ego4D's absolute volume
and batch-share are UNCHANGED from RUN-1 (this run only adds AudioSet as a third source, per the
original RUN-2 spec -- it does not attempt to restore Ego4D's 22.2% share, so the Ego4D gate here
is "did not get any worse than RUN-1," not "recovered to the 22.2%-share run's numbers").
Checkpoint dir: `checkpoints/m2_run2_vggsound199k_ego4d17k_audioset8k5`. Gates (pre-registered):
VGGSound R@1>=52%, Ego4D sibling-excl R@1 >= RUN-1's 11.57%/10.68%, AND within-modality cosine
improves toward <=0.25 (RUN-1 measured 0.483/0.429 -- the one threshold every prior run has
missed). All three will be reported regardless of outcome, not averaged into one story.

(Process note: first launch attempt used `conda run -n jepa-omni torchrun ...` and produced zero
stdout for several minutes despite `flush=True` on every print -- `conda run` fully buffers piped
child stdout until process exit, which would have meant zero visibility into eval gates for the
entire 20,000-step run. Confirmed via GPU utilization that training was in fact proceeding
correctly; killed and relaunched after ~3 minutes using the direct torchrun binary path +
`PYTHONUNBUFFERED=1` + `stdbuf -oL -eL tee` instead, matching the pattern an older diagnostic run
in this repo already used -- live line-buffered output confirmed working on relaunch, ~3 min of
compute lost, no data or checkpoint corruption.)

## 2026-07-27 -- RUN-2 redesigned mid-flight: correcting a wrong premise, real Ego4D expansion

User asked why RUN-2 didn't add more Ego4D data given RUN-1's own finding (batch-share, not
volume, drives Ego4D's gain -- diluting from 22.2%->8.0% cost ~half the measured gain). Killed
RUN-2 at step 200/20000 (cheap, <1% in) to investigate real headroom before relaunching.

**Correction to an initial wrong claim**: I had cited "57,822 already-scored candidates vs 23,303
kept" from memory as unused headroom. Verified directly: 57,822 is the TOTAL already-scored count
(kept+dropped), not unused candidates. Of 1,808 raw Ego4D files on disk, 420 were never scored --
but ALL 420 have NO audio stream (confirmed via ffprobe on every one), exactly why the original
filter's `has_audio_stream()` check skipped them. Zero real headroom there. Reported this
correction to the user rather than proceeding on the wrong premise.

**Real lever 1 (executed): raise per-file candidate cap 50->150 on the 946 known-safe train files
only** (never touching the 350 frozen held-out files -- confirmed file-disjoint by construction,
these files were invisible to the original held-out draw). Real ceiling check: at cap=500 (near-
exhaustive), the 946 train files can only yield 105,390 raw candidates total -- confirms literal
50/50 vs VGGSound (197k) is mathematically impossible from the existing downloaded corpus alone,
no matter how high the cap goes. Generated 42,568 net-new candidate windows at cap=150 (diffed
against already-scored keys), scored via new `scripts/ego4d_score_expand.py` (4-GPU, ~50min).
Applied floor=0.10 + VAD<0.84 + a fresh 5%-per-tag cap (same style as the AudioSet fix): **18,900
new kept windows**, 255 distinct tags, no category over 6.8%. Extracting features now via new
`scripts/extract_features_ego4d_expand.py` (additive to the existing cache, same schema/clip_id
convention). This alone takes Ego4D train from 17,140 -> ~36,040 windows.

**Real lever 2 (in progress): download new videos from the untouched full Ego4D corpus.** User
asked directly whether we could pull more raw Ego4D via the existing AWS access rather than just
re-mining already-downloaded files. Verified via the `ego4d` CLI + `ego4d.json` full metadata
(9,821 videos total, 8,585 never downloaded, ~3,425 video-hours): our existing 1,236 local
`video_540ss` files came ONLY from the AV/Social benchmark filter (task #61) -- a subset
specifically curated for CONVERSATIONAL content, which is exactly what our VAD-speech-exclusion
filter throws out. The untouched 8,585 videos are dominated by non-conversational, task-oriented
`scenarios` metadata (Crafting/knitting/sewing 1420, Cooking 1078, Construction 923, Cleaning 801,
etc.) -- likely a HIGHER post-filter keep-rate than the AV-benchmark corpus, not just more volume.
Selected 3,000 videos (activity-balanced, no scenario >15% of the draw, excluding conversational
scenarios and any AV-benchmark-flagged videos), ~1,100 video-hours, ~888GB, launched via
`ego4d --video_uid_file`. Projected: ~108k more candidate windows at cap=150, ~40-45k more kept
after filtering -- would bring Ego4D toward ~125-129k combined, a ~40% batch share alongside
VGGSound's 197,462 (fixed) with NO AudioSet in this run (explicitly dropped per user instruction
to keep the batch-share test clean). Download + scoring + extraction is real, multi-hour work
(~12-13h projected total) -- not free/instant like RUN-1. Will extract features for this second
batch, combine with lever-1's expansion, do one final combined per-tag-cap pass for corpus
balance, then launch the redesigned RUN-2 (VGGSound 197k + Ego4D ~125-129k, no AudioSet).

**Lever 1 COMPLETE**: all 18,900 new windows extracted, 0 failures across all 4 GPU shards
(~93 min wall-clock). Verified directly: `feature_cache_ego4d_train_v1` now holds exactly 36,040
`.pt` files (17,140 original + 18,900 new, confirmed by file count, not just script exit status).
`data/ego4d_train_clip_ids.txt` rebuilt from the real cache directory listing (36,040 lines).
Ego4D train volume has already more than doubled from lever 1 alone, before lever 2 (new-video
download, in progress, ~998GB of a since-revised ~950GB-1TB projection -- the activity-selected
subset runs longer per video on average than the full-corpus baseline used for the original
estimate) lands. Not yet launching RUN-2 -- waiting for lever 2's scoring+extraction so the
final per-tag balance pass covers the FULL combined pool at once, not two inconsistent passes.

**Lever 2 download COMPLETE**: 4,158/4,158 files on disk (1,236 original + 2,922 new), 1.4TB
total, 0 errors. Of the 3,000 targeted UIDs: 78 genuinely unavailable (S3 403/missing, matches
the download log's own count), 1,023 have no audio stream -- consistent with this Ego4D video
format's known ~34% no-audio base rate (see `has_audio_stream()` docstring in
ego4d_av_relevance_filter.py, 420/1236 on the original corpus), not a targeting problem specific
to the task-oriented scenario selection. 1,899 usable videos -> 178,285 candidate windows at
cap=150 (uniform_window_starts, same convention as lever 1). Scoring launched across 4 GPUs
(~44,571 candidates/shard) via the same `scripts/ego4d_score_expand.py`, projected ~3.5h based on
lever 1's measured throughput (42,568 candidates in ~93min). Once scored: apply floor=0.10 +
VAD<0.84 + a combined per-tag cap across BOTH lever-1's and lever-2's kept pools together (one
balance pass, not two), extract features for the net-new lever-2 windows, rebuild
`data/ego4d_train_clip_ids.txt` with the full combined set, then launch the redesigned RUN-2
(VGGSound 197,462 fixed + Ego4D [36,040 + however many lever-2 contributes], no AudioSet).

**Lever 2 scoring COMPLETE**: 178,285/178,285 candidates scored (4 GPUs, ~93 min). floor=0.10 +
VAD<0.84 -> 131,606 passed. Applied the SAME 5%-of-own-pool per-tag cap methodology as lever 1
(pragmatic choice: capping against lever-1's already-extracted 18,900 windows' tag distribution
would mean discarding already-extracted work to rebalance -- instead each lever's cap is applied
against its own pool, consistent methodology, zero wasted compute) -> **98,451 new kept windows**,
351 distinct tags, no category over 6.7%. `checkpoints/vjepa21_shelved/ego4d_newvideo_kept.json`.

**Projected final Ego4D train size: 36,040 (existing+lever1) + 98,451 (lever2) = 134,491.**
Against VGGSound's fixed 197,462: **~40.5% Ego4D batch share** -- matches the n=3000 projection
almost exactly, and is a genuine, order-of-magnitude step up from RUN-1's 8.0% / the
22.2%-share reference run, without touching VGGSound's corpus scale. Feature extraction for the
98,451 new windows launched across 4 GPUs (~24,613/shard) -- this is ~5.2x lever 1's extraction
job, projected ~8h wall-clock. Once done: rebuild `data/ego4d_train_clip_ids.txt` with the full
134,491-window set, smoke-test the 2-source (VGGSound+Ego4D, no AudioSet) mixed dataloader, then
launch the redesigned RUN-2.

**Lever 2 extraction COMPLETE**: 98,451/98,451 windows, 0 failures across all 4 shards (~615 min /
10.25h wall-clock). Ego4D train cache verified at exactly 134,491 `.pt` files (36,040 lever-1 +
98,451 lever-2). `data/ego4d_train_clip_ids.txt` rebuilt from the real cache directory listing.

**Batch-size/negatives investigation (user request)**: grepped historical 200x200-negatives
(batch-size=50/GPU) evidence. Found two clean 6000-step, 51k-corpus, no-fusion-bridge diagnostic
runs: `logs/m2_diag192.log` (192x192) final step 6000 R@1 43.75%/46.88%, shuffle_gap=0.6111 vs
`logs/m2_diag_negscale.log` (200x200) final step 6000 R@1 45.31%/46.88%, shuffle_gap=0.5958.
**Verdict: a wash, not a confirmed gain** -- vision->ambient R@1 identical, ambient->vision +1.56pp
for 200-neg but shuffle_gap actually lower, both within single-run step-to-step noise (no
multi-seed comparison exists). Also flagged: neither reference run used the fusion bridge RUN-2
actually needs (+50M params, real activation memory), and no 199k-corpus comparison exists at all
for either negative count -- both references are 51k-corpus only.

Per user instruction ("if the system doesn't OOM use 200, revert to 192 if it does"): ran a fresh
30-step smoke test with batch-size=50 AND the real fusion bridge AND the actual 2-source (VGGSound
197,462 + Ego4D 134,491, no AudioSet) mixed dataloader -- the one comparison that actually matters
since it's the real recipe, not a proxy. **No OOM** -- ran cleanly, `negatives=200x200` confirmed,
all 4 GPUs settled at ~94-96GB/97GB (near-full but stable). Batch composition confirmed exactly as
projected: `197462 vggsound, 134491 ego4d = 331953 total (59.5% vggsound, 40.5% ego4d)`.

**RUN-2 (final) LAUNCHED**: 20,000 steps, 4-GPU DDP, batch-size=50/GPU (negatives=200x200, per
user's tested preference over the original 192x192), lam_fusion=1.0/fusion_layers=2, source-
disjoint sampler, VGGSound 197,462 (fixed, full scale) + Ego4D 134,491 (no AudioSet this run, per
explicit user instruction to keep the batch-share test clean). Checkpoint dir:
`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200`. Gates unchanged from the original RUN-2 spec:
VGGSound R@1>=52%, Ego4D sibling-excl R@1>=RUN-1's 11.57%/10.68% (this run's ~40.5% Ego4D share is
a genuine attempt at RECOVERY toward the 22.2%-share run's 18.40%/18.40%, not just "no worse"),
within-modality cosine improving toward <=0.25. All three reported regardless of outcome.
