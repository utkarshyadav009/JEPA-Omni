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

## Next row due

Whatever M4d/production work comes next — genuinely-paired EasyCom Phase 2
data, Ego4D LAM proactive-attention integration, or a broader recording
setup to test generalization beyond EasyCom's single acoustic domain. Also
outstanding: the fresh-vs-stale World-State falsifier flagged above, and
resolving where the disputed 16-frame/fp16/3.6s/1.16s figures actually
came from (a different session, a different project, or a
misremembering) before M5 Day-2/Day-3 work resumes.
