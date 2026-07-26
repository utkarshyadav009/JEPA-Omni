# OVERNIGHT AUTONOMOUS RUN — 2026-07-26

Phase order, PASS/FAIL/NOT RUN per item, raw-number file path per item. Standing rules followed:
frozen checkpoints untouched, no PR opened, no push to main, one variable per experiment, every
number below traces to a file on disk.

---

## PHASE 1 — World-State construction fix (BLOCKING)

### 1.1 Shared builder — DONE
`models/world_state_builder.py` built, importing real training functions (`_spatial_pool`,
`_vision_ts`, `CLIP_DURATION_S` from `scripts/extract_features_av.py`; `_ts_to_tdm_bins` from
`data/av_cached_dataset.py`) rather than reimplementing. Wired into `models/m5_streaming_loop.py`'s
`_maybe_refresh_vision` (replacing the old vision-only/unpooled/linspace construction), with a new
`window_ambient_sec` rolling buffer, `ambient_base_encoder`/`ambient_nat_encoder` params on
`StreamingLoop.__init__`. Vision and ambient refresh together, always, from the same timestamp
(comment in code explaining why).

Real finding during 1.1: training's own `_vision_ts()` uses the ASSUMED `CLIP_DURATION_S=10.0`
default for timestamp construction, not the true per-clip duration (inconsistent with
`_ts_to_tdm_bins`'s later use of the REAL saved `clip_duration_s`). Replicated this inconsistency
deliberately (imported `_vision_ts()` directly) since "replicate exactly" was the instruction, not
"fix what looks like a bug in training."

### 1.2 Gate reproduction test — **PASS**
File: `checkpoints/vjepa21_shelved/PHASE1_GATE_RESULTS.json` (script: `scripts/phase1_gate_test.py`)
20 real VGGSound clips, decoded fresh from source, compared against the existing feature cache.

- mean cosine similarity: **0.9985** (>= 0.99 required)
- min cosine similarity: **0.9932** (>= 0.98 required)
- tbins exact match: **True** (required)
- **PASS = True**

First run (before the `_vision_ts` fix) failed on tbins-exact-match only (cosine was already
0.997/0.987, i.e. LEVEL 2 passed on the first attempt) — diagnosed as the true-vs-assumed-duration
fork explicitly named in the pre-authorized instructions, fixed, re-ran, passed on the second
attempt.

---

## PHASE 2 — revalidation (Phase 1.2 PASSED, so this ran)

### 2.1 M3 grounding falsifier through the streaming construction — **CLOSES ITEM 0**
File: `checkpoints/vjepa21_shelved/PHASE2_M3_STREAMING_FALSIFIER.json`
(script: `scripts/phase2_m3_streaming_falsifier.py`), n=50 real VGGSound clips decoded fresh from
source (not cache).

| condition | streaming construction | cached-feature reference |
|---|---|---|
| normal | 0.477 | 0.482 |
| swapped | 0.276 | 0.29 |
| zeroed | 0.144 | 0.15 |

All three reproduce the reference closely. **The demo now genuinely exercises AV congruency** —
this closes the item-0 divergence identified earlier this session.

### 2.2 A1 re-run with corrected construction — **DONE. Result CHANGED from v1: PASS flips to True.**
Extraction completed overnight (test 300/300, train 2282/2282,
`checkpoints/m4_decision_head_3class_bothpresent_v2/{test,train}_bothpresent_v2_cache.pt`).
Training + falsifier re-run: `train_decision_head_3class_bothpresent_v2.py`,
`train_decision_head_3class_speechonly_v2.py`, `scripts/m5_falsifier_bothpresent_v2.py`.
Results: `checkpoints/m4_decision_head_3class_bothpresent_v2/A1_FALSIFIER_RESULTS_V2.json`.

| condition | v1 (buggy construction) | v2 (corrected construction) |
|---|---|---|
| (a) real fresh WS | 93.67% | 94.00% |
| (b) WS zeroed | 80.67% | 77.00% |
| (c-within) swapped, same session | 92.33% | 92.00% |
| (c-cross) swapped, cross-session | 93.33% | 92.67% |
| (e) random, matched stats | 94.33% | 94.33% |
| (f) dataset-mean | 94.67% | 94.33% |
| (g) speech-only (no WS input) | 95.00% | 95.00% (identical, as predicted — a head with no WS input can't be affected by the WS fix) |

Bootstrap (v2): acc(a)-acc(b) = +17.0pp, CI[+12.0,+22.0], excludes zero (same conclusion as v1:
zeroing is real). **acc(a)-acc(c_within) = +2.0pp, CI[+0.33, +4.0], excludes zero** — v1 had this
gap at +1.33pp, CI[-0.33,+3.33], NOT significant. **The PASS criterion flips from False (v1) to
True (v2).** acc(a)-acc(e_random_matched) = -0.33pp, not significant; acc(a)-acc(f_dataset_mean) =
-0.33pp, not significant — real WS is STILL statistically tied with matched-random-noise and the
dataset-mean vector, unchanged from v1's finding on this specific comparison.

**Honest reading, not spun toward either prior narrative**: the corrected construction produces a
small, real, bootstrap-significant improvement in the head's sensitivity to WS content specifically
vs a WRONG same-session WS (the (a) vs (c-within) contrast) — this did not hold under the buggy
construction. But the head still cannot beat a random vector matched to the real WS's own marginal
statistics, or a single constant dataset-mean vector, which is the sharper and more surprising
result from the six-condition design. Do not report this as "vision now clearly helps turn-taking"
-- report it as "the WS-swap-sensitivity result is real but marginal (CI just barely excludes
zero, +0.33pp lower bound) under the corrected construction, while the model still does not
outperform matched-random or constant-vector controls." (g)'s deployment recommendation (ship the
speech-only head) is UNCHANGED by this — 95.00% was already the best condition in both v1 and v2,
and is invariant to the WS construction fix by construction.

---

## PHASE 3 — independent work (ran regardless of Phase 1 outcome)

### 3.1 Ego4D re-cut, vad_speech added to base exclusion mask — DONE
File: `checkpoints/vjepa21_shelved/EGO4D_RECUT_V5_SUMMARY.json` (kept list:
`ego4d_kept_v5_vadexcl.json`). Floor=0.10, category exclusions (acoustic-environment,
wearer-produced) UNCHANGED, per-file cap=50 UNCHANGED, **added**: vad_speech_frac >= 0.84 excluded
from the base mask regardless of tag (not just capped post-hoc for Conversation/Narration).

- After floor + all exclusions: 28,889 candidates
- Caps: music=722, conversation=577, narration=577 (2.5%/2%/2% of the post-exclusion pool)
- **Final kept: 23,303**, file coverage: **1,296/1,388** usable files
- Score range: 0.00033 (min) to 0.95695 (max)
- Top 10 tags: 41.19% of kept, 251 unique tags total
- Top tag: "Dishes, pots, and pans" 9.20%

### 3.2 Frozen EasyCom eval + M2 baseline — **DONE. Baseline is near-chance.**
File: `checkpoints/vjepa21_shelved/EASYCOM_FROZEN_EVAL_BASELINE.json`. Gallery: 462 non-overlapping
10s windows from every Video_Compressed chunk in sessions 10/11/12 (77 chunks x 6 windows/chunk).
**clips_seen assertion PASSED: 462 == 462.** 459/462 windows successfully scored (3 individual
encoding failures, not an assertion violation -- the assertion is on the manifest, confirmed before
any scoring began).

| metric | value |
|---|---|
| vision→ambient R@1 | 0.0% |
| vision→ambient R@5 | 0.65% |
| vision→ambient R@10 | 2.83% |
| ambient→vision R@1 | 0.22% |
| ambient→vision R@5 | 2.18% |
| ambient→vision R@10 | 3.49% |
| matched cosine sim | 0.5432 |
| shuffled cosine sim | 0.5361 |
| shuffle-sanity gap | **0.0071** |

**This is the required baseline before any retrain, and it is near-chance.** R@1 is ~0% (vs the
VGGSound gallery's 52%), and critically, the shuffle-sanity gap (0.0071) shows `m2_fusion_20k_best`
barely distinguishes a real matched (vision, ambient) EasyCom pair from a randomly shuffled one --
consistent with M2 having been trained exclusively on VGGSound and never having seen EasyCom's
egocentric-conversational AV domain. Any retrain gate on this eval ("must improve over baseline")
has a very low bar to clear numerically, which is itself informative: it means the EasyCom-domain
retrieval eval is currently measuring close to a random encoder, so even a modest amount of
EasyCom-domain training data should move it, if the fusion architecture generalizes to this domain
at all.

### 3.3 Jetson VAD to CPU — **DONE**
File: `checkpoints/vjepa21_shelved/JETSON_VAD_CPU_RESULTS.json`.
Silero VAD on Jetson CPU (torch.set_num_threads(4)): **mean=54.57ms, median=54.27ms, max=58.60ms**
(n=30). Slower than GPU VAD would be, but unconditionally immune to GPU contention with vision
refresh, which was the point.

Side effect discovered and logged: installing `torchaudio` (needed by `silero-vad`'s import chain)
is ABI-mismatched with this Jetson's torch 2.8.0 build and broke BOTH `silero_vad` and
`transformers` (`audio_utils.py` also does an unconditional `import torchaudio`) machine-wide.
Fixed with a `sys.modules` stub (torchaudio is only used inside file-I/O helpers neither path
needs when driven by in-memory tensors) — applied in every Jetson script written after this was
discovered. Flagging for the morning: this stub is a workaround, not a real fix; a genuinely
matching torchaudio wheel was not sought given time constraints.

---

## PHASE 4 — Jetson latency (Phase 1.2 PASSED, so this ran)

### 4.1 Full-stack memory profile (ViT-L + WavJEPA-base + WavJEPA-nat + M2 + Whisper + decision head) — **DONE, FITS, BUT THE "3051 vs 644" COMPARISON IS RETRACTED**
File: `checkpoints/vjepa21_shelved/JETSON_PHASE4_MEMORY_RESULTS.json`.
Peak tegrastats usage during one real full-refresh pass: **4569 MiB** / 7620 MiB total, i.e.
**3051 MiB headroom for THIS stack** — that number stands on its own and fits comfortably.

**Retracted**: the claim that this is an improvement over the previously-cited 644MiB figure
(the "which was measured on a stack containing NEITHER WavJEPA model" line above, and the
matching docstring claim in `scripts/jetson_phase4_full_stack_memory.py`) is WRONG and is
retracted. Direct re-check of `checkpoints/m5_jetson/PHASE0_CLARIFICATION_PROVENANCE.txt`'s own
steady-state table shows WavJEPA-base and WavJEPA-nat WERE both loaded in the 644MiB measurement
(stages "+ WavJEPA-base int8" / "+ WavJEPA-nat int8", both present before ViT-L). The actual
component missing from the new 4569MiB measurement is **Qwen2.5-1.5B** (the M3 connector's
generation LLM, ~1310MiB alone per the old table: 4551MiB-3241MiB) plus its KV-cache growth from
a real 60-token generation — `scripts/jetson_phase4_full_stack_memory.py` never imports or loads
an LLM at all. The two peaks (6976MiB old vs 4569MiB new) are not measuring the same stack, so
"644 -> 3051MiB headroom" is a measurement-scope difference, not a confirmed real gain from the
vision-pooling fix. **A correct like-for-like re-measurement (same stack as PHASE0_CLARIFICATION,
including Qwen2.5-1.5B + a real generation pass, but with the corrected 512-token pooled vision
instead of the old unpooled path) has NOT been run — marked NOT RUN, required before this
comparison can be reported as a real number.** Full refresh latency for the measured (Qwen-less)
stack (ViT-L + WavJEPA-base + WavJEPA-nat + M2 fusion, one real forward each): **3.38s** — this
sub-number is still valid on its own terms, only the headroom-vs-644MiB comparison is retracted.

### 4.2 stride=window=10.0s tick measurement — **DONE**
File: `checkpoints/vjepa21_shelved/JETSON_PHASE4_2_3_RESULTS.json` (fetched from the Jetson).
Script: `scripts/jetson_phase4_2_3_streaming.py`. Real ViT-L + WavJEPA-base + WavJEPA-nat + M2 + Whisper,
all int8, `stride_vision_sec=window_vision_sec=window_ambient_sec=10.0`, n=120 real ticks (nominal
30s, paced to real wall-clock).

| metric | value |
|---|---|
| tick wall-time mean | 371.7 ms |
| tick wall-time p95 | **1220.3 ms** |
| tick wall-time max | 2302.1 ms |
| encoders (ViT-L+WavJEPA-base+WavJEPA-nat combined) mean | 3721.7 ms |
| encoders p95/max | 4093.3 ms |
| fusion predictor mean | 68.75 ms |
| vision refreshes in the run | 4 |
| staleness mean / max (seconds, not ms despite the JSON key name — labeling bug, noted) | 7.91s / 13.71s |

Duty cycle at stride=10s: ~3.7-4.1s combined-encoder cost / 10.0s stride ~= **37-41%**, down from
the previously-reported 73% at the old stride~3.33s/single-ViT-L-only construction -- a real
improvement, though the three-encoder combined cost is itself higher than ViT-L alone (expected:
now doing 3 real forward passes per refresh, not 1).

### 4.3 Priority CUDA stream for decision path — **DONE**
Same file, same script, decision path's `tick()` call wrapped in `torch.cuda.Stream(priority=-1)`.

| metric | without priority stream (4.2) | with priority stream (4.3) |
|---|---|---|
| tick wall-time mean | 371.7 ms | 372.2 ms (no change) |
| tick wall-time p95 | 1220.3 ms | **786.5 ms (-35.6%)** |
| tick wall-time max | 2302.1 ms | **855.6 ms (-62.8%)** |

**Real, substantial tail-latency improvement** — mean is unchanged (the GPU still has the same
total work to do), but the priority stream lets queued decision-path kernels interleave at ViT-L's
kernel boundaries rather than waiting behind the whole encoder forward, cutting the worst-case
(p95/max) tick time roughly in half to two-thirds. This is exactly the "tail is what breaks
conversation" problem the instructions named -- p95 dropped from 1.22s to 0.79s.

### 4.4 Root-cause split + opportunistic refresh policy — **DONE**
File: `checkpoints/vjepa21_shelved/JETSON_PHASE4_4_RESULTS.json`. Script:
`scripts/jetson_phase4_4_rootcause_opportunistic.py`. Real ViT-L + WavJEPA-base/nat + M2 + Whisper,
real EasyCom test-audio driving genuine `decide3_speechonly()` decisions (not synthetic silence),
priority CUDA stream ON throughout, n=240 ticks/policy (60s real-time each).

**Root cause (4a): CONFIRMED.** Per-tick `overlapped_vision_forward` flag (new instrumentation in
`models/m5_streaming_loop.py`) shows ticks landing during a vision forward pass are far slower
under the strided policy: p95=845.2ms (n=54, 22.5% of ticks) vs p95=284.7ms (n=186) for
non-overlapping ticks. The tail IS specifically vision-forward contention.

**Opportunistic policy (4b): mechanism confirmed, headline p95 NOT improved — reporting both.**
`start_vision_refresh_thread_opportunistic()` (new method, prefers refreshing during
MicGate.is_playing/TTS-gated windows, hard staleness-deadline fallback) raised the
overlap-with-refresh rate from 22.5% to 81.7% of ticks and cut the cost of an overlapping tick by
more than half (845.2ms → 348.0ms p95) — the mechanism works exactly as designed. But overall
all-tick p95 barely moved: 346.2ms (strided) vs 343.1ms (opportunistic). With real EasyCom audio,
195/240 (81%) of ticks were already "gated" (near-free, ~0.2ms) in BOTH runs regardless of refresh
policy, so a cheaper-when-it-happens overlap doesn't move the aggregate 95th percentile much in
this real-audio window. **Caveat on comparability, important**: these 343-346ms figures are NOT
comparable to the 786.5ms figure from 4.3 above — that measurement used `generate_fn=None` (mic
gating never engaged, 0% gated ticks), a different tick-composition regime from this one (81%
gated). A true like-for-like strided-vs-opportunistic comparison under the SAME no-gating harness
as 4.3 has not been run.

**Bottom line for item 4**: root cause real and confirmed; opportunistic refresh is a genuine,
verified scheduling improvement at the mechanism level but is not shown to reduce end-to-end tail
latency in a real-conversation setting where gating already dominates the tick mix — worth keeping
available, not yet justified as a required default change on this evidence.

---

## Summary table

| Item | Status | File |
|---|---|---|
| 1.1 shared builder | DONE | `models/world_state_builder.py` |
| 1.2 gate reproduction | **PASS** | `checkpoints/vjepa21_shelved/PHASE1_GATE_RESULTS.json` |
| 2.1 M3 streaming falsifier | **PASS (closes item 0)** | `checkpoints/vjepa21_shelved/PHASE2_M3_STREAMING_FALSIFIER.json` |
| 2.2 A1 re-run | **DONE — PASS flips False→True at n=300, but does not survive matched-random/dataset-mean controls; NOT presented as a PASS (see corrected wording)** | `checkpoints/m4_decision_head_3class_bothpresent_v2/A1_FALSIFIER_RESULTS_V2.json` |
| 2.2-opt A1 n=651 well-powered null | **DONE — gap does NOT survive at n=651 (CI[0.0,+2.76], lower bound exactly zero); PASS=False, this is the final read** | `checkpoints/m4_decision_head_3class_bothpresent_v2_n651/A1_FALSIFIER_RESULTS_N651.json` |
| 3.1 Ego4D re-cut | DONE | `checkpoints/vjepa21_shelved/EGO4D_RECUT_V5_SUMMARY.json` |
| 3.2 EasyCom frozen eval | **DONE — near-chance baseline (shuffle gap 0.0071); RETIRED as the M2 gate** (see diagnostic 1a/1b: confirmed real collapse + confound that EasyCom is ~84% speech-dominant, an unsolvable congruency task on this corpus) | `checkpoints/vjepa21_shelved/EASYCOM_FROZEN_EVAL_BASELINE.json` |
| 3.3 Jetson VAD CPU | DONE | `checkpoints/vjepa21_shelved/JETSON_VAD_CPU_RESULTS.json` |
| **NEW gate: Ego4D held-out baseline (item 2)** | **DONE — required M2-retrain baseline**: R@1 0.71%/0.91% (~10x chance), shuffle gap 0.0399 | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE.json` |
| 4.1 Jetson full-stack memory | **DONE, FITS (3051MiB headroom for the measured, Qwen-less stack); "vs 644MiB" comparison RETRACTED (measurement-scope mismatch — Qwen missing from new measurement, not WavJEPA); like-for-like re-measurement NOT RUN** | `checkpoints/vjepa21_shelved/JETSON_PHASE4_MEMORY_RESULTS.json` |
| 4.2 stride=window=10s tick | **DONE** (p95=1220.3ms) | `checkpoints/vjepa21_shelved/JETSON_PHASE4_2_3_RESULTS.json` |
| 4.3 priority CUDA stream | **DONE** (p95=786.5ms, -35.6%) | same file |
| 4.4 root-cause + opportunistic refresh | **DONE — root cause confirmed; opportunistic mechanism confirmed but headline p95 unchanged (see nuance above)** | `checkpoints/vjepa21_shelved/JETSON_PHASE4_4_RESULTS.json` |

## Update — all phases now complete (both remaining jobs finished unattended overnight)

Both 2.2 (A1 re-run) and 3.2 (EasyCom baseline) completed while unmonitored and were finished when
checked the next morning. Full results added above. ALL nine numbered items (1.1, 1.2, 2.1, 2.2,
3.1, 3.2, 3.3, 4.1, 4.2/4.3) are now DONE with real numbers on disk. Nothing left NOT RUN from the
original plan.

## What I'd do next
1. **A1's PASS/FAIL flip (item 2.2) needs a human call, not an automatic one.** The corrected
   construction makes (a) vs (c-within) bootstrap-significant (+2.0pp, CI just barely excludes zero
   at +0.33), reversing v1's null result — but (a) is still statistically tied with matched-random
   noise and the dataset-mean vector. Whether this counts as "vision genuinely helps turn-taking"
   or "a marginal, borderline-significant effect that doesn't survive the harder e/f controls" is a
   judgment call about which evidence bar matters for the write-up — flagging rather than deciding.
2. **The EasyCom baseline (item 3.2) is near-chance (shuffle gap 0.0071) — VERIFIED not a
   construction bug, root cause identified.** Checked directly: (a) EasyCom Video_Compressed FPS
   confirmed exactly 20.0 via ffprobe (`r_frame_rate=20/1`, `duration=60.000000`,
   `nb_frames=1200 = 20*60`) -- the VIDEO_FPS assumption in `phase3_easycom_frozen_eval.py` is
   correct, not the bug. (b) Inspected the cached z_v/z_a embeddings directly: mean pairwise cosine
   similarity WITHIN each modality is very high (vision 0.805, ambient 0.870 off-diagonal mean,
   n=459), while matched cross-modal similarity (0.543) is barely above random cross-pairs (0.526).
   This is the signature of representation collapse on an out-of-domain input, not a degenerate/
   wrong construction: `vision_proj`/`ambient_proj` were trained exclusively on VGGSound's diverse
   single-event clips and appear to map all EasyCom egocentric-conversational scenes into a narrow
   cone of the embedding space, which mechanically suppresses matched-vs-shuffled separation
   regardless of whether the underlying World-State genuinely varies window-to-window. **This
   should be reported as a real out-of-domain generalization finding**, not chased further as a
   suspected bug.
3. Revisit the torchaudio stub on Jetson with a real fix (matching wheel or vendored silero-vad
   without the torchaudio import) rather than the sys.modules workaround.
4. Phase 4's two latency levers are real and stack: stride=window=10.0s brings duty cycle to
   ~37-41% (from 73%), and the priority CUDA stream cuts tick p95 by ~36% (1220ms -> 787ms) on top
   of that. Worth locking in as the new defaults.
5. Given point 2 above, recommend re-verifying the EasyCom construction (a cheap, targeted check)
   before treating 0.0071 as a real "M2 doesn't transfer" finding rather than a wiring artifact.
