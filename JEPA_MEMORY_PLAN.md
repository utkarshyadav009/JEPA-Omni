# JEPA_MEMORY_PLAN.md — North-star track: JEPA-space multimodal memory

Working log + plan for BACKLOG.md's North-star item (`bmo-pipeline-vision` memory):
**BMO recognizes people/things by their highly-abstracted joint visual+audio JEPA
embeddings, not pixels/faces.** Runs ASYNCHRONOUSLY on mercury (4x Blackwell),
alongside — not blocking — the production/optimization push.

Style follows `RESULTS_TABLE.md` / `NEGATIVE_RESULTS.md` / `PERCEPTION_FINDINGS.md`:
every number carries provenance, gates are pre-registered before the run, negative
results are recorded with the same weight as positive ones.

Started 2026-08-10.

---

## STATUS SUMMARY (updated 2026-08-11) — read this first

| phase | state | headline number |
|---|---|---|
| 1A vision falsifier (EasyCom) | done | identity IS in frozen ViT-L: within-session top-1 **0.860** vs 0.446 chance; cross-session AUC 0.639 (FAIL, → head required) |
| 1B voice falsifier (VoxCeleb2) | done | cross-session 250-way top-1 **0.250** vs 0.004 chance (62x), AUC 0.832 |
| 1C follow-ups | done | M2 adds nothing to ViT-L for identity; household-scale voice ID **0.758 @ N=5** untrained |
| 2a voice identity head | done | TAR@FAR=1% **0.273 → 0.419** (+54%); overfit, data-limited |
| 2b query predictor | done | within-clip **0.891** vs 0.167 chance, swapped-query **0.004** |
| 2c joint AV identity head | superseded | TAR@FAR=1% 0.691 (33.8k clips / 1.4k spk) |
| unified query architecture | done | `m2+vision+ambient` best; **M2 and ViT-L are complementary** |
| 3 memory store + G7 | done | **G7 PASS** at the 1% operating point |
| DDP+GradCache | done | R@1 **0.458 → 0.715 (+56%)**, gradient-equivalence verified |
| voice head, full corpus | done | 122k clips / 4k spk: TAR@FAR=1% **0.705** (+68% over 1.4k-spk head) |
| λ-within two-term loss | done | **λ=0.3 recommended**; clean dose-response, swap stays 0.003 |
| **joint AV head, full corpus** | **done** | 107k clips / 4.4k spk: TAR@FAR=1% **0.765**, AUC **0.966** — BEST |
| **final G7 (joint, full)** | **done** | **PASS** @1% FAR: 51.5% correct, **0.4% wrong-name**, 1.1% FAR |

**The two architecture decisions that came out of measurement, not assumption:**
1. The identity head reads **raw pooled ViT-L**, not the M2 world-state (M2 discards ~16pp
   of identity, and fusing it back recovers none of it).
2. The query predictor reads **M2 AND ViT-L AND ambient together** — M2 is NOT redundant
   (`m2+vision` 0.478 beats both `m2` 0.385 and `vision` 0.447), so AV congruence is real
   and earned. Nothing in the existing pipeline is discarded.

**Biggest open risk:** recognition rate. At the recommended 1-2% FAR operating point only
~46-52% of genuine household queries are accepted. The head is badly overfit (train loss
0.0006), so this is a data problem, and the queued full-corpus retrain targets it directly.

---

## 0. Session preflight (2026-08-10) — measured, not assumed

**All 4 Blackwells are free.** `nvidia-smi`: 4x NVIDIA RTX PRO 6000 Blackwell Server
Edition, `0MiB / 97887MiB` each, 0% util, **no running processes**, P8 idle @ 25-27C.
Nothing to kill, nothing to wait for. Full 4-GPU DDP is available immediately.

**Disk is the real blocker, and it is hard.**

| Mount | Size | Avail | Note |
|---|---|---|---|
| `/mnt/Raid-Storage-2` | 7.0T | **0** (100%) | every feature cache lives here |
| `/` (nvme7n1p2) | 879G | **9.2G** (99%) | `checkpoints/` lives here |
| `/dev/shm` (tmpfs) | 756G | 750G | volatile — lost on reboot |

RAID breakdown (`du -sh`): `action100m_videos` 2.3T, `feature_cache_action100m` 1.5T,
`ego4d_probe` 1.5T, `feature_cache_vgg51k` 764G, `feature_cache_ego4d_train_v1` 518G,
`easycom` 145G, `action100m_preview` 37G.

**Consequence: no plan step below may assume new bulk feature extraction on RAID.**
Phase 1 and 2 are deliberately designed to fit in the 9.2G root + tmpfs. See §6.

---

## 1. Verified baselines

### 1a. M2 — locked joint AV predictor

`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`
(197,007 VGGSound + 134k Ego4D windows, 200x200 in-batch negatives, no AudioSet).

| Eval | v→ambient R@1 | ambient→v R@1 | Gate | Verdict |
|---|---|---|---|---|
| VGGSound retrieval (step19000, locked) | 53.27% | 53.72% | ≥52% | **PASS** |
| VGGSound retrieval (step20000, log tail) | 52.69% | 51.20% | — | step19000 is genuinely better |
| Ego4D held-out, sibling-excluded | 27.60% | 27.00% | ≥18.40% | **PASS** |
| Ego4D within-modality cosine (v / a) | 0.4358 | 0.3893 | ≤0.25 | **NEVER MET**, any checkpoint |

Training-end diagnostics (`logs/m2_run2_final.log`, step 19999): `pred=0.4250`,
`sigreg=1.9852`, `eff_rank=27.9/49`, `contrastive=0.0965`, `c_acc=0.980`,
`matched_cos=0.7217`, `shuffled_cos=0.0226`, `shuffle_sanity_gap=0.6990`.
`best_loss=0.4409`.

### 1b. M3 — connector (superseded, kept as the baseline it replaced)

`checkpoints/m3_multigran_richcaption_v2/last.pt`, `logs/train_m3_richcaption_v2.log`:
`FINAL test_loss=1.6724`, `mean_word_overlap_f1=0.317` over 30 sampled generations.

Grounding falsifier (`logs/m3_multigran_falsifier_rerun.log`, n=200, normal vs
swapped vs mode vs random):

| caption field | stream | normal | swapped | mode | random |
|---|---|---|---|---|---|
| gpt_action_brief | visual | 0.431 | 0.096 | 0.115 | 0.078 |
| gpt_action_detailed | visual | 0.302 | 0.173 | 0.096 | 0.154 |
| gpt_summary_brief | visual | 0.355 | 0.125 | 0.074 | 0.120 |
| gpt_summary_detailed | visual | 0.281 | 0.192 | 0.158 | 0.169 |
| gpt_sound_acoustic | acoustic | 0.440 | 0.283 | 0.210 | 0.282 |

Frozen-LLM reference: M3 F1 0.471 / 0.268 / 0.274, cos 0.724. M4b cos 0.517 / 0.152 / 0.517.
**M3 connector is DROPPED** per `bmo-pipeline-vision`; superseded by the embedding predictor below.

### 1c. M2 embedding predictor (the "M2/M3 prediction model" this track builds on)

`train_m2_embed_predictor.py`: frozen `AVJepaPredictor.encode_pre_pool_tokens` →
trainable `Predictor(mode=mlp)` → bidirectional InfoNCE vs `TextTarget`
(EmbeddingGemma-300M, `gpt_action_detailed`). 19.54M trainable params.
Score = mean(VGGSound R@1 both directions) + mean(Action100M R@1 both directions).

**Read directly out of each checkpoint's own `results_log`** (authoritative — see the
correction in §2b), not from prose:

| run | ckpt | step | VGGSound R@1 | Action100M R@1 | score |
|---|---|---|---|---|---|
| action100m_isolated (`brief`) | last | 5999 | — | 2.5 / 2.5 | 2.50 |
| action100m_isolated_detailed | last | 5999 | — | 5.6 / 6.6 | 6.10 |
| mlp_combined (`brief`) | last | 4999 | 12.9 / 12.4 | 2.7 / 2.7 | 15.35 |
| mlp_combined_detailed | last | 5999 | 16.6 / 13.9 | 8.8 / 8.0 | 23.65 |
| ddp_gradcache_bs2048 | last | 499 | 16.4 / 16.5 | 9.2 / 9.0 | 25.55 |
| ddp_gradcache_bs4096 | best | 359 | 29.8 / 30.1 | 20.7 / 21.7 | 51.15 |
| ddp_gradcache_bs8192 (240 steps) | best | 239 | 29.2 / 27.0 | 19.4 / 20.6 | 48.10 |
| ddp_gradcache_bs8192_lr424 (sqrt-LR) | best | 439 | 32.2 / 33.6 | 24.4 / 25.7 | 57.95 |
| ddp_gradcache_bs8192_stepcount_followup | best | 599 | 35.4 / 35.3 | 23.5 / 27.2 | 60.70 |
| **ddp_gradcache_bs16384 (BEST OVERALL)** | **best** | **799** | **34.3 / 34.6** | **27.8 / 27.4** | **62.05** |
| ddp_gradcache_bs16384 | last | 999 | 34.4 / 33.8 | 25.1 / 27.7 | 60.50 |
| ddp_gradcache_bs16384_softinfonce | best | 699 | 32.3 / 31.2 | 26.0 / 28.3 | **58.90** |
| ddp_gradcache_bs16384_softinfonce | last | 799 | 32.4 / 32.8 | 26.3 / 26.1 | **58.80** |

**Reference checkpoint for everything downstream in this track:**
`checkpoints/m2_embed_predictor_mlp_ddp_gradcache_bs16384/best.pt` (step 799, score 62.05).

### 1d. Jetson inference reality (the deployment envelope this track must live inside)

`scripts/jetson_embed_predictor_latency.py`, real Orin, 2026-08-04 — the predictor path
is essentially free, perception is everything:

| stage | latency |
|---|---|
| AV encode (64f) | 2366ms — ViT-L 1907ms, WavJEPA-base 159ms, WavJEPA-nat 287ms |
| M2 predictor | 160ms |
| **embedding Predictor** | **5ms** |
| **nearest-neighbour lookup** | **3ms** |
| **TOTAL** | **2534ms**, memory 6472/7620 MiB |

At the production 16-frame setting (`PERCEPTION_FINDINGS.md`, 2026-08-07): ViT-L 292ms,
WavJEPA-base 252ms, WavJEPA-nat 469ms, predictor+glue 232ms, **total ~1247ms**.

**The load-bearing number for this whole track: retrieval against a stored embedding
bank costs 3ms.** A memory *lookup* is free on-device. Only *enrolling* costs perception
time, and perception is already off the response critical path.

---

## 2. New findings this session

### 2a. Soft-InfoNCE is a NEGATIVE result — never recorded until now

`checkpoints/falsifier_tracking.md` ends mid-experiment: it records the Soft-InfoNCE
implementation, the CPU unit test, and "queued for the next training round", but the run
itself **completed on 2026-08-04 and its result was never written down.**

Read from the checkpoint's own `results_log`:

| | best | matched step 799 |
|---|---|---|
| hard InfoNCE (bs16384) | **62.05** @ step 799 | **62.05** |
| Soft-InfoNCE (`--soft-temp 0.05`) | 58.90 @ step 699 | 58.80 |

**Soft-InfoNCE lost by ~5.1% relative, and the matched-step comparison (58.80 vs 62.05 at
step 799) rules out early-stopping as the explanation.** It was also behind at *every*
logged eval step, not just at the end (step 99: 15.0/14.9 vs 18.8/17.1; step 399:
28.2/28.2 vs 30.1/29.8). This is a clean, consistent negative.

Caveat that keeps it from being fully decisive: this run used the **trainable-proj**
similarity source, the exact circularity the session itself identified. The
`--soft-infonce-frozen-sim` variant (using `TextTarget.encode_text_frozen_raw()`) was
implemented, verified on CPU, and **never run** — no checkpoint directory exists for it.
So the honest verdict is: *soft targets from a moving similarity source hurt; the frozen-source
variant is untested.* One cheap run closes this (§5, Phase 0).

### 2b. Small provenance correction to `falsifier_tracking.md`

Line ~2231 quotes the bs16384 step-799 result as "VGGSound R@1 33.6%/35.0%, Action100M
R@1 26.2%/28.1%". The checkpoint's own `results_log` at step 799 says **34.3/34.6 and
27.8/27.4**. The *score* (62.05) is correct and unchanged; only the per-dataset split is
misquoted. Nothing downstream depends on it — noted so the table in §1c is trusted over
the prose.

---

## 3. The V-JEPA 2.1 question, answered with the real numbers

The user's recollection was right that a Jetson latency check was done. It exists, on
disk, and it is decisive. Both 2.1 checkpoints are already downloaded — no re-download
needed: `checkpoints/vjepa21_shelved/vjepa2_1_vitb_dist_vitG_384.pt` (1.66GB) and
`vjepa2_1_vitl_dist_vitG_384.pt` (5.15GB).

### 3a. Clean head-to-head, Blackwell, n=20, warmed up
`checkpoints/vjepa21_shelved/HEAD_TO_HEAD_LATENCY.json`

| encoder | latency (median) | peak activation |
|---|---|---|
| **V-JEPA 2 ViT-L 256px bf16 (current production)** | **47.9ms** | 931.8 MB |
| V-JEPA 2.1 ViT-B 384px | 64.1ms | 1155.6 MB |
| V-JEPA 2.1 ViT-L 384px | 174.6ms | 2561.2 MB |

2.1 ViT-B is **slower than the current ViT-L** despite 3.5x fewer params — the 384²
token penalty dominates parameter count.

### 3b. Jetson, real hardware, decisive
`checkpoints/vjepa21_shelved/B_2B_JETSON_VITL_DECISIVE.txt` +
`vjepa21_vitl_jetson_results.json` (2026-07-25, MAXN_SUPER, clocks confirmed by sysfs,
int8 via the same `q_int8_cpu_then_move` recipe as production)

| encoder | Jetson forward | tegrastats | torch max_alloc | output |
|---|---|---|---|---|
| V-JEPA 2 ViT-L 256px | 2.43–2.45s | 3398–5354 MiB | 1373–1376 MiB | (1, 8192, 1024) |
| **V-JEPA 2.1 ViT-L 384px** | **37.98s** | 3585 MiB | 1499 MiB | (1, **18432**, 1024) |

**15.6x slower**, far worse than the Blackwell ratio (3.64x) predicts — memory was NOT
the blocker, latency was. Standing decision, already recorded as final:
**2.1 ViT-L is dissertation/server-side only; the Jetson demo track stays on V-JEPA 2 ViT-L, permanently.**

### 3c. Correction to the recollection — the base model was NEVER Jetson-tested

2.1 **ViT-B was measured on Blackwell only** (64.1ms above) and dropped there, before any
Jetson test. There is no `vjepa21_vitb_jetson_results.json` — I searched
`checkpoints/vjepa21_shelved/`, `scripts/`, `logs/`, and the prior session's scratchpad
(`dc0bf6a0-…`); the only Jetson 2.1 script is `scripts/jetson_b1_vjepa21_vitl_test.py`.
So the ViT-B-on-Jetson number does not exist. It *can* be produced in ~1 hour of Jetson
time if wanted — but see §3d for why it probably shouldn't gate this track.

### 3d. Honest re-diagnosis: "the predictor doesn't have fine details" is only partly a 2.1 problem

The premise is right — the predictor genuinely lacks fine detail. But this repo has
already measured *three* causes, and the encoder version is the most expensive and least
evidenced of them:

1. **Caption specificity — measured, large, already fixed.** `gpt_action_brief` averages
   3.1 words (37.5% are ≤2 words); `gpt_action_detailed` averages 26.1 words. Switching
   field alone moved isolated Action100M R@1 from 2.5–3.1% → 5.6–8.8% (2–2.5x), and it
   is why the current 62.05 checkpoint exists at all. **Detail in = detail out.**
2. **Spatial pooling — measured, large, NOT yet ablated.** `scripts/extract_features_av.py::_spatial_pool`
   mean-pools each temporal bin's 256 spatial tokens (16x16) down to **16** (4x4) via
   `avg_pool2d(kernel=4, stride=4)` at *extraction* time. That is a **16x discard of
   spatial detail**, applied before the predictor ever sees a token. This is the single
   most likely mechanical cause of "can't answer fine-grained questions about the scene"
   and it is far cheaper to test than swapping encoders.
3. **Encoder version — the expensive one.** 2.1 at 384px yields 18,432 tokens vs the
   current 8,192. `ITEM3_THROUGHPUT_PREFLIGHT.txt` measured 18,432 tokens as an outright
   **CUDA OOM at batch 32 on a 96GB GPU**; 8,192 tokens already cost 1911ms/step and
   90.8GB peak vs the pooled 512-token config's 211.8ms/step and 16.1GB. Plus the
   re-extraction disk cost against a **100%-full RAID** (§0), at 37.75MB/clip unpooled.

**Recommendation: order the detail work 2 → 1 → 3, not 3 first.** A pooling ablation
(4x4 → 8x8 = 64 tokens/bin) is a controlled single-variable experiment that reuses the
existing pipeline, needs no new encoder, and is bounded by GPU time we currently have
free. If detail is still missing after that, 2.1 ViT-B server-side becomes the justified
next step rather than a guess. This is a sequencing recommendation, not a refusal — if
you want 2.1 first, the checkpoints are already local and §5 Phase D can be pulled forward.

---

## 4. The architectural problem the memory idea has to solve first

This is the part that changes the shape of the plan, so it goes before the phases.

### 4a. Caption-InfoNCE space is trained to be identity-INVARIANT

The embedding predictor's entire objective is to land next to an EmbeddingGemma encoding
of a caption like *"a person kneels and adjusts a bicycle chain."* Two **different people**
performing the same action produce near-identical captions — so InfoNCE actively pulls
their embeddings **together**. Identity is nuisance variation under this loss.

**Reusing `z_p` (the 1536-d predictor output) as the memory key is therefore expected to
fail at telling people apart, by construction.** Any version of "store `z_p`, look it up
later" inherits this. The memory needs its own head over the frozen M2 tokens, trained
with an identity objective — not a reuse of the caption-aligned vector.

### 4b. The one domain we have with real identities is exactly where M2 collapses

EasyCom is the only corpus on disk with repeated, tracked people (12 sessions,
7 participant slots/session, 84 `Participant_Photos`, per-participant `Close_Microphone_Audio`,
`Face_Bounding_Boxes`, `Head_Bounding_Boxes`). It is the natural identity benchmark.

It is also, measurably, where the current M2 world-state falls apart
(`VGGSOUND_COLLAPSE_CHECK_1545.json`, `EASYCOM_FROZEN_EVAL_BASELINE.json`):

| within-modality mean off-diagonal cosine | VGGSound (1545) | EasyCom (462) |
|---|---|---|
| vision | 0.0308 | **0.8048** |
| ambient | 0.0289 | **0.8703** |
| shuffle-sanity gap | 0.6604 | **0.0071** |

EasyCom windows collapse into a narrow cone under the same checkpoint that behaves
cleanly on VGGSound. Compounding it, EasyCom is ~pure speech: only 19/462 windows
(4.11%) carry any non-speech acoustic event above the 0.10 floor, mean speech
probability 0.839 (`EASYCOM_EVENT_COMPOSITION.json`).

**Read for this plan:** an out-of-domain generalization gap is already documented on the
exact corpus the memory track needs. Phase 1 must *measure* identity separability rather
than assume it, and a FAIL there is a genuinely likely outcome that changes the design.

### 4c. WavJEPA is an ambient encoder, not a speaker encoder

The north-star description is "a particular voice signature." WavJEPA-base/nat are
trained on environmental/ambient audio, are fed **duplicated mono** for the nat branch
(`world_state_builder.py:156`), and produce ~orthogonal outputs on real speech
(`PERCEPTION_FINDINGS.md`: per-token cos(base, nat) = 0.008). There is no evidence in
this repo that they encode speaker identity.

**But the production stack already carries a speech encoder that plausibly does:**
`models/m4_speech.py::MoonshineSpeechEncoder` (`UsefulSensors/moonshine-base`, 416-d),
already resident on the Jetson, already running per turn at **37ms**. Its encoder states
are computed anyway for STT — using them as the audio-identity feature is **zero
additional on-device latency**. That should be tested as the audio identity source
alongside WavJEPA, not instead of considering it.

### 4d. What this means for the thinker→predictor query loop

The north-star's item 1 ("thinker asks perception for more detail") and item 4 (memory)
are separable, and memory is the cheaper one to land first:
- **Memory** = an extra head + a vector store + a threshold. Lookup measured at 3ms.
  No M2 retrain required for v1.
- **Query-conditioned predictor** = a real M2+predictor retrain with a query input path.
  Expensive, and its payoff is bounded by §3d's detail problem, which is unsolved.

So the plan lands memory first, and treats the query loop as Phase E, gated on the
detail work.

---

## 5. The plan

Pre-registered gates. Every phase writes its result back to this file — including
failures — before the next phase starts.

### Phase 0 — close the open baseline (0.5 day, 4 GPUs, no new disk)
1. Run the never-executed `--soft-infonce-frozen-sim` variant at the bs16384 config
   (573,053 pairs, lr 3e-4, 800 steps) so the soft-target question is actually settled
   rather than half-answered. **Gate: beats 62.05 → adopt; else record as closed-negative.**
2. Re-score `bs16384/best.pt` on the fixed held-out sets to confirm 62.05 reproduces from
   the checkpoint on disk (guards against the §2b class of provenance drift).
3. Append both to this file.

*Cost: ~4h GPU. Disk: checkpoints only (~80MB each), fits in the 9.2G root headroom.*

### Phase 1 — THE FALSIFIER: does the frozen M2 representation carry person identity at all? (1–2 days)

This is the phase that decides whether the whole idea is viable, and it is deliberately
first, cheap, and capable of failing.

- Build a **global identity map** for EasyCom. Participant IDs are per-session slots
  (1–7), *not* global identities — already confirmed: ID=1 is the same physical person
  across Sessions 1, 5, 10, 11; ID=3 is a different person between Session 2 and 11; ID=2
  is an anonymized silhouette in two sessions. Cluster the 84 `Participant_Photos` into
  global IDs (one-time, human-verified — this must not be guessed).
- Extract, for N windows per participant: frozen M2 `encode_pre_pool_tokens`, the pooled
  World-State, WavJEPA-base/nat ambient features, and Moonshine encoder states from that
  participant's `Close_Microphone_Audio`.
- Measure identity separability four ways, **session-disjoint** (enroll on one session,
  recognize in another — never within-session, which would leak clothing/lighting/seating):
  linear probe accuracy, k-NN top-1, within-vs-between-identity cosine margin, and
  the same metrics on a **matched-random control** (the repo's own standard).

**Pre-registered gates** (set now, before the numbers exist):
- **G1 (viability):** session-disjoint k-NN top-1 identity accuracy ≥ **3x chance** on at
  least one feature stream. Below that, the frozen features do not carry usable identity
  and Phase 2 is not attempted as designed.
- **G2 (attribution):** the winning stream must beat the matched-random control by a
  margin whose bootstrap CI excludes 0. Guards against the §4b collapse producing a
  vacuous number.
- **G3 (which modality):** report vision / WavJEPA-ambient / Moonshine-speech / joint
  separately. A likely outcome given §4b–4c is *speech carries identity, ambient does not,
  vision is degraded by collapse* — that result is informative and reshapes Phase 2 rather
  than killing it.

*Disk: EasyCom windows only, a few GB — fits in tmpfs. No RAID writes.*

### Phase 2 — the identity head (2–3 days, gated on G1/G2)
- New `models/jepa_identity_head.py`: frozen M2 pre-pool tokens (+ whichever streams pass
  G3) → small trainable head → L2-normalized **identity embedding**, trained with a
  supervised metric loss (ArcFace/sub-center or supervised-contrastive — chosen by a
  head-to-head at this scale, per the repo's standing "run the comparison, don't assume"
  practice that already overturned `llama_last8` twice).
- Held out: **whole identities**, not whole sessions. Open-set is the real task — BMO must
  handle a stranger, not just re-rank a closed roster.

**Pre-registered gates:**
- **G4:** open-set verification TAR@FAR=1% ≥ 0.60 on held-out identities.
- **G5:** ≥ 15pp above the Phase-1 frozen-feature k-NN baseline (the head must earn its
  parameters).
- **G6:** the caption-space control — the same eval using `z_p` from
  `bs16384/best.pt` — must be *worse*, confirming §4a empirically instead of by argument.

### Phase 3 — the memory store + the "I don't know you" threshold (2 days)
- `models/jepa_memory.py`: enroll(embedding, label) → persistent bank; query(embedding) →
  (label, confidence) with a **calibrated rejection threshold**. Few-shot enrollment
  (1/3/5 shots) curve reported.
- Consolidation: multiple enrollments per identity → running centroid + spread, so BMO's
  memory of someone sharpens with exposure instead of storing duplicates.
- **G7:** on held-out identities never enrolled, false-accept rate ≤ 5% at the operating
  threshold. Confidently greeting a stranger by the maker's name is the failure mode that
  matters most here; this gate exists specifically to catch it.
- **G8:** Jetson latency for a 200-entry bank ≤ 20ms (the measured 3ms NN lookup gives
  ~6x headroom; this gate just prevents silent regression).

### Phase 4 — thinker integration (2 days)
- The recognition result reaches the thinker as an **embedding prefix**, not a text
  injection — using the already-PROVEN `llama_batch.embd` path
  (`scripts/prototype_llama_embd_input.py`: HF embeddings fed to the GGUF produced
  byte-identical output to the token path, no C++ fork, `llama-cpp-python 0.3.34`).
- Falsifier: swapped-identity and zeroed-identity controls, mirroring the standing M3/M4b
  rule — the thinker must actually *use* the identity signal, not pattern-match a slot.
  Append a row to `checkpoints/falsifier_tracking.md`.

### Phase D (parallel, independent) — the "fine detail" track
- **D1: spatial-pooling ablation.** Re-extract a bounded VGGSound subset at 8x8 (64
  tokens/bin) instead of 4x4, retrain the embedding predictor at matched config, compare
  to 62.05. Single variable. **This is the highest-evidence detail lever (§3d) and needs
  no new encoder.** Disk-bounded: ~4x the vision cache for the subset only, into tmpfs.
- **D2 (gated on D1): V-JEPA 2.1 ViT-B, server-side only.** Checkpoint already local.
  Requires the disk decision in §6 first. 2.1 ViT-L stays ruled out on-device — final.
- **D3 (optional, ~1h Jetson):** produce the missing 2.1 ViT-B Jetson number so §3c stops
  being a gap in the record, even though §3a already predicts it loses to the current ViT-L.

### Phase E — query-conditioned predictor (north-star item 1)
Deferred behind Phase D by design: a query path that asks for detail the representation
doesn't contain cannot succeed. Revisit once D1 has a number.

---

## 6. Decisions — ANSWERED 2026-08-10

- **Start with Phase 1 only.** The soft-InfoNCE loose end (Phase 0) is deferred; go
  straight at the identity falsifier, since it is the gate that decides viability.
- **Phase D stages in `/dev/shm`** (750G tmpfs). No existing cache is deleted, no RAID
  writes. Volatile is acceptable — D1's deliverable is a number, not a persisted asset.
- Phase D ordering unchanged: pooling ablation before V-JEPA 2.1.

## 6a. Open decision — closing the cross-session VISION gap

Phase 1 left exactly one hole: a multi-identity cross-session vision number. Options,
cheapest first:

- **(i) VoxCeleb1 via Academic Torrents / OpenSLR.** VoxCeleb1 keeps the
  `id/video/segment` structure AND ships an official verification trial list
  (`veri_test.txt`), which is the field-standard protocol. Oxford no longer serves it
  directly; academictorrents has it. Cost: a torrent pull into `/dev/shm`.
- **(ii) Accept `voxceleb2-mp4-binary` as an A1-style UPPER BOUND.** Already downloadable,
  but segments may share a source recording, so it measures something between
  within-recording and cross-recording. Only worth it if labelled as an upper bound, like A1.
- **(iii) Re-pair the audio and video mirrors.** Not possible as checked — the mp4 mirror's
  flattened segment index cannot be joined back to the wds paths.
- **(iv) Skip it and let Phase 2 answer it.** Train the identity head on the data we have
  and measure cross-session generalization as a Phase-2 result rather than a Phase-1 one.

**Recommendation: (i), falling back to (iv).** VoxCeleb1's official trial list would give a
number directly comparable to published work, which matters for the dissertation framing.

### Original decision list (kept for provenance)

1. **Disk.** RAID is at 0 bytes free. Phases 0–4 are designed to avoid it entirely, but
   Phase D needs real space. Options, in my recommended order:
   (a) stage D1 in `/dev/shm` (750G, volatile — fine for an ablation that produces a
   number, not an asset); (b) reclaim `ego4d_probe` (1.5T) or `action100m_preview` (37G)
   if either is genuinely spent; (c) do nothing and cap Phase D at whatever fits in tmpfs.
   **I'd take (a) and not touch existing caches** — but I won't delete anything without
   you saying so explicitly.
2. **Phase D ordering.** I recommend pooling-ablation-before-2.1 (§3d). If you'd rather
   see 2.1 numbers first, say so and I'll pull D2/D3 forward — the checkpoints are already
   on disk, so it's a scheduling choice, not a blocker.
3. **Phase 1 identity labelling.** Clustering the 84 EasyCom participant photos into
   global identities needs a human confirmation pass (the falsifier record already shows
   the slot IDs mislead). I'll prepare the contact sheet; you confirm the groupings.

---

## 7. Run log

*(append one entry per run, newest last — result, gate verdict, provenance path)*

### 2026-08-10 — EasyCom identity structure, fully mapped (SUPERSEDES the first version of this entry)

**CORRECTION.** The first version of this entry concluded "EasyCom has no cross-session
identity, a cross-session eval is impossible." **That was wrong, and it was wrong because
I sampled only the first 3 chunks of each session.** Participant 1 barely appears early in
a session, so the sample made him look absent. Over ALL chunks he appears in **all 12
sessions, 79,080 frames, median face box 71–195px**. The corrected structure is below; the
cross-session eval IS possible, with one genuine identity.

**Verified identity structure (28 single-session guests + 1 cross-session identity):**

How it was established (exact, not inferred):

1. **md5'd all 84 `Participant_Photos`** → only **31 unique images**. Three images repeat:
   - `49a3dec8` — slot **1** in all 12 sessions. Viewed directly: a **real person** (man,
     cap, glasses). This is the fixed device-wearer.
   - `2b371fb3` — slot **2** in all 12 sessions. Viewed directly: an **anonymized blue
     silhouette placeholder**, not a person.
   - `a161935d` — 32 occurrences across many slots/sessions. Viewed directly: a **black
     "X"** = "this slot is unused in this session".
   The remaining **28 images are singletons**, one session each.

2. **Viewed all 28 singleton photos.** Every one is a visually distinct person. No
   cross-session repeat identified. (Caveat: photos are low-res; S7/4 and S7/6 look
   similar, but they are same-session simultaneous participants and therefore necessarily
   different people.)

3. **`Close_Microphone_Audio` availability matches the real-photo map exactly** — audio
   exists for precisely the 28 guest slots and for nobody else:

   | sessions | participant IDs with close-mic audio |
   |---|---|
   | 1, 3, 5, 6, 7, 8, 9, 10, 12 | 4, 6 |
   | 2, 11 | 3, 5, 7 |
   | 4 | 3, 4, 6, 7 |

   Total = 28, matching the 28 unique photos exactly.

4. **Slot 1 is NOT the wearer — slot 2 is.** p1 appears in the face bounding boxes; p2
   never does, has the silhouette photo, and has no close mic. So **p2 = the glasses
   wearer** (invisible in their own video by construction) and **p1 = a real, recurring,
   visible person** — almost certainly the experimenter/operator, given ~17–44 utterances
   per session and frequent close-to-camera framing.

   **p1 is therefore a genuine cross-session identity**, confirmed three ways: byte-identical
   photo in all 12 sessions; direct visual comparison of crops pulled from S1/S7/S8/S10
   (same man — cap, glasses, reddish beard); and presence in every session's boxes:

   | | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | p1 frames | 3737 | 4735 | 3097 | 8479 | 6662 | 7515 | 9459 | 9641 | 6835 | 8122 | 6985 | 3813 |
   | median box px | 99 | 88 | 195 | 85 | 71 | 138 | 126 | 94 | 90 | 110 | 97 | 144 |

   Clothing genuinely varies across sessions (blue plaid / dark / grey / black beanie), so
   this is real appearance variation, not one outfit repeated. He also speaks in every
   session (112–250s each, ~37 min total) — but has **no close mic**, so his audio can only
   come from the glasses array. That makes every audio-bearing stream **channel-confounded**
   for the cross-session test (guests = close mic, p1 = array). Only vision is clean there.

5. **A burned-in participant-photo legend leaks the answer key.** Frames are **2123×1080,
   not 1920×1080**: the left **203px (= 2123−1920)** is a legend strip rendering **every
   participant's photo**, identical in every frame of a session. Any crop touching it would
   read identity directly and would also uniquely identify the session. Verified by dumping
   a full frame. All crops are now hard-clamped to `x >= 203` with an assert.
   Separately verified: bounding boxes are in **raw 2123-wide coordinates** (checked by
   cropping with and without a +203 offset and looking — raw is correct); do not "fix" them.

Also confirmed usable for a *within*-session design: `Face_Bounding_Boxes` carry per-frame
`Participant_ID` + box coords (1200 frames/chunk @ 20fps), 323 chunks × 60s ≈ 5.4h video
across 12 sessions. So per-person crops are directly available; only the *cross-session*
axis is missing.

**Resulting Phase 1A design (user approved "A then B", 2026-08-10).** Two evals, reported
separately because they are not the same claim:

- **A1 — within-session, cross-chunk, N-way identification** over the 28 guests. Gallery =
  first 60% of a session's chunks, query = last 40%. **An UPPER BOUND, always labelled as
  one**: same day, clothing, seat and lighting, so part of any score is appearance matching.
  It is still the right first test — a representation that fails here cannot support the
  north-star at all.
- **A2 — cross-session verification for p1**, leave-one-session-out: enroll p1 on 11
  sessions, test on the held-out session against that session's guests as impostors. This
  is the real, un-inflated cross-session number. n=1 genuine identity, so it is
  statistically limited — but it is exactly the deployed task (recognize the one person you
  know, reject everyone else). **Audio-bearing streams are excluded from A2's gate as
  channel-confounded** (guests close-mic vs p1 array-mic); only `vitl_crop` is clean.

Then **(B) VoxCeleb1/2** — the canonical audio-visual person-ID benchmark, many identities
with multiple *separate* videos each, staged in `/dev/shm`. That is where a claimable
multi-identity cross-session number comes from.

Code: `scripts/jepa_memory_phase1a_extract.py` (frozen-feature extraction, 6 streams) and
`scripts/jepa_memory_phase1a_eval.py` (A1/A2 + shuffle controls + bootstrap CIs + gates).

### 2026-08-10 — PHASE 1A RESULT: identity IS there, but M2 discards it and cross-session FAILS

3,768 person-windows, 29 identities, 12 sessions, ~4 min on 4 GPUs. Extraction:
`scripts/jepa_memory_phase1a_extract.py`; scoring: `scripts/jepa_memory_phase1a_eval.py`;
raw: `checkpoints/JEPA_MEMORY_PHASE1A_RESULTS.json`.

**A1 — within-session, cross-chunk, N-way identification (UPPER BOUND, see caveat)**

| stream | top-1 | 95% CI | chance | label-shuffled |
|---|---|---|---|---|
| **vitl_crop** (raw V-JEPA2 ViT-L on the person crop) | **0.860** | [0.841, 0.879] | 0.446 | 0.483 |
| z_p (caption-aligned control) | 0.733 | [0.708, 0.756] | 0.446 | 0.495 |
| m2_prepool_mean | 0.704 | [0.678, 0.729] | 0.446 | 0.491 |
| m2_world_state | 0.703 | [0.677, 0.729] | 0.446 | 0.489 |
| wavjepa | 0.644 | [0.618, 0.670] | 0.446 | 0.514 |
| moonshine | 0.565 | [0.537, 0.592] | 0.446 | 0.446 |

**A2 — cross-session verification, p1 leave-one-session-out (the real test)**

| stream | AUC | TAR@FAR=1% | control AUC | note |
|---|---|---|---|---|
| **vitl_crop** | **0.639** | 0.124 | 0.098 | the only channel-clean stream |
| wavjepa | 0.882 | 0.327 | 0.364 | CONFOUNDED — close-mic vs array-mic |
| m2_world_state / m2_prepool_mean | 0.862 | 0.192 | 0.417 | CONFOUNDED — same |
| z_p | 0.852 | 0.233 | 0.448 | CONFOUNDED — same |
| moonshine | 0.744 | 0.083 | 0.576 | CONFOUNDED — same |

**Gates:** G1-A **PASS** (0.860 ≥ 0.80). G2-A **PASS** (CI lower bound 0.841 far above both
chance 0.446 and shuffle 0.483). G6 **PASS**. A2 cross-session **FAIL** (0.639 < 0.70).

**What this actually means — four findings, in order of consequence:**

1. **Person identity IS linearly present in the frozen perceptual features.** vitl_crop hits
   0.860 vs 0.446 chance, with the label-shuffled control at 0.483 (i.e. exactly chance, so
   the result is not a metric artifact). The north-star's core premise survives its first
   real test.
2. **The M2 predictor DESTROYS most of that identity signal: 0.860 → 0.703.** Its input
   carries identity; its output has ~16pp less of it. This is not surprising in hindsight —
   M2 is trained for cross-modal *congruence*, where who the person is counts as nuisance
   variation — but it is now measured rather than argued. **Concrete consequence for
   Phase 2: the identity head must attach to the raw pooled ViT-L tokens, NOT to the M2
   world-state.** That inverts the original plan, which assumed M2 tokens were the source.
3. **§4a is confirmed in direction but overstated in degree.** The caption-aligned z_p does
   lose to vitl_crop (0.733 vs 0.860, G6 PASS), so caption-InfoNCE space is genuinely worse
   for identity — but at 0.733 vs 0.446 chance it is *far* from identity-free. "Reusing z_p
   fails by construction" was too strong; the accurate claim is "z_p is a materially
   degraded identity space, and there is a better one available for free."
4. **Cross-session recognition does NOT work off frozen features alone (AUC 0.639).** This
   is the honest headline and it FAILS the pre-registered gate. Note the failure is *worse*
   than it looks: the control AUC of 0.098 shows a random-guest centroid ranks guests far
   above p1, which means part of even the 0.639 is p1's distinctive framing/scale (he is
   often close to camera) rather than his identity.

**This is a "Phase 2 is required, not optional" result, not a kill.** Zero-shot frozen
features failing cross-session verification is the *expected* outcome — face/speaker
recognition has always needed metric learning on top of a backbone. What Phase 1A
establishes is that the signal exists to train on (A1) and where to attach the head
(finding 2). What it cannot establish is a claimable multi-identity cross-session number:
n=1 genuine identity over 12 folds is far too thin. **That is exactly what (B) VoxCeleb is
for, and it is now the gating dependency for Phase 2 rather than a nice-to-have.**

Caveats held open, not resolved: A1 is same-day/same-clothing/same-seat and is an upper
bound; A2 rests on one person; audio identity is untestable cross-session on EasyCom
because p1 has no close mic.

### 2026-08-10 — PHASE 1B (voice half): cross-session voice identity CONFIRMED on VoxCeleb2

Phase 1A could not test audio identity cross-session at all — EasyCom's only recurring
identity (p1) has no close mic, so every audio stream there was microphone-channel
confounded. Phase 1B removes that confound entirely.

**Data + protocol.** `gaunernst/voxceleb2-dev-wds` (5,991 speakers, 779 shards, audio
only) **preserves the original `speaker/youtube_video_id/segment` key** — verified
directly (`id02139/yCPbcLeT5SI/00147.m4a`). The middle field is a distinct source
recording: different day, room, microphone and channel. Enrolling on one set of source
videos and testing on **held-out source videos** is therefore a genuine cross-session
test. 150 shards staged in `/dev/shm` (15GB), 250 speakers x 6 source videos x 5
segments = **3,182 clips, 1,590 enroll / 1,592 query, 0 decode failures.**

**Rejected mirror, and why it matters:** `blueskyheaven/voxceleb2-mp4-binary` (279GB) is
the only mirror found that has VIDEO, but it **flattens the path to a bare segment index**
— the source-video grouping is gone, so any split on it silently leaks same-recording
segments. Checked directly rather than assumed (`file_name` values are `00001`-style
segment numbers, not `id/video/segment`). Not used. This is why 1B covers the voice half
only.

| stream | 250-way top-1 | 95% CI | chance | shuffled | AUC | TAR@FAR=1% |
|---|---|---|---|---|---|---|
| **wavjepa** (base+nat mean — exactly M2's `ambient` input) | **0.250** | [0.229, 0.272] | 0.0040 | 0.0044 | **0.832** | 0.259 |
| moonshine (already Jetson-resident at 37ms/turn) | 0.166 | [0.149, 0.185] | 0.0040 | 0.0031 | 0.826 | 0.161 |

**Gate: PASS** (0.250 = 62x chance; shuffle control sits exactly at chance, so this is not
a metric artifact). Stable under sample size — the earlier 30-shard run gave 0.221/0.136,
the 150-shard run 0.250/0.166, same ordering and magnitude.

**Findings:**
1. **Voice identity survives a real change of session.** Different YouTube recordings mean
   different rooms and microphones, so channel variation works *against* the score here
   rather than inflating it, unlike EasyCom's p1 confound.
2. **WavJEPA beats Moonshine for speaker identity (0.250 vs 0.166), and the reason is
   principled**: Moonshine's encoder is trained for ASR, an explicitly speaker-*invariant*
   objective, so discarding voice identity is what it is supposed to do. The "free on the
   Jetson" argument for Moonshine does not survive contact with the measurement — the
   voice-identity half should be built on WavJEPA, which M2 already consumes.
3. **Frozen features are nowhere near deployment-grade, and this is the important number.**
   A dedicated speaker-embedding model (ECAPA-TDNN class) reaches ~99% top-1 / ~1% EER on
   VoxCeleb; 25% top-1 / AUC 0.83 is a real signal but roughly 4x off usable. Same verdict
   the vision side reached in 1A.

**Phase 1 combined verdict — consistent across both modalities:** identity information is
genuinely present in the frozen perceptual features (vision A1 0.860 vs 0.446 chance; voice
0.250 vs 0.004 chance), and in both cases the frozen features alone are insufficient for
cross-session recognition (vision AUC 0.639; voice top-1 0.250). **Phase 2's trained
identity head is therefore REQUIRED, and it now has measured baselines to beat and two
concrete architectural decisions already made for it: attach vision to the raw pooled ViT-L
tokens (not the M2 world-state, which loses 16pp of identity), and build voice on WavJEPA
(not Moonshine).**

**Still open — the one real gap:** a multi-identity *cross-session* VISION number. EasyCom
gives n=1 (p1, AUC 0.639) and VoxCeleb2's only video mirror has unusable grouping. Options
for closing it are in §6a.

### 2026-08-10 — PHASE 1C: fusion + realistic operating point (answers three direct questions)

`scripts/jepa_memory_phase1c_followups.py`, reusing the cached 1A/1B embeddings (seconds,
no new GPU work). Raw: `checkpoints/JEPA_MEMORY_PHASE1C_FOLLOWUPS.json`.

**Q1 — "M2 discards 16pp; what if we add it back?" Measured: adding it back does nothing.**

| streams (L2-normed, concatenated) | A1 top-1 | 95% CI | vs ViT-L |
|---|---|---|---|
| raw ViT-L only | **0.860** | [0.841, 0.879] | — |
| ViT-L + M2 ("add it back") | 0.846 | [0.826, 0.865] | −0.014 |
| ViT-L + z_p | 0.824 | [0.803, 0.845] | −0.036 |
| ViT-L + M2 + z_p | 0.799 | [0.778, 0.820] | −0.062 |
| ViT-L + WavJEPA (joint AV) | 0.827 | [0.807, 0.847] | −0.033 |
| M2 + z_p (no raw ViT-L) | 0.726 | [0.700, 0.750] | −0.134 |
| everything | 0.784 | [0.762, 0.805] | −0.076 |

**M2 contributes no identity information that raw ViT-L does not already contain** — the
fusion is statistically indistinguishable from ViT-L alone (CIs overlap heavily), and every
larger fusion is *worse*, because equal-weight concatenation lets weaker streams dilute the
cosine. For identity, M2 is a strictly lossy view of its own input.

**The practical consequence is the opposite of a problem: nothing needs to be added back,
and no M2 retrain is required.** The ViT-L tokens are already computed on every production
perception tick — the identity head taps a point in the *existing* pipeline that is
currently being thrown away. Cost: one extra head, no new encoder.

**Q2 — "rich-caption training on Action100M + VGGSound compensated for it, no?"
Measurably right in direction, with a structural ceiling.**

z_p (0.733) beats the M2 world-state it is computed from (0.703) by ~3pp. Caption training
DID recover identity-correlated information — the mechanism is real: `gpt_action_detailed`
captions average 26.1 words and describe appearance ("a woman with long dark hair in a grey
sweater"), so training to predict them preserves appearance-correlated features that M2's
congruence objective had discarded.

The ceiling is structural, not a training-budget issue: captions describe *appearance
categories*, never identity. Two different people with long dark hair get the same caption,
so InfoNCE actively pulls them together. That is why z_p recovers +3pp but is still 12.7pp
below raw ViT-L, and more caption data cannot close it.

**Crucially, this does NOT reflect on the predictor's actual job.** Scene prediction does
not need identity, so M2's identity loss costs the retrieval objective nothing — the 62.05
checkpoint is unaffected by any of this. Identity and scene-prediction are different
quantities; a weakness in one is not a weakness in the other.

**Q3 — "the voice embedding produced when I speak — can we do that?" Yes, and it already
half-works with zero training.** The 250-way number was the wrong operating point; a
household gallery is 2-10 people.

| enrolled people (N) | WavJEPA top-1 | 95% CI | chance | Moonshine |
|---|---|---|---|---|
| 2 | **0.896** | [0.399, 1.000] | 0.500 | 0.885 |
| 3 | 0.845 | [0.500, 1.000] | 0.333 | 0.819 |
| 5 | **0.758** | [0.473, 0.967] | 0.200 | 0.730 |
| 10 | 0.646 | [0.416, 0.842] | 0.100 | 0.606 |
| 20 | 0.540 | [0.385, 0.682] | 0.050 | 0.493 |
| 100 | 0.343 | [0.294, 0.390] | 0.010 | 0.256 |
| 250 | 0.250 | — | 0.004 | 0.166 |

Enrollment depth at N=5 (WavJEPA) — **more enrollments genuinely sharpen the memory**,
which is the Phase-3 consolidation design validated in advance:

| clips enrolled per person | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| top-1 | 0.584 | 0.666 | 0.688 | 0.728 | **0.760** |

All cross-session (held-out source recordings). Wide CIs at small N are real speaker-pair
variance — some pairs are easy, some genuinely confusable — not instability in the estimate.

**Caveat that separates this from deployment:** VoxCeleb is single-speaker YouTube audio.
Production audio is a room mic carrying BMO's own TTS, other speakers, and room noise, so
the user's speech must be isolated (VAD/diarization) before embedding. The numbers above
are the clean-input ceiling for the frozen encoder, not a deployment estimate.

### 2026-08-10 — PHASE 2a: voice identity head — helps where it matters, data-limited

`models/jepa_identity_head.py` + `train_identity_head.py`. 16,472 clips / 1,400 speakers
(4 GPUs, 8.5 min, 0 failures), **speaker-disjoint** 1,120 train / 280 test, cross-video
enrol/query inside the test speakers. Results on the 280 **never-seen** speakers:

| gallery N | frozen mean | frozen stats | TRAINED head | chance |
|---|---|---|---|---|
| 2 | 0.913 | 0.914 | 0.892 | 0.500 |
| 5 | 0.772 | 0.747 | 0.762 | 0.200 |
| 10 | 0.662 | 0.657 | 0.668 | 0.100 |
| 20 | 0.555 | 0.553 | **0.576** | 0.050 |
| 50 | 0.423 | 0.438 | **0.454** | 0.020 |
| 250 | 0.273 | 0.281 | 0.266 | 0.004 |
| **AUC** | 0.838 | 0.836 | **0.895** | — |
| **TAR@FAR=1%** | 0.273 | 0.255 | **0.419** | — |

**Read: the head buys verification, not identification.** Top-1 is roughly unchanged, but
TAR@FAR=1% goes 0.273 → **0.419 (+54% relative)** and AUC +5.7pp. That is the right
trade for BMO: the deployed question is "is this the person I know, or a stranger?" at a
low false-accept rate — which is literally Phase 3's G7 gate — not "rank 250 candidates."

**Two honest caveats.**
1. **Statistics pooling bought nothing** (frozen stats ≈ frozen mean everywhere, and
   slightly *worse* on TAR). The x-vector/ECAPA precedent did not transfer to frozen
   WavJEPA features. Recorded as a small negative result; the gain is entirely the head's.
2. **The head is badly overfit and therefore data-limited, not architecture-limited.**
   Train loss reached 0.0006 — memorisation — on ~14.7 clips/speaker. A first run with only
   ~1,040 optimizer steps was *worse than frozen* (250-way 0.069 vs 0.273); fixing that
   needed AAM **margin warm-up** (full margin from step 0 stalls convergence on a frozen,
   weak backbone) plus 15x more steps. Only 150 of 779 available shards were used. **More
   data per speaker is the obvious next lever, before any architecture change.**

### 2026-08-10 — PHASE 2b: QUERY-CONDITIONED PREDICTOR — works, and the falsifier is decisive

`models/query_predictor.py` (30.9M params, Perceiver-style: the query seeds latents that
cross-attend the frozen perceptual tokens, so a follow-up question re-reads the SAME
perception instead of forcing a re-encode) + `train_query_predictor.py`.

**Data — both captioned corpora, per the "use the right data" steer.** VGGSound v2 is
really a **3 aspects x 2 granularities grid** (action 4.8/37.0 words, summary 8.9/58.1,
sound 12.2/18.0 — the 6th field `gpt_sound_acoustic_v1_original` exists only in v2, which
is also the corrected file the locked M3 used), giving **172,593** train clips with all 6
fields. Action100M adds **345,754** clips on exactly the brief↔detailed axis. Ego4D's 134k
cached clips are deliberately excluded — no captions, so no query/answer supervision.
K differs per corpus (6 vs 2) so batches draw from one corpus at a time.

The loss carries both negative types in one matrix: cross-CLIP (answer must describe THIS
scene) and **within-CLIP** (the same clip's other field captions — all true, differing only
in what was asked). The second is what forces the query to carry information; without it
the model can ignore the query and still score perfectly.

**Result at step 249 of 3000 (still training):**

| corpus | within-clip field acc | chance | **swapped-query** | cross-clip R@1 |
|---|---|---|---|---|
| VGGSound (6 fields) | **0.580** | 0.167 | **0.002** | 0.226 |
| Action100M (2 fields) | **0.970** | 0.500 | **0.030** | 0.018 |

**The swapped-query control is the decisive number, and it lands far BELOW chance
(0.002 vs 0.167).** A model ignoring the query would sit *at* chance. Scoring near zero
means it is actively following the question — ask the wrong one and it confidently returns
the wrong field. That is exactly the behaviour "the thinker asks perception for detail"
requires, and it is measured on **held-out query phrasings never seen in training**, so it
is intent-following rather than string-matching.

### 2026-08-10 — UNIFIED ARCHITECTURE: M2 kept, ViT-L added alongside it (per user direction)

User direction: feed the query predictor ViT-L directly, **but do not throw M2 away — AV
congruency is essential.** That is the right call and it is now the architecture rather
than an either/or.

`QueryPredictorConfig.source_dims` makes the predictor cross-attend over the
**concatenation of several token streams**, each with its own input projection and a
learned source embedding so the model knows which stream a token came from:

| stream | dim | what it uniquely provides |
|---|---|---|
| `m2` | 1024 | `encode_pre_pool_tokens` — the cross-modal FUSED view. **Audio-visual congruence is M2's trained job and is not reproducible from either raw stream alone.** This is why M2 stays. |
| `vision` | 1024 | the cached spatially-pooled ViT-L tokens = M2's own INPUT, read directly. Phase 1C measured M2 discards information present here (identity 0.860 → 0.703). |
| `ambient` | 768 | raw WavJEPA base+nat tokens, read directly. |

All three are **free from the existing AV cache — no re-extraction**, because
`AVCachedDataset` already returns the pooled ViT-L tokens as `feats["vision"]`. Padding
masks are built per stream (the `m2` stream's token order is `[vision; ambient]` per
`AVJepaPredictor._embed`, so its mask is the concatenation of the two input masks).

**4-way ablation launched** (3000 steps each, batch 96, identical otherwise) to measure
what each stream actually contributes rather than assuming:
`m2` (baseline, GPU0) · `vision` (GPU1) · `m2+vision` (GPU2) · `m2+vision+ambient` (GPU3).

Baseline (`m2` only) trajectory — within-clip saturates early, cross-clip R@1 is the metric
still moving, which is exactly the split predicted for a contrastive objective (within-clip
negatives are fixed at K per clip regardless of batch; cross-clip negatives scale with it):

| step | VGG within-clip | swapped | VGG cross-clip R@1 |
|---|---|---|---|
| 249 | 0.580 | 0.002 | 0.226 |
| 999 | 0.862 | 0.004 | 0.321 |
| 1749 | 0.878 | 0.005 | 0.357 |
| 2249 | 0.889 | 0.004 | 0.361 |

**On "would DDP help":** yes for `cross_clip_r1` specifically — that is the half whose
negatives scale with batch size, and this repo already has proven DDP+GradCache for exactly
this in `train_m2_embed_predictor.py` (batch 64→2048 matched 6000 steps of gains in ~150;
4096→8192 = +17.5% relative). It will do ~nothing for `within_clip_acc`, whose candidate set
is fixed at K per clip — and that metric is already effectively solved (0.889 vs 0.167
chance, swapped 0.004). Two standing cautions from this repo's own measurements apply:
bigger batches need proportionally MORE optimizer steps, and **do not sqrt-scale the LR**
(measured 57.95 vs 60.70, a real loss).

### 2026-08-11 — ABLATION RESULT: the unified architecture wins. M2 and ViT-L are complementary.

All four runs completed (3000 steps, batch 96, identical otherwise). Held-out eval, held-out
query phrasings:

| token sources | VGG within-clip | swapped | **VGG cross-clip R@1** | A100M within | A100M R@1 | combined |
|---|---|---|---|---|---|---|
| `m2` (baseline) | 0.895 | 0.004 | 0.385 | 0.965 | 0.043 | 1.2794 |
| `vision` (ViT-L direct) | 0.835 | 0.006 | 0.447 | 0.956 | 0.064 | 1.2826 |
| `m2 + vision` | 0.844 | 0.005 | **0.478** | 0.949 | 0.064 | 1.3224 |
| **`m2 + vision + ambient` (unified)** | **0.897** | 0.004 | 0.458 | 0.964 | **0.082** | **1.3630** |

**Three findings, and they settle the architecture question:**

1. **Feeding ViT-L directly was the right call.** Cross-clip R@1 goes 0.385 → 0.447 on the
   raw stream alone — a **+16% relative** gain over reading M2's output. The same lesson as
   the identity head: M2's output does not contain everything its input did.
2. **But M2 is NOT redundant — the two are complementary.** `m2+vision` (0.478) beats BOTH
   `m2` alone (0.385) and `vision` alone (0.447). If M2 were merely a lossy view of ViT-L,
   adding it back would do nothing; instead it adds **+7% relative** on top of raw vision.
   That is direct evidence that the cross-modal fused representation carries information
   neither raw stream has — i.e. **AV congruence is real, earned, and worth keeping.** The
   user's instinct not to throw M2 away is confirmed by measurement, not deference.
3. **The full unified model is the best overall** (combined 1.3630): best within-clip
   query-sensitivity (0.897), best Action100M R@1 (0.082), with VGG R@1 0.458 slightly
   behind `m2+vision`'s 0.478. Adding the raw ambient stream recovers the query-sensitivity
   that `vision`/`m2+vision` gave up (0.835/0.844 → 0.897) — sensible, since two of the six
   VGGSound query types are about SOUND, and those need a direct audio path.

**Recommendation: adopt `m2 + vision + ambient` as the standing query-predictor
architecture.** It keeps the whole existing pipeline (M2 unchanged, its checkpoint
untouched), adds two direct token paths that cost nothing to compute (both already in the
AV cache), and is best or tied-best on every metric except one.

Swapped-query stays at 0.004-0.006 across every configuration — far below the 0.167 chance
level — so query-following is robust to the architecture change, not an artifact of one setup.

**PROCESS FAILURE, recorded so it is not repeated:** the joint AV identity extraction was
supposed to auto-start when the baseline finished, but the chaining script's
`while pgrep -f "query_predictor_v1"` matched its OWN command line and spun forever. Four
GPUs sat idle for hours. Two lessons: never `pgrep` a pattern that appears in the waiting
script's own argv, and never `pkill -f <pattern>` from a shell whose command line contains
that pattern (it kills the caller — this then happened too). Fixed by launching from a
script FILE via `setsid`, so no pattern can self-match.

### 2026-08-11 — PHASE 2c: JOINT AV IDENTITY HEAD — the joint head roughly doubles the deployment metric

33,798 clips / 1,400 speakers from VoxCeleb2 mp4 (face-crop video WITH its own audio, so
vision and voice are the same person at the same instant). **Speaker-disjoint**: trained on
1,120 speakers, evaluated on **280 never-seen** speakers. Enrol/query = index-range split
(APPROXIMATE cross-video — see `train_identity_head_av.py`; the clean cross-session voice
number remains Phase 2a's).

| gallery N | vision froz | vision head | audio froz | audio head | joint froz | **joint head** |
|---|---|---|---|---|---|---|
| 2 | 0.849 | 0.844 | 0.923 | 0.911 | 0.881 | **0.959** |
| 5 | 0.684 | 0.704 | 0.839 | 0.824 | 0.694 | **0.897** |
| 10 | 0.577 | 0.639 | 0.777 | 0.757 | 0.584 | **0.857** |
| 50 | 0.410 | 0.507 | 0.618 | 0.613 | 0.412 | **0.708** |
| 280 | 0.309 | 0.402 | 0.478 | 0.470 | 0.310 | **0.557** |
| **AUC** | 0.821 | 0.861 | 0.889 | 0.918 | 0.822 | **0.952** |
| **TAR@FAR=1%** | 0.341 | 0.493 | 0.480 | 0.599 | 0.341 | **0.691** |

**Four findings:**

1. **The joint head wins decisively and roughly doubles the deployment metric.**
   TAR@FAR=1% goes 0.341 (frozen) → **0.691**; AUC 0.822 → 0.952; 280-way top-1 0.310 →
   0.557. At household scale it is 0.959 (N=2) / 0.897 (N=5) / 0.857 (N=10).
2. **Adding vision was the right call — but not for the reason expected.** On this corpus
   **voice alone beats face alone** (audio frozen 0.478 vs vision frozen 0.309 at 280-way).
   That is NOT evidence that faces carry less identity than voices in general — VoxCeleb2
   faces are small, heavily-compressed YouTube crops, and **V-JEPA2 is not a face-recognition
   model**, whereas WavJEPA is a strong general audio representation. A dedicated face
   encoder (ArcFace-class) would almost certainly invert this. The honest claim is
   *V-JEPA2 features are weaker for faces than WavJEPA features are for voices.*
3. **Fusion REQUIRES the trained head — concatenating frozen features does not work.**
   Joint-frozen (0.310) is dragged down to the weaker modality, essentially matching
   vision-frozen (0.309) and far below audio-frozen (0.478). Only after training does joint
   (0.557) beat both branches. This retro-justifies the head's per-modality trunks +
   learned present/absent embeddings over naive concatenation, and it mirrors the Phase-1C
   finding that equal-weight concatenation lets a weak stream dilute a strong one.
4. **Vision contributes real complementary information** despite being the weaker branch:
   joint head (0.557) beats audio head (0.470) by +0.087 at 280-way and +0.092 on
   TAR@FAR=1%. The face is not redundant with the voice.

### 2026-08-11 — DDP+GradCache: R@1 was negative-count bound, and batch scaling confirms it

`train_query_predictor_ddp.py` — 3-phase GradCache adapted from
`train_m2_embed_predictor.py`'s proven implementation. 4 GPUs, micro-batch 64 x 4 chunks x
4 ranks = **effective global batch 1024 clips / 2048 candidates per step** (vs 96 / 576
single-GPU). ~4.6s/step.

**Verified BEFORE spending GPU hours** (`/dev/shm/verify_gc.py`) — GradCache's 3-phase
gradient vs a direct single-batch backward on identical data:
`loss 4.683039 vs 4.683039 (diff 0.00e+00), gradient relative L2 error 1.365e-07,
cosine 1.00000000` → **PASS**. A silent GradCache bug trains happily while being wrong,
so this is checked, not assumed.

**Matched-step comparison, VGGSound cross-clip R@1:**

| step | batch 96 (single GPU) | **batch 1024 (DDP)** |
|---|---|---|
| 249 | 0.261 | **0.458** |
| 2999 (final) | 0.458 | — still running |

**At matched step 249 the DDP run is +75% relative (0.458 vs 0.261), and it already equals
the single-GPU run's FINAL 3000-step result in 249 steps — ~12x fewer optimizer steps.**
This is the same shape as the repo's own embed-predictor finding (batch 64→2048 matched
6000 steps of gains in ~150) reproduced at the query-predictor scale, and it confirms the
pre-registered diagnosis that cross-clip R@1 is bound by negative count, not step count.

As predicted, within-clip query-sensitivity is NOT the beneficiary (0.770 DDP vs 0.732
single at matched step) — its candidate set is fixed at K per clip regardless of batch.
Swapped-query stays at 0.003.

Two correctness requirements specific to this trainer, both handled: all ranks must draw
the SAME corpus per step (VGGSound K=6 vs Action100M K=2 would otherwise make the gathered
matrix ragged — corpus is derived from the step number via a shared seed), and the asked
field + query phrasing are sampled once and reused across Phase 1/Phase 3 so the replayed
forward matches the cached gradient. `qp` and `text_target.proj` are broadcast from rank 0,
guarding the real bug from commit `0efdbf5` (sync_grads averages gradients but never fixes
divergent initialisation).

### 2026-08-11 — PHASE 3: the memory store is built, and G7 PASSES

`models/jepa_memory.py` (enrol / query / consolidate / calibrate / persist) +
`scripts/jepa_memory_phase3_gate.py`. Every design choice traces to a measurement:
cosine on L2-normalised embeddings because the head was trained with an ANGULAR margin;
a running centroid + spread per identity because Phase 1C measured that enrolment depth
sharpens recognition (1 clip 0.584 → 8 clips 0.760); and a threshold **calibrated** to a
target false-accept rate rather than guessed.

**Gate run on the REAL trained joint-AV head**, 280 speaker-disjoint identities the head
never saw. Household of 5 enrolled from low-index clips, queried on high-index clips;
strangers split into a **calibration** pool (threshold fitting only) and a disjoint
**test** pool (scoring only), so FAR is never fit on what it is scored against.

| target FAR | correct | "I don't know you" | measured FAR | threshold |
|---|---|---|---|---|
| 1% | 0.462 | 0.533 | **0.013** | 0.396 |
| 2% | 0.517 | 0.475 | 0.023 | 0.355 |
| **5%** | **0.603** | 0.378 | **0.053** | 0.288 |
| 10% | 0.681 | 0.288 | 0.100 | 0.236 |
| 20% | 0.751 | 0.201 | 0.194 | 0.184 |

**G7 (false-accept on never-enrolled ≤ 5%): PASS** — 0.013 at the 1% operating point,
with the 95% CI upper bound (0.040) still inside the gate. Calibration tracks the target
closely at every point (1%→1.3%, 5%→5.3%, 10%→10.0%), so the threshold fitting is working
as designed rather than by luck.

**The important structural property: almost every error is "unknown", not "wrong person".**
correct + unknown = 0.995 at the 1% point, so only ~0.5% of household queries are
misidentified as a DIFFERENT enrolled person. For a companion robot that is exactly the
right failure mode — "I don't recognise you" rather than confidently using the wrong name.

**Honest limitation: recognition rate is the weak side.** At a strict 1% FAR only 46% of
genuine household queries are accepted; 5% FAR buys 60%. That is a threshold-placement
consequence of the head's separation quality (TAR@FAR=1% = 0.691 measured in Phase 2c),
not a bug in the memory — and it is exactly what the full-corpus retrain is meant to
improve, since the current head is badly overfit (train loss 0.0006 at ~24 clips/speaker).
Wide CIs are real household-composition variance: some groups of 5 are easy, some contain
confusable pairs.

### 2026-08-11 — END-TO-END DEMO, and the wrong-name/FAR trade the operating point really buys

`scripts/jepa_memory_demo.py` runs both capabilities on real clips and real checkpoints.

**Part 1 — the thinker asks perception for detail.** Same clip, six different questions,
all using HELD-OUT phrasings never trained on. 11/12 correct across two clips, e.g.:

```
Q: Briefly, what am I looking at?
A: A military parade featuring soldiers marching in formation.
Q: Tell me in detail what the surroundings look like.
A: The video depicts a military parade where soldiers from different branches march in
   perfect synchronization. The soldiers wear distinct uniforms repre...
Q: In detail, what is the sound and where is it coming from?
A: The music is rhythmic with drums and other percussion instruments...
```
The single miss was "Sum up the action in a few words" retrieving the *summary*-brief
line rather than the *action*-brief line — two one-sentence descriptions of the same
scene, i.e. the least meaningful confusion available.

**Part 2 — recognition, with a real failure.** A 4-person household at 5% target FAR:
0/20 false-accepts on strangers (correctly "UNKNOWN"), but **one enrolled person was
returned as a different enrolled person's name.** That is the failure mode that matters
most in a home, so it is quantified rather than waved away:

| target FAR | correct | unknown | **WRONG NAME** |
|---|---|---|---|
| 1% | 0.462 | 0.533 | **0.0046** |
| 2% | 0.517 | 0.475 | 0.0086 |
| 5% | 0.603 | 0.378 | **0.0188** |
| 10% | 0.681 | 0.288 | 0.0308 |
| 20% | 0.751 | 0.201 | 0.0485 |

**This corrects an over-claim in the Phase 3 entry above.** "Almost every error is
unknown, not wrong person" is true only at the STRICT end: 0.46% wrong-name at 1% FAR,
but 1.9% at 5% and 4.9% at 20%. Loosening the threshold to buy recognition rate buys
wrong-name errors at roughly the same pace. The demo drew 5% FAR, where ~1.9% wrong-name
is expected — so one error in a 4-person household is an unremarkable draw, not a bug.

**Recommended operating point: 1-2% target FAR**, accepting ~46-52% recognition, because
a missed greeting is recoverable and a wrong name is not. That ordering should be
revisited once the full-corpus retrain improves the head's separation.

### 2026-08-11 — DDP RUN COMPLETE: R@1 0.458 → 0.715 (+56%), with one real trade-off

Finished clean (`DDP_EXIT=0`), best checkpoint step 1999,
`checkpoints/query_predictor_ddp_b1024/best.pt`.

| | single-GPU batch 96 | **DDP batch 1024** | change |
|---|---|---|---|
| VGGSound cross-clip R@1 | 0.458 | **0.715** | **+56%** |
| VGGSound within-clip | 0.897 | 0.857 | −4.5% |
| swapped-query | 0.004 | 0.005 | unchanged |
| Action100M R@1 | 0.082 | 0.085 | +4% |
| combined | 1.355 | **1.572** | +16% |

Full R@1 trajectory: 0.458 (249) → 0.569 (499) → 0.647 (749) → 0.630 (999) → 0.699 (1499)
→ **0.715 (1999, best)** → 0.705 (2499) → 0.708 (2999). The step-999 dip was noise, not
saturation; real flattening only appears after ~2000.

**The trade-off is real and mechanistic, not noise: within-clip query-sensitivity dropped
0.897 → 0.857.** With 1024 clips in the batch the candidate pool is 2048-6144 entries, of
which only K−1 are the anchor clip's *other* fields — so within-clip negatives become a
vanishing fraction of the softmax and the gradient shifts toward cross-clip discrimination.
Batch scaling therefore buys scene-grounding at a small cost to query-sensitivity.

**Concrete fix for a future round (not run):** decouple the two objectives instead of
letting one softmax arbitrate — either a separate within-clip cross-entropy term over just
the anchor's K captions, added to the global one, or an up-weight on within-clip negatives
proportional to batch size. That should recover the 0.04 without giving back the 0.257.

### 2026-08-11 — TWO-TERM LOSS implemented + verified (prepared, not yet trained)

Implements the fix identified above for the one regression batch-scaling caused
(within-clip 0.897 → 0.857). In both `train_query_predictor.py` and
`train_query_predictor_ddp.py`, behind `--lambda-within` (**default 0.0 = the original
single-softmax loss, so every recorded number stays reproducible**):

```
L = L_global                      # over ALL gathered candidates -> scene grounding
  + lambda_within * L_within      # over ONLY this clip's K captions -> query-sensitivity
```

**Why this is the right shape.** Within-clip negatives are the only pressure forcing the
query to matter, and one softmax lets batch scaling drown them:

| batch | K | within-clip share of the candidate pool |
|---|---|---|
| 96 | 6 | 5/576 = **0.87%** |
| 1024 | 6 | 5/6144 = **0.08%** |

~10x dilution — which is exactly the size of effect observed. The new term is **K-way
regardless of batch**, so its gradient share cannot be diluted by scaling.

**GradCache compatibility is not an accident and was checked.** The within term uses only
`z_q_local` and the anchor's own captions from `z_t_local` — purely local, no all_gather —
so it is a function of the same leaf tensors and Phase 3's replay needs no change.
Re-ran the gradient-equivalence falsifier with `lambda_within=0.5`:
`loss 5.601333 vs 5.601333 (diff 0.00e+00), gradient relative L2 error 1.363e-07,
cosine 0.99999999` → **PASS**. 2-rank runtime smoke also clean (`within=0.500` at K=2 init,
i.e. chance, as it should be untrained).

**Not yet trained** — GPUs are occupied by the full-corpus identity extraction. Ready to
launch:
```
torchrun --nproc_per_node=4 train_query_predictor_ddp.py \
  --steps 3000 --micro-batch 64 --n-micro 4 --token-sources m2,vision,ambient \
  --lambda-within 0.5 --out-dir checkpoints/query_predictor_ddp_b1024_lw05
```
Target: recover within-clip toward 0.897 while holding R@1 near 0.715. Worth a small
sweep (0.3 / 0.5 / 1.0) since the balance is the whole question — and the falsifier that
matters is unchanged: swapped-query must stay near 0.005.

### 2026-08-11 — FULL-CORPUS RETRAIN: the head was data-limited, and fixing that ~doubled it

**122,235 clips / 4,000 speakers** (all 779 shards), 0 failures — vs 16,472 / 1,400 before
(**7.4x clips, 2.9x speakers**). Speaker-disjoint 3,200 train / **800 unseen** test (vs 280).

| gallery N | frozen mean | frozen stats | **TRAINED head** | chance |
|---|---|---|---|---|
| 2 | 0.919 | 0.923 | **0.950** | 0.500 |
| 5 | 0.799 | 0.804 | **0.897** | 0.200 |
| 10 | 0.705 | 0.706 | **0.851** | 0.100 |
| 20 | 0.596 | 0.610 | **0.801** | 0.050 |
| 50 | 0.480 | 0.490 | **0.711** | 0.020 |
| 250 | 0.306 | 0.319 | **0.536** | 0.004 |
| **AUC** | 0.848 | 0.845 | **0.957** | — |
| **TAR@FAR=1%** | 0.302 | 0.284 | **0.705** | — |

**Against the small-corpus head (280-speaker test):**

| metric | 1,400 spk head | **4,000 spk head** | change |
|---|---|---|---|
| 250-way top-1 | 0.266 | **0.536** | **+101%** |
| AUC | 0.895 | **0.957** | +6.9% |
| TAR@FAR=1% | 0.419 | **0.705** | **+68%** |

**Diagnosis confirmed: the head was DATA-limited, not architecture-limited.** Same
architecture, same hyper-parameters, same recipe — only the corpus changed. The earlier
head was ~tied with or worse than frozen features at small N (N=5: 0.762 vs 0.772); the
new one **beats frozen at every single gallery size**, by +10 to +23 points. Train loss
also relaxed from 0.0006 to 0.0043, i.e. materially less memorisation.

**Two consequences worth acting on:**

1. **The voice-only head at full scale (TAR@FAR=1% = 0.705) now EXCEEDS the JOINT
   audio-visual head (0.691)** — but the joint head was trained on only 33,798 clips /
   1,400 speakers. This is a data-scale artefact, not evidence that vision hurts. The
   obvious next win is **retraining the joint AV head at full scale**, which needs more of
   the VoxCeleb2 mp4 mirror (13 of 56 parquet shards local, ~65GB of 279GB).
2. **The Phase 3 operating point should improve directly.** G7's weak side was recognition
   rate (46% correct at 1% FAR) and that was a consequence of head separation quality —
   TAR@FAR=1% has since gone 0.419 → 0.705, so the gate should be re-run on the new head.

### 2026-08-11 — G7 RE-RUN on the full-corpus head: PASS, and now on the CLEAN protocol

Re-ran the Phase 3 gate against `jepa_identity_head_voice_full` (800 unseen speakers).
`scripts/jepa_memory_phase3_gate.py` gained a `--mode voice` path that uses the **real
cross-video split** — enrol on some source RECORDINGS, query on held-out ones — rather than
the mp4 mirror's approximate index-range split. So this number is on a strictly better
protocol than the earlier AV gate, not just a bigger model.

| target FAR | correct | unknown | measured FAR | wrong-name |
|---|---|---|---|---|
| 1% | 0.420 | 0.574 | **0.0113** | 0.006 |
| 2% | 0.493 | 0.497 | 0.0209 | 0.010 |
| 5% | 0.597 | 0.384 | 0.0506 | 0.020 |
| 10% | 0.671 | 0.296 | 0.0985 | 0.033 |

**G7: PASS** (1.13% ≤ 5%), calibration tracking target closely at every point.

**Honest reading — the head got much better, the GATE numbers barely moved.** TAR@FAR=1%
went 0.419 → 0.705 (+68%), yet gate `correct` at 1% FAR went 0.462 → 0.420. Not a
contradiction, and worth stating precisely: the earlier gate ran on the **AV** head with an
**index-range** split (same-recording contamination inflating it), on 280 test speakers;
this runs on a **voice-only** head with a **genuine cross-video** split, on 800 speakers.
The task got harder at the same time the model got better, and the two roughly cancelled.
**They are not comparable, and the honest conclusion is that the earlier gate number was
optimistic** — this one is the trustworthy figure. The full-scale JOINT AV retrain (running)
is what should actually move the gate, since it adds the vision branch back on top of the
data scale-up.

### 2026-08-11 — All three approved tasks launched

1. **G7 re-run** — done, above.
2. **lambda-within sweep** — 0.3 / 0.5 / 1.0, run SEQUENTIALLY at the full 4-GPU
   batch-1024 config rather than 3 runs on 1 GPU each. The dilution effect being tested is
   batch-dependent (5/6144 of candidates), so a smaller-batch sweep would understate the
   fix. 1500 steps each (the metric of interest, within-clip, saturates early). First
   signal at lw=0.3 is immediate: **train within-clip accuracy 0.520 → 0.980 by step 50,
   0.996 by step 100** — the term is doing exactly what it was designed to do; the open
   question is whether held-out within-clip recovers toward 0.897 without costing R@1.
3. **Full-scale joint AV retrain** — mp4 download running in parallel (30/56 shards, network
   only, no GPU contention), chained to extract + retrain once the sweep frees the GPUs.

### 2026-08-11 — LAMBDA-WITHIN SWEEP: the fix works, with a textbook dose-response

All runs at the identical 4-GPU batch-1024 config, 1500 steps, only `--lambda-within`
varying. Matched-step comparison (the baseline's own 0.715 came at step 1999 of a
3000-step run, so only matched steps are compared):

**Step 999:**

| λ_within | VGG within | VGG R@1 | A100M within | A100M R@1 | swap | combined |
|---|---|---|---|---|---|---|
| 0.0 (baseline) | 0.863 | 0.630 | 0.933 | 0.072 | 0.004 | 1.493 |
| **0.3** | **0.882** | **0.657** | **0.961** | **0.087** | 0.003 | **1.539** |
| 0.5 | 0.883 | 0.623 | 0.971 | 0.075 | 0.003 | 1.506 |
| 1.0 | 0.907 | 0.604 | 0.982 | 0.088 | 0.003 | 1.511 |

**Step 1499:**

| λ_within | VGG within | VGG R@1 | A100M within | A100M R@1 | combined |
|---|---|---|---|---|---|
| 0.0 (baseline) | 0.854 | 0.699 | 0.937 | 0.075 | 1.553 |
| **0.3** | **0.883** | 0.681 | **0.961** | **0.090** | **1.564** |
| 0.5 | 0.887 | 0.639 | 0.981 | 0.077 | 1.526 |

**Findings:**

1. **The mechanism is confirmed by a clean monotonic dose-response**: within-clip rises
   (0.863 → 0.882 → 0.883 → 0.907) and R@1 falls (0.630 → 0.657 → 0.623 → 0.604) as λ
   increases. A hypothesised mechanism producing an orderly dose-response across four
   settings is much stronger evidence than a single lucky config.
2. **λ=1.0 fully recovers the regression and then some** — within-clip 0.907 exceeds the
   original batch-96 value of 0.897 — but pays 0.095 of R@1 to do it. Too expensive.
3. **λ=0.3 is the recommended setting.** At step 999 it **strictly dominates the baseline
   on all four metrics** (within +0.019, R@1 +0.027, and both Action100M metrics up). At
   step 1499 it recovers +0.029 of the 0.043 within-clip regression while giving back only
   0.018 of R@1 — i.e. **~2/3 of the loss recovered for ~7% of the gain**, which is the
   trade the fix was designed to make. Best combined score at BOTH matched steps.
4. **Action100M improves at every λ > 0** (within 0.937 → 0.961-0.982, R@1 up too). With
   K=2 there is only ONE within-clip negative, so it is the most diluted case and benefits
   most from being given its own term.
5. **Swapped-query stays at 0.003-0.004 throughout** — slightly BETTER than baseline. The
   within-clip gain is genuine query-following, not memorisation of caption length/style,
   which is exactly what this control exists to rule out.

**Standing recommendation: `--lambda-within 0.3` for future query-predictor runs.**

Final step-1499 row for λ=1.0, completing the table: within **0.905**, R@1 0.633,
A100M within **0.987**, combined 1.538. Confirms the dose-response holds to the end of
training — λ=1.0 buys the highest within-clip of any setting (0.905, above the original
batch-96 0.897) and pays the most R@1 for it. Ranking by combined score at step 1499:
**λ=0.3 (1.564) > λ=0.0 (1.553) > λ=1.0 (1.538) > λ=0.5 (1.526)**.

### 2026-08-11 — FULL-SCALE JOINT AV HEAD: vision EARNS its place, and this is the best head yet

**106,736 clips / 4,420 speakers** (40 of 56 mp4 shards), speaker-disjoint 3,536 train /
**884 unseen** test — vs 33,798 / 1,400 / 280 for the first joint head (**3.2x** on every axis).

| gallery N | vision froz | vision head | audio froz | audio head | joint froz | **joint head** |
|---|---|---|---|---|---|---|
| 2 | 0.859 | 0.875 | 0.941 | 0.946 | 0.847 | **0.966** |
| 5 | 0.679 | 0.775 | 0.841 | 0.882 | 0.691 | **0.917** |
| 10 | 0.595 | 0.713 | 0.769 | 0.836 | 0.595 | **0.880** |
| 50 | 0.428 | 0.585 | 0.610 | 0.707 | 0.429 | **0.771** |
| 280 | 0.328 | 0.469 | 0.481 | 0.563 | 0.327 | **0.631** |
| **AUC** | 0.820 | 0.891 | 0.888 | 0.950 | 0.820 | **0.966** |
| **TAR@FAR=1%** | 0.350 | 0.571 | 0.464 | 0.694 | 0.351 | **0.765** |

**1. VISION EARNS ITS PLACE — this is the clean answer to the open question.** At matched
data scale, matched protocol, matched everything: **joint 0.765 > audio-only 0.694 >
vision-only 0.571** on TAR@FAR=1%. Adding the face to the voice is worth **+0.071 TAR
(+10.2% relative)**, and the same ordering holds at every gallery size. The earlier worry —
that full-scale voice-only (0.705) beat the small-scale joint head (0.691), implying vision
might be dead weight — was **confirmed to be a data-scale artefact**, exactly as diagnosed.
The user's original push-back ("I don't just identify people by their voice, most of the
time it's their face") is now supported by a controlled measurement.

**2. Best head produced by this track.** vs the first joint head: TAR@FAR=1% 0.691 → **0.765**
(+10.7%), 280-way top-1 0.557 → **0.631** (+13.3%), AUC 0.952 → **0.966** — while the test
set got 3.2x larger (harder, not easier).

**3. The trained head is now essential, not incremental.** joint-frozen is 0.351 TAR and
0.327 top-1 — *worse than audio-frozen alone* (0.464 / 0.481), i.e. naive concatenation of
frozen features is still actively harmful. The head takes the same inputs from 0.351 to
0.765, a **2.2x** improvement. This re-confirms the Phase-2c finding at 3.2x the scale.

**4. Vision remains the weaker branch** (0.571 vs 0.694 TAR) for the reason given in
Phase 2c: V-JEPA2 is not a face-recognition model and VoxCeleb2 faces are small compressed
YouTube crops. A dedicated face encoder is the obvious future lever — but the *complementarity*
is real regardless of which branch is stronger.

### 2026-08-11 — FINAL G7 on the full-scale joint head: PASS, and recognition finally moves

884 speaker-disjoint identities, household of 5, 8 enrolments each, 150 random households.

| target FAR | correct | unknown | measured FAR | wrong-name | G7 |
|---|---|---|---|---|---|
| **1%** | **0.515** | 0.481 | **0.0112** | **0.004** | **PASS** |
| 5% | 0.665 | 0.321 | 0.0511 | 0.014 | FAIL (marginal) |

**Recognition rate finally moved.** Across the three gate runs at the 1% operating point:

| head | test ids | correct @1% FAR | wrong-name |
|---|---|---|---|
| joint, small corpus (index-range split) | 280 | 0.462 | 0.005 |
| voice-only, full corpus (clean cross-video) | 800 | 0.420 | 0.006 |
| **joint, full corpus** | **884** | **0.515** | **0.004** |

Best recognition AND lowest wrong-name rate, on the largest test set. Earlier gate runs
traded one against the other; this one improves both at once.

**The 5% row FAILS its own gate at 0.0511 vs a ≤0.05 threshold** — by 0.0011, with the CI
[0.032, 0.078] straddling the line. Reported as a FAIL rather than rounded to "essentially
5%", because the threshold was pre-registered. It does not change the recommendation: the
**1% operating point is the deployment setting** (0.4% wrong-name), and it passes cleanly
with CI upper bound 0.023.

**Deployment recommendation, final:**
`checkpoints/jepa_identity_head_av_full/head_joint.pt` at a **1%-FAR-calibrated threshold**
(~0.33) — 51.5% of household queries recognised, 48.1% answered "I don't know you", **0.4%
wrong name**, 1.1% of strangers falsely accepted.

### 2026-08-14 — THINKER↔PERCEPTION LOOP CLOSED: the thinker can now ask, and get a grounded answer

`models/m5_perception_query.py::PerceptionQueryEngine` + `scripts/perception_query_e2e.py`.

**Built as a TOOL, not a new protocol.** `models/m5_tools.py` already parses
`<tool_call name=.../>`, executes a handler, and folds the result into the spoken line —
built, tested, already carrying time/date/weather. Perception is just another tool, so the
thinker needs **no new output format and no retraining to use it**, and
`assets/tool_call.gbnf` keeps the syntax valid for free.

**Retrieval, not an embedding prefix — deliberately, for v1.**
`scripts/prototype_llama_embd_input.py` proved llama.cpp accepts a raw embedding prefix, so
feeding `z_q` straight in is mechanically possible. Not done yet because the thinker has
never been trained to INTERPRET JEPA-space vectors, and this project already paid for that
lesson twice (M3 soft-prompt connector dropped as slow + confidently wrong; M4b/Ultravox
projector plateaued at WER 0.94). Retrieval is the path measured to work on-device.

**A genuinely NEW and harder measurement.** Every earlier query-predictor number scored
against a small candidate set — within-clip picked 1 of 6, cross-clip R@1 picked 1 of ~600
at a single fixed granularity. Deployment retrieves against a bank of tens of thousands
spanning every clip AND every granularity at once, where near-duplicate captions from other
scenes are real competitors.

| bank | correct CLIP | chance | correct FIELD | swapped | both | latency |
|---|---|---|---|---|---|---|
| 6,000 captions / 1,000 clips | **0.442** | 0.00100 | **0.946** | 0.000 | 0.421 | 17.2 ms |
| 48,000 captions / 8,000 clips | **0.189** | 0.00013 | **0.953** | 0.003 | 0.181 | 17.7 ms |

Per-query-type field accuracy at the 6k bank: action-brief 0.900, action-detailed 0.975,
summary-brief 0.925, summary-detailed 0.975, sound-brief 0.950, sound-detailed 0.950.

**Findings:**

1. **Query-following is essentially solved and bank-size-INDEPENDENT** (0.946 → 0.953 as the
   bank grew 8x), with the swapped-query control at **0.000-0.003** — ask the wrong question
   and it essentially never returns the right field type. That is the cleanest evidence
   available that the answer tracks the QUESTION, not just the scene.
2. **Scene grounding degrades with bank size, as it must**: 0.442 → 0.189 for an 8x bigger
   bank. Still **1,450x chance**, but this is the honest deployment number and it is much
   lower than the 0.715 R@1 measured against a ~600-clip eval pool. **Any future claim about
   this system must state its bank size.**
3. **Retrieval latency is flat in bank size** (17.2 → 17.7 ms for 8x more candidates) — it is
   one matmul, so the bank can grow a lot before this is the bottleneck.
4. **Tool round-trip works end to end**: a real emitted `<tool_call name=look query="Describe
   the room and setting in detail."/>` parsed → executed → returned a scene-grounded
   paragraph.

**Safety behaviours built in, because a confident wrong answer is the failure that matters:**
`ask()` returns None (and the tool says so in words) when there is no perception at all, or
when perception is older than `max_age_s` — BMO saying "I can't see anything right now"
beats describing a room it saw a minute ago. An optional `min_score` floor allows refusing
weak matches.

**Honest limitation, stated plainly: the answer vocabulary IS the candidate bank.** The
engine can only say things the bank contains. That is why the bank is built from the full
caption corpora rather than a hand-written list, and why `ask()` returns the retrieval score
so a caller can decline to speak on a weak match.

### 2026-08-14 — WIRED INTO THE STREAMING LOOP (opt-in, non-breaking)

`StreamingLoop.__init__` gained `perception_query_engine=None`, and
`_maybe_refresh_vision()` now publishes each tick's perception to it.

**Why that hook specifically:** `_maybe_refresh_vision` is the ONE place that builds
`feats`/`tbins`, and publishing from the SAME object the world-state was just built from
means "what do you see?" and the decision path can never disagree about which moment they
describe. Refreshing them separately would recreate the item-0 divergence class of bug
(three independent reimplementations of feature construction, each wrong differently) in a
new place.

`PerceptionQueryEngine.update_from_features(feats, tbins, duplex_loop)` is the live
equivalent of training's `build_sources()`. It reads its own `source_dims`, so the call
site never needs to know which streams a checkpoint uses — swapping checkpoints needs no
loop change. The `m2` stream comes from the EXISTING `DuplexLoop.compute_pre_pool()` (the
same method M3 used), not a reimplementation.

**Measured cost, stated rather than hidden:** the loop already calls
`compute_world_state()`, and `encode_pre_pool_tokens` shares everything with it except the
final attentive pool, so requesting the `m2` stream costs **one extra M2 backbone pass
(~160ms on Jetson)**. It is paid on the vision-refresh thread, already off the response
critical path. It is deliberately NOT deduplicated: deriving the world-state from the
pre-pool tokens here would mean reimplementing the attentive pool — the exact divergence
class this hook was placed to avoid. A checkpoint configured without the `m2` stream avoids
the cost entirely.

**Verified in isolation** (fake stack, no GPU, no encoders — this is a WIRING test; the
semantic behaviour was proven separately in the e2e run above):

| behaviour | result |
|---|---|
| ask before any refresh | `None` (refuses, correct) |
| publish on refresh | 1.4 ms, `compute_pre_pool` called exactly once |
| new scene → perception republished | yes (age < 0.5s) |
| engine raises mid-publish | caught, logged, **loop survives** |
| `engine=None` | no-op, zero cost, no crash |

Back-compat confirmed: the parameter defaults to `None`, and the latency dict returned by
`_maybe_refresh_vision` keeps its original shape (a new `perception_query_publish_ms` key
appears only in the logs, not in the returned dict).

Also added `load_perception_query_engine()` so a caller (e.g. `build_bmo_stack`) can
construct the engine without duplicating checkpoint-loading — including the easy-to-miss
restore of the **co-trained `text_target.proj`**, without which retrieval geometry silently
mismatches training and every answer degrades for no visible reason.

**NOT yet done (the remaining deployment gaps, both flagged rather than glossed):**
1. Never run on the Jetson — no on-device latency/memory figure for this path.
2. `build_bmo_stack()` does not construct the engine yet; production is unchanged until it
   does. A precomputed bank file should be shipped rather than encoding captions at boot.
3. The face-detector gap still blocks the identity half (not this query half, which needs
   no crops).

### 2026-08-14 — ON THE JETSON: it runs, and the bottleneck is NOT where I expected

Real hardware, after `reboot` + `jetson_preflight.sh` (**PASS**: free 6848 MiB, large blocks
6524 MiB, competing services stopped). `~/jetson_perception_query_results.json`, copied to
`jetson_artifacts/benchmarks/home/`.

**Memory — comfortable, not tight:**

| load step | used | available |
|---|---|---|
| start (post-preflight) | 2528 MiB | 5091 MiB |
| + EmbeddingGemma (query encoder) | 3238 MiB | 4382 MiB |
| + QueryPredictor + 24k-caption bank | 3383 MiB | **4237 MiB** |

The whole capability costs **~855 MiB** (EmbeddingGemma ~700, predictor + bank ~145). The
bank was shipped fp16 (74 MB instead of 147 MB) specifically to protect this budget.

**Latency — the surprise:**

| stage | median |
|---|---|
| **query encode (EmbeddingGemma)** | **263.5 ms** |
| QueryPredictor forward | 138.5 ms |
| bank lookup (24,000 candidates) | **2.6 ms** |
| **total per question** | **403.8 ms** |

**The bottleneck is encoding the QUESTION, not searching the bank.** Retrieval over 24,000
candidates is 2.6 ms — 0.6% of the total — so the bank could grow ~10x before it matters.
65% of the cost is running a 300M-param text encoder over a ~10-word question. That inverts
the intuition the design was built on (I had assumed retrieval scale would be the risk) and
it points at a concrete, cheap optimization: **cache query embeddings**. The thinker asks
from a small, repetitive space of questions, so an LRU cache would take repeat questions to
~141 ms, and precomputing the ~30 most likely phrasings at boot would cover most traffic.

Sample answer on synthetic perception (random tokens, so the CONTENT is meaningless — this
measures the path, not the semantics): a fluent, well-formed scene paragraph at score 0.528.

**Real integration bug this caught, which mercury testing could not:** the Jetson's
`models/text_target.py` predated `encode_text_frozen_raw()`, so the first run crashed with
`AttributeError`. Every mercury test passed because mercury had the newer file. This is
precisely the class of gap that on-device testing exists to find — the fix was pushing
`text_target.py` + `av_jepa_predictor.py` alongside the new modules.

**Status: the query path is measured end-to-end on real hardware.** Still NOT done:
`build_bmo_stack()` is wired but has not been run as part of a FULL stack boot (this test
loaded the engine in isolation), so the interaction with the load-order-sensitive
LLM/TTS/perception sequence is unverified.

### 2026-08-14 — JETSON DESCRIBE-DEMO: it runs, it fits (without TTS), and TWO real blockers found

`scripts/jetson_describe_demo.py` — CSI camera → V-JEPA2 + WavJEPA → M2 → QueryPredictor →
retrieval → fast-tier LLM. **No STT, no VAD, no decision head, no M3** (per instruction:
BMO describing what it sees, not a conversation). Face engine + Xorg deliberately LEFT
RUNNING (they own the display; `bmo_app.service` that preflight stops is an unrelated
Streamlit labeling app, verified). Reboot → `jetson_preflight.sh` **PASS** (6881 MiB free)
→ run. **No new quantization applied** (per instruction); perception uses the pre-existing
production int8 path.

**It works, and the output is genuinely scene-grounded:**
> perception: *"In this video segment, we observe a person interacting with their
> environment inside a building. Initially focused on an air conditioning unit mounted on
> the ceiling…"*
> BMO: *"The person is opening a door to collect a coffee cup, while the AC unit hums in
> the background."*

The room really does have a ceiling AC/vent and a door — this is real grounding, not a
generic caption.

**Memory — fits, but tight (no TTS):**

| stage | Δ | avail |
|---|---|---|
| after preflight | — | ~6100 MiB |
| + fast tier v2 GGUF | 273 | 4787 |
| + ViT-L int8 | 577 | 4210 |
| + WavJEPA base int8 | 646 | 3564 |
| + WavJEPA nat int8 | 615 | 2742 |
| + M2 int8 | −457 (quantize frees) | 3199 |
| + EmbeddingGemma | 999 | 2200 |
| + QueryPredictor + bank | 665 | **1535** |
| after 3 rounds (activations) | 812 | **737** |

**BLOCKER 1 — TTS is broken on this device, and it is NOT memory.** `import onnxruntime`
**alone aborts**, with 5145 MiB free:
```
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
.../stl_vector.h:1130: Assertion '__n < this->size()' failed.
```
Isolated in three steps (decoder alone / with sherpa / neither) — the pip `onnxruntime
1.23.2` fails to identify the Tegra CPU, gets a short vector, and indexes out of bounds.
This kills the **NeuCodec INT8 ONNX decoder**, which is the deployed TTS decode path.
Note `sherpa_onnx` ships its OWN `libonnxruntime.so.1.18.1` and works fine — so SenseVoice
STT is unaffected while NeuTTS is dead. **TTS fit is therefore UNMEASURED, not "fits".**

**BLOCKER 2 (the big one) — the Jetson was in 7W power mode, not MAXN_SUPER.**
`nvpmodel -q` = **7W (mode 3)**: only **4 of 6 cores online**, CPU capped **960 MHz**
(vs 1728), GPU **408 MHz** (vs 1020). Every historical benchmark in this project was taken
at MAXN_SUPER + `jetson_clocks`. That explains the demo's perception latency of
**3.8–12.4 s** against the documented **1247 ms** at 16 frames — roughly the 2.5x GPU
clock deficit compounded by CPU-side capture/pre-processing. **All latency numbers in this
entry are 7W-mode numbers and must not be compared against the historical MAXN_SUPER
figures.** Switched to MAXN_SUPER (`nvpmodel -m 2`, needs a reboot) + `jetson_clocks`,
re-preflighted (PASS, 6914 MiB free) and re-ran. **The fix is dramatic — 7x on perception:**

| stage | 7W mode | **MAXN_SUPER** | speedup |
|---|---|---|---|
| perception (ViT-L + WavJEPA) | 3828-12368 ms | **1089-1732 ms** | **~7x** |
| M2 pre-pool | 40-78 ms | 24-39 ms | 1.9x |
| query (encode + predict + lookup) | 1033-1641 ms | **240-553 ms** | ~3.6x |
| fast-tier LLM | 1552-7274 ms | **364-504 ms** | ~6x |
| **total per round** | **14971-18401 ms** | **3892-4994 ms** | **~4x** |

MAXN_SUPER perception (1089-1732 ms) now lands right on the documented 1247 ms figure,
confirming the 7W mode was the entire discrepancy rather than anything about this pipeline.

**And the answers became scene-RESPONSIVE.** At 7W all three rounds returned an identical
caption; at MAXN_SUPER each round differs (AC unit / wooden wardrobe / smoke detector in an
office with a cluttered desk) — the pipeline is tracking a changing view instead of
returning a constant. The room genuinely contains a ceiling vent, a wardrobe-like cabinet
and a cluttered desk, so these are grounded, if imperfect, reads.

**Third finding — a silent-garbage bug in MY demo script (scope corrected 2026-08-14):**
`scripts/jetson_describe_demo.py` originally opened the CSI sensor via OpenCV's default
V4L2 backend, which returns *unconverted* Bayer/YUV that OpenCV reports as a SUCCESSFUL
read — in practice a flat green frame. That run described a blank image and produced three
identical nonsense captions ("a tuning fork being tapped") while looking perfectly healthy.
Fixed: `nvarguscamerasrc` first, plus a per-channel std check (`_frame_is_degenerate`).

**IMPORTANT SCOPE CORRECTION:** this was a defect in the demo script written for this
track, **NOT in the deployed codebase**. `face_engine/motion_tracker.cpp` already goes
through nvarguscamerasrc / the ISP-Argus pipeline correctly and never had this bug — 
confirmed by the motion-tracking agent on the device. The earlier phrasing here could have
been read as implicating the deployed tracker; it does not.

**Open, needs your eye:** the camera is mounted sideways. `--rotate` defaults to 90° CCW,
but the dim frame did not let me confirm the direction visually — you know the physical
mount, so this wants a quick confirmation.

### 2026-08-14 — MAX BANK + QUANTIZATION + FULL STACK: what fits, and two hard findings

**Bank scaled 5x:** all held-out VGGSound (13,679 clips x 6 granularities) + 30,000
held-out Action100M clips x 2 → **121,104 unique captions** after dropping 20,970
duplicates/placeholders. 355 MiB fp16 (`checkpoints/perception_bank_max_fp16.pt`).

**FINDING 1 — int8 via torchao is a NO-OP on the Jetson.** `[quant] encoder 577.7 MiB ->
577.7 MiB`, and EmbeddingGemma still cost +995 MiB on load. Cause: torchao prints
*"Skipping import of cpp extensions due to incompatible torch version. Please upgrade to
torch >= 2.11.0 (found 2.8.0)"* — the Jetson's torch 2.8.0 cannot use torchao's quantized
kernels. **The 105 MiB saving measured on mercury does NOT transfer to the device.** Any
real on-device saving needs a different mechanism (llama.cpp-style GGUF, or a bitsandbytes/
custom int8 path that does not depend on torchao's extensions).

**FINDING 2 — the full stack with thinker + max bank does NOT fit.** Measured load walk:

| stage | Δ | avail |
|---|---|---|
| start (post-preflight) | — | 4922 |
| + fast tier v2 | 816 | 4093 |
| **+ thinker v3 (Qwen3-0.6B Q8)** | **1103** | 2990 |
| + ViT-L int8 | 767 | 2223 |
| + WavJEPA base | 414 | 1809 |
| + WavJEPA nat | 727 | 1082 |
| + M2 (quantize frees) | −484 | 1566 |
| + EmbeddingGemma | 995 | **571** |
| + 355 MiB bank | — | **OOM** (`NVML_SUCCESS == r` / NvMap error 12) |

The thinker alone is **1103 MiB**. Thinker + max bank cannot coexist with perception in
7.6 GB. Options, in order of preference: drop the thinker from the perception build (the
fast tier already phrases descriptions well), shrink the bank, or — best — make the bank
unnecessary (see `PERCEPTION_GENERALIZATION_PLAN.md`).

**Real bug fixed along the way:** loading the bank called `F.normalize(bank.float())`, which
materialised a **2x fp32 copy of the whole bank** (355 → ~710 MiB) at the worst possible
moment. `build_bank` already L2-normalises, so this was pure waste. Now: sample-check the
norms, skip if already unit-norm, and otherwise normalise in 8192-row chunks.

**Quantization verdict (all measured on mercury, `EMBEDDINGGEMMA_QUANT_EVAL.json`):**

| config | retrieval field acc | encoder-output cos | query latency | note |
|---|---|---|---|---|
| bf16 (baseline) | 0.939 | — | 14.8 ms | |
| **int8 linears** | **0.939** | **0.9998** | 18.1 ms | safe; ~105 MiB **on mercury only** |
| int8 dynamic-activation | 0.933 | — | **163 ms (11x slower)** | activation-aware but far too slow |
| int8 embedding table | **0.272** | **0.31-0.58** | — | **BREAKS THE ENCODER** — do not use |
| true AWQ | — | — | — | unavailable (needs `SupportsActivationPreScaling`; int4 path needs `mslk>=1.0.0`) |

### 2026-08-14 — 4-WAY STREAM ABLATION: SigLIP2 is a big win, and ONE ear is enough

All four arms trained on the **identical** clip pool (VGGSound 171,430 + Action100M 69,339 =
240,769), identical hyper-parameters, 3000 steps, `lambda_within=0.3`, single GPU each.
Best checkpoint per arm, VGGSound held-out:

| arm | streams | ambient | within-clip | swap | **cross-clip R@1** |
|---|---|---|---|---|---|
| **A** | m2+vision+ambient | mean | 0.906 | 0.003 | **0.441** |
| **B** | **+ scene** | mean | 0.911 | 0.002 | **0.564** |
| **C** | **+ scene** | **base only** | 0.886 | 0.003 | **0.566** |
| **D** | scene + vision | mean | 0.891 | 0.007 | 0.546 |

**1. SigLIP2 helps a lot: A → B = 0.441 → 0.564, +28% relative.** The largest single-change
gain in this track, and it targets exactly the failure the user identified (the stack
answering "opening a microwave oven" to a bedroom).

**2. WavJEPA-nat does NOT earn its cost — one ear is enough.** C (base only) 0.566 vs
B (base+nat mean) 0.564: identical within noise, and base-only is marginally *ahead*. This
independently reproduces the user's own earlier M2 ablation (base 37.99% vs mean 37.15%,
`logs/m2_ablation_audio_*.log`). **Dropping nat frees the measured 469 ms that is the single
largest perception component on the Jetson** (`PERCEPTION_FINDINGS.md`).
**Mechanism**: nat is a BINAURAL model being fed `wav.unsqueeze(0).expand(2,-1)` — duplicated
mono (`world_state_builder.py:156`) — so it has never received the stereo signal it was
trained for. It has been running out of distribution for its entire deployed life.

**THE 4-MIC CAVEAT, and why we drop nat anyway (decision recorded 2026-08-14).**
BMO's ReSpeaker IS a 4-mic array, so real spatial audio is physically available — nat was
built for exactly that, and this ablation does NOT prove nat is a bad model. It proves nat is
useless *on duplicated mono*. Two separate questions, and only the second was tested.

Dropping it regardless, because the deciding constraint is the **Jetson Orin Nano**, not model
quality:
  * nat costs a measured **469 ms per perception tick** — the single largest component,
    larger than ViT-L at 16 frames (292 ms).
  * The full describe stack already runs with only **827 MiB** free, and the thinker alone
    (1103 MiB) does not fit alongside the max bank.
  * Feeding it real stereo is not a config change: `world_state_builder.build_world_state_
    features()` mixes to mono BEFORE the encoders (`wav.mean(0)`), the cached corpora
    (VGGSound/Action100M) are mono, and **M2 was trained on the base+nat mean of mono** —
    so real-stereo nat would need a re-extraction AND an M2 retrain to be used honestly.
  * Measured payoff for all that: **+0.00 to −0.002 R@1** on this benchmark.

**So: drop nat now; the door stays open.** If a future build wants spatial hearing (sound
localisation, "who is speaking, and where"), that is a different capability than congruence
and would justify revisiting — with real stereo capture, a re-extraction, and its own metric.
Recorded here so the reasoning is not lost and nat is not silently written off as a bad model.

**3. Audio + M2 contribute only ~+0.02 ON THIS METRIC (D 0.546 vs B/C ~0.565) — and that is
NOT evidence they are unnecessary.** This is a caption-retrieval metric over VGGSound and
Action100M, which is overwhelmingly visual; 4 of the 6 VGGSound question types are visual.
M2's trained job is audio-visual **congruence** — knowing a sound belongs to what is being
looked at — which this benchmark barely probes. The honest conclusion is *for scene
description, vision+scene is nearly sufficient*, not *M2 is useless*. Testing congruence
needs a congruence task (e.g. matched-vs-mismatched AV pairs), which this ablation did not run.

**Recommended configuration: `m2 + vision + ambient(base) + scene`** — arm C. Best R@1
(0.566), keeps the JEPA trunk and AV congruence intact, and drops 469 ms of Jetson latency.

### 2026-08-14 — AV CONGRUENCE EVAL: the ears ARE load-bearing, and nat is not free after all

`scripts/eval_av_congruence.py`. The stream ablation could not settle whether audio/M2 earn
their place, because caption retrieval over VGGSound is visually guessable (you can SEE the
guitar). This test removes that shortcut: **swap the audio between clips, then ask what it
hears.** Retrieving the caption of the clip the AUDIO came from = following the ears;
retrieving the clip you can SEE = inferring sound from pictures.

| arm | has audio | **follows EARS** | follows EYES | matched control |
|---|---|---|---|---|
| **A** m2+vision+ambient(mean) | yes | **0.650** | 0.350 | 0.956 |
| **B** +scene, ambient(mean) | yes | **0.609** | 0.391 | 0.958 |
| **C** +scene, ambient(**base only**) | yes | **0.562** | 0.438 | 0.953 |
| **D** scene+vision (no audio) | no | **0.070** | 0.930 | 0.923 |

**1. The ears are real.** D follows the eyes 93% of the time — it is not guessing (that would
be ~0.5), it systematically answers sound questions from the picture. Every audio-carrying arm
follows the ears instead. **This is the measurement the stream ablation could not make**: on
caption retrieval D looked only 0.02 behind, but on the job M2 exists to do it is catastrophic
(0.070 vs 0.650). M2 + WavJEPA are load-bearing for a companion that actually listens.

**2. This REVISES the "one ear is enough" conclusion — nat is not free.**

| arm | R@1 (scene ablation) | audio-following |
|---|---|---|
| B (base+nat) | 0.564 | **0.609** |
| C (base only) | 0.566 | **0.562** |

Identical on caption retrieval, but **dropping nat costs 4.7 points of audio-following
(0.609 → 0.562)**. The earlier "nat is free" claim was an artifact of measuring on a metric
that barely uses audio at all. Corrected here rather than left standing.

Note also A (no scene, mean) has the HIGHEST audio-following at 0.650 — adding the strong
visual scene stream slightly reduces reliance on the ears (0.650 → 0.609). Sensible: a better
picture makes guessing-from-pictures more attractive. It is a trade, not a bug, but it is
worth watching if audio grounding matters more than description quality.

**REVISED RECOMMENDATION.** Not a simple "drop nat":
  * If the product needs BMO to genuinely hear (sound events, "what was that noise?", not
    confusing a TV picture of a dog with a real dog) -> **keep arm B** (`m2+vision+ambient
    (mean)+scene`) and pay the 469 ms.
  * If perception latency is the binding constraint and description quality is what matters
    -> **arm C** is still defensible: identical R@1, 469 ms cheaper, and still follows the
    ears 0.562 of the time versus 0.070 without audio.
The honest framing is a **latency-vs-listening trade with a measured price**, not a free win.

Unified-config R@1 **plateaued at ~0.46 from step ~2000** (0.462 / 0.449 / 0.458 / 0.466 /
0.458 across the last five evals), so step-count is exhausted as a lever — consistent with
the prediction that cross-clip R@1 is negative-count-bound, not step-bound. Testing batch
directly: batch 192 running; **batch 288 OOMed** at 94.97GiB (tried to allocate 11.50GiB on
top of 90.19GiB in use), which sets the single-GPU ceiling between 192 and 288 for the
3-stream unified config and is exactly why DDP+GradCache (not a bigger single batch) is the
real path past this.

### 2026-08-10 — track opened
- 4x Blackwell confirmed idle, 0MiB used, no processes.
- Disk state recorded (§0). RAID 100% full — a real constraint on Phase D, not on 0–4.
- Baselines pulled from checkpoints' own `results_log` and training logs (§1).
- **Soft-InfoNCE recorded as a negative result** (58.90 vs 62.05), previously unlogged (§2a).
- `falsifier_tracking.md` step-799 per-dataset numbers corrected (§2b); score unchanged.
- V-JEPA 2.1 evidence located and summarized (§3); ViT-B-on-Jetson confirmed **never measured**.
- Nothing trained yet. No GPU time consumed beyond checkpoint reads.

---

## 2026-08-15 — SigLIP2 target space: built, FALSIFIED, fixed

Triggered by the user catching a real inconsistency: *"why the fuck is embedding gemma there,
I thought we got siglip2 to replace embedding gemma?"* `ARCHITECTURE.md` §6 had claimed
SigLIP2 would replace the bank **and** EmbeddingGemma; what had actually been built used
SigLIP2 only as a fourth input stream while retrieval still ran through EmbeddingGemma's
space. Both were therefore resident on the Jetson (1,441 + 578 MiB), which is why the
core-pipeline fit test OOM'd at 659 MiB free.

### A.1 — The claim, corrected before any training
SigLIP2 **scores** text, it does not **produce** it, so a candidate list is inherent to
retrieval and is not an artifact of EmbeddingGemma. The "no bank to curate" line was wrong
and is corrected in place. What SigLIP2 genuinely removes is the *corpus-shaped* 121k bank
and the text encoders.

Measured tower split (`siglip2-base-patch16-224`, fp16), which drove the design:

| | params | fp16 |
|---|---|---|
| vision tower | 92.9 M | **177 MiB** |
| **text tower** | 282.3 M | **538 MiB** |

The text side is 3x the vision side because of a 256k-token Gemma vocab embedding — the same
table that broke under int8 on EmbeddingGemma. **Consequence: pre-encode candidates offline
and ship only vectors; no text tower need ever load on-device.**

### A.2 — Corpus fix (required before any comparison was valid)
Action100M scene coverage was 80,000 / 399,934, so `--restrict-to-scene` gutted the corpus
345,754 → 69,339 (20%). Extracted the missing **319,934** segments: 64 shards x 4 threads,
**~249 clips/s, ~22 min, bad=0**. Coverage now 345,754 → **345,751**. The scene-restricted
pool (517,181) now matches the unrestricted pool (518,347) to **0.2%**, which is what makes
the 3-stream vs 4-stream comparison legitimate.

`scripts/extract_siglip2_scene.py` gained `--skip-existing` (resumable top-up) and
`--limit 0` (whole corpus).

### A.3 — The head-to-head (all runs: 518k pool, batch 1024, negatives 2048/6144, λ_within 0.3)

| run | target space | streams | VGG within | VGG R@1 | A100M R@1 |
|---|---|---|---|---|---|
| `query_predictor_ddp_lw0.3` *(ref)* | EmbeddingGemma + proj 1536 | 3 | **0.883** | 0.681 | 0.090 |
| `sig_runA_matched3stream` | SigLIP2 frozen (Identity) | 3 | 0.654 | 0.489 | 0.051 |
| `sig_runB_scene4stream` | SigLIP2 frozen (Identity) | 4 | 0.688 | 0.627 | 0.069 |
| `sig_runC_proj1536` | SigLIP2 + proj 1536 | 4 | 0.747 | **0.739** | 0.091 |
| **`sig_runD_proj768`** ← deploy | SigLIP2 + proj **768** | 4 | **0.811** | **0.737** | 0.085 |

**NEGATIVE RESULT — the frozen target space.** Run A is the strict apples-to-apples against
the reference and lost badly. The within-clip curve was **FLAT from step 249**
(0.651 / 0.645 / 0.642 / 0.655 / 0.658 / 0.654) — a ceiling, not slow convergence.

**The first root-cause hypothesis was ALSO wrong.** The guess was that SigLIP2's 64-token
space collapses long captions. Measured on 400 held-out clips' 6 caption fields:

| text space | within-clip cos | cross-clip cos |
|---|---|---|
| SigLIP2 raw | 0.7556 | 0.6648 |
| EmbeddingGemma raw | 0.7306 | 0.6075 |
| **EmbeddingGemma + trained proj** | **0.4340** | **0.1533** |

SigLIP2's raw space is barely worse than EmbeddingGemma's raw space. **The trainable
projection is where the representation learning happens** — it spreads a space crammed
between 0.61–0.73 out to 0.15–0.43. The frozen design deleted the load-bearing component.

**FIX.** Two things had been conflated: *no text encoder on-device* comes from pre-encoding
(a projection is a matmul applied offline at bank-build time) and *frozen target space* comes
from Identity. Only the second cost accuracy, and it was never required. Restoring the Linear
(B → D, nothing else changed) moved within-clip **0.688 → 0.811** and R@1 **0.627 → 0.737**.

**proj-768 beats proj-1536**: same R@1 (0.737 vs 0.739), much better within-clip (0.811 vs
0.747), and half the bank. Cheaper *and* better.

**Net vs reference: R@1 0.737 vs 0.681 (+8.2% rel) with on-device text machinery 933 MiB →
177 MiB.** Still behind on within-clip (0.811 vs 0.883); swapped-query control 0.006 vs a
0.167 chance level, so the query is genuinely read.

Accepted cost: banks are checkpoint-coupled again (rebuild on retrain — build-time, not
memory). Zero-shot tag scoring must read `encode_text_frozen_raw`, never the projected target.

### A.4 — Candidate sets: captions vs tags (run D, 512 held-out clips)

| path | space | metric | result |
|---|---|---|---|
| captions via **predictor** | learned | caption R@1 | **0.705** |
| captions via zero-shot | raw | caption R@1 | 0.619 |
| tags via **predictor** | learned | tag p@5 | **0.418** (shuffled 0.021, gap **+0.396**) |
| tags via zero-shot | raw | tag p@5 | 0.387 (shuffled 0.048, gap +0.339) |

1. **The JEPA streams earn their place**: predictor 0.705 > zero-shot SigLIP2 0.619. SigLIP2
   alone is NOT sufficient — V-JEPA2 + WavJEPA + M2 + query conditioning add real signal.
2. **Tags survive the predictor path** (0.418 > 0.387 zero-shot); the extrapolation worry was
   unfounded. Caveat: tag p@5 is a word-overlap proxy, not comparable to caption R@1, and is
   only meaningful by its gap to the shuffled control.

**The room frame (the actual deployment condition), zero-shot:**

| candidates | top results |
|---|---|
| **tags (2.04 MiB)** | *an office chair* (+0.118), *a cluttered room*, *desk*, *a tidy room*, *a desk*, *chair* |
| captions (177 MiB) | *"a musician practicing the clarinet within the confines of an office space"* |

**Tags return grounded correct facts; captions hallucinate a clarinet player.** Strongest
evidence yet for handing the thinker tags and letting it compose the sentence.

### A.5 — AV congruence re-measured in the new space (audio swapped, asked what it HEARS)

| run | follows EARS | matched control |
|---|---|---|
| **`sig_runD_proj768`** (base audio, **no nat**) | **0.608** | **0.967** |
| `sig_runB_scene4stream` (frozen) | 0.502 | 0.502 |
| `sig_runA_matched3stream` (frozen) | 0.467 | 0.541 |
| *old EmbeddingGemma arm B (base+nat)* | *0.609* | — |
| *old EmbeddingGemma arm C (base only)* | *0.562* | — |

**SUPERSEDES the 2026-08-14 finding that "dropping nat costs 4.7 points of audio-following."**
That was measured in EmbeddingGemma geometry. In the new space, run D reaches **0.608 with
base audio only** — matching old arm B (0.609, which needed nat) and beating old arm C
(0.562). **Dropping nat now costs nothing measurable**, retiring the 469 ms nat forward on
the Jetson for free.

Both frozen runs sit at matched_control ≈ 0.50 (chance for a 2-way choice) — they could not
reliably tell a clip's own sound caption from another's at all. Run D is 0.967. Independent
confirmation the projection was load-bearing.

### A.6 — Artifacts and code
* `models/text_target.py::SigLIP2TextTarget` — frozen SigLIP2 base, optional trainable proj
  via `shared_dim` (`None` = the falsified Identity config, kept only to reproduce it).
  `encode_text_frozen_raw` = raw (QUERY side); `encode_text` = through proj (TARGET side).
* `models/text_target.py::PreEncodedTextSpace` — device-side text-encoder stand-in, pure
  lookup, no weights. Word-overlap fallback on miss, recorded in `last_fallbacks`;
  `strict=True` refuses. Measured: *"what do you hear right now?"* → *"What do you hear?"*
  (0.429), gibberish → 0.000.
* `scripts/encode_captions_siglip2.py` — 1,361,635 unique captions pre-encoded in **36 s** on
  4 GPUs. Stores **RAW** vectors so one cache stays valid across every `--siglip-shared-dim`
  and every retrain.
* `scripts/build_bank_siglip2.py --ckpt` — applies the trained proj **offline**; this is what
  keeps the text tower off the device despite a learned target space.
* `scripts/build_candidate_vocab.py` — 1,372 tags (187 hand-curated home/room + corpus-mined)
  = **2.04 MiB**; plus 30 query phrasings = **0.05 MiB**.
* `scripts/eval_candidate_sets.py`, `scripts/eval_av_congruence.py` (now takes `--arms` and
  `--text-backbone`, reads geometry off the checkpoint).
* `train_query_predictor_ddp.py` — gained the scene stream, `--restrict-to-scene`,
  `--audio-mode`, `--text-cache-dir`, `--siglip-shared-dim`; `shared_dim` now comes from the
  target, not a hardcoded 1536.
* `load_perception_query_engine` reads `shared_dim`/`query_dim` off the checkpoint and raises
  on encoder mismatch.

### A.7 — Process notes
* A run was **stopped deliberately** at ~step 900/3000 (`sig_arm{B,C}_restrictedpool_stopped`)
  once it was clear its Action100M pool was 20% of the corpus. Stopping + re-running cost
  ~4 h against ~7.5 h for finishing a result that would be superseded.
* An earlier comparison at equal STEP COUNT was **unfair and corrected**: reference `steps`
  was 1500 vs 3000, and cosine `T_max = steps`, so at step 749 the reference was at 50% of
  its anneal (LR 1.50e-4) and the new run at 25% (LR 2.56e-4). Compare at matched schedule
  FRACTION, not step number.
* GradCache makes the micro/world split irrelevant (verified rel-L2 1.365e-07), so 64x8x2 and
  64x4x4 are the same 1024-clip batch with the same 2048/6144 negative pool.
