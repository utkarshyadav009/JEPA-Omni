# RUN-4 — padding-fix-only retrain: implementation spec

**Status: PREPARED, NOT APPLIED.** No training code has been modified. Two things gate
the launch — a GPU-memory blocker (§4) and one open decision (§5).

Single variable vs RUN-2 = the padding defect (`docs/ERRATA_PROPOSED.md` E-1/E-7).
Everything else byte-identical: 197,462 VGGSound + 134,491 Ego4D, 40.5% Ego4D batch share,
negatives 200×200, `contrast_temp` 0.05, `lam_sigreg` 0.03, `lam_pred` 0.0,
`lam_contrastive` 1.0, `lam_fusion` 1.0, 4-GPU DDP, 20,000 steps, same schedule and seed.

---

## 1. The three code changes

### 1.1 Mask padding in the training path

`models/av_jepa_predictor.py` already carries the optional `key_padding_mask` argument on
`encode_world_state`, `encode_pre_pool_tokens` and `encode_source_tokens` (added during the
audit, verified a bit-exact no-op when not passed). RUN-4 additionally needs it on
`forward()` and `world_state()`, and the training loop must pass it at every call site:

| `train_m2.py` line | call | why it matters |
|---|---|---|
| 1121 | `model(feats, tbins, mask)` — prediction loss | inert at `lam_pred=0.0`, but the mask keeps the logged `pred=` diagnostic honest |
| 1129 | `raw.world_state(feats, tbins)` — SIGReg | SIGReg currently shapes a world-state computed over pad tokens |
| 1142 / 1157 | `raw.encode_source_tokens(feats, tbins)` | feeds both the contrastive head and the fusion bridge |
| 1227 | `pool_and_project(...)` | **the pooling leak** — `.mean(1)` over the padded length |
| 608 / 625 / 686 | GradCache + eval paths | same two leaks |

Pooling must use a masked mean (`scripts/temporal_probe/padded_eval.py:masked_mean`, already
written and used throughout the audit).

### 1.2 Fixed `T_a` — and a prerequisite bug that must be fixed first

**`train_m2.py:233 _cap_ambient_len` truncates `feats["ambient"]` and `tbins["ambient"]` but
NOT `padding_mask["ambient"]`.** (`train_m3.py:205`'s version does take and truncate the
mask; `train_m2`'s does not — it has no `pad_mask` parameter at all.)

This is inert today because nothing consumes the mask. **RUN-4 is precisely the change that
starts consuming it**, at which point the mask is longer than the tensor it describes —
a silent misalignment or a shape crash. This must be fixed in the same patch.

Measured `T_a` distribution (1,500 clips sampled from each corpus,
`docs/artifacts/temporal_probe/p13_fixed_Ta_analysis.json`):

| corpus | min | p1 | p50 | p99 | max | mean |
|---|---|---|---|---|---|---|
| VGGSound | 594 | 896 | 996 | 998 | 1042 | 993.2 |
| Ego4D | 990 | 996 | 996 | 996 | 999 | 996.0 |

| fixed `T_a` | % cropped | % padded | mean pad tokens |
|---|---|---|---|
| 896 | 99.0% | 0.4% | 0.1 |
| 960 | 98.6% | 1.4% | 0.8 |
| **992** | **98.1%** | **1.8%** | **1.4** |
| 1000 | 0.0% | 99.9% | 5.2 |
| 1024 (current `MAX_AMBIENT_T`) | 0.0% | 100.0% | 29.2 |

**Recommended: `T_a = 992`.** 98.1% of clips become an exact-length crop, only 1.8% need any
padding at all, and the mean pad is 1.4 tokens. Length then carries essentially no
information even before masking — which is the belt-and-braces the spec asks for. At the
current 1024 every clip is padded by ~29 tokens, which is the leak's raw material.

**Crop policy — recommend deterministic head-truncation**, i.e. lower `MAX_AMBIENT_T` from
1024 to 992 and keep the existing `[:max_t]` behaviour. Random-cropping is the alternative
named in the spec, but it introduces a per-step stochastic augmentation, which is a *second*
experimental variable in a run scoped to one. Cropping 996→992 discards 4 tokens ≈ 40 ms of
audio at the 100 Hz token rate.

### 1.3 The `scene` mask — belongs to a different run

E-6's latent bug (`build_sources` adds masks for `m2`/`vision`/`ambient` but not `scene`,
and `QueryPredictor.forward` substitutes an all-False mask for a missing stream) is in
**`train_query_predictor.py`, the query-predictor path**. RUN-4 is an M2 run and never
executes it. Fixing it here changes nothing about RUN-4 and cannot be validated by it.

**Recommendation: fix it in the query-predictor retrain that consumes RUN-4's checkpoint,
not in RUN-4.** It remains inert until K becomes variable (the extractor emits a fixed K=8;
all 900 re-extracted tensors are (8,768)).

## 2. Pre-flight checks

| check | status |
|---|---|
| checkpoint sha256 `e1a8231e…` | ✅ verified |
| `0eb3337` checkpoint-selection fix in effect | ❌ **it was never a code fix** — see E-14 and §5 |
| `_cap_ambient_len` truncates the pad mask | ❌ **it does not** — §1.2 |
| GPU memory headroom | ❌ **blocked** — §4 |

## 3. Evaluation (three galleries, corrected harness, deterministic order, 5 seeds)

| # | gallery | contamination | RUN-2 corrected reference |
|---|---|---|---|
| 1 | `data/vggsound_eval_1545.txt` | 0% | v→a **29.90** / a→v **28.28** |
| 2 | `data/vggsound_eval_1545_balanced.txt` | 90.9% | 39.29 / 38.25 |
| 3 | `data/vggsound_testsplit_contaminated_1545.txt` (md5 `dbd40511…`) | **100%** | 39.48 / 37.54 |

Gallery 3 is the substitute for the Ego4D axis, which is impossible (E-8: the 674 held-out
windows exist only as video that no longer exists on this machine, and 0 of them are in the
surviving feature cache). It is a **comparability** axis, not a clean one: RUN-4 keeps
RUN-2's corpus, so all 1,545 of its clips are training data for RUN-4 too. Galleries 2 and 3
together give two independent contamination deltas (E-5, E-13).

**The question RUN-4 answers:** how much did the training-time shortcut cost the model?
Any change on gallery 1 is attributable to the padding fix alone.

## 4. BLOCKER — RUN-4 cannot fit in current GPU memory

| | |
|---|---|
| RUN-2's documented envelope at batch 50/GPU | **94.9 GB of 95 GB** (`train_m2.py:242`) |
| free per card at time of writing | **71.8 GB** (another user holds 25.5 GB × 4, ~100% util) |

94.9 > 71.8: it will OOM. Reducing the batch is not an option — batch size sets the InfoNCE
negative pool and the 40.5% Ego4D batch share, so it would introduce a second variable and
defeat the design. **RUN-4 waits for the cards.**

## 5. OPEN DECISION — `best.pt` selection (E-14)

`train_m2.py:1344` still selects `best.pt` by training `loss_ema`, never by held-out R@1;
commit `0eb3337` never touched the file. Two options:

* **(a) Launch unchanged; select post hoc from the `stepN000.pt` checkpoints by held-out
  R@1 and ignore `best.pt`.** Zero code change, preserves the single-variable design. This
  is exactly how RUN-2's real best checkpoint was identified. **Recommended.**
* **(b) Fix the selection criterion.** Correct, but adds a second change to a run scoped as
  padding-fix-only.

## 6. Smoke check before the full run

6,000 steps. **Do not gate on the old 33.46/34.24 reference — it was leak-inflated.**
Confirm only: loss is not NaN, no collapse, and the trajectory is sane.
**Abort on:** loss NaN, R@1 < 5% at step 3,000, or `shuffle_sanity_gap` < 0.3.
Checkpoint selection during the smoke run is irrelevant under option (a).
