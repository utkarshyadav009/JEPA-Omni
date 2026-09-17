# ERRATA — PROPOSED, NOT APPLIED

Every entry below is a **proposal**. Nothing in `docs/BASELINE_1545.md`,
`docs/EVIDENCE_LEDGER.md`, `docs/EVIDENCE_LEDGER_V2.md`, `docs/ICLR_RESULTS.md`,
`docs/METHODOLOGY_FORENSICS.md` or `checkpoints/RESULTS_TABLE.md` has been edited.
A human approves each entry individually before any published table changes.

Raised: 2026-09-11. Checkpoint under test throughout:
`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`,
sha256 `e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8`
(verified before every run listed here; the scripts abort on mismatch).

---

## E-1 — VGGSound 1,545-gallery R@1 is inflated ≈23 points by a padding leak

**SEVERITY: HIGH. This is the headline retrieval number.**

| | a→v R@1 | a→v R@5 | a→v R@10 | v→a R@1 | v→a R@5 | v→a R@10 |
|---|---|---|---|---|---|---|
| **published** | 53.27 | 81.62 | 88.67 | 53.72 | 80.32 | 88.09 |
| **proposed (leak-free)** | **28.28** | **56.25** | **68.16** | **29.90** | **56.83** | **68.67** |
| delta | −24.99 | −25.37 | −20.51 | −23.82 | −23.49 | −19.42 |

Published source: `logs/m2_run2_final.log:1325-1334` (in-training eval at step 19000).

### Cause

`data/av_cached_dataset.py:av_collate_fn` (lines 226-264) pads the ambient stream to
the longest clip **in the batch**, fills the pad slots with zeros, and returns a
`padding_mask` naming them. No consumer in the M2 eval path ever passed that mask on:

* `models/av_jepa_predictor.py:_backbone` accepts a `key_padding_mask`; every caller
  passed `None`, so pad tokens were attended to in all 8 layers. After `in_proj` a
  zero-feature pad token is **not** a zero vector — it carries the layer bias, the
  modality embedding and `temporal_emb[0]` — so it is live content to attention.
* `train_m2.py:pool_and_project` (lines 417-419) does `.mean(1)` over the **padded**
  length, so a clip batched with a longer neighbour is divided by a larger denominator.

A clip's pad count is `max_T_a(batch) − T_a(clip)`. It is therefore a **per-clip
quantity injected identically into that clip's vision embedding and its ambient
embedding** — a shared nuisance variable that retrieval can match on instead of
audio-visual correspondence.

### Evidence

1. **Batch size 1 (no padding can exist; the two code paths are mathematically
   identical) gives ≈30, not ≈53** — and the two paths agree with each other:
   `diag_bs1_leak.json` v→a 30.16 / a→v 29.13; `diag_bs1_fix.json` 30.03 / 29.26.
2. **The padding-corrected path at batch 64 reproduces batch-size-1**: 29.90 / 28.28
   (`p02_cell4_fix_det.json`), and is essentially seed-invariant (std ≤ 0.04).
3. **The leak is not a duration fingerprint.** The gallery has only 53 distinct `T_a`
   values and 2.3% of clips have a unique one (`duration_only_control.json`) — far too
   coarse to drive 53% R@1.
4. **The leak is not an attention-sink / register effect.** Appending a *uniform* pad
   count to every clip (no clip-specific information) monotonically **hurts**:
   P=0 → 30.16, P=10 → 23.43, P=50 → 20.91, P=100 → 17.48, P=200 → 13.72, P=400 → 10.74
   (`pad_sweep.json`).
5. **DECISIVE — a pure-noise shared ID reproduces the gain.** Giving each clip a
   *random* pad count in [0,10), drawn once per clip so it is identical for that clip's
   vision pass and its ambient pass but unrelated to anything about the clip, lifts R@1
   from 30.16/29.13 to **50.36/48.54** (`pad_sweep_random.json`). The real gallery's
   mean pad is ≈8 tokens per clip (0.83% of ~990, `ambient_padding_stats.json`) — the
   same regime. Larger random bounds lose the gain because representation damage
   (evidence 4) overtakes the ID signal: [0,50) → 31.52, [0,100) → 30.61, [0,200) → 31.59.

### Corroboration

The Ego4D held-out number does **not** go through this path (see E-4) and is unaffected
at 27.60/27.00. Under the published VGGSound figure the VGGSound↔Ego4D gap was ~26
points and unexplained; under the corrected figure it is ~2 points. The corrected
number is the one consistent with the rest of the repository.

### Commands

```
# leak-free, deterministic batch order
python scripts/temporal_probe/bin_scramble_eval.py --arms A0 --no-world-state \
  --fix-padding --out docs/artifacts/temporal_probe/p02_cell4_fix_det.json
# batch-size-1 reference (no padding exists)
python scripts/temporal_probe/bin_scramble_eval.py --arms A0 --no-world-state \
  --batch-size 1 --out docs/artifacts/temporal_probe/diag_bs1_leak.json
# the decisive control
python scripts/temporal_probe/pad_sweep.py --pads "" --rand-pads 10,50,100,200 \
  --out docs/artifacts/temporal_probe/pad_sweep_random.json
```

Gallery `data/vggsound_eval_1545.txt` (sha256 `89307c6d4104…`), n = 1,545, all runs
assert `dataset_len == clips_seen == 1545`.

---

## E-2 — The published R@1 carries no error bar, and the run-to-run range is ≈2.5 points

The eval loader shuffles with the **unseeded global RNG**: `train_m2.py:352-356` sets
`shuffle=(sampler is None)`, and `scripts/eval_checkpoint_gallery.py` passes
`distributed_sampler=False`, so `sampler is None`. Batch composition therefore changes on
every invocation, and via E-1's mechanism so does every clip's embedding.

Five runs, identical checkpoint, nothing changed but batch order
(`noisefloor_batchorder_{0..4}.json`, also `p02_cell1_*`):

| | mean ± std | range |
|---|---|---|
| v→a R@1 | 53.01 ± 1.11 | 2.66 |
| a→v R@1 | 52.91 ± 0.85 | 2.26 |

`docs/BASELINE_1545.md:62` and `docs/EVIDENCE_LEDGER.md:39` report these as bare point
estimates. **Proposed:** if E-1 is not adopted, the figure must at minimum carry ±1.1.
If E-1 *is* adopted the issue disappears — the corrected path is seed-invariant
(std ≤ 0.04, range ≤ 0.06).

---

## E-3 — `docs/METHODOLOGY_FORENSICS.md` RUN-2 row: InfoNCE temperature is absent, not 0.03

Row 16 of the chronological table (line 51) is widely read as giving the contrastive
temperature as 0.03. The `0.03` sits in the **SIGReg λ** column (header, line 34); the
temperature is not recorded in that table at all. The actual RUN-2 values are recorded
verbatim in `logs/m2_run2_final.log:42`:

```
lam_sigreg=0.03 lam_pred=0.0 lam_pooled=0.0 lam_contrastive=1.0
contrast_dim=256 contrast_temp=0.05 ... negatives=200x200
```

**Proposed:** add a `contrast_temp` column, value **0.05** for RUN-2. No published metric
changes. (`docs/EVIDENCE_LEDGER.md:210`'s 0.05 is correct but describes the superseded
`m2_fusion_20k_best` checkpoint, not RUN-2.)

---

## E-4 — State explicitly which numbers are NOT affected

Not an error; a missing statement that E-1 makes necessary.

**Unaffected — does not use `av_collate_fn`, so no padding exists:**
`scripts/phase_ego4d_heldout_gallery_score_v2.py:195-207` decodes one window at a time
and calls `world_state_builder.build_world_state_features` at batch size 1. The Ego4D
674-window sibling-excluded figures (27.60 / 27.00) stand as published.

**Affected — same unmasked path, needs re-measurement:** `scripts/eval_fresh_holdout.py`
(imports `contrastive_retrieval_eval` directly), `scripts/m2_modality_dropout_eval.py`,
`scripts/m2_effective_rank_large_n.py` (both call `encode_world_state` on collated
batches with the mask dropped).

**Partially affected — attention leak only, pooling leak absent:** the query-predictor
and M3/M4 connector paths. `train_query_predictor.py:build_sources` correctly builds and
forwards a padding mask to `QueryPredictor`, and `train_m3.py:342` / `train_m4.py:153` do
the same for the connector — but the M2 backbone call that produces their input
(`encode_pre_pool_tokens`) was itself unmasked, so the real tokens were already
contaminated before the downstream mask was applied. **This is training code and was not
touched.** Quantifying it requires a separate decision.

**Baseline rows:** each baseline ran its own inference script
(`scripts/{cavmae,audioclip,equiav,avsiam,...}_retrieval.py`) and does not import
`av_collate_fn`. Confirmation by reading each file is pending (P0.4).

---

## E-5 — Gallery-contamination gap: survives the fix, but is ≈40% smaller and ≈20× more precise

**The paper's headline methodological claim. It holds, at a reduced magnitude.**

Both galleries n = 1,545, same checkpoint (sha256 `e1a8231e…`), same script, deterministic
batch order, 5 batch-order seeds per cell for the uncertainty estimate.
`data/vggsound_eval_1545.txt` (0/1,545 training overlap) vs
`data/vggsound_eval_1545_balanced.txt` (1,404/1,545 = 90.9% overlap, md5 `2eeaceef…`,
sha256 `93dbf962…`). The two galleries share 141 clips with each other.

### R@1, balanced − held-out

| path | v→a Δ | a→v Δ |
|---|---|---|
| **leaked (as published)** | +15.63 ± 1.57 | +16.80 ± 1.38 |
| **padding fixed** | **+9.36 ± 0.07** | **+9.97 ± 0.00** |
| attributable to the leak | +6.27 | +6.83 |

Published claim was a "≥15.4 point lower bound" (68.61 vs 53.14). The leaked path
reproduces it closely here (68.64 ± 1.10 vs 53.01 ± 1.11), confirming the original
protocol was faithfully re-executed before being corrected.

### Full table, padding fixed

| gallery | v→a R@1 | v→a R@5 | v→a R@10 | a→v R@1 | a→v R@5 | a→v R@10 |
|---|---|---|---|---|---|---|
| held-out | 29.93 ± 0.04 | 56.87 ± 0.03 | 68.67 ± 0.05 | 28.28 ± 0.00 | 56.25 ± 0.00 | 68.16 ± 0.00 |
| balanced (contaminated) | 39.29 ± 0.07 | 75.29 ± 0.03 | 85.98 ± 0.04 | 38.25 ± 0.00 | 73.20 ± 0.00 | 85.11 ± 0.00 |
| **Δ** | **+9.36 ± 0.08** | **+18.43 ± 0.04** | **+17.31 ± 0.06** | **+9.97 ± 0.00** | **+16.95 ± 0.00** | **+16.95 ± 0.00** |

Two observations that cut in opposite directions and should both be reported:

* At **R@1** the fix **shrinks** the gap (+15.6 → +9.4): roughly 40% of the published
  contamination effect was the padding leak, not memorisation.
* At **R@5/R@10** the fix **enlarges** it (+14.6 → +18.4, +10.9 → +17.3). Under the leaked
  path the contaminated gallery was near ceiling (94.85 / 97.86), which compressed the
  gap; removing the leak un-saturates it.

The uncertainty on the delta falls from ±1.57 to ±0.07 — the corrected measurement is
about twenty times more precise, because the corrected path is nearly seed-invariant.

**Proposed:** `docs/GALLERY_CONTAMINATION.md`'s "≥15.4 point lower bound" becomes
**≥9.4 points (v→a) / ≥10.0 points (a→v)**, with the R@5/R@10 figures reported alongside
so the effect is not understated. Not applied.

### Command

```
python scripts/temporal_probe/bin_scramble_eval.py --arms A0 --no-world-state \
  --fix-padding --eval-subset data/vggsound_eval_1545_balanced.txt \
  --batch-order-seed <0..4> --out docs/artifacts/temporal_probe/p03a_bal_fix_s<seed>.json
```
Raw: `docs/artifacts/temporal_probe/p03a_*.json` (24 runs),
summary `docs/artifacts/temporal_probe/p03a_contamination_summary.json`.

---

## E-6 — Downstream query-predictor numbers are CLEAN (no change proposed)

Recorded so the audit is complete and so nobody re-opens this.

Batch size 1 removes padding entirely, so it is the decisive test. Identical clip set,
order, query phrasings (same seed) and checkpoint; only the batch size differs.

| arm | metric | logged | bs=48 | bs=1 | Δ |
|---|---|---|---|---|---|
| `sig_runD_proj768` | `within_clip_acc` | 0.8114 | 0.8181 | 0.8189 | +0.0008 |
| | `swapped_query_acc` | 0.0056 | 0.004541 | 0.004274 | −0.0003 |
| | `cross_clip_r1` | 0.7372 | 0.73558 | 0.73397 | −0.0016 |
| `sig_runA_matched3stream` | `within_clip_acc` | 0.6541 | 0.6546 | 0.6546 | **0.0000** |
| | `swapped_query_acc` | 0.005876 | 0.005876 | 0.005876 | **0.0000** |
| | `cross_clip_r1` | 0.4888 | 0.48878 | 0.50321 | +0.0144 |

Why the query-predictor path is structurally immune to E-1's mechanism: one side of the
retrieval is TEXT, which has no access to a clip's pad count, so no shared nuisance
variable can exist; `QueryPredictor.forward` already receives and applies the mask
(`build_sources` → `qp(src, qe, msk)`); and it pools over 8 fixed LATENTS, not over
tokens. `within_clip_acc` / `swapped_query_acc` compare captions of the SAME clip, so any
per-clip nuisance cancels — which is why they come out bit-identical.

`sig_runD` was scored on a RE-EXTRACTED 900-clip scene subset (the original
`/dev/shm/scene_all` is gone), so its absolute value is a near-match, not a reproduction;
the bs48-vs-bs1 DELTA is exact, both sides using the identical subset.

**Ears-following (0.650 → 0.070) cannot be tested at batch size 1 at all**:
`scripts/eval_av_congruence.py` swaps audio WITHIN a batch (`perm = torch.roll(...)`) and
skips `B < 2`. The correct single-variable test — masking the one unmasked operation, the
m2 backbone pass — was run instead on identical batches with identical swap pairing:

| metric | published | reproduced | m2 masked | Δ |
|---|---|---|---|---|
| `audio_following_rate` (arm A) | 0.6500 | 0.6500 | 0.6531 | +0.0031 |
| `matched_control_acc` | 0.95625 | 0.95625 | 0.95469 | −0.0016 |

Artifacts: `p04_sig_runA.json`, `p04_sig_runD.json`, `p04_congruence_mask_ab.json`.

**LATENT BUG, not fixed (training code):** `train_query_predictor.build_sources` adds a
padding mask for `m2`/`vision`/`ambient` but NOT for `scene`, and `QueryPredictor.forward`
substitutes an all-False mask for any stream missing one. Inert today because the
extractor emits a fixed K=8 frames per clip (confirmed: all 900 re-extracted tensors are
(8,768), so `collate` never pads scene). It activates the moment K becomes variable.
Affects `sig_runB/C/D` and `abl_B/C/D`.

---

## E-7 — Training-side residual leak in `encode_pre_pool_tokens` (quantified, NOT fixed)

`AVJepaPredictor.encode_pre_pool_tokens` runs the backbone with `key_padding_mask=None`,
so the M2 tokens consumed by the query predictor and the M3/M4 connectors during TRAINING
were computed with pad tokens participating in all 8 attention layers. The downstream
modules mask the pad POSITIONS but cannot undo their influence on the surviving ones.

Measured on the 1,545 gallery at batch 64 (the QP's micro-batch),
`docs/artifacts/temporal_probe/p03d_pre_pool_leak.json`:

| quantity | value |
|---|---|
| padding as a fraction of slots | 0.54% |
| relative L2 change to REAL tokens | **0.515** |
| cosine between real tokens with/without padding | 0.835 |
| pad tokens' post-backbone norm ÷ real tokens' | **0.9997** |
| relative L2 vs the batch-1 (no-padding) reference | 0.538 |

The last row is the explanation: a zero-FEATURE pad token is not a zero VECTOR. After
`in_proj` it carries the layer bias, the modality embedding and `temporal_emb[0]`, and
leaves the backbone with 99.97% of a real token's norm. Half a percent of slots displaces
the real tokens by half their own magnitude.

Downstream METRICS moved ≤0.02 (E-6) because those modules trained around it. This is a
defect in the trained representation, not in the reported numbers. **Proposed: fix in
RUN-4's training path, not here.**

---

## E-8 — Ego4D held-out evaluation is currently UNREPRODUCIBLE (data loss, not an error)

`scripts/phase_ego4d_heldout_gallery_score_v2.py` decodes raw video from
`/mnt/Raid-Storage-2/utkarsh-data/ego4d_probe/v2/clips/`. That directory now holds **0
mp4 files**; `find` across both Ego4D trees returns zero video files (91 MB of metadata
and 6.1 GB of annotations survive).

The held-out windows are not recoverable from the surviving 518 GB feature cache either:
all 674 were checked against the 134,491 cached windows — **0 present**, and file overlap
is **0 of 350**. That independently confirms the split's `file_disjoint_verified: true`
was honest, and simultaneously means the held-out set exists only as video that is gone.

RUN-2's Ego4D figures (v→a/a→v R@1 27.60 / 27.00) **stand as published** and are
unaffected by E-1 (that harness runs at batch size 1, so no padding exists) — but they
**cannot be re-verified, and no future checkpoint can be compared against them** without
re-downloading Ego4D. Any RUN-4 spec that evaluates on the 674 windows must be revised.

---

## E-9 — Sub-10-second temporal work is not possible from existing data

Ego4D window ORDER is recoverable: cache ids are `ego4d_<uuid>_w<index>`, 59.1% of
adjacent index pairs are consecutive, 1,042 of 2,821 files have a run of ≥10 consecutive
windows and 357 have ≥30 (`docs/artifacts/temporal_probe/p07_ego4d_ordering.json`).

But the stride is **10 s, non-overlapping**: every `start_sec` in
`EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json` is a multiple of 10, and window index × 10 =
start second. So the finest measurable Δ is 10 s. Phase 2's Δ ∈ {1,2,5} s and any RUN-5
temporal-prediction arm at those horizons are **unobtainable** from cached features, and
re-extraction at a finer stride needs the raw video, which is gone (E-8).

---

## E-10 — A previously recorded explanation for the reproduction gap is wrong

`docs/METHODOLOGY_FORENSICS.md:74-79` records a 2026-08-23 reproduction of
`eval_checkpoint_gallery.py` on `step19000.pt` giving a→v 53.59/81.10/88.03 and v→a
52.75/80.00/87.12, and attributes the ~0.5–1pp gap to:

> "bf16 autocast non-determinism plus the fact that the training-time
> `/dev/shm/jepa_m2_cache` is gone and the re-run used the on-disk
> `feature_cache_vgg51k` (a separate extraction pass under the same manifest)."

**That explanation is falsified.** The cause is the unseeded `shuffle=True` on the eval
loader (`train_m2.py:352-356`, reached because `eval_checkpoint_gallery.py` passes
`distributed_sampler=False`) interacting with unmasked ambient padding — E-1 and E-2.
Holding batch order fixed makes the eval bit-reproducible across runs (world-state tensor
sha256 identical); it is not autocast noise. The cache swap is also not the cause: the
2026-08-23 values sit inside the batch-order distribution measured here
(v→a 53.01 ± 1.11, a→v 52.91 ± 0.85, range ≈2.5).

**Proposed:** replace that paragraph with a pointer to E-1/E-2. The 2026-08-23 numbers
themselves need no correction — only the stated cause. Not applied.

---

## E-11 — All eight baselines are CLEAN (no change proposed), and the head-to-head is a TIE

Batch-size-1 test on a fixed 300-clip subset (`baseline_bs1_subset_300.txt`, sha256
`4efcb0b9…`), except AudioCLIP which hardcodes `assert len(clip_ids) == 1545` and so was
run on the full gallery at batch 16 vs 1.

| model | batch | a→v R@1 | v→a R@1 | Δ at bs=1 | verdict |
|---|---|---|---|---|---|
| ImageBind | 8 → 1 | 50.33 → 50.33 | 52.33 → 52.33 | **0.00 on all six** | immune |
| AVSIAM | 32 → 1 | 6.67 → 6.67 | 8.00 → 8.00 | **0.00 on all six** | immune |
| EquiAV | 32 → 1 | 42.67 → 42.00 | 39.67 → 39.67 | ±0.67 = 2/300 clips, both directions | immune |
| AudioCLIP (n=1545) | 16 → 1 | 0.19 → 0.19 | 0.91 → 0.91 | **0.00 on all six** | immune |

AudioCLIP's run reproduces `docs/BASELINE_1545.md`'s published row **exactly**
(0.19/0.58/1.17, 0.91/2.72/3.95), confirming the protocol was re-executed faithfully.

Structural reason, verified by reading each script: every baseline pads to a **fixed,
config-level** length — AudioCLIP 220,500 samples, ImageBind 204 frames, CAV-MAE /
CAV-MAE-Sync / AVSIAM 1024, LanguageBind 1036, EquiAV per-clip — then `torch.stack`s
equal-shaped tensors. The pad length is a constant of the model config, never of the
batch. This is the opposite of `av_collate_fn`, which pads to the longest clip IN THE
BATCH. None imports `av_collate_fn`.

**Also immune: the AudioSet A-column.** `scripts/audioset_extract_features.py:43-45` pads
or crops every waveform to a fixed `NSAMP = 160000` before batching, so every clip in a
batch has identical length. No `av_collate_fn`, no per-clip nuisance. The AudioSet
mAP / mAUC / d-prime figures stand unchanged.

### The head-to-head, on the same gallery

E-1 lowers our VGGSound figure, so the comparison must be restated — **on the 1,545-clip
gallery, against the published baseline rows**:

| system | a→v R@1 | v→a R@1 |
|---|---|---|
| **Ours, RUN-2 corrected** | **28.28** | **29.90** |
| ImageBind (1200.8M) | 29.45 | 29.64 |
| EquiAV (211.8M) | 24.66 | 21.81 |
| AVSiam Base (333.2M) | 2.59 | 3.04 |
| AudioCLIP (134.1M) | 0.19 | 0.91 |

We are **within ~1 point of ImageBind in both directions** — ahead by 0.26 on v→a, behind
by 1.17 on a→v. The honest summary is **a tie with the strongest baseline**, at 155.9M
trainable parameters against ImageBind's 1200.8M, and still clearly ahead of EquiAV.

**Proposed:** restate the comparison in `docs/BASELINE_1545.md` as a tie with ImageBind
once E-1 is adopted. Not applied.

---

## E-12 — `docs/ICLR_RESULTS.md` §2 cites source files that no longer contain its numbers

§2's baseline table (lines 116-123) reports the **old 1532/1545** measurement while citing
per-model JSONs that were **re-measured to the full 1545** on 2026-09-09
(`docs/BASELINE_1545.md:139`, "all 13 missing videos recovered"). Seven of eight rows are
therefore stale relative to the file they name. Read from the JSONs, 2026-09-11:

| model | source JSON (canonical, n=1545) | ICLR §2 (stale) | match |
|---|---|---|---|
| ImageBind | **29.45 / 29.64** | 29.70 / 29.70 | ✗ |
| EquiAV | **24.66 / 21.81** | 24.80 / 21.80 | ✗ |
| CAV-MAE | 12.23 / 14.24 | 12.23 / 14.24 | ✓ (already 1545) |
| LanguageBind | **7.64 / 10.29** | 7.57 / 10.31 | ✗ |
| Wav2CLIP | **5.31 / 6.47** | 5.35 / 6.46 | ✗ |
| AVSiam | **2.59 / 3.04** | 2.48 / 3.00 | ✗ |
| CAV-MAE Sync | **2.01 / 4.92** | 2.02 / 4.90 | ✗ |
| AudioCLIP | **0.19 / 0.91** | 0.20 / 0.91 | ✗ |

Differences are small (≤0.25) and no conclusion changes, but the ImageBind figure quoted in
the abstract should be **29.45 / 29.64**. **Proposed:** refresh §2 from the JSONs and state
the gallery as 1545/1545. Raw: `p14_baseline_reconciliation.json`. Not applied.

---

## E-13 — A cleaner contamination measurement: 0% vs 100%, matched size and class structure

E-5 measured contamination as clean-vs-*balanced* (0% vs 90.9% contaminated), and the
balanced gallery differs from the clean one in construction as well as contamination. A
better-controlled pair is now available and **corroborates E-5 closely**.

Accounting, read from disk (`p13_testsplit_accounting.json`):

```
official VGGSound test split         15,446
  ... in the RUN-2 training corpus   13,894   (90.0%)
  ... in the 1,545 eval list          1,545   (100% of the eval list)
  ... never extracted at all              7
```

**Two facts worth recording in their own right.** First, **`data/vggsound_eval_1545.txt` is
itself a subset of the official test split** — all 1,545 of its clips are in `data/test.csv`.
Our held-out gallery is therefore not an arbitrary draw: it is (1,545 of) the 1,552
official-test clips that were never trained on. Second, and consequently, **a genuinely
clean gallery larger than what we already have cannot be built from the official test
split** — only 7 further clips qualify. RUN-5's exclusion is the only route to a bigger one.

New gallery: `data/vggsound_testsplit_contaminated_1545.txt`, md5
`dbd405117674715acda08dd08a65f3c2`, sha256 `4b7bd2ba81e6636f…`. Built by sampling 1,545
(seed 0) from the 13,894 official-test clips that ARE in the RUN-2 training corpus. All
cached. **Overlap with the training corpus 1545/1545 = 100.0%. Overlap with the clean
gallery: 0.** Class structure is closely matched: 305 vs 307 classes, 5.07 vs 5.03
clips/class, both from the same source split.

| gallery | contamination | v→a R@1 | v→a R@5 | v→a R@10 | a→v R@1 | a→v R@5 | a→v R@10 |
|---|---|---|---|---|---|---|---|
| `vggsound_eval_1545` | **0%** | 29.90 | 56.83 | 68.67 | 28.28 | 56.25 | 68.16 |
| `vggsound_testsplit_contaminated_1545` | **100%** | 39.48 | 74.63 | 86.28 | 37.54 | 73.85 | 84.98 |
| **Δ** | | **+9.58** | **+17.80** | **+17.61** | **+9.26** | **+17.60** | **+16.82** |

Corrected path, deterministic batch order, checkpoint sha256 `e1a8231e…`, n=1,545 both.

Against E-5's balanced-gallery delta of **+9.36 / +9.97** at R@1 and +18.43 / +16.95 at
R@5, two independently constructed contaminated galleries agree to within ~0.7 points in
every column. **The contamination effect is +9.3 to +10.0 R@1 and +16.8 to +18.4 R@5.**

**Proposed:** cite E-13 as the primary contamination result (it is the better-controlled
pair) with E-5 as independent corroboration. Not applied.

---

## E-14 — The `best.pt` checkpoint-selection bug was never fixed in code

Commit `0eb3337` ("Correct RUN-2 result: best.pt was mislabeled") is widely referenced as
having fixed checkpoint selection. **It changed four files and none of them is
`train_m2.py`**: `checkpoints/falsifier_tracking.md`,
`presentation/M2_M5_Supervisor_Update.md`, and two PNGs. It corrected the *claim* and
re-scored the checkpoints; it did not touch the selection logic.

`train_m2.py:1344` still reads:

```python
if loss_ema < best_loss:
    best_loss = loss_ema
    save_checkpoint(os.path.join(ckpt_dir, "best.pt"), ...)
```

`git log -L 1344,1345:train_m2.py` shows the line unchanged since `ab7bae5` ("M2 work
started"). **`best.pt` is still selected on training `loss_ema`, never on held-out R@1** —
exactly the defect that made RUN-2's `best.pt` (step 13,960) worse than `step19000.pt` on
every measured metric.

No published number is affected: RUN-2's reported results come from `step19000.pt`, chosen
correctly after the fact. But **any future run inherits the bug**, and the next time there
is no plateau to notice, it may go uncaught.

**Proposed, for RUN-4:** either (a) launch unchanged and select post hoc from the
`stepN000.pt` checkpoints by held-out R@1, ignoring `best.pt` — zero code change, preserves
RUN-4's single-variable design; or (b) fix the selection criterion, which adds a second
change to a run scoped as padding-fix-only. (a) is recommended. Not applied either way.

---

## E-15 — `docs/ICLR_RESULTS.md` §15, the abstract crib, is stale in 6 of its rows

**SEVERITY: HIGH — this section is titled "the numbers most likely to go in the abstract",
so it is the most likely route for a retracted number to reach a submission.**

`docs/ICLR_RESULTS.md` was last written 2026-09-09 (`4a83912`). It therefore predates the
padding-leak discovery (E-1), the contamination re-measurement (E-13), the baseline
reconciliation (E-12), RUN-4 in its entirety, the naming retirement, and RUN-5. E-12 already
covers §2; **§15 has never been covered by any errata entry**, and it is the section a paper
draft would be written from.

| § 15 row | published | status | corrected source |
|---|---|---|---|
| M2 AV congruency R@1 | a→v **53.27** / v→a **53.72** | **E-1 — padding leak, ≈24 pts** | RUN-2 corrected **28.28 / 29.90**; the system result is now RUN-4 `step18000` **41.88 (a→v) / 41.77 (v→a)** (`CANONICAL_NUMBERS.md` §1.1) |
| vs chance | **823×** | **arithmetic on the leaked number** | chance = 0.0647 %; corrected RUN-2 **437×**, RUN-4 **647× (a→v) / 645× (v→a)** |
| best same-gallery baseline | ImageBind **29.70** @ n=1532 | **E-12 — stale, and n differs** | **29.45** a→v @ **n=1545** (`data/imagebind_retrieval_results.json`) |
| gallery contamination | **≥15.4 pts**, "state as a LOWER BOUND, never an estimate" | **E-13 — superseded, and the framing now misleads** | **+9.58 / +9.26 R@1** and **+17.80 / +17.60 R@5**. Under the leak the contaminated gallery was near ceiling (94.85/97.86), which compressed R@1. Reporting R@1 alone inverts the finding — see `PAPER_ASSETS.md` asset 2 |
| corpus scale, matched steps | 33.46 → **44.27** @ 6,000 steps | **in-training evals on the leaked path** — not re-measured | do not quote until re-measured through the corrected harness |
| Ego4D batch share | 18.40 → 11.57 → 27.60 | same leaked path for the VGGSound column | the 27.60 / 27.00 Ego4D figure itself stands, but see E-8 |
| Ego4D transfer | 27.60 / 27.00 | **numerically stands**; E-8 caveat missing | **PERMANENTLY UNREPRODUCIBLE.** If the caveat cannot travel in the table, cut the row |
| "World-State" (§15 AudioSet row, and throughout) | — | **name retired** | "audio-visual scene representation" (`FORWARD_INFORMATION_PROBE.md`) |

**Rows that survive unchanged:** the 90.0 % test-split overlap (re-derived, `CANONICAL_NUMBERS.md`
§3.1); the SigLIP2 / WavJEPA-nat / ears-following / query-predictor ablations (E-6 confirmed
clean, batch-size-1 re-scored); the AudioSet mAP values; the latency progression.

### Second finding: there is no abstract document

`find -iname "*abstract*"` returns nothing. §15 is a *crib for* an abstract; no abstract has been
written. So there is no drafted abstract carrying these numbers — the exposure is prospective,
not already-published.

**Proposed:** do not edit §15 in place. Supersede it. `docs/CANONICAL_NUMBERS.md` (single source,
with provenance per row) and `docs/PAPER_ASSETS.md` (per-claim inventory with the caveat that must
travel) were written for exactly this purpose and are current. Add a banner at the head of
`ICLR_RESULTS.md` pointing there, and draft any abstract from those two files.

**Not applied.**
