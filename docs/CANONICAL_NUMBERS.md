# CANONICAL_NUMBERS — the single source for every number quoted in the paper

**This file is the only place numbers should be quoted from.** If a figure appears in a
draft and not here, it is not canonical. Where a dispute is live it is marked **DISPUTED**
and both sides are given — nothing is silently resolved.

Last updated 2026-09-12. Corrections proposed against published tables live in
`docs/ERRATA_PROPOSED.md` (14 entries); **none has been applied to a published table.**

Locked checkpoint everywhere unless stated:
`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`,
sha256 `e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8`
(every script verifies this and aborts on mismatch).

Galleries:

| file | n | md5 / sha256 | contamination vs RUN-2 corpus |
|---|---|---|---|
| `data/vggsound_eval_1545.txt` | 1,545 | sha256 `89307c6d4104…` | **0%** |
| `data/vggsound_eval_1545_balanced.txt` | 1,545 | md5 `2eeaceef6866895cf02bb4204fb63835` | 90.9% |
| `data/vggsound_testsplit_contaminated_1545.txt` | 1,545 | md5 `dbd405117674715acda08dd08a65f3c2` | **100%** |

---

## 1. Our system — VGGSound 1,545 gallery

**Protocol:** padding-corrected eval (`--fix-padding`), deterministic batch order,
5 batch-order seeds, n = 1,545, full-gallery assertion `dataset_len == clips_seen == 1545`.

| direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| **v→a** | **29.93 ± 0.04** | 56.87 ± 0.03 | 68.67 ± 0.05 |
| **a→v** | **28.28 ± 0.00** | 56.25 ± 0.00 | 68.16 ± 0.00 |

Deterministic single run: v→a 29.90 / 56.83 / 68.67, a→v 28.28 / 56.25 / 68.16.

* Source: `docs/artifacts/temporal_probe/p02_summary.json`, `p02_cell{3,4}_fix_*.json`
* Command: `python scripts/temporal_probe/bin_scramble_eval.py --arms A0 --no-world-state --fix-padding --out <path>`
* **Supersedes the published 53.27/53.72** (ERRATA E-1). Those were inflated ≈24 points by a
  padding-derived shared-nuisance leak, established three ways: batch-size-1 agreement,
  a uniform-pad control that *hurts*, and a random-pad control that reproduces the gain.
* The published figure had **no error bar**; the leaked path's run-to-run range was ≈2.5
  points (E-2). The corrected path is nearly seed-invariant.

## 2. Baselines — same gallery, n = 1,545

Canonical values are the per-model JSONs, **not** `docs/ICLR_RESULTS.md` §2, which is stale
for 7 of 8 rows (E-12).

| model | params (M) | a→v R@1 | v→a R@1 | contamination |
|---|---|---|---|---|
| **ImageBind** | 1200.8 | **29.45** | **29.64** | held-out |
| **Ours (RUN-2 corrected)** | **155.9 trainable** | **28.28** | **29.90** | held-out |
| EquiAV | 211.8 | 24.66 | 21.81 | held-out |
| CAV-MAE | 190.7 | 12.23 | 14.24 | **in-distribution** |
| LanguageBind | 709.3 | 7.64 | 10.29 | held-out |
| Wav2CLIP | 163.0 | 5.31 | 6.47 | **DISPUTED** (in-distribution in the table, held-out in the JSON — §13.2) |
| AVSiam (Base) | 333.2 | 2.59 | 3.04 | held-out |
| CAV-MAE Sync | 190.7 | 2.01 | 4.92 | **in-distribution** |
| AudioCLIP | 134.1 | 0.19 | 0.91 | held-out |

* Source: `data/{model}_retrieval_results.json`; reconciliation `p14_baseline_reconciliation.json`
* All eight verified **immune** to the padding leak (E-11): each pads to a fixed
  config-level length then `torch.stack`s equal-shaped tensors, so batch-dependent padding
  is structurally impossible. Confirmed empirically at batch size 1 for ImageBind, AVSIAM
  and AudioCLIP (bit-identical) and EquiAV (±0.67 = 2/300 clips, both directions).

### 2.1 The head-to-head claim

> **A tie with ImageBind at roughly one-eighth the trainable parameters** (155.9M vs
> 1200.8M): ahead by 0.26 on v→a, behind by 1.17 on a→v. Clearly ahead of EquiAV.

**Do not write this as a win.** Under the pre-correction number it looked like one; it is not.

## 3. Gallery contamination

**E-13 is primary** (better controlled), **E-5 corroborates**. Both padding-corrected,
deterministic order, n = 1,545 per gallery.

| pair | contamination | v→a Δ R@1 | v→a Δ R@5 | a→v Δ R@1 | a→v Δ R@5 |
|---|---|---|---|---|---|
| **E-13** clean vs official-test-split contaminated | 0% vs **100%** | **+9.58** | **+17.80** | **+9.26** | **+17.60** |
| **E-5** clean vs balanced | 0% vs 90.9% | +9.36 ± 0.08 | +18.43 ± 0.04 | +9.97 ± 0.00 | +16.95 ± 0.00 |

**Headline: +9.3 to +10.0 R@1 and +16.8 to +18.4 R@5**, from two independently constructed
galleries agreeing to within ~0.7 points in every column.

* **Report R@1 AND R@5/R@10.** Against the published "≥15.4 point" claim, R@1 *shrinks* to
  ~9.4 while R@5 *grows* to ~18. "The effect shrank 40%" is a false summary — under the leak
  the contaminated gallery was near ceiling (94.85/97.86), which compressed the gap.
* E-13's pair is matched on source split, size, and class structure (305 vs 307 classes,
  5.07 vs 5.03 clips/class) with **zero overlap** between the two galleries.
* Source: `p03a_contamination_summary.json`, `p13_contaminated_gallery.json`

### 3.1 Two facts about the held-out gallery worth stating

* **`vggsound_eval_1545.txt` is itself a subset of the official VGGSound test split** — all
  1,545 clips appear in `data/test.csv`. It is not an arbitrary draw: it is 1,545 of the
  1,552 official-test clips never trained on.
* **A clean gallery larger than this cannot be built from that split** — only 7 further
  clips qualify. 13,894 of 15,446 (90.0%) are in the training corpus.

## 4. Ego4D transfer — **cannot be re-verified**

| direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| v→a | 27.60 | 58.01 | 73.59 |
| a→v | 27.00 | 58.16 | 74.04 |

674 sibling-excluded windows / 350 files, seed 43, `file_disjoint_verified: true`.

* **Unaffected by the padding leak** — that harness decodes one window at a time at batch
  size 1, so no padding exists (E-4).
* **PERMANENTLY UNREPRODUCIBLE (E-8).** The raw Ego4D video is gone from this machine
  (`find` returns 0 mp4 files), and all 674 held-out windows are absent from the surviving
  134,491-window cache (file overlap 0 of 350 — which independently confirms the split was
  honestly file-disjoint). No future checkpoint can be compared against these numbers.

## 5. Query predictor — clean, no correction needed

| arm | within-clip | swapped-query | cross-clip R@1 | n |
|---|---|---|---|---|
| `sig_runD_proj768` | 0.8114 | 0.0056 | **0.7372** | 624 |
| `sig_runA_matched3stream` | 0.6541 | 0.0059 | **0.4888** | 624 |

Batch-size-1 re-scoring moved these by ≤0.016; `within_clip` and `swapped_query` were
**bit-identical** for `sig_runA` (E-6). Structurally immune: one side of the retrieval is
text, which has no access to a clip's pad count.

* **Caveat on `sig_runD`:** its SigLIP2 scene features were lost with `/dev/shm`. The
  re-scoring used a **re-extracted 900-clip subset**, so the bs48-vs-bs1 *delta* is exact but
  the *absolute* value is a near-match (bs48 reproduction 0.8181/0.7356 vs logged
  0.8114/0.7372), not a reproduction.
* Ears-following **0.650 → 0.070** stands; reproduced exactly at 0.6500 and moved +0.0031
  when the one unmasked operation was masked (`p04_congruence_mask_ab.json`).

## 6. Temporal-structure probe — one line

> The M2 fused latent is an **audio-visual scene representation, not a world-state**: it has
> no recurrence, `lam_pred = 0.0` so no predictive term was ever trained, vision's temporal
> axis is provably unused (|Δ| ≤ 0.13 R@1, world-state cosine 0.99998), persistence is a
> plateau rather than a decay (0.835 at 20 s → 0.802 at 60 s, i.e. scene identity), and a
> learned forward map beats copying at R@1 (3.43 vs 0.43) but loses on cosine and R@5.

Full treatment, all three phases: `docs/TEMPORAL_STRUCTURE_PROBE.md`.

## 7. RUN-4 — COMPLETE. The mechanism is LENGTH NORMALISATION, not masking.

20,000 steps, `checkpoints/m2_run4_padfix_ta896/`, `RUN4_EXIT=0`, 20 evals, **zero NaN**,
full gallery asserted (`clips_seen=1545`). All 20 tagged checkpoints retained; per E-14 the
best is selected post hoc by held-out R@1, **never** from `best.pt`.

### 7.1 Matched-length grid (P2.3) — n=1,545, 5 seeds, one harness, both models both lengths

| model | `T_a` | v→a R@1 | v→a R@5 | v→a R@10 | a→v R@1 | a→v R@5 | a→v R@10 | gap | eff_rank |
|---|---|---|---|---|---|---|---|---|---|
| RUN-2 | **~996 (own)** | 29.84 | 56.88 | 68.57 | 28.28 | 56.25 | 68.09 | 0.4300 | 37.72 |
| RUN-2 | 896 | 22.21 | 48.40 | 59.57 | 19.42 | 45.05 | 53.79 | 0.4300 | 37.68 |
| RUN-4 | ~996 | 4.27 | 14.05 | 21.42 | 20.91 | 47.64 | 59.42 | 0.3600 | 73.12 |
| **RUN-4** | **896 (own)** | **41.35** | **72.50** | **80.46** | **41.68** | **72.36** | **80.45** | **0.6400** | **74.26** |

**Each model in its own training regime: RUN-4 is +11.5 (v→a) / +13.4 (a→v) R@1 over RUN-2**,
+15.6/+16.1 at R@5, +11.9/+12.4 at R@10.

**The `T_a` confound is controlled, not merely acknowledged.** RUN-2 evaluated at RUN-4's
length (896) scores **worse**, not better — 22.21/19.42 against its own 29.84/28.28. The
shorter window is a handicap, so RUN-4's gain cannot be attributed to 896 being an easier
setting. Neither single length is a fair comparison; the diagonal is the defensible one.

Source: `docs/artifacts/temporal_probe/p23_matched_length_grid.json`.

### 7.2 P2.2 control — what actually causes the gain

Identical config, 6,000 steps, `T_a=896`, evaluated at 896. **The only difference is the mask.**

| | a→v R@1 | v→a R@1 | matched cos | gap | eff_rank |
|---|---|---|---|---|---|
| control, **mask OFF** | 32.56 | 34.89 | 0.6732 | 0.6321 | 25.75 |
| RUN-4 smoke, **mask ON** | 33.01 | 33.01 | 0.6740 | 0.6324 | 26.13 |
| Δ | −0.45 | +1.88 | −0.0008 | −0.0003 | −0.38 |

**Identical within noise.** Masking padding adds essentially nothing once the ambient length
is fixed.

**Therefore the claim is "removing the length-derived shortcut from training", NOT "masking
padding in training."** This is consistent with the measured distribution: at `T_a=896`,
truncation alone leaves only 0.4% of clips padded at a mean of 0.1 tokens, so there is almost
nothing left for a mask to do.

**The effective-rank doubling tracks the same cause.** It rises in the control too (25.75 vs
26.13, both ~2× RUN-2's ~12.5), so it is shortcut removal rather than masking specifically.

### 7.3 How to state this

> Removing the length-derived shortcut from M2's training — by fixing the ambient sequence to
> a constant 896 tokens — raises held-out VGGSound retrieval from **28.28/29.90** to
> **41.68/41.35** R@1 (n=1,545, each model evaluated at its own training length), and roughly
> doubles the world-state's effective rank. A control run isolates the cause: masking the
> padding contributes nothing measurable once the length is fixed.

**Caveats that must travel with it:** RUN-2 and RUN-4 see different amounts of audio
(~996 vs 896 tokens, ≈1 s), which is why the control row and the P2.2 decomposition are part
of the result rather than an appendix. RUN-4 was still improving at step 20,000.

## 8. AVE external gallery — feasibility

4,097 distinct YouTube ids, 28 categories. Overlap with the VGGSound **training** corpus
**867 (21.2%)**; with our AudioSet mirror **0**. **3,230 (78.8%) survive** as genuinely
external, all 28 categories retained, median 125 clips/category. Source:
`p25_ave_overlap.json`. Requires a YouTube scrape, so it inherits that yield risk.

---

## Live disputes — marked, not resolved

| # | dispute | status |
|---|---|---|
| E-1 | 53.27/53.72 vs corrected 28.28/29.90 | correction **proposed**, not applied |
| E-12 | ICLR §2 stale for 7 of 8 baseline rows | correction **proposed**, not applied |
| §13.2 | Wav2CLIP contamination: table says in-distribution, JSON says held-out | **unresolved** |
| §13.3 | RUN-2 step20000 direction order conflicts between two docs | **unresolved** |
| E-14 | `best.pt` still selected on training `loss_ema`; `0eb3337` never touched `train_m2.py` | **live bug**; RUN-4 selects post hoc instead |
