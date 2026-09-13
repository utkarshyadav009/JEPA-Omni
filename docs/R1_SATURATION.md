# R@1 Saturation — why top-1 retrieval under-reports representation quality

**Claim.** Over RUN-4's last 5,000 training steps, held-out R@1 is flat while R@5 and the
representation's effective rank keep climbing monotonically. The model is still improving;
**R@1 has stopped registering it.**

This is not a curiosity about one run. It is the **third independent instance** of the same
failure mode in this project, and the reason every retrieval claim here carries R@5 alongside
R@1.

---

## 1. The measurement

All 20 tagged RUN-4 checkpoints, scored on the held-out 1,545-clip gallery through the corrected
harness at the model's own training length (`T_a = 896`), **mean of 3 batch-order seeds**.

| step | v→a R@1 | a→v R@1 | v→a R@5 | a→v R@5 | eff_rank | seed range R@1 |
|---|---:|---:|---:|---:|---:|---:|
| 14000 | 38.53 | 38.92 | 69.19 | 69.64 | 64.16 | 0.07 |
| 15000 | 39.33 | 39.29 | 69.82 | 70.49 | 67.87 | 0.13 |
| 16000 | 41.68 | 41.42 | 70.51 | 71.39 | 69.74 | 0.13 |
| 17000 | 41.44 | 41.42 | 71.74 | 72.30 | 72.75 | 0.07 |
| 18000 | **41.77** | **41.88** | 71.97 | 72.10 | 73.53 | 0.06 |
| 19000 | 41.40 | 41.81 | 72.30 | 72.23 | 74.12 | 0.06 |
| 20000 | 41.34 | 41.68 | **72.51** | 72.36 | **74.26** | 0.07 |

**Steps 16,000 → 20,000:**

| metric | change | seed noise | trend? |
|---|---:|---:|---|
| v→a R@1 | −0.34 (span 0.43) | 0.06–0.13 | **none** — non-monotonic, within ~3× noise |
| v→a R@5 | **+2.00** | — | rising, monotonic from 16k |
| eff_rank | **+4.52** | — | rising, strictly monotonic |

Effective rank increases at *every single step* across the plateau. R@5 increases at every step
but one. R@1 goes up, down, up, down. Only R@1 is flat.

## 2. Why this happens

R@1 asks one question: is the correct clip ranked first out of 1,545? Late in training the model
is not moving items *into* rank 1 — it is tightening the neighbourhood *around* rank 1 and
spreading probability mass across more dimensions. R@5 sees the first effect. Effective rank
sees the second. R@1 sees neither, because a top-1 indicator is a step function and cannot
report sub-threshold movement.

The consequence is asymmetric and therefore dangerous: **R@1 can fail to rise while the
representation improves, and it can also rise while the representation gets worse** (§3.2).
A flat R@1 is not evidence of convergence.

## 3. The same lesson, twice before

### 3.1 Gallery contamination — R@1 shrinks while R@5 grows

Against the published "≥15.4 point" contamination claim, the corrected measurement gives:

| | v→a Δ R@1 | v→a Δ R@5 |
|---|---:|---:|
| E-13 (0% vs 100% contaminated) | **+9.58** | **+17.80** |
| E-5 (0% vs 90.9% balanced) | +9.36 ± 0.08 | +18.43 ± 0.04 |

Read on R@1 alone, the contamination effect looks like it *shrank by 40%*. That is a false
summary. Under the padding leak the contaminated gallery was near ceiling (94.85 / 97.86 R@1),
which compressed the gap by construction. On R@5 the effect is **larger** than published, not
smaller. The metric moved, not the phenomenon.

### 3.2 The padding leak — R@1 rose because of a shortcut

The published 53.27 / 53.72 R@1 was inflated ≈24 points by a batch-dependent padding
shared-nuisance leak. A rising top-1 number gave no signal that anything was wrong; the leak was
caught by a batch-size-1 control, a uniform-pad control that *hurt*, and a random-pad control
that reproduced the gain — never by watching R@1.

This is the mirror image of §1: there, R@1 under-reported real improvement; here, it
over-reported improvement that was not real.

## 4. What we do about it

* **Never quote R@1 alone.** Every retrieval claim in this project reports R@5, and where the
  representation itself is the subject, effective rank.
* **Never select a checkpoint on R@1 alone.** `step18000` is the R@1 argmax but wins by 0.43
  over a 5,000-step plateau whose seed noise is 0.06–0.13. It is the argmax, not a meaningfully
  better model. On R@5 or effective rank the argmax would be `step20000`. We report `step18000`
  as the selected checkpoint *and* state that the plateau is flat, rather than implying the
  selection found something.
* **A flat R@1 is not a stopping criterion.** RUN-4 was stopped at 20,000 steps because that was
  the pre-specified budget, not because it converged. Effective rank was still rising when it
  stopped, and we do not know where that ends.

## 5. Provenance

| item | value |
|---|---|
| checkpoints | all 20 of `checkpoints/m2_run4_padfix_ta896/step*.pt` |
| gallery | `data/vggsound_eval_1545.txt`, n = 1,545, 0% contamination |
| protocol | corrected harness, `T_a = 896`, deterministic order, 3 batch-order seeds |
| eff_rank | participation ratio of the scene representation over the full gallery |
| artifacts | `docs/artifacts/temporal_probe/p30_shard{0,1,2,3}.json` |
| script | `scripts/temporal_probe/p30_select_checkpoint.py` |
| canonical | `docs/CANONICAL_NUMBERS.md` §1.1 and §9 |

**Retracted claim carried here for the record:** "RUN-4 was still rising at step 20,000" was
stated from partial data through step 16,000. The full sweep shows R@1 flat from 16,000. R@5 and
effective rank *were* still rising — which is the finding, and is not the same claim.
