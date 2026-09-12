# P2.8 — Predictive World-State Experiment: design, and a data prerequisite that must be settled first

**Status: SPEC ONLY. Nothing implemented, nothing trained.** Separate research track, not
part of the ICLR RUN-2/RUN-4 submission (decision 5).

Objective under test:

```
L = L_contrastive  +  lam_pred * L_predict  +  L_SIGReg
L_predict = 1 - cos( P(W_t), W_{t+delta} )
```

---

## 1. BLOCKER — only 23% of the training corpus can supply a prediction pair, and Δ=5 s is impossible

A future-prediction loss needs **sequential pairs from the same source file**. Measured
against the actual RUN-2/RUN-4 corpus:

| corpus | clips | share | can supply `(W_t, W_{t+Δ})`? |
|---|---|---|---|
| VGGSound | 199,007 | **59.7%** | **No.** Independent 10 s clips with no temporal continuation. |
| Ego4D | 134,491 | 40.3% | Partially — ordered windows, but a **10 s non-overlapping** stride |

Usable pairs, counting only genuinely consecutive window indices:

| Δ | usable pairs | % of Ego4D | **% of the full corpus** |
|---|---|---|---|
| 10 s | 77,831 | 57.9% | **23.3%** |
| 20 s | 85,295 | 63.4% | **25.6%** |
| 30 s | 81,977 | 61.0% | **24.6%** |
| 60 s | 77,117 | 57.3% | **23.1%** |

**Two consequences that change the experiment as specified:**

1. **Δ = 5 s is unobtainable.** The extraction stride is 10 s and non-overlapping; the raw
   Ego4D video is gone (E-8/E-9). "Δ = 1 temporal step" *is* 10 s here. The requested ladder
   {1 step, 5 s, 10 s, 20 s, 30 s} collapses to **{10, 20, 30} s** (60 s optionally).
2. **`lam_pred` would be active on only ~23% of training samples.** The other 77% have no
   future to predict. That is not a small detail: it means the predictive term sees a
   quarter of the gradient steps the contrastive term does, so a `lam_pred` sweep is
   confounded with an effective-weight-per-sample factor of ~4. Any λ reported must be
   stated as **λ per participating sample**, not per step, or the numbers will not transfer.

### Options, in order of preference

**(A) Run it on Ego4D-only batches at Δ ∈ {10,20,30} s.** No new data. Costs the VGGSound
portion of the contrastive signal, so it is *not* a clean single-variable change against
RUN-4 — it changes corpus composition and objective together.

**(B) Wait for Epic-Kitchens (P2.6).** 100 h continuous, any stride, 180,000 windows at 2 s
for 0.68 TB. Δ ∈ {1,2,5} s becomes available and **100% of that corpus can supply pairs**.
This is the design the experiment actually wants, and it is gated only on the licence call.

**(C) Mixed batches, masking `L_predict` to the pairs that exist.** Keeps the corpus intact,
but bakes in the 4× effective-weight asymmetry above and makes λ hard to interpret.

**Recommendation: (B).** The experiment is worth doing properly once, and Epic-Kitchens
removes both constraints. (A) is a reasonable pilot if an answer is wanted before the
licence decision — but it must be reported as a pilot on a different corpus, not as a
result comparable to RUN-4.

## 2. Design, once data allows

### 2.1 Predictor `P`

Small and fixed across the sweep, so `lam_pred` is the only variable: a 2-layer MLP
(1024 → 1024 → 1024, GELU), mirroring the Phase-3 probe so the two are directly
comparable. **Δ is a conditioning input, not a separate head** — one predictor taking
(W_t, Δ) generalises across horizons and avoids 3–4× the parameters.

### 2.2 `lam_pred` sweep

`{0 (=RUN-4 control), 0.05, 0.2, 1.0}` **set by gradient-norm matching at step 0**, not by
borrowed constants — `docs/PREDICTIVE_OBJECTIVE_SURVEY.md` §2 records why: CAV-MAE's
λ_c=0.01 sits on the *contrastive* term beside a dominant reconstruction loss, the mirror of
this configuration, and our own SIGReg λ was scaled 0.03→0.00375 by analogy and the run
collapsed anyway. Log both gradient norms every step.

### 2.3 Metrics — every one of these, every arm

| metric | why |
|---|---|
| temporal cosine decay `cos(W_t, W_{t+Δ})` | **the trivial-solution detector.** If this rises toward 1, the model is winning by making `W` static |
| future-prediction cosine | the loss, reported honestly |
| future R@1/R@5/R@10 | the metric that matters, with **gallery size stated** |
| shuffled-future control | pair `W_t` with a random future; must collapse to chance |
| effective rank | collapse detector |
| **identity baseline** `W_t → W_t` | the bar |
| **learned predictor vs identity** | **the actual test** |
| present-state retrieval (1,545 gallery) | must not be sacrificed |

### 2.4 The trivial solution, stated concretely

`L_predict = 1 - cos(P(W_t), W_{t+Δ})` is minimised perfectly by a **constant** `W`. SIGReg
resists that (it shapes toward N(0,I), penalising collapse) and the contrastive term resists
it (a constant `W` cannot discriminate), but neither directly penalises **temporal**
staticness. Phase 2 already measured the baseline to beat: the current representation's
`cos(W_t, W_{t+10s})` is **0.8699**, and it plateaus rather than decaying.

**Pre-registered failure condition: if temporal cosine rises above ~0.93 while future R@1
does not improve over identity, the arm has found the trivial solution and is rejected** —
regardless of how far `L_predict` fell.

## 3. Evaluation — reuse Phase 3 exactly

`scripts/temporal_probe/phase3_forward.py`, unchanged, on the new checkpoint: file-disjoint
split, IDENTITY / corpus-mean / per-dim-rescaled baselines, cosine **and** R@1/R@5 with the
gallery size stated. Against the current representation, whose numbers are already measured:

| Δ | method | cosine | R@1 | gallery |
|---|---|---|---|---|
| 10 s | IDENTITY | **0.8569** | 0.43 | 2,533 |
| 10 s | ridge | 0.7733 | 3.16 | 2,533 |
| 10 s | 2-layer MLP | 0.7858 | **3.43** | 2,533 |

**Note the split that already exists and will complicate any claim:** on the *current*
representation the learned map beats identity ~8× at R@1 but **loses** on cosine and R@5.
Copying `W_t` lands nearest `W_t` itself, which is in the gallery as an earlier window's
future, so identity retrieves a systematic off-by-one. Any P2.8 result must be reported on
**all three** of cosine, R@1 and R@5 — a win on one alone is not a win.

## 4. P2.8.4 conditioning ablation — do second, and note the confound

`P(W_t, A_t, V_t)` vs `P(W_t)`. If the former wins substantially, the bottleneck is
discarding information needed for prediction.

**Caveat to state in any write-up:** the conditioned predictor has strictly more parameters
and strictly more input, so it *should* win slightly on capacity alone. The interesting
quantity is the **size** of the gap relative to a parameter-matched `P(W_t)` control, not
its sign. Include that control.

## 5. Acceptance criteria (P2.8.5), pre-registered

All six required. Any one failing ⇒ report the negative result and retain the current
description.

1. above-chance held-out future retrieval (chance stated with the gallery size)
2. improvement over IDENTITY on **future R@1 *and* not a regression on R@5**
3. improvement over the current non-predictively-trained representation
4. present-state retrieval retained (1,545 gallery, no material drop vs RUN-4)
5. no effective-rank collapse
6. survives the shuffled-future control

**And explicitly: a falling `L_predict` is not evidence of anything.** The seven historical
cosine-regression runs reached low loss at **chance retrieval** (0.07–0.20% R@1 on a 1,545
gallery where chance is 0.065%) — `docs/PREDICTIVE_OBJECTIVE_SURVEY.md` §5. The same trap is
open here.
