# Information-existence probes — does `W_t` know anything about `t+10 s`?

**Motivation.** The RUN-5 pilot showed the tested post-hoc *predictors* do not beat persistence.
That is a weak test of *information content*: a predictor fighting `future = current + change`
looks bad whether there is no change information, or the change information exists but direct
future-state prediction cannot extract it against a strong persistence baseline.

**Answer: the information exists. This is not Case A.**

> `W_t` linearly predicts **22.7 %** of the variance of the *change* `ΔW = W_{t+10} − W_t`,
> against a persistence baseline of **0.000** and a null of **−0.002**.

**But the directional part of that information largely lives *before* fusion, not in `W`.**

---

## 1. Predeclared method

Metrics were fixed **before any probe ran** (script docstring, committed first).

`R² = 1 − SS_res/SS_tot` on **held-out participants**, `SS_tot` about the **train** mean.
`R² = 0` means "no better than predicting the training mean".

Reported per target: `R²_forward`, `R²_backward`, `R²_shuffled` (null), `R²_persistence`
(use `t`'s own value of the property).

**Signal criterion:** `R²_fwd − R²_shuf ≥ 0.05` **and** `R²_fwd ≥ 3 × bootstrap SE`.
**Beyond persistence** additionally requires `R²_fwd − R²_persist ≥ 0.05`.

**Two inputs, identical probe and targets — the three-way diagnostic:**

| input | contents |
|---|---|
| **W** | `W_t`, the fused scene representation (1024) |
| **VA** | `[vision_mean_t ; ambient_mean_t]`, pre-fusion frozen encoder summaries (1024+768) |

Δ = 10 s, 10 s windows ⇒ pairs share **zero** input. Eval on the 10 s-spaced within-file grid,
participant-held-out split (170 eval videos, 7 unseen kitchens), 119,686 train / 6,409 eval pairs.
Ridge only; λ chosen on a held-out slice of **train**, never on eval.

> **Caveat.** `vision_mean` / `ambient_mean` are **mean-pooled** over the window, not full token
> sequences. They are a cheap, faithful proxy for "pre-fusion features", **not** a full-token
> probe — that needs the 610 GB feature cache and is not cheap.

### 1.1 A broken null, found and fixed

The first null permuted the **target** index `j`. That is invalid for difference targets: under a
shuffled `j`, `ΔW = W_j − W_i` has much larger variance, inflating `SS_tot` and therefore `R²`.
**The null outscored the real pairing** (`dW_full`: null 0.337 vs real 0.227) — a signature of a
broken null, not an absent signal, and it would have produced a false "no information" verdict on
precisely the most important target.

**Corrected null: permute the *input* rows, leaving pairs and targets untouched.** The target
distribution is then identical and only the input→target correspondence is destroyed. The
corrected null sits at −0.002 for `dW_full`, as it should.

## 2. Headline — change is predictable, and persistence cannot do it

| input | target | R²_fwd | R²_bwd | null | persistence | verdict |
|---|---|---:|---:|---:|---:|---|
| **W** | **ΔW (full)** | **0.2267** ±0.0035 | 0.1952 | −0.0022 | **0.0000** | **SIGNAL, beyond persistence** |
| W | ΔW (top-8 PCA) | 0.2433 | 0.1653 | −0.0021 | n/a | SIGNAL |
| W | ‖ΔW‖ | 0.2352 | 0.1383 | 0.0404 | n/a | SIGNAL |

The persistence baseline for `ΔW` is "predict zero", which scores exactly 0.000 by construction.
**The probe reaches 0.227.** Whatever the pilot's failure was, it was **not** the absence of
change information in `W`.

All 24 probe cells clear the signal criterion. The full future state is also predictable
(`W:W_full` 0.387 vs persistence 0.208, beyond persistence).

## 3. The three-way diagnostic

| target | **W** fwd | W fwd/bwd | **VA** fwd | VA fwd/bwd |
|---|---:|---:|---:|---:|
| W_full | **0.3873** | 1.07× | 0.3094 | 1.20× |
| W_pca8 | **0.5003** | 1.12× | 0.4740 | 1.29× |
| **ΔW full** | **0.2267** | **1.16×** | 0.0698 | **9.83×** |
| **ΔW pca8** | **0.2433** | **1.47×** | 0.1607 | **8.59×** |
| ‖ΔW‖ | 0.2352 | 1.70× | **0.2770** | 3.62× |
| vis_full | 0.3319 | 1.14× | **0.4786** | 1.30× |
| aud_full | 0.4769 | 1.05× | **0.5818** | 1.07× |
| vis_norm | 0.1818 | 0.99× | **0.3282** | 1.25× |
| aud_norm | 0.5353 | 1.08× | **0.6029** | 1.15× |

### 3.1 Fusion discards modality-predictive information

Forward R², VA minus W: **vision +0.147, ‖vision‖ +0.147, audio +0.105, ‖audio‖ +0.068.**

Pre-fusion summaries predict the **future modality content** substantially better than `W` does.
`W` wins only on predicting future `W` (−0.078 for VA), which is unsurprising — that is `W`'s own
space. **This is the Case C signature.**

### 3.2 The directional asymmetry lives before fusion — the sharpest result here

On change targets, `W` is nearly **symmetric** (1.16×, 1.47×) while `VA` is strongly
**directional** (9.83×, 8.59×).

**Read this carefully, because the absolute numbers cut the other way.** `W` predicts change
*better* in absolute terms (0.227 vs 0.070). What `VA` has is not more change information but
far more **directional** change information. The natural reading:

* `W` largely encodes **how much** will change — a magnitude-like, time-symmetric quantity
  ("this is a high-activity moment"). Consistent with `‖ΔW‖` being `W`'s most asymmetric change
  target (1.70×) yet still modest.
* `VA` retains **which way** it will change, and that part is 8–10× stronger forward than
  backward — but it is a small share of total variance (0.070).

**Fusion into `W` preserves — even improves — predictability while largely destroying
directionality.** That is exactly the failure mode the pilot would exhibit: plenty of information,
almost none of it directional, so a forward predictor cannot beat a symmetric persistence baseline.

## 3.3 CAPACITY CONTROL — the directional gap is **not** dimensionality

`VA` has 1,792 dims against `W`'s 1,024, so part of its advantage could have been capacity rather
than content. `VA` PCA-reduced to **1,024** on train, everything else identical:

| target | **W** fwd / bwd | ratio | **VA@1024** fwd / bwd | ratio |
|---|---:|---:|---:|---:|
| ΔW full | 0.2267 / 0.1952 | 1.16× | 0.0558 / **−0.0004** | **∞** |
| ΔW pca8 | 0.2433 / 0.1653 | 1.47× | 0.1374 / 0.0097 | **14.2×** |
| ‖ΔW‖ | 0.2352 / 0.1383 | 1.70× | 0.2534 / 0.0630 | **4.0×** |
| vis_full | 0.3319 | | **0.4807** | |
| aud_full | 0.4769 | | **0.5560** | |

**The gap survives matching and sharpens.** At equal dimensionality the pre-fusion features
predict forward change at R² 0.056 and backward change at **−0.0004 — exactly zero**. That is a
clean arrow of time. `W`, built from the same windows, predicts backward change almost as well as
forward (1.16×). The modality-prediction advantage also survives (+0.149 vision, +0.079 audio).

**Case C is established: the information entering the fusion has a temporal direction, and what
leaves the fusion has largely lost it.**

## 3.4 MLP probe — **INCONCLUSIVE. Do not read its numbers as a result.**

The predeclared rule licensed a small MLP wherever the linear probe found signal, to test whether
`W`'s directional content is **non-linearly** recoverable (which would shift weight back to
Case B). **That probe does not work, and its outputs are not interpretable.**

**The bar: an MLP can represent a linear map, so it must at least match ridge on the same data
before any of its numbers mean anything.** Ridge scores **0.2267** on `W → ΔW`. The MLP never got
there:

| fix attempted | best R² | vs ridge 0.2267 |
|---|---:|---|
| as first written (no early stopping) | −0.026 | fails; *worse* with more steps (−0.270 at 15k) — overfitting |
| + early stopping on a train-internal split (the same discipline ridge's λ gets) | **+0.083** | still 2.7× short |
| + objective aligned to the metric (centre targets, don't scale — standardising made it optimise a different quantity from R²) | −0.085 | still fails |

Each fix was a correctness issue, not a hyper-parameter: the first comparison was rigged
(unregularised MLP vs regularised ridge), and the second optimised per-dimension standardised MSE
while R² pools raw squared error.

**Stopping here deliberately.** Continuing would become the hyper-parameter sweep this
investigation is explicitly not allowed to run, and a probe that needs sweeping to beat ridge
cannot support a clean claim either way.

**Consequence for the conclusions:** the question "is `W`'s directional content non-linearly
recoverable?" is **OPEN**, not answered negatively. Case C rests on the linear evidence in §3.1–3.3,
which is sound and capacity-controlled. **Case B is neither confirmed nor excluded.**

## 4. What each result rules in and out

| finding | rules OUT | rules IN |
|---|---|---|
| ΔW predictable from `W` at R² 0.227, persistence 0.000 | **Case A** — "no future information" | information exists; the pilot's failure was extraction, not absence |
| `W` change prediction nearly symmetric (1.16×) | that `W` holds strong directional structure | why an InfoNCE forward predictor cannot beat persistence |
| `VA` change prediction 9.83× directional | that directionality is absent from the system | **Case C** — fusion discards it |
| VA > W on future modality targets by +0.07…+0.15 | that `W` is a lossless summary for prediction | compression is costing predictive content |
| Null at −0.002, all controls clean | measurement artifacts | the linear probe is sound |
| VA@1024 backward ΔW = −0.0004 vs forward 0.0558 | that the gap is dimensionality | **Case C established** — capacity-controlled |
| MLP never matches ridge (0.083 vs 0.227 best) | *nothing* — the probe is invalid | only that this instrument does not work |

**Classification: Case B *and* Case C, with C the more actionable.** The InfoNCE future-state
objective was a poor extractor (B), *and* the representation it was extracting from has had its
directional content largely removed by fusion (C).

## 5. What this does NOT establish

* **Not** that a redesigned predictor will work. R² 0.227 on ΔW is real but modest, and `W`'s
  directional component is weak (1.16×).
* **Not** that full token features would do better than the mean-pooled proxy. Untested; needs
  the 610 GB cache.
* **Not** that the retrieval metric and R² agree. A 0.227 R² on ΔW need not convert into R@1
  gains against a persistence baseline that is very strong at R@1.
* **Not** a licence to reinterpret the pilot. The pre-registered pilot **failed** and stays failed.
* **Nothing here is about action-conditioned prediction.** `P(W_{t+Δ} | W_t, a_t)` is untested and
  is a separate research direction, not a fix for this experiment.

## 6. Recommended next step — and it is not joint RUN-5

**Do not launch joint RUN-5 as specified.** The spec's objective trains a predictor on top of the
fused `W`, and §3.2 says the directional information has already been discarded *by the fusion*
before that predictor sees anything. Adding a larger predictor downstream of the bottleneck does
not address the bottleneck.

Ordered by cost:

1. **Confirm §3.2 with a matched-capacity control (cheap, ~minutes).** `VA` has 1,792 input dims
   against `W`'s 1,024, so some of its advantage may be capacity rather than content. Re-run the
   key rows with `VA` PCA-reduced to 1,024 dims. **If the directional gap survives, Case C is
   established; if it collapses, §3.2 is an artifact and the recommendation changes.** This should
   run before anything else.
2. ~~A small MLP probe~~ — **attempted, INCONCLUSIVE (§3.4).** If the Case B question is worth
   settling, it needs a probe that demonstrably matches ridge first; that is a methods task, not
   a result, and it is not a prerequisite for the decision below.
3. **A redesigned RUN-5 is now justified** — Case C survived the capacity control. Its objective
   must preserve predictive information *through the fusion*, not bolt a larger predictor onto the
   existing `W`, because §3.3 locates the loss at the fusion itself.

**Step 1 did not collapse the gap — it sharpened it.** The clean negative in
`RUN5_PILOT_RESULT.md` stands as written, and this probe suite explains *why* it happened: the
information exists, but the fusion strips its direction before any downstream predictor sees it.
That pair is publishable on its own, and it also identifies a specific, testable architectural
cause rather than leaving the failure unexplained.

## 7. Provenance

| item | value |
|---|---|
| representation | RUN-4 `step18000`, sha256 `27b33c8c…`, frozen |
| corpus / split | Epic-Kitchens 648 videos; participant-held-out, 7 unseen kitchens |
| pairs | 119,686 train (2 s lattice) / 6,409 eval (10 s grid) |
| Δ | 10 s, zero input overlap |
| probe | ridge, λ ∈ {10, 10², 10³, 10⁴} chosen on a held-out train slice |
| error bars | 200-sample bootstrap over eval queries |
| script / artifact | `scripts/temporal_probe/p6_information_probes.py`, `p6_information_probes.json` |
