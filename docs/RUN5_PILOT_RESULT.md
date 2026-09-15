# RUN-5 PILOT — FAILED the pre-registered gate. Joint training does not run.

**Question.** Can a predictor trained on a *frozen* RUN-4 representation extract forward
information that RUN-4 does not itself expose?

**Pre-registered answer: no.** A 4.2M-parameter predictor trained with InfoNCE on **239,136**
Epic-Kitchens pairs is **1.45 points worse than doing nothing at all.**

**The conclusion this licenses, stated narrowly and deliberately:**

> **The tested post-hoc predictors did not recover directional future information from `W`
> beyond persistence.**

**Interpretation, after the fusion diagnosis (`docs/FUSION_BOTTLENECK.md`):**

> The failure of post-hoc future prediction is **not adequately explained** by predictor
> capacity, negative construction, or participant-held-out splitting — all three were tested and
> rejected. Capacity-controlled linear probes instead indicate that **directional future
> information is present in the pre-fusion features but is substantially attenuated in the
> learned fusion into `W`**: vision alone predicts forward change with backward prediction at
> *exactly zero*, while `W` predicts backward change almost as well as forward (1.14×). This
> **localises the bottleneck to the fusion**. It is an observational comparison, not a causal
> demonstration.

**This does NOT show that future prediction is impossible.** It shows that *the current
representation* and *the tested prediction objective* did not provide measurable directional
prediction. It is a statement about two specific things we built, not about the problem.

---

## 1. The pre-registered gate, and the result against it

Criteria fixed in `RUN5_SPEC.md` §2.1 **before any number here existed**. Δ = 10 s, within-file
micro R@1, 3 seeds, 170 held-out videos across 7 held-out kitchens, 6,240 queries,
mean gallery 52.6, chance 1.857 %.

| # | criterion | required | observed | verdict |
|---|---|---|---|---|
| 1 | beat PERSISTENCE (IDENTITY) | ≥ +2.0, ≥3× SE | **−1.448 ± 0.078** (18.6× SE the **wrong way**) | **FAIL** |
| 2 | forward − backward | ≥ +2.0, ≥3× SE | **+1.261 ± 0.171** (7.4× SE, magnitude short) | **FAIL** |
| 3 | shuffle at chance | ratio < 1.5 | 1.977 vs 1.857 → **1.06×** | PASS |
| 3 | falsifier at chance | ratio < 1.5 | 4.142 vs 3.672 → **1.13×** | PASS |
| 4 | no collapse | see §5 | predictor output *higher* rank than its input | PASS |

**Two of four fail, and criterion 1 fails in the wrong direction. This is not a near miss.**

### 1.1 The pre-registered numbers, all three seeds

| method | forward R@1 | backward R@1 |
|---|---:|---:|
| **IDENTITY (persistence)** | **16.763** | **17.003** |
| ridge | 14.215 | 12.436 |
| **pilot predictor** | **15.315 ± 0.078** | 14.055 ± 0.171 |
| chance | 1.857 | 1.857 |

Seeds: **15.160 / 15.401 / 15.385** forward. Not a seed accident.

**Persistence is symmetric again.** IDENTITY scores *higher backward than forward*
(17.003 vs 16.763, **+0.240**) — the past is marginally easier to retrieve than the future. This
is the second corpus on which that holds, after Ego4D in P3.2.

**Every learned map loses to copying**, as on Ego4D: ridge −2.5, the trained predictor −1.4.
P3.2 replicates on a different corpus, at 3.9× the pairs, with a contrastive objective and a
participant-disjoint split.

---

## 2. EXPLORATORY grid — negatives × split

**Everything in this section is exploratory.** The pre-registered cell is
(`cross_video`, `participant`) and it failed. Per `RUN5_SPEC.md` §2.3, nothing below may be
presented as the pre-registered outcome, and no cell here was used to reinterpret §1.

It was run to test the two explanations that could most plausibly have explained the failure
away — a misspecified objective, and an over-harsh split.

| negatives | split | IDENTITY | ridge | pilot | vs IDENTITY | fwd−bwd | shuf/ch | fals/ch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cross_video *(pre-reg)* | participant | 16.76 | 14.21 | 15.32 ± 0.08 | **−1.45** (18.3σ) | +1.06 ± 0.14 | 1.12 | 1.12 |
| mixed | participant | 16.76 | 14.21 | 12.05 ± 0.11 | **−4.71** (41.8σ) | +0.61 ± 0.41 | 1.14 | 1.11 |
| same_video | participant | 16.76 | 14.21 | 9.66 ± 0.23 | **−7.10** (30.4σ) | +0.49 ± 0.29 | 1.05 | 1.15 |
| cross_video | video | 17.00 | 14.93 | 15.60 ± 0.07 | **−1.40** (19.5σ) | +0.91 ± 0.14 | 1.02 | 1.09 |
| mixed | video | 17.00 | 14.93 | 13.57 ± 0.30 | **−3.43** (11.3σ) | +0.59 ± 0.36 | 1.01 | 1.10 |
| same_video | video | 17.00 | 14.93 | 11.46 ± 0.16 | **−5.54** (34.2σ) | +1.18 ± 0.23 | 1.06 | 1.17 |

**Not one of six configurations beats persistence.** Best −1.40, worst −7.10. All shuffle and
falsifier controls sit within 1.01–1.17× chance.

### 2.1 Explanation 1 — "the negatives were too easy" — REJECTED

The diagnosis after the pilot was that cross-video negatives are separable by scene identity
(a kitchen is not another kitchen), so the objective saturates at 95–98 % in-batch accuracy while
the real task degrades. It predicted that harder, within-scene negatives would close the gap.

**They made it monotonically worse, in both splits**: −1.45 → −4.71 → −7.10 (participant) and
−1.40 → −3.43 → −5.54 (video). **The diagnosis was wrong.**

### 2.2 Explanation 2 — "the split was too harsh" — REJECTED

| negatives | IDENTITY participant → video | pilot participant → video |
|---|---|---|
| cross_video | 16.76 → 17.00 (**−0.24**) | 15.32 → 15.60 (−0.29) |
| mixed | 16.76 → 17.00 (−0.24) | 12.05 → 13.57 (−1.52) |
| same_video | 16.76 → 17.00 (−0.24) | 9.66 → 11.46 (−1.80) |

**The participant split costs ~0.24 points on IDENTITY. The pilot fails by 1.45.** The split
cannot account for the failure. Putting 31 of 34 kitchens on both sides of a video-level split
buys almost nothing — within-recording persistence, not scene familiarity, carries the signal.

One real interaction: the split matters *more* for harder-trained predictors (−0.29, −1.52,
−1.80). The harder the training, the more the model depends on having seen the kitchen — it is
increasingly fitting scene identity rather than temporal structure.

### 2.3 Control — was `same_video` hardness, or just fewer negatives?

`same_video` at `n_per_video=8` has only 7 negatives, so its deficit was confounded with
negative *count*. Two matched controls settle it:

| scheme | negatives | pilot | vs IDENTITY | predictor eff_rank |
|---|---|---:|---:|---:|
| cross_video | 255 easy | 15.32 ± 0.08 | −1.45 | 35.97 |
| cross_video | **7 easy** | 13.43 ± 0.26 | −3.33 | 11.55 |
| same_video | 7 hard | 9.66 ± 0.23 | −7.10 | 60.45 |
| same_video | **63 hard** | 9.33 ± 0.20 | −7.43 | 63.55 |

| factor | effect |
|---|---|
| **hardness** at ~7 negatives | **3.77 points** |
| **hardness** at many negatives | **5.98 points** |
| count, easy negatives (255→7) | 1.89 points |
| count, hard negatives (63→7) | **−0.33 — nothing** |

**Hardness dominates, and `same_video` is not a count artifact**: going from 7 to 63 hard
negatives changes the result by −0.33. The monotonic hardness reading stands at matched count.

### 2.4 An anti-correlation worth recording

Predictor-output effective rank runs **opposite** to retrieval quality: `cross_video`@7 has
eff_rank **11.55** and scores **13.43**; `same_video`@63 has eff_rank **63.55** and scores
**9.33**. The highest-rank predictor output is the worst predictor. Whatever the hard-negative
objective is spreading the output across, it is not information about the future.

---

## 3. Why the failure looks the way it does

**The predictor got worse with training.** At 300 steps it scored **17.115**, *above* IDENTITY.
At 8,000 steps, **15.315**. Optimising the training objective degraded the evaluation task.

That pattern survives the grid, and the grid rules out the obvious reading of it. The remaining
description — which is a description, not a mechanism — is that **the more closely training is
made to resemble the evaluation task, the worse the model does at it**. Discriminating moments
≥10 s apart within one recording does not appear to be learnable from `W` under these objectives,
so training on it degrades the persistence structure the model would otherwise inherit for free.

**Why this is not yet a claim about information.** A predictor fighting
`future = current + change` can look bad for two very different reasons: there is no change
information, or the change information exists but direct future-state prediction is a poor way
to extract it against a strong persistence baseline. **§6 is designed to separate those.**

---

## 4. Corrected chance calculation

Chance is the **query-weighted mean of 1/G**, not `1/mean(G)`. Gallery sizes here run from 1 to
370 grid windows, and by Jensen's inequality `E[1/G] ≫ 1/E[G]`. The first implementation used
the latter and reported a falsifier that was *exactly at chance* as **2× chance**. Fixed:
falsifier 3.797 vs 3.672 → 1.03×.

## 5. Collapse, and a threshold that was mis-calibrated

| | effective rank |
|---|---:|
| frozen RUN-4 `W` on this Epic-Kitchens eval set | **26.63** |
| pilot predictor's output | **35.60** |
| (RUN-4 `W` on the VGGSound 1,545 gallery, for reference) | 73.53 |

No collapse: predictor output is *higher* rank than its input, cross-video cosine 0.063, no
zero-variance dimension.

**The spec's `eff_rank ≥ 37` threshold was mis-calibrated and would have been wrong either way.**
It came from RUN-4's 73.53 on VGGSound; the *frozen, uncollapsed* representation scores 26.63 on
Epic-Kitchens because the corpus is one narrow domain rather than 309 diverse classes. It would
have failed a working representation. **Effective-rank thresholds are corpus-specific and must be
set against the frozen baseline on the same corpus.**

## 6. Implementation and control defects found and fixed

All four are recorded because **each would have produced a better-looking, wrong number.**

| # | defect | consequence had it stood |
|---|---|---|
| 1 | Falsifier partner was "the next file" — the **same participant 95 % of the time**, i.e. the same kitchen | a "mismatched" control that shares scene identity; read 4.78 vs 1.78 |
| 2 | Falsifier target index was the query's own position carried over and **clamped** into a shorter file | positional correlation with content; targets piled on the last window |
| 3 | Chance computed as `1/mean(G)` instead of query-weighted `mean(1/G)` | an at-chance falsifier reported as 2× chance |
| 4 | Backward control was the **forward-trained** predictor run backwards | worse backward by construction; would have measured our training direction, not the data |

**Defect 4 mattered most, and it is worth stating exactly.** With the forward-trained predictor
run backwards, the forward−backward gap is **+3.082 ± 0.154 (20.0× SE)** — which **PASSES**
criterion 2 (≥ +2.0 and ≥ 3× SE). With a separately trained backward predictor it is
**+1.261 ± 0.171 (7.4× SE)**, which **FAILS** on magnitude.

**The pilot would have appeared to half-pass on a measurement artifact**, and the artifact was
large, clean and highly significant — 20× SE is not a marginal call. A forward-trained model is
worse backwards *by construction*; that gap measures our training direction, not the data.

## 7. What this does and does not establish

**Does:**
* The tested post-hoc predictors did not recover directional future information from `W` beyond
  persistence — under six objective/split configurations plus two matched controls.
* P3.2's core finding replicates on an independent corpus: persistence is symmetric, copying
  beats every learned map.
* Neither the negative construction nor the split explains the failure.

**Does not:**
* **Does not show there is no learnable temporal information.** That question is not yet
  answered; §8 is the experiment that addresses it.
* **Does not show joint predictive training would fail.** The pilot asks only whether the
  information is *already* recoverable from a frozen `W`.
* **Does not show `W` contains no future information.** Direct future-state prediction against a
  strong persistence baseline is a weak test of information content. See §8.

## 8. Next — cheap information probes, not another predictor

Before joint RUN-5 is considered at all, the open question is whether `W_t` contains **any**
measurable information about `W_{t+Δ}`. That is being tested with linear/ridge probes on the
already-cached representations, predicting simple properties of the future rather than the whole
future state, and including **ΔW** as a target so the persistence baseline is not doing the work.

The three-way diagnostic that matters: if the frozen token summaries predict the future but `W`
does not, the fusion into `W` is discarding predictive information; if neither does, the corpus
offers little predictable signal at this horizon; if `W` does but the InfoNCE predictor could not
recover it, the objective is the bottleneck.

Results: `docs/RUN5_INFORMATION_PROBES.md`.

## 9. Provenance

| item | value |
|---|---|
| representation | RUN-4 `step18000`, sha256 `27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4`, frozen |
| corpus | Epic-Kitchens, 648 videos, 308,897 windows, 1 s stride |
| split (primary) | participant-held-out — 478 train / 170 eval videos, kitchens `P01 P06 P11 P16 P21 P26 P31`, **0 overlap** |
| split (secondary) | video-held-out — 518 / 130 videos, 31 of 34 kitchens on both sides |
| training pairs | 239,136 at Δ = 10 s |
| predictor | LayerNorm → 1024→2048 GELU → 2048→1024, ~4.2M params |
| objective | InfoNCE, τ = 0.05, AdamW 3e-4 cosine, 8,000 steps |
| eval | 10 s-spaced within-file grid (zero input overlap), query's own window excluded |
| seeds | 0, 1, 2 (every cell) |
| scripts | `scripts/temporal_probe/p5_pilot.py`, `p5_grid_report.py` |
| artifacts | `p5_pilot.json`, `p5_exp_{cross_video,mixed,same_video}_{participant,video}.json`, `p5_ctl_samevideo_n64.json`, `p5_ctl_crossvideo_bs8.json` |

**Consistency gate:** the re-run of (`cross_video`, `participant`) after the refactor reproduces
**−1.448** exactly, so the grid and the pre-registered pilot are the same measurement.
