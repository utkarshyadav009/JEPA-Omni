# RUN-5 PILOT — FAILED the pre-registered gate. Joint training does not run.

**Question.** Can a predictor trained on a *frozen* RUN-4 representation extract forward
information that RUN-4 does not itself expose?

**Answer: no.** A 4.2M-parameter predictor trained with InfoNCE on **239,136** Epic-Kitchens
pairs is **1.45 points worse than doing nothing at all.**

---

## 1. The gate, and the result against it

Criteria fixed in `RUN5_SPEC.md` §2.1 **before any number here existed**. Δ = 10 s, within-file
micro R@1, 3 seeds, 170 held-out videos across 7 held-out kitchens, 6,240 queries,
mean gallery 52.6, chance 1.857 %.

| # | criterion | required | observed | verdict |
|---|---|---|---|---|
| 1 | beat PERSISTENCE (IDENTITY) | ≥ +2.0, ≥3× SE | **−1.448 ± 0.078** (18.6× SE the **wrong way**) | **FAIL** |
| 2 | forward − backward | ≥ +2.0, ≥3× SE | **+1.261 ± 0.171** (7.4× SE, but magnitude short) | **FAIL** |
| 3 | shuffle at chance | ratio < 1.5 | 1.977 vs 1.857 → **1.06×** | PASS |
| 3 | falsifier at chance | ratio < 1.5 | 4.142 vs 3.672 → **1.13×** | PASS |
| 4 | no collapse | see §4 | predictor output *higher* rank than its input | PASS |

**Two of four fail. The gate is not met, and criterion 1 fails in the wrong direction —
this is not a near miss.**

## 2. The numbers

| method | forward R@1 | backward R@1 |
|---|---:|---:|
| **IDENTITY (persistence)** | **16.763** | **17.003** |
| ridge | 14.215 | 12.436 |
| **pilot predictor** | **15.315 ± 0.078** | 14.055 ± 0.171 |
| chance | 1.857 | 1.857 |

Seeds are tight: 15.160 / 15.401 / 15.385. This is not a seed accident.

**Persistence is again symmetric.** IDENTITY scores *higher backward than forward*
(17.003 vs 16.763, +0.240). The past is marginally easier to retrieve than the future — the
third independent corpus-level confirmation of P3.2's finding, now on Epic-Kitchens rather than
Ego4D.

**Every learned map loses to copying**, exactly as on Ego4D: ridge −2.5, the trained predictor
−1.4, against IDENTITY. P3.2's result reproduces on a different corpus, at 3.9× the pairs, with
a contrastive objective and a participant-disjoint split.

## 3. Why it failed — a diagnosis, not an excuse

**The predictor got worse with training.** At 300 steps it scored **17.115**, *above* IDENTITY.
At 8,000 steps it scored **15.315**, well below. Meanwhile in-batch InfoNCE accuracy rose to
**95–98 %**.

Those two facts together identify the problem: **the training task and the evaluation task are
not the same task.** Training negatives are futures from *other videos*, which are separable by
scene identity alone — a kitchen is not another kitchen — so the objective saturates on a
discrimination that is trivial. Evaluation requires ranking the true future against **52 other
moments from the same recording**, where scene identity is constant and useless. Optimising the
easy task past the point of saturation actively degraded the hard one.

**This is a hypothesis about the negative construction, and the spec forbids acting on it here.**
§2.3 freezes negative construction before results are inspected. Re-running with same-video
temporally-distant negatives would be a **new exploratory experiment**, and must be reported as
such — never as the pre-registered outcome. It is recorded in §6 as the strongest candidate, not
adopted.

## 4. Collapse, and a threshold that was mis-calibrated

| | effective rank |
|---|---:|
| frozen RUN-4 `W` on this Epic-Kitchens eval set | **26.63** |
| pilot predictor's output | **35.60** |
| (RUN-4 `W` on the VGGSound 1,545 gallery, for reference) | 73.53 |

No collapse: the predictor's output is *higher* rank than its input, cross-video cosine is 0.063,
and no dimension has zero variance.

**But the spec's "eff_rank ≥ 37" threshold was mis-calibrated and would have been wrong either
way.** It was derived from RUN-4's 73.53 on VGGSound. On Epic-Kitchens the *frozen, uncollapsed*
representation scores 26.63 — because the corpus is one narrow domain (kitchens) rather than 309
diverse classes. A threshold of 37 would have failed a representation that is working correctly.
**Effective-rank thresholds are corpus-specific and must be set against the frozen baseline on
the same corpus.** Carried into joint RUN-5.

## 5. What this does and does not establish

**Does:**
* The predictive information RUN-5 is looking for is **not latent in RUN-4's representation** in
  a form a frozen-representation predictor can reach — under this objective, on this corpus.
* P3.2's core finding **replicates on an independent corpus**: persistence is symmetric, and
  copying beats every learned map.

**Does not:**
* **Does not show that joint predictive training would fail.** The pilot only asks whether the
  information is *already there*. Joint training asks whether it can be *put* there, which is a
  different question and remains open.
* **Does not show the objective is right.** §3's diagnosis suggests the negatives may be badly
  chosen, which would make this a test of a flawed objective rather than of the hypothesis.
* **Does not rule out the split as a factor.** The participant-held-out split is harsher than
  anything RUN-4 was measured on. The video-held-out comparison (spec §3) has **not** been run.

## 6. Recommendation — do not launch joint RUN-5 yet

The spec says stop on failure, and that is the right call: joint RUN-5 is the expensive step, and
launching it now would spend it on an objective whose pilot says the signal is absent *and* whose
diagnosis says the objective may be misspecified.

**Two cheap exploratory runs would sharpen the decision.** Both are minutes on a free GPU, both
must be reported as exploratory, and neither can be presented as the pre-registered result:

1. **Same-video temporally-distant negatives** (§3's diagnosis). If the pilot beats IDENTITY
   under hard negatives, the objective was wrong and joint RUN-5 becomes worth running with the
   corrected objective. If it still loses, the conclusion is much stronger.
2. **Video-held-out split**, as the spec already anticipated. The gap between splits measures how
   much of the signal is scene identity, which is a result in itself.

**A third option is to accept the negative and write it up.** Combined with P3.2, "ordinary
multimodal alignment produces no forward information, and none can be extracted post hoc from the
frozen representation" is a clean, defensible, publishable pair of results.

## 7. Provenance

| item | value |
|---|---|
| representation | RUN-4 `step18000`, sha256 `27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4`, frozen |
| corpus | Epic-Kitchens, 648 videos, 308,897 windows, 1 s stride |
| split | participant-held-out — 478 train / 170 eval videos, held-out kitchens `P01 P06 P11 P16 P21 P26 P31`, **0 kitchen overlap** |
| training pairs | 239,136 at Δ = 10 s |
| predictor | LayerNorm → 1024→2048 GELU → 2048→1024, ~4.2M params |
| objective | InfoNCE, τ = 0.05, batch 256, one pair per video per batch, AdamW 3e-4 cosine, 8,000 steps |
| eval | 10 s-spaced within-file grid (zero input overlap), query's own window excluded |
| seeds | 0, 1, 2 |
| script / artifact | `scripts/temporal_probe/p5_pilot.py`, `docs/artifacts/temporal_probe/p5_pilot.json` |

**Three defects the controls caught before they could corrupt this result**, all recorded because
each would have produced a *better-looking* and wrong number: the falsifier partner was the same
participant 95 % of the time; chance was computed as `1/mean(G)` instead of the query-weighted
`mean(1/G)`; and the backward control was the forward-trained predictor run backwards, which
would have measured our training direction rather than the data.
