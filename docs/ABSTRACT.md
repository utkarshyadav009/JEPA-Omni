# ICLR abstract — drafted from `CANONICAL_NUMBERS.md` and `PAPER_ASSETS.md`

**Every number below traces to `CANONICAL_NUMBERS.md`. Nothing is drawn from
`ICLR_RESULTS.md` §15, which is stale in six rows (E-15).**

Two framings are given because the choice is the author's, not mine. **A** leads with the
measurement findings; **B** leads with the system. The evidence supports A more strongly —
the system number is real but its headline comparison is non-equivalent (§2.1), whereas the
leak, the contamination correction and the fusion result are all fully controlled.

---

## A — measurement-first (recommended), 200 words

> Audio-visual retrieval benchmarks are sensitive to defects that a top-1 metric does not
> reveal. We report three. First, a **padding-derived shared-nuisance leak**: batching pads the
> audio stream to the longest clip present, injecting a per-clip quantity into both embeddings
> that retrieval can match on instead of audio-visual correspondence. On a held-out 1,545-clip
> VGGSound gallery this inflates R@1 by ≈24 points (53.27→29.90 v→a). We isolate the mechanism
> with three controls, and show the cure is **length normalisation, not masking**: fixing the
> audio sequence to a constant length raises held-out R@1 to **41.77 (v→a) / 41.88 (a→v)** and
> roughly doubles the representation's effective rank (74.26 vs 37.72), while masking adds
> nothing once length is fixed. Second, **gallery contamination**: a gallery overlapping the
> training corpus inflates R@1 by +9.58 and R@5 by +17.80: reporting R@1 alone inverts the
> apparent size of the effect, because the contaminated gallery saturates. Third, we test
> whether the fused representation is a *world state*. Across four probes it is not: predicting
> its future is no easier than predicting its past (gap −0.04±0.15). Directional information
> exists **before** fusion and is attenuated by it; an explicitly predictive fusion objective
> recovers it transiently, then trades it back for retrieval quality (r=−0.75).

## B — system-first, 185 words

> We present an audio-visual predictor that fuses frozen V-JEPA2 and WavJEPA features into a
> compact scene representation, reaching **41.77 (v→a) / 41.88 (a→v) R@1** on a held-out
> 1,545-clip VGGSound gallery with 155.9M trainable parameters. The result rests on a
> correction: we identify a **padding-derived shared-nuisance leak** that inflated our own
> previously circulated number by ≈24 points, isolate it with three controls, and show the cure
> is **length normalisation rather than masking**. We further quantify **gallery contamination**
> at +9.58 R@1 / +17.80 R@5, and show that reporting R@1 alone inverts the apparent size of the
> effect. Finally we ask whether the fused representation encodes temporal dynamics. It does
> not: across four probes, predicting the future state is no easier than predicting the past
> (gap −0.04±0.15), and copying the current state outperforms every learned forward map.
> Capacity-controlled probes localise the loss to the **fusion**: directional information is
> present in the pre-fusion features (backward prediction R²=−0.0009, exactly zero) and largely
> absent afterwards. Training the fusion with an explicit predictive objective recovers that
> information transiently but trades it against retrieval quality, yielding a Pareto frontier
> rather than a gain.

---

## Claim ledger — every number, its source, and the caveat that must travel

| # | claim in the abstract | value | source | caveat |
|---|---|---|---|---|
| 1 | system R@1 | **41.77 (v→a) / 41.88 (a→v)** | CN §1.1, RUN-4 `step18000`, n=1,545, 3 seeds, range 0.06 | always write the direction (§1.2). `41.35/41.68` is `step20000`, a different checkpoint |
| 2 | leak magnitude | 53.27 → **29.90** (v→a), ≈24 pts | CN §1, E-1 | this is RUN-2 corrected, **not** the system number — do not conflate rows 1 and 2 |
| 3 | leak mechanism | **length normalisation, not masking** | CN §7.2, P2.2 control | the mask-off control is within noise; saying "we fixed it by masking" is wrong |
| 4 | effective rank | **74.26 vs 37.72** | CN §7.1, full-gallery, corrected harness | ≈2×, not 6×. The in-training ~12.5 figure is withdrawn |
| 5 | contamination | **+9.58 R@1 / +17.80 R@5** | CN §3, E-13 primary | **must quote R@1 and R@5 together.** "The effect shrank 40%" is a false summary |
| 6 | forward−backward gap | **−0.04 ± 0.15** (RUN-4) | `FORWARD_INFORMATION_PROBE.md` | pre-registered threshold was ≥2.0 at ≥3×SE; IDENTITY's artifact floor is ≈0.6 |
| 7 | pre-fusion directionality | forward 0.0446, backward **−0.0009** | `FUSION_BOTTLENECK.md` §2 | capacity-matched at 768 dims; the signal is real but **small** (R² 0.045) |
| 8 | fusion trade-off | **r = −0.75** | `RUN5_DECAY_ANALYSIS.md` | six checkpoints of one run; `step1000` clears every mechanism criterion and retrieves at 9.26 |
| 9 | trainable parameters | **155.9M** | CN §2 | quote as a fact about our model **only** — never as efficiency *relative to* a baseline (§2.1) |

## Sentences that must never appear

Each is blocked by a specific audited finding, not by caution.

| forbidden | why |
|---|---|
| "beats ImageBind" / "outperforms ImageBind" | **CN §2.1.** We train on 197k VGGSound clips; ImageBind has never seen VGGSound. The gallery is held out at the *clip* level, not the *distribution* level, so the ~12-point margin is in-domain vs zero-shot and confounds representation quality with domain adaptation |
| "state of the art" | same; no equivalent-footing comparison exists |
| "X× more parameter-efficient than …" | parameter efficiency is only meaningful between models measured on the same footing |
| "world model" / "world state" | retired across four probes (`FORWARD_INFORMATION_PROBE.md`). Use **audio-visual scene representation** |
| "predictive representation" | `lam_pred = 0.0` in every measured run; RUN-5's predictive objective **failed** its pre-registered gate |
| "contamination shrank by 40%" | R@1 shrinks only because the contaminated gallery saturates; R@5 *grows*. The framing inverts the finding |
| "still improving at 20,000 steps" | retracted — R@1 is flat from step 16,000 (`R1_SATURATION.md`) |
| any Ego4D number without E-8 | 27.60 / 27.00 is **permanently unreproducible**; if the caveat cannot travel in the table, cut the row |

## Not in either draft, and deliberately

* **Ego4D transfer (27.60 / 27.00).** Real, but unreproducible (E-8). An abstract cannot carry
  that caveat; the body can.
* **Query-predictor results (0.7372 / 0.4888).** Clean (E-6) but **RUN-2-based**, with no
  downstream retrain planned. Putting them beside a RUN-4 headline implies they sit on it.
* **RUN-5's retrieval (42.59 / 42.14).** Higher than RUN-4, but from a run whose mechanism gate
  **failed**; leading with it would advertise a failed experiment as a system improvement. The
  margin is also under 1 point against a 0.13 seed range.
* **AudioSet probe, AVE feasibility, latency progression.** Body material.

## Status

**Draft. Not submitted, not reviewed.** Numbers are canonical as of 2026-09-17; the 15 errata
remain **proposed, not applied**, so any figure quoted from a published table still needs the
corresponding entry approved first.
