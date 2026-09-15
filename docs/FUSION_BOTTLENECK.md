# Fusion bottleneck diagnosis — what `W` preserves, and what it discards

**One-line finding.** The directional future signal is **visual**, it is **small**, and it is
**progressively attenuated** as vision is mixed with audio and then compressed into `W`.

**Wording discipline.** Everything below is an *observational* probe comparison. It **localises**
the attenuation to the fusion; it does not prove the fusion *causes* it, because `W` and the
pre-fusion features are not interchangeable objects — only a trained intervention could establish
causation. Language throughout is "consistent with" and "localises to", never "the fusion destroys".

---

## 1. Method

Four inputs, **all PCA-reduced to a common 768 dims** (the natural minimum, `ambient_mean`'s
width) so capacity cannot explain any difference. Same predeclared metrics as P6: R² on held-out
participants about the train mean; forward, backward, input-permuted null, persistence. Ridge
only — the MLP probe is inconclusive (`RUN5_INFORMATION_PROBES.md` §3.4) and **was not tuned
further**.

| input | raw dims | contents |
|---|---:|---|
| **V** | 1024 | `vision_mean(t)` — V-JEPA2 |
| **A** | 768 | `ambient_mean(t)` — WavJEPA |
| **VA** | 1792 | both concatenated |
| **W** | 1024 | the fused scene representation |

119,686 train / 6,409 eval pairs, Δ = 10 s (zero input overlap), participant-held-out,
7 unseen kitchens.

**A null defect found and fixed, for the second time.** `dV = V(t+10) − V(t)` derives from the
*input* `V(t)`, so computing it inside the input-permuted null changed the **target**
distribution and inflated `SS_tot` — the null *outscored* the real pairing (`dV:V` 0.346 vs
0.276). Corrected by taking null targets from the unshuffled pack. All change-target nulls now
sit at ≈ 0.000. The same class of defect as the P6 `dW` null; worth noting that it recurred in a
new place and was again caught only because a null scoring above its own signal is impossible.

## 2. Where the arrow of time lives — predicting ΔW

| input | R²_fwd | R²_bwd | **fwd/bwd** | null |
|---|---:|---:|---:|---:|
| **V** (vision only) | 0.0446 | **−0.0009** | **∞** | −0.001 |
| A (audio only) | 0.0652 | 0.0485 | 1.34× | 0.000 |
| VA (both) | 0.0629 | 0.0127 | **4.97×** | −0.001 |
| **W** (fused) | **0.2299** | 0.2008 | **1.14×** | −0.002 |

**Read the two columns against each other — they disagree, and that is the finding.**

* **Vision alone is perfectly directional.** It predicts forward change at 0.045 and backward
  change at **−0.0009 — exactly zero**. A clean arrow of time.
* **Audio is essentially symmetric** (1.34×). It carries change information but not direction.
* **`W` has by far the largest magnitude (0.230, 5× vision) and the least direction (1.14×).**

So `W` is the *best* predictor of ΔW and the *worst* at knowing which way time runs. Its advantage
is persistence-like — "how much will change" — not "what comes next".

**A monotone attenuation is visible across the pipeline:**

```
V alone      ∞      (backward = exactly 0)
VA           4.97×  (audio dilutes it)
W            1.14×  (fusion attenuates what remains)
```

## 3. What fusion costs on modality-specific futures

| target | best pre-fusion | `W` | **difference** |
|---|---:|---:|---:|
| `V_next` | **0.4877** (V) | 0.3297 | **−0.158** |
| `A_next` | **0.6001** (A) | 0.4739 | **−0.126** |
| `dV` | **0.2758** (V) | 0.1330 | **−0.143** |
| `dA` | **0.2258** (A) | 0.0890 | **−0.137** |

A consistent **0.13–0.16 R²** shortfall, in all four cases, at matched capacity. `W` is a lossy
summary *for the purpose of predicting future modality content*.

**Where `W` wins:** predicting its own future (`W_next` 0.3912 from `W` vs 0.2816 from VA). That
is expected — `W_next` is `W`'s own space — and it is again nearly symmetric (1.06×).

## 4. Is the directional signal visual, audio, or cross-modal?

**Primarily visual, and not cross-modal.**

* Vision→ΔW is purely directional (backward exactly 0); audio→ΔW is not (1.34×).
* Cross-modal prediction is near-zero in both directions: `A_next` from V is 0.026,
  `V_next` from A is 0.029. The two modalities barely predict each other's futures at all.
* `VA` (4.97×) is *less* directional than `V` alone. Concatenating audio dilutes rather than adds.

## 5. Magnitudes — the honest caveat

**The directional signal is real but small.** Vision→ΔW forward is **R² 0.045**: about 4.5 % of
change variance, from a perfectly directional but weak predictor. Preserving it perfectly through
a redesigned fusion would not automatically produce large retrieval gains, and R² on ΔW need not
convert into R@1 against a persistence baseline that is very strong at R@1.

**This is a well-localised effect, not a large one.** Any RUN-5 built on it should be sized and
justified accordingly.

## 6. Classification

**CASE A.** Pre-fusion features contain directional information (vision: backward exactly zero)
that `W` largely does not retain (1.14×), and the modality-future shortfall is consistent at
0.13–0.16 R² across four independent targets. Capacity is controlled. The effect **survives**.

This is consistent with the learned fusion attenuating or discarding directional predictive
structure, and it **localises the bottleneck to the fusion** rather than to the predictor, the
negatives, or the split — all three of which were tested and rejected in `RUN5_PILOT_RESULT.md`.

## 7. Design principles for a redesigned RUN-5 — NOT IMPLEMENTED

**Why this is not the failed pilot.** The pilot attached a predictor to a **frozen** `W`; its
gradient never reached the fusion, so it could only read what the fusion had already discarded.
The proposed design puts the predictive loss **upstream of `W`'s formation**, so it can change
what the fusion keeps. That distinction is the entire point, and it is what makes this a new
experiment rather than a rescue of the old one.

```
VIDEO + AUDIO
      │
      ▼
V-JEPA2 + WavJEPA        (frozen, always)
      │
      ▼
   FUSION  f_θ   ◄── predictive gradient MUST reach here
      │
      ├──────────────►  AV alignment loss          (non-regression)
      ├──────────────►  future objective           (the new term)
      └──────────────►  SIGReg                     (anti-collapse)
      │
      ▼
      W
```

Principles the evidence actually supports:

1. **The predictive gradient must flow into `f_θ`.** Any design where `W` is formed first and
   predicted from second reproduces the pilot.
2. **Consider targeting ΔV, not only ΔW.** §2 and §4 say the directional signal is *visual*, and
   §3 says `W` loses 0.143 R² of `dV`. A term that asks `W` to retain what predicts future
   *vision* targets the measured deficit directly; asking it to predict its own ΔW targets the
   quantity it is already best at and least directional about.
3. **Retain modality-specific information explicitly.** An auxiliary head reconstructing
   `V_next` / `A_next` from `W` would penalise exactly the 0.13–0.16 shortfall in §3.
4. **Anti-collapse is SIGReg's job**, and it becomes load-bearing here for the first time (the
   targets are no longer frozen). LeJEPA's defaults are λ=0.05 and 1024 slices; ours are 0.03 and
   256 — raise them or justify not doing so (`PREDICTION_LITERATURE.md` §3).
5. **Non-regression is mandatory**: VGGSound n=1,545 R@1 within 2.0 of **41.77 (v→a) / 41.88 (a→v)**.
   A predictive gain paid for with retrieval collapse is not a result.
6. **Pre-register**, as before: persistence/IDENTITY as the bar, separately-trained backward
   control, shuffle and mismatched-file falsifier, forward−backward threshold, and explicit stop
   conditions. Every one of those caught a real defect this round.
7. **Effective-rank thresholds must be set against the frozen baseline on the same corpus**
   (26.63 on Epic-Kitchens, not VGGSound's 73.53).

**Not in scope, and not to be retrofitted:** action conditioning. A passive predictor
`W_t → W_{t+Δ}` is evidence of temporal prediction, not an embodied world model, which is more
naturally `P(W_{t+Δ} | W_t, a_t)`. Even a successful RUN-5 does not license calling `W` a world
model.

## 8. Provenance

| item | value |
|---|---|
| representation | RUN-4 `step18000`, sha256 `27b33c8c…`, frozen |
| corpus / split | Epic-Kitchens 648 videos; participant-held-out, 7 unseen kitchens |
| pairs | 119,686 train (2 s lattice) / 6,409 eval (10 s grid), Δ = 10 s |
| capacity match | every input PCA-reduced to 768 dims, fitted on train |
| probe | ridge, λ on a held-out train slice; **no MLP, no sweep** |
| script / artifact | `scripts/temporal_probe/p7_fusion_diagnosis.py`, `p7_fusion_diagnosis.json` |
