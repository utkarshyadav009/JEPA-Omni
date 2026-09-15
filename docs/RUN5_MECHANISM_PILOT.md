# RUN-5 v2 mechanism pilot — gate FAILS on M2. The mechanism itself works.

**Verdict: the pre-registered gate FAILED.** Three of four criteria pass; M2 misses at
**1.81× against a required 2.00×**. Per `RUN5_SPEC_V2.md` §6.1 the gate requires all four, so
**this is a fail and is not to be reported as a pass.**

**But the causal claim the pilot was built to test came back positive**, and that is a separate
statement from the gate.

---

## 1. Results

3,000 steps, `lam_future = 1.0`, Δ = 10 s, participant-held-out (7 unseen kitchens),
119,686 train / 31,702 eval pairs. Both checkpoints run through the identical path.

| | RUN-4 | RUN-5 pilot | gain | required | |
|---|---:|---:|---:|---|---|
| **M1** `W→ΔV` R² | 0.1315 | **0.2326** | **+0.1011** | ≥ +0.050 | **PASS** |
| **M2** `W→ΔW` fwd/bwd | 1.15× | **1.81×** | +0.66× | ≥ 2.00× | **FAIL** |
| **M3** `W→V_next` R² | 0.3320 | **0.4343** | **+0.1023** | ≥ +0.050 | **PASS** |
| **M4** `W→ΔA` R² | 0.0839 | 0.0902 | +0.0063 | ≤ ΔV gain | **PASS** |

### The baseline re-measurement validates the protocol

RUN-4's numbers were **re-measured on the 2 s token grid**, not quoted from the 1 s world-state
files, because RUN-5's `W` can only be recomputed from tokens. The two grids agree closely:

| | spec (1 s grid) | re-measured (2 s grid) | Δ |
|---|---:|---:|---:|
| `W→ΔV` | 0.1330 | 0.1315 | −0.0015 |
| `W→V_next` | 0.3297 | 0.3320 | +0.0023 |
| `W→ΔW` ratio | 1.14× | 1.15× | +0.01× |

So the gains are real and not a grid artifact.

## 2. What passed, and why it matters more than the arithmetic

**The fusion now retains what it was discarding.** `W→ΔV` went 0.1315 → 0.2326 against a
pre-fusion ceiling of **0.2758** (vision alone, capacity-matched, `p7_fusion_diagnosis.json`):

```
RUN-4 W      0.1315  ├──────────────── 70.1% of the gap closed ────────────┤
RUN-5 W      0.2326                                                    ▲
pre-fusion V 0.2758                                                    │ ceiling
```

**This is the causal test of the Case C diagnosis, and it passed.** The probe evidence was
observational — it localised the attenuation to the fusion but could not prove causation.
Changing only the fusion objective recovered **70 %** of the deficit. The fusion was indeed
where the information was being lost.

**M4 confirms the mechanism is the visual one claimed.** ΔA gained +0.0063 against ΔV's +0.1011 —
a 16× difference. The objective did not simply make `W` better at predicting everything; it made
it better at predicting future *vision*, which is exactly where the measurements said the
directional signal lives.

## 3. Why M2 failed, stated precisely

The ratio is a quotient, and both halves moved in the right direction:

| | RUN-4 | RUN-5 | |
|---|---:|---:|---|
| `W→ΔW` **forward** | 0.2258 | **0.3339** | **+0.1081** |
| `W→ΔW` **backward** | 0.1958 | **0.1845** | **−0.0113** |
| ratio | 1.15× | 1.81× | |

**Forward rose by 0.108 while backward fell by 0.011.** That is the signature of directional
learning, not of a uniformly better predictor — a model that had merely got better at everything
would have raised both. The ratio improved by 57 %.

It still missed 2.00×, and **the threshold was fixed in advance precisely so this call could not
be made after seeing the number.** 1.81× is a fail.

## 4. What this does NOT show

* **Not a pass.** The gate required four of four.
* **Not evidence the outcome tier would pass.** §6.2 (beat persistence at retrieval) is
  untested here, and `FUSION_BOTTLENECK.md` §5 warns the directional signal is small in absolute
  terms — R² gains need not convert to R@1 against a strong persistence baseline.
* **The VGGSound R@1 of 13.92 / 13.79 is NOT a regression and must not be read as one.** The
  pilot ran at AV batch **24** (96 negatives) against RUN-4's **50** (200), because RUN-4 already
  used 94.9 of 95 GB and a second graph OOM'd. Fewer negatives lowers the metric mechanically.
  Non-regression (O5) is only meaningful at matched batch size, in the full run.
* **Nothing here makes `W` a world model.** No action information exists in this design.

## 5. The decision this leaves

The spec's stop conditions are specific, and **the literal trigger did not fire**: "M1–M3 fail →
stop" (M1 and M3 passed) and "M4 fails → stop and re-diagnose" (M4 passed). The composite gate
nevertheless fails, because it requires all four.

That tension is recorded rather than resolved unilaterally. Three readings are defensible:

1. **Honour the gate: stop.** 1.81× < 2.00×, pre-registered, no further compute. Publish the
   causal confirmation of Case C, which is a genuine result on its own.
2. **Run the full 20,000 steps.** The pilot used 15 % of the budget and M2 was still improving.
   This must be declared a **new pre-registered run** with M2 ≥ 2.00× unchanged — not a retry
   with a softened threshold.
3. **Re-diagnose M2 first.** The ratio may be the wrong statistic: `ΔW` is `W`'s own space and
   `W` was always best-and-least-directional there, which is exactly why the spec targeted ΔV
   instead. A directionality criterion on ΔV would test the same claim more directly — but
   choosing it now, after seeing M2 fail, would be post-hoc and must be labelled as such.

**No full run launches without an explicit decision.** Option 2 costs ~34 h of GPU.

## 6. Provenance

| item | value |
|---|---|
| RUN-5 pilot | `checkpoints/RUN-5/mechanism_pilot/step3000.pt`, 3,000 steps, exit 0 |
| RUN-4 baseline | `checkpoints/m2_run4_padfix_ta896/step18000.pt`, sha256 `27b33c8c…` |
| config | `lam_future=1.0`, future batch 8, AV batch 24, `lam_sigreg=0.03`, `lam_fusion=1.0` |
| data | EK 2 s token cache, Δ=10 s (offset 5), participant-held-out |
| pairs | 119,686 train / 31,702 eval |
| probe | ridge, λ on a held-out train slice — identical to `p7_fusion_diagnosis.py` |
| script / artifact | `scripts/temporal_probe/p8_mechanism_eval.py`, `p8_mechanism.json` |

**Deviation on record:** AV batch 24 rather than RUN-4's 50, forced by memory. It does not affect
M1–M4, which are probes of `W`, but it makes the pilot's retrieval numbers non-comparable.
