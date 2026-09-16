# RUN-5 v2 full run — mechanism gate FAILS. The pilot's effect did not survive training.

**Verdict: FAIL on M1, M2 and M3.** The full 20,000-step run retains **9 %** of the mechanism
effect the 3,000-step pilot showed. The predictive term's influence on the fusion is
**transient**: it shapes `W` early and is washed out by continued training.

**I recommended this run on an inference that turned out to be wrong.** I argued the mechanism
was "demonstrably working and M2 was still improving", so more steps should clear 2.00×. M2 went
the other way — 1.81× at 3,000 steps, **1.30×** at 20,000.

---

## 1. The result

| | RUN-4 | pilot 3k | **full 20k** | required |
|---|---:|---:|---:|---|
| **M1** `W→ΔV` | 0.1315 | 0.2326 | **0.1407** | +0.05 → **FAIL** (+0.0091) |
| **M2** `W→ΔW` fwd/bwd | 1.15× | 1.81× | **1.30×** | ≥2.00× → **FAIL** |
| **M3** `W→V_next` | 0.3320 | 0.4343 | **0.3361** | +0.05 → **FAIL** (+0.0041) |
| **M4** `W→ΔA` | 0.0839 | 0.0902 | 0.0587 | ≤ ΔV gain → PASS |

**Effect surviving from pilot to full run: ΔV 9 %, V_next 4 %.**

**The comparison is valid.** RUN-4's baseline re-measured **identically** in both evaluations
(ΔV 0.1315/0.1315, V_next 0.3320/0.3320, ratio 1.15×/1.15×), so the protocol is stable and the
pilot-vs-full difference is real, not measurement drift.

## 2. The training objective kept improving while the representation stopped

The `future` loss fell monotonically throughout — **1.00 → 0.188 → 0.136 → 0.115**. The head got
steadily better at predicting `ΔV` from `W`, on training participants.

Meanwhile `W` itself became **no more linearly informative about `ΔV`** on held-out participants
(0.2326 → 0.1407). Those two facts together are the finding:

> **The predictor absorbed the task instead of the fusion retaining the information.**

A 4.2M-parameter head has enough capacity to extract a fixed amount of signal from a `W` that is
drifting toward whatever the AV objective prefers. Early in training the fusion is still plastic
and the predictive gradient moves it; later, the head can satisfy the loss on its own, the
gradient reaching the fusion weakens, and the AV term reasserts itself.

**This is a hypothesis consistent with the numbers, not a demonstrated mechanism.** §5 names the
experiment that would test it.

## 3. What did survive

* **M4 passes, and more cleanly than in the pilot.** `ΔA` *fell* (0.0839 → 0.0587) while `ΔV`
  rose slightly. Whatever the objective did, it was visual-specific, not a generic lift.
* **`ΔW` backward fell in both runs** (0.1958 → 0.1845 pilot, → 0.1884 full). The backward
  direction is consistently suppressed; it is the forward gain that fails to persist.
* **O5 non-regression passes comfortably.** Selected `step20000` scores **42.59 v→a / 42.14 a→v**
  against RUN-4's 41.77 / 41.88, with effective rank 78.24 vs 73.53. The predictive term costs
  nothing in AV retrieval.
* **RUN-5 did not plateau the way RUN-4 did.** `R1_SATURATION.md` recorded RUN-4 flat from step
  16,000; RUN-5 climbs 41.27 → 42.59 over its last 4,000 steps with effective rank rising
  monotonically. One run, so this is an observation, not a claim.

## 4. What this does and does not establish

**Does:**
* A predictive gradient into the fusion **can** make `W` retain directional information — the
  pilot showed 70 % of the deficit recovered, and that measurement stands.
* Under this objective, at this λ, that effect **does not persist to convergence**.
* The failure is not a capacity, negative-construction, split, or retrieval-metric artifact; all
  of those were tested and excluded earlier.

**Does not:**
* **Does not show predictive fusion objectives cannot work.** It shows *this* formulation —
  a fixed-λ MSE term through a 4.2M head against a frozen `ΔV` target — does not hold its effect.
* **Does not retroactively invalidate the pilot.** Both measurements are correct; they describe
  different points in training.
* **Does not make the Case C diagnosis wrong.** The causal test passed at 3,000 steps. What
  failed is making the change *stick*.

## 5. Recommendation — stop, per the pre-registered stop condition

`RUN5_SPEC_V2.md` §7: *"M1–M3 fail → The objective does not retain what it was designed to
retain. **Stop.** Do not enlarge `P_φ`, re-tune λ_future, or change the target. Report as a
negative."* All three failed. **I am stopping, and I have not run the outcome tier**, which the
stop condition makes moot.

The honest write-up is now a three-part result, and it is a good one:

1. Ordinary multimodal alignment produces no directional forward information (P3.2).
2. The information exists pre-fusion and is attenuated by the fusion (P6/P7, capacity-controlled).
3. A predictive gradient into the fusion recovers 70 % of it — **transiently**. The effect decays
   to 9 % by convergence as the predictor absorbs the task.

**One cheap diagnostic is worth considering before closing**, and it is diagnosis rather than
rescue: score M1–M3 at `step{1000,3000,6000,10000,15000,20000}` of *this same run*. That turns
"the effect decayed" into a measured decay curve and directly tests the absorption hypothesis in
§2. It trains nothing and re-uses existing checkpoints.

Anything beyond that — freezing the head, scheduling λ, a bottlenecked predictor — would be
tuning to rescue the hypothesis, and is out of scope under the stop condition.

## 6. Provenance

| item | value |
|---|---|
| RUN-5 full | `checkpoints/RUN-5/full/`, 20,000 steps, exit 0, 12.15 h, 0 NaN |
| selected | `step20000` by held-out R@1 over all 20 tagged checkpoints, 3 seeds (E-14; `best.pt` ignored) |
| config | AV batch 50 (200×200 negatives, RUN-4 parity), `lam_future=1.0`, future batch 16, Δ=10 s |
| RUN-4 baseline | `step18000`, sha256 `27b33c8c…`, re-measured on the identical path |
| artifacts | `p8_mechanism_full.json`, `p8_mechanism.json` (pilot), `r5_selection.json` |
