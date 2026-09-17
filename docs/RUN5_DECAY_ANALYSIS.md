# Decay analysis — the mechanism effect is real, and it is traded away for retrieval

**The curve is clean, not noisy.** Both mechanism metrics decline monotonically across six
checkpoints of a **single run**, while VGGSound retrieval rises monotonically over the same
checkpoints. Pearson **r = −0.75** between retrieval R@1 and `W→ΔV`.

**The headline is sharper than the "head absorption" story I floated, and less comfortable:**

> The predictive objective and the audio-visual objective are **in tension**. Under this
> formulation the representation **trades directional future information for retrieval quality**
> as training proceeds. The mechanism effect is strongest exactly where the model is least useful.

---

## 1. The curve

Six checkpoints, one run, identical config, seed, data order and batch size — only training time
varies. RUN-4's baseline was encoded once and re-measured identically for the **third** time
(ΔV 0.1315, V_next 0.3320, ratio 1.15×).

| step | `W→ΔV` | `W→V_next` | `ΔW` fwd/bwd | `W→ΔA` | VGGSound v→a R@1 |
|---:|---:|---:|---:|---:|---:|
| **1000** | **0.2231** | **0.4311** | **2.19×** | 0.1042 | **9.26** |
| 3000 | 0.2273 | 0.4265 | 1.67× | 0.0781 | 21.94 |
| 6000 | 0.2257 | 0.4229 | 1.55× | 0.0729 | 29.69 |
| 10000 | 0.2116 | 0.4063 | 1.40× | 0.0570 | 35.64 |
| 15000 | 0.1644 | 0.3550 | 1.33× | 0.0513 | 40.15 |
| **20000** | **0.1407** | **0.3361** | **1.30×** | 0.0587 | **42.59** |
| *RUN-4* | *0.1315* | *0.3320* | *1.15×* | *0.0839* | *41.77* |

```
W->dV                                   VGGSound R@1
0.23 ●──●──●                                    ●  42.6
          ╲●                              ●
0.16        ╲●                       ●
0.14          ●                 ●                 
0.13 ····················· RUN-4      ●
     1k 3k 6k 10k 15k 20k        ●  9.3
                                 1k 3k 6k 10k 15k 20k
```

**Two processes on two timescales, not one decay:**

* **Directionality erodes immediately.** The ratio falls from step 1,000 and is strictly
  monotone — 2.19 → 1.67 → 1.55 → 1.40 → 1.33 → 1.30 — decelerating toward RUN-4's 1.15×. The
  largest single drop is the very first interval.
* **Information survives longer, then goes.** `ΔV` is *flat* through step 6,000 (0.2231, 0.2273,
  0.2257 — spread 0.004), then declines, with most of the loss between 10k and 20k.

## 2. The finding: `step1000` passes the mechanism gate and is a useless model

| criterion | `step1000` | required | |
|---|---|---|---|
| M1 `ΔV` gain | +0.0916 | ≥ +0.050 | **PASS** |
| M2 ratio | **2.19×** | ≥ 2.00× | **PASS** |
| M3 `V_next` gain | +0.0991 | ≥ +0.050 | **PASS** |
| M4 `ΔA` gain | +0.0203 | ≤ ΔV gain | **PASS** |
| **O5 VGGSound R@1** | **9.26** | within 2.0 of 41.77 | **FAIL by 30.5 points** |

**All four mechanism criteria pass at step 1,000 — the only checkpoint that ever clears M2 — and
that checkpoint retrieves at 9.26 R@1 against RUN-4's 41.77.**

**This is not a pass and must never be reported as one.** The gate was evaluated on the
checkpoint selected by held-out R@1 (`step20000`, E-14). Selecting `step1000` *because* it clears
the mechanism gate is precisely the post-hoc selection the protocol exists to prevent — the same
move that would have converted the v1 pilot's +3.082 backward-control artifact into a "PASS".

What it establishes is a **Pareto frontier**: under this objective one can have directional
temporal structure **or** audio-visual retrieval quality, but not both.

## 3. Two explanations, and I cannot separate them

**(a) Head absorption.** The 4.2M predictor head learns to extract the signal itself, the
gradient reaching the fusion weakens, and the AV term reasserts itself. Consistent with the
training `future` loss falling monotonically (1.00 → 0.188 → 0.136 → 0.115) while held-out `ΔV`
falls.

**(b) Objective tension.** The AV contrastive loss actively pulls `W` toward a retrieval-optimal
solution that is less predictive. Consistent with r = −0.75 against retrieval.

**The data support (b) at least as well as (a), and one observation sits awkwardly with (a):**
directionality begins eroding at step 1,000 while information stays flat to step 6,000. Pure head
absorption predicts both should track the head's convergence together. They do not.

**I previously offered (a) as the explanation. That was premature** — it was fitted to two points
(pilot vs full) before this curve existed. Both remain live.

**The discriminating experiment** (not run, and out of scope under the stop condition): freeze
`P_φ` after step 1,000 and continue training. If `ΔV` still decays, absorption is not the
mechanism and tension is. A λ_future sweep would test the same thing from the other side.

## 4. Cross-checks that hold

* **Batch size was not a confound.** Full-run `step3000` (batch 50): ΔV 0.2273, ratio 1.67×.
  Pilot `step3000` (batch 24): ΔV 0.2326, ratio 1.81×. The pilot-vs-full difference was training
  time, not configuration.
* **`step20000` reproduces exactly** between this run and the independent `p8` evaluation
  (ΔV 0.1407, ratio 1.30×).
* **RUN-4's baseline is stable to four decimal places across three independent evaluations.**
* **M4 holds throughout**: `ΔA` sits below RUN-4's 0.0839 at every checkpoint from 3,000 on. The
  effect was visual-specific at every point in training, never a generic lift.

## 5. What this changes

**It does not reopen the gate.** `RUN5_SPEC_V2.md`'s stop condition stands: M1–M3 failed on the
selected checkpoint, and this analysis was diagnosis, not rescue.

**It upgrades the negative result.** "The effect decayed" was one observation from two runs. It
is now a measured monotone curve across six checkpoints of one run, anti-correlated with
retrieval at r = −0.75, with a named Pareto trade-off and a stated discriminating experiment.
The four-part story is:

1. Ordinary multimodal alignment produces no directional forward information (P3.2).
2. That information exists **pre-fusion** and is attenuated by the fusion (P6/P7, capacity-controlled).
3. A predictive gradient **into** the fusion recovers it — 70 % at 3,000 steps, and at 1,000 steps
   enough to clear every mechanism criterion.
4. **It cannot be kept.** The recovery is traded away for retrieval quality as training proceeds,
   monotonically, to near-baseline by convergence.

Point 4 is a stronger and more useful claim than "RUN-5 failed", and it is the one worth writing up.

## 6. Provenance

| item | value |
|---|---|
| checkpoints | `checkpoints/RUN-5/full/step{1000,3000,6000,10000,15000,20000}.pt` |
| baseline | RUN-4 `step18000`, sha256 `27b33c8c…`, encoded once and reused |
| probe | ridge, λ on a held-out train slice, capacity-matched — identical to `p7`/`p8` |
| split | participant-held-out, 7 unseen kitchens, 119,686 train / 31,702 eval pairs |
| retrieval column | `r5_selection.json`, held-out 1,545 gallery, 3 seeds |
| script / artifact | `scripts/temporal_probe/p10_decay.py`, `p10_decay.json` |
