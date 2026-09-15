# RUN-5 SPECIFICATION v2 — a predictive *fusion* objective

**Status: specification only. No training code written or modified.**
**Supersedes `RUN5_SPEC.md`, whose pilot FAILED and is not being rescued.**

---

## 0. Why this is a new experiment, not a retry

| | RUN-5 v1 (failed) | RUN-5 v2 (this) |
|---|---|---|
| where the predictive loss sits | on a **frozen** `W` | **inside `f_θ`**, before `W` exists |
| what it can change | the predictor only | **what the fusion keeps** |
| what it could see | only what fusion had already discarded | the pre-fusion features themselves |

The v1 pilot could not have worked given what we now know: `FUSION_BOTTLENECK.md` localises the
attenuation to the fusion, and v1's gradient never reached it. **The failure is not evidence
against v2; it is the evidence that motivates v2.**

**What the diagnosis licenses, and what it does not.** It is an *observational* probe comparison.
It shows pre-fusion features carry directional information that `W` largely does not retain. It
does **not** prove the fusion causes the loss. **RUN-5 v2 is itself the causal test**: if a
predictive gradient into `f_θ` recovers the deficit, the localisation was right.

---

## 1. Architecture and where the gradient goes

```
VIDEO + AUDIO
      │
      ▼
V-JEPA2 ViT-L  +  WavJEPA base/nat          FROZEN, always
      │
      ▼
  f_θ = AVJepaPredictor.world_state()        ◄── the fusion. THE PREDICTIVE
      _embed → 8-layer backbone → attentive       GRADIENT MUST REACH HERE
      pool (pool_query)  [av_jepa_predictor.py:219]
      │
      ├────────────►  L_AV       InfoNCE, unchanged from RUN-4   (non-regression)
      ├────────────►  L_future   NEW                              (§2)
      └────────────►  L_SIGReg   anti-collapse                    (§5)
      │
      ▼
      W  (1024, un-normalised)
```

`world_state()` is grad-enabled by construction (it exists so SIGReg can backprop). The new term
attaches to the **same call**. Nothing in the existing loss graph is restructured.

**`lam_pred` stays off.** Code inspection (`RUN5_SPEC.md` §0) established it is *within-window
masked cross-modal* prediction, not temporal. The new term is named **`lam_future`** so the two
can never be conflated in a config, log line, or table.

| | |
|---|---|
| **trainable** | `f_θ` (backbone + `pool_query` + `in_proj` + `temporal_emb`), the projections, and `P_φ` — RUN-4's trainable set (155.9M) plus a small head |
| **frozen** | V-JEPA2 and both WavJEPA towers. Always. The cached features exist so this stays true. |

## 2. The predictive objective — target, and why this target

**Primary target: `ΔV = vision_mean(t+Δ) − vision_mean(t)`**, the *change* in frozen V-JEPA2
features. `P_φ: W_t → ΔV̂`, a 2-layer MLP (1024→2048→1024).

```
L = L_AV  +  λ_future · L_future  +  λ_SIGReg · L_SIGReg
L_future = MSE( P_φ(W_t),  ΔV )          ΔV computed from the FROZEN encoder
```

**Three reasons this target and not `ΔW`, each from a measurement:**

1. **The directional signal is visual.** Vision→ΔW predicts forward at R² 0.0446 and backward at
   **−0.0009 — exactly zero**. Audio→ΔW is symmetric (1.34×). The arrow of time is in vision.
2. **`W` loses exactly this.** `dV` from pre-fusion V is 0.2758; from `W` it is 0.1330 — a
   **0.143 deficit**. The term penalises the measured gap directly.
3. **`ΔW` is the wrong target.** `W` is already *best* at predicting `ΔW` (0.2299) and *least*
   directional about it (1.14×). Training on it would reinforce the magnitude-like, time-symmetric
   component that is the problem.

**The target is frozen, which removes a whole failure mode.** `ΔV` comes from a frozen encoder, so
the predictive term cannot be satisfied by collapse, and no stop-gradient/EMA machinery is needed.
This side-steps the unresolved LeJEPA-vs-V-JEPA2 disagreement (`PREDICTION_LITERATURE.md` §2)
rather than betting on either side. It also means **MSE is safe here**: `ΔV` lives in a different
space from `W`, so "copy the input" is not a solution — the identity-map problem that forced
InfoNCE in v1 does not arise.

**Secondary (reported, not optimised):** `ΔA`. Audio is symmetric, so it is a control — if the
directional gain appears on `ΔA` too, the effect is not the visual mechanism claimed.

## 3. Temporal horizon and receptive-field reasoning

**Δ = 10 s primary; 20 s secondary.**

Receptive field, **measured** (`p4_receptive_field.json`):

| encoder | measured RF | evidence |
|---|---|---|
| V-JEPA2 ViT-L | **full 10 s window** | no zero cell in the influence matrix; earliest output responds to the last input slice at 0.36 of self-response |
| WavJEPA base/nat | ≈2.25 s median, 4.25 s max | **76 %** of the influence matrix exactly zero |

`W` fuses both and vision spans the whole window, so a fused target needs **Δ ≥ window length =
10 s**. Δ < 10 s shares raw video input with the query and is not prediction at any stride.

This is the same principle as V-JEPA 2's EK-100 anticipation at τ_a = 1 s, which gets
input/target disjointness by ending the clip before the target; we get it by requiring
Δ ≥ window length. **A shorter query window would permit a smaller Δ and is reachable from the
kept token features — explicitly out of scope here.**

## 4. Data, splits, negatives

| corpus | windows | role | why |
|---|---:|---|---|
| VGGSound | 199,007 | **L_AV only** | 10 s clips, no temporal pairs exist |
| Epic-Kitchens | 154,610 @2 s | **L_AV + L_future** | 151,388 pairs at Δ=10 s (5 windows) |
| Ego4D | 134,491 @10 s | **L_future**, held back | out-of-corpus generalisation check, not training |

**Batch composition:** each step draws an AV batch from VGGSound+EK and a future batch from EK.
`L_future` is computed only on the future batch. Mixing ratio fixed at **1:1** and **not swept**.

**Split: participant-held-out on EK** — 478 train / 170 eval videos, 7 unseen kitchens, **0
kitchen overlap**. Chosen on evidence, not preference: a video-level split puts **31 of 34
kitchens on both sides**, and the participant split costs only **0.24 R@1** on the persistence
baseline. Video-held-out is run as a **secondary diagnostic**; the gap measures how much signal is
scene identity.

**Negative construction (for `L_AV` only, unchanged from RUN-4):** InfoNCE, τ = 0.05, 192
negatives, one clip per item. `L_future` is a regression and has no negatives — which removes the
negative-construction axis that consumed six grid cells in v1 and explained nothing.

## 5. Anti-collapse

**SIGReg on `W_t`, raised to LeJEPA's defaults: λ = 0.05, 1024 slices** (RUN-4 ran 0.03 / 256).
In RUN-4 this did not matter — frozen targets made collapse impossible and SIGReg only shaped
geometry, as `models/sigreg.py` itself documents. Here `W` is shaped by a predictive gradient for
the first time, so running the anti-collapse term *below* the paper's defaults is an unnecessary
risk. Stop-gradient is **not** used and **not** needed (§2).

**Collapse diagnostics, every eval — any one failing aborts the run:**

| metric | threshold | source of the threshold |
|---|---|---|
| eff_rank of `W` on the EK eval set | **≥ 26.63 × 0.8 = 21.3** | the **frozen RUN-4 baseline on this corpus** |
| mean cos(`W_t`, `W_{t+10}`) | ≤ 0.90 | RUN-4: 0.79 |
| mean cross-video cos | ≤ 0.30 | RUN-4 floor: 0.063 |
| min per-dimension std | > 0 | — |

**The first threshold is corpus-calibrated on purpose.** v1 used `eff_rank ≥ 37`, derived from
VGGSound's 73.53. The frozen, *uncollapsed* representation scores **26.63** on Epic-Kitchens, so
that threshold would have failed a working representation.

## 6. Pre-registered success criteria — fixed before any v2 number exists

Two tiers, because **the mechanism can succeed while the outcome fails, and that is itself a
result.** Both are declared now so neither can be selected after the fact.

### 6.1 MECHANISM — did the objective do what it is designed to do?

Linear ridge probe, identical protocol to `p7_fusion_diagnosis.py`, capacity-matched 768 dims.

| # | criterion | RUN-4 | required |
|---|---|---:|---|
| M1 | `W → dV` forward R² | 0.1330 | **≥ 0.183** (+0.05, i.e. ≥⅓ of the way to the 0.2758 pre-fusion ceiling) |
| M2 | `W → dW` forward/backward ratio | 1.14× | **≥ 2.00×** |
| M3 | `W → V_next` forward R² | 0.3297 | ≥ 0.380 (+0.05) |
| M4 | `ΔA` control | — | the `dA` gain must **not** exceed the `dV` gain — otherwise the effect is not the visual mechanism claimed |

### 6.2 OUTCOME — did it convert to retrieval?

Within-file forward retrieval, Δ = 10 s, 10 s-spaced grid, 3 seeds — the **identical** protocol
the v1 pilot failed.

| # | criterion | required |
|---|---|---|
| O1 | beat PERSISTENCE (IDENTITY, 16.76) | **≥ +2.0** and ≥ 3× SE |
| O2 | forward − backward, **separately trained** backward predictor | **≥ +2.0** and ≥ 3× SE, sign held at Δ=20 s |
| O3 | shuffle control | at chance (< 1.5× ) |
| O4 | mismatched-file falsifier — **different participant, random target index** | at chance (< 1.5×) |
| O5 | **NON-REGRESSION**: VGGSound n=1,545 R@1 | within **2.0** of **41.77 (v→a) / 41.88 (a→v)** |

O2's backward control **must be a separately trained backward predictor.** In v1 the naive
version (forward model run backwards) gave **+3.082 ± 0.154, a clean PASS at 20× SE**, while the
correct control gave +1.261, a FAIL. That artifact was large and highly significant, not marginal.

## 7. Stop conditions

| outcome | action |
|---|---|
| **M1–M3 fail** | The objective does not retain what it was designed to retain. **Stop.** Do not enlarge `P_φ`, re-tune λ_future, or change the target. Report as a negative. |
| **M4 fails** (ΔA gains ≥ ΔV gains) | The mechanism is not the visual one claimed. **Stop and re-diagnose** before any further training. |
| **Mechanism passes, outcome fails** | **A result, not a failure to fix.** "Directional information can be retained through fusion, but does not convert to retrieval gains against a persistence baseline." Report it; do not tune to rescue. §5 of `FUSION_BOTTLENECK.md` predicts this is possible — the signal is only R² 0.045. |
| **O5 breached** | Abort. A predictive gain paid for with AV retrieval collapse is not a result. |
| **Any collapse metric fails** | Abort immediately. |
| **Both tiers pass** | RUN-5 succeeds. Even then, see §9. |

**Protocol freeze.** Δ, exclusion rules, splits, baselines, thresholds and λ_future are frozen
before results are inspected. Any post-hoc change is a new exploratory experiment, reported as
such, never presented as the pre-registered outcome.

## 8. Execution order

1. **Mechanism pilot first** — 3,000 steps, evaluate §6.1 only. Cheap, and it tests the causal
   claim before the full budget. If M1–M3 fail here, stop.
2. Full run, 20,000 steps, monitored at launch, RUN-4 checkpoints untouched (`RUN-5/` namespace).
3. Evaluate §6.1 and §6.2 on completion; post-hoc checkpoint selection by held-out R@1 (E-14).
4. Ego4D out-of-corpus check.
5. Preserve, back up, compare against the immutable RUN-4 baseline.

Every RUN-5 result records: RUN-4 checkpoint sha256 `27b33c8c…`, RUN-5 code commit, config,
dataset version, stride, Δ, sample counts, seed, RUN-5 checkpoint sha256, harness version.

## 9. What success would and would not mean

**Would:** that directional predictive information can be retained through the fusion, and that
the bottleneck diagnosis was causally correct.

**Would NOT make `W` a world model.** A passive predictor `W_t → W_{t+Δ}` is evidence of temporal
prediction. An embodied world model is more naturally `P(W_{t+Δ} | W_t, a_t)`. **No action
information exists anywhere in this design.** Action conditioning is a separate research
direction and must not be retrofitted onto this experiment or its vocabulary. The naming
discipline from `FORWARD_INFORMATION_PROBE.md` stands: the object is an audio-visual scene
representation until an experiment earns it a stronger name.

**Estimated cost:** ~34 h on one free GPU for the full run (RUN-4 took 30 h; the added terms are
small relative to the fusion forward pass). The mechanism pilot is ~1 h.
