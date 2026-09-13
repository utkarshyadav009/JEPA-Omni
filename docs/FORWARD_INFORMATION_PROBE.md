# Forward-Information Probe (P3.2)

**Question.** Does the M2 fused representation `W(t)` carry information about the *future*
beyond symmetric temporal persistence and scene identity?

**Answer: no.** Under this protocol, for both RUN-2 and RUN-4, predicting `W(t+Δ)` is no easier
than predicting `W(t−Δ)`. The representation exhibits *symmetric persistence*, not forecasting.
This closes the naming question: `W(t)` is an **audio-visual scene representation**, not a
world model.

**Scope.** This is a property of a representation trained by ordinary multimodal alignment
(`lam_pred = 0.0`). It is **not** evidence that an explicitly predictive objective would fail.
See *What this does NOT show*.

---

## 1. The headline — forward minus backward

Within-file micro R@1, ridge map, mean over 3 seeds, Ego4D held-out cache, chance = 2.198 %.

| model | Δ | forward | backward | **gap** | ±SE |
|---|---:|---:|---:|---:|---:|
| RUN-2 `step19000` | 10 s | 9.13 | 8.52 | **+0.61** | 0.40 |
| RUN-2 `step19000` | 20 s | 3.89 | 3.75 | +0.13 | 0.15 |
| RUN-2 `step19000` | 30 s | 2.97 | 3.05 | −0.08 | 0.10 |
| RUN-2 `step19000` | 60 s | 2.10 | 2.43 | −0.32 | 0.13 |
| **RUN-4 `step18000`** | **10 s** | **9.78** | **9.82** | **−0.04** | 0.15 |
| RUN-4 `step18000` | 20 s | 3.98 | 3.92 | +0.06 | 0.23 |
| RUN-4 `step18000` | 30 s | 3.21 | 3.03 | +0.18 | 0.28 |
| RUN-4 `step18000` | 60 s | 2.14 | 2.00 | +0.14 | 0.10 |

### The pre-registered threshold, and the failure against it

Fixed **before** any number was inspected:

> A forward-minus-backward gap of **≥ 2.0 points absolute R@1**, at **≥ 3× the standard error**
> across files and seeds, at **Δ = 10 s**, with the **same sign sustained at Δ = 20 s**.

| model | required | observed at Δ=10 s | ×SE | sign held at 20 s | **verdict** |
|---|---|---:|---:|---|---|
| RUN-2 `step19000` | ≥ 2.0, ≥ 3× SE | +0.61 | 1.5× | yes (+0.13) | **FAIL** |
| RUN-4 `step18000` | ≥ 2.0, ≥ 3× SE | −0.04 | 0.3× | n/a (sign flips) | **FAIL** |

RUN-2 reaches under a third of the required magnitude at half the required significance.
RUN-4's gap is indistinguishable from zero.

### The artifact floor on this metric

`IDENTITY` — copying `W(t)` unchanged — is **direction-blind by construction**: it cannot
encode an arrow of time. Any forward−backward gap it shows is therefore pure gallery-composition
artifact (the forward gallery drops the first Δ windows of each file, the backward gallery drops
the last Δ).

| model | Δ=10 s | Δ=20 s | Δ=30 s | Δ=60 s |
|---|---:|---:|---:|---:|
| RUN-2 IDENTITY gap | +0.17 ±0.31 | +0.51 ±0.25 | +0.33 ±0.10 | −0.14 ±0.09 |
| RUN-4 IDENTITY gap | +0.55 ±0.41 | +0.61 ±0.13 | +0.46 ±0.18 | −0.08 ±0.12 |

**The artifact floor is ≈ +0.6 points.** Every learned-map gap in §1 lies at or below it.
RUN-4's positive IDENTITY gaps are individually several × their SE (Δ=20 s is 4.7× SE) — which
is precisely why significance alone was never sufficient, and why the pre-registered criterion
also demanded a 2.0-point magnitude. A reliable 0.6-point effect that a direction-blind baseline
reproduces is not forward information.

---

## 2. Receptive field, and why Δ ≥ 10 s is the floor

| encoder | checkpoint | temporal extent of one window |
|---|---|---|
| V-JEPA2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` | 64 frames sampled **uniformly across the full 10 s** |
| WavJEPA-base/nat | — | 100 Hz over the **same full 10 s** |

One window's receptive field is therefore **the whole 10 s**, for both modalities. Two
consequences, both *forced* by that number rather than chosen:

1. **Δ < 10 s is not a prediction horizon at any stride.** A target less than 10 s away shares
   raw input content with the query. "Predicting" it is partly retrieval of material already
   inside the encoder's input. Δ < 10 s is therefore **not reported**, at any stride.
2. **The within-file gallery excludes everything within ±10 s of the query.**

This also retires an earlier claim: Epic-Kitchens was once argued to be necessary to unlock
Δ ∈ {1, 2, 5} s. It is not — those Δ are invalid for *any* corpus at this receptive field.
Epic-Kitchens buys scale and scene diversity, not smaller Δ.

### The ±10 s band reduces to offset 0

The Ego4D cache is a **10 s window at a 10 s non-overlapping stride**. At that stride, offset
±1 is exactly 10 s away and overlaps the query by *exactly zero* frames and zero audio samples.
The ±10 s exclusion band therefore reduces to **excluding offset 0 alone — the query itself**.
Δ = 10 s is the tightest valid horizon, and the cache sits exactly on it: 77,831 consecutive
pairs, zero input overlap, already on disk.

---

## 3. Supporting results

All within-file, Ego4D held-out cache, 100 files, ~5,101 queries at Δ=10 s, mean gallery 51.7,
3 seeds, chance 2.198 %. Each model evaluated at **its own training length** (RUN-2 uncapped,
RUN-4 `T_a = 896`).

### 3.1 IDENTITY beats every learned map

| model | Δ=10 s | IDENTITY | rescaled | ridge | MLP | corpus-mean |
|---|---|---:|---:|---:|---:|---:|
| RUN-2 | macro R@1 | **14.19** | 14.17 | 9.83 | 9.50 | 2.29 |
| RUN-2 | micro R@1 | **13.60** | 13.59 | 9.13 | 8.86 | 2.06 |
| RUN-4 | macro R@1 | **15.50** | 15.50 | 10.57 | 9.04 | 2.29 |
| RUN-4 | micro R@1 | **14.81** | 14.81 | 9.78 | 8.38 | 2.06 |

Copying `W(t)` unchanged outperforms ridge by ~4.5 points and an MLP by ~6.4 points. Fitting a
map on the corpus *loses* accuracy. There is no learnable forward transformation to find —
the best available predictor of the future state is the current state, unmodified.

The `corpus_mean` control sits at 2.06–2.29 against a 2.198 % chance, confirming the gallery
carries no exploitable prior.

### 3.2 Decay to chance by Δ = 60 s

Ridge, micro R@1, chance 2.198 %:

| model | 10 s | 20 s | 30 s | 60 s |
|---|---:|---:|---:|---:|
| RUN-2 | 9.13 | 3.89 | 2.97 | **2.10** |
| RUN-4 | 9.78 | 3.98 | 3.21 | **2.14** |

Both land **at chance** at Δ = 60 s. Whatever `W(t)` knows about `W(t+Δ)` is gone inside a
minute. IDENTITY decays the same way (RUN-2 13.60 → 1.70; RUN-4 14.81 → 1.80), falling *below*
chance at 60 s as persistence stops helping and the un-adapted state becomes actively misleading.

### 3.3 Temporal-shuffle control — the Δ=10 s signal is real, just not directional

Windows shuffled within each file, everything else identical:

| model | Δ | intact | shuffled | chance |
|---|---:|---:|---:|---:|
| RUN-2 | 10 s | 9.13 | **2.03** | 2.198 |
| RUN-2 | 60 s | 2.10 | 1.93 | 2.198 |
| RUN-4 | 10 s | 9.78 | **1.99** | 2.198 |
| RUN-4 | 60 s | 2.14 | 1.87 | 2.198 |

Destroying temporal order collapses the result to chance. The Δ=10 s signal is genuine temporal
structure and not a within-file confound — it is simply *symmetric*. The backward-shuffled arm
behaves identically (2.03 / 1.91 at Δ=10 s).

### 3.4 Mismatched-pair falsifier (stage e, run first on purpose)

Query from one file, target drawn from a different file's gallery. Run **before** any other
stage, so that a failure would have invalidated everything downstream.

| model | Δ=10 s R@1 | chance | ratio | gallery |
|---|---:|---:|---:|---:|
| RUN-2 `step19000` | 0.019 | 0.019 | **1.0×** | 5,129 |
| RUN-4 `step18000` | 0.000 | 0.019 | **0.0×** | 5,267 |

At chance at every Δ. **PASS.**

### 3.5 Residual prediction adds nothing

Fitting `g : W(t) → W(t+Δ) − W(t)` instead of the state directly (micro R@1):

| model | Δ | direct ridge | residual ridge |
|---|---:|---:|---:|
| RUN-2 | 10 s | 9.13 | 9.14 |
| RUN-2 | 60 s | 2.10 | 2.11 |
| RUN-4 | 10 s | 9.78 | 9.80 |
| RUN-4 | 60 s | 2.14 | 2.14 |

Statistically identical. Subtracting persistence out leaves nothing behind — **because
persistence was all there was.** This is the cleanest single statement of the negative result.

### 3.6 Persistence, rescaled per model

Raw cosine is not comparable across models: their different-file floors differ (RUN-2 0.0727,
RUN-4 0.0362). Rescaled as `(cos − floor) / (cos(Δ=0) − floor)`:

| model | floor | 10 s | 20 s | 30 s | 60 s |
|---|---:|---:|---:|---:|---:|
| RUN-2 `step19000` | 0.0727 | 0.842 | 0.798 | 0.778 | 0.753 |
| RUN-4 `step18000` | 0.0362 | 0.783 | 0.731 | 0.708 | 0.680 |

Both plateau: a 50 s increase from Δ=10 s to Δ=60 s costs only 0.089 (RUN-2) and 0.103 (RUN-4).
**RUN-4 is slightly *less* persistent than RUN-2** at every Δ — consistent with its lower floor
and higher effective rank (74.26 vs 37.72), i.e. a less anisotropic representation space. The
padding fix did not make the representation more temporally sticky.

### 3.7 Macro vs micro, and per-file spread

Macro (mean over files) and micro (pooled over queries) agree throughout — RUN-2 IDENTITY at
Δ=10 s is 14.19 macro vs 13.60 micro. Long recordings are not driving the pooled numbers.

Per-file spread is nonetheless wide: median file R@1 13.31, IQR **[8.78, 18.92]** over 100 files.
Single-file numbers from this probe would be unreliable; only the 100-file aggregate is.

### 3.8 Self-retrieval calibration — undefined by construction

The planned self-retrieval row (how well `W(t)` retrieves *its own* window) **cannot be measured
under this design, and this is the correct answer rather than a gap.** The ±10 s exclusion band
removes offset 0, and offset 0 *is* the query's own vector. Self-retrieval and the exclusion are
mutually exclusive by construction; reporting a number would require disabling the exclusion that
makes every other row valid.

`IDENTITY` is the available ceiling and is reported in its place: it is the same measurement
(un-transformed `W(t)` scored against the gallery) with the query's own window legitimately absent.

---

## 4. What this does NOT show

* **It does not show that a predictive objective would fail.** This is a measurement of a
  representation produced by *ordinary multimodal alignment*. The finding is that predictive
  structure **did not emerge spontaneously** — not that it cannot be learned. Read correctly,
  this probe is the **baseline experiment that motivates RUN-5**: the null was tested and held,
  so the question becomes whether directional temporal structure must be **explicitly** trained.
* **`lam_pred = 0.0` in every run measured here.** Neither RUN-2 nor RUN-4 was ever asked to
  predict anything across time. Nothing in this document makes either a trained world model, and
  no result here should be cited as evidence about predictive training. (Separately: the existing
  `lam_pred` code path implements *within-window masked cross-modal* prediction, not temporal
  prediction, so turning it on would not test this hypothesis either.)
* **It does not show the representation is temporally uninformative.** Δ=10 s sits far above
  chance (9.1–9.8 ridge, 13.6–14.8 IDENTITY vs 2.198 chance) and the shuffle control confirms
  that signal is real temporal structure. The claim is narrower and exact: **the structure has no
  measurable arrow of time.**
* **It is protocol-bounded.** One corpus (Ego4D held-out), one window length (10 s), one stride
  (10 s), 100 files, Δ ∈ {10, 20, 30, 60} s, two checkpoints, linear and one-hidden-layer maps.
  A different corpus, a longer context, or a nonlinear predictor with temporal context could in
  principle reach a different result. Nothing here rules that out.
* **It says nothing about downstream task performance.** Retrieval quality, transfer, and the
  query-predictor results are unaffected by this finding.

---

## 5. Verdict

> Predicting the past is as easy as predicting the future.

`W(t)` is a compact audio-visual scene representation whose similarity to nearby windows is
**symmetric persistence**. Combined with the two earlier probe phases — no recurrence, no
cross-window state — the reviewer's challenge is answered, and answered in the negative.

**The name "world-state" is retired.** See `docs/TEMPORAL_STRUCTURE_PROBE.md` for Phases 0–2 and
`docs/CANONICAL_NUMBERS.md` for the naming verdict line.

---

## 6. Provenance

| item | value |
|---|---|
| RUN-2 checkpoint | `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` (uncapped `T_a`) |
| RUN-4 checkpoint | `checkpoints/m2_run4_padfix_ta896/step18000.pt`, `T_a = 896` |
| RUN-4 sha256 | `27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4` |
| checkpoint selection | P3.0 post-hoc sweep of all 20 tagged steps, 3 seeds — `step18000` selected |
| cache | `/mnt/Raid-Storage-2/utkarsh-data/feature_cache_ego4d_train_v1` |
| window / stride | 10.0 s / 10.0 s (non-overlapping) |
| files / queries | 100 files, 5,101 queries at Δ=10 s (5,001 / 4,901 / 4,601 at 20 / 30 / 60 s) |
| seeds | 3 |
| scripts | `scripts/temporal_probe/p32_stages_abcd.py`, `p32_forward_info.py` |
| artifacts | `docs/artifacts/temporal_probe/p32_abcd.json`, `p32e_falsifier.json` |
| stage order | **e** (falsifier) → a (persistence) → b (forward) → c (backward) → d (residual) |

**Pending (P4.15 step 10).** The receptive field in §2 is derived from encoder *configuration*
(`fpc64` over a 10 s window; WavJEPA at 100 Hz over the same window). An independent **empirical**
measurement is scheduled and will be appended here. It can only tighten the Δ floor's
justification, not relax it — 64 frames spanning 10 s is an upper bound on locality either way.
