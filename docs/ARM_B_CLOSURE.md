# ARM_B_CLOSURE (P3.3) — why Arm B does not run in its current form

**Decision: Arm B is closed as specified.** Not because the research question is answered, but
because *the experiment it describes has already been run, at zero GPU cost, as the P3.2 probe* —
and it returned a decisive negative.

---

## 1. Arm B and P3.2 are the same experiment

Arm B, as written in `docs/P2_8_PREDICTIVE_WORLDSTATE_SPEC.md`:

> fit `g: W(t) → W(t+Δ)`, ridge and a 2-layer MLP over frozen scene representations

P3.2 stage (b) fits exactly that — ridge and a one-hidden-layer MLP, over frozen `W`, on the
Ego4D cache at Δ ∈ {10, 20, 30, 60} s, with a backward control, a temporal-shuffle control, a
residual variant, and a mismatched-pair falsifier run first.

**Arm B's own analysis was performed. The results:**

| Δ = 10 s, within-file micro R@1 | RUN-2 | RUN-4 | chance |
|---|---:|---:|---:|
| **IDENTITY** (copy `W(t)` unchanged) | **13.60** | **14.81** | 2.198 |
| ridge (Arm B's model) | 9.13 | 9.78 | 2.198 |
| MLP (Arm B's model) | 8.86 | 8.38 | 2.198 |
| corpus mean | 2.06 | 2.06 | 2.198 |

**Both of Arm B's learned maps lose to copying the input unchanged** — ridge by 4.5 points, the
MLP by 6.4. And the forward−backward gap is +0.61 ± 0.40 (RUN-2) and −0.04 ± 0.15 (RUN-4)
against a pre-registered ≥2.0, with a direction-blind `IDENTITY` baseline showing artifact gaps
of the same size.

Running Arm B on a larger corpus would fit the same two model classes to the same absent signal.
**More data does not create structure that is not there.** There is no version of "ridge over
frozen `W`" that finds a forward direction the frozen `W` does not encode.

## 2. The three options, costed

### Option A — Ego4D-only, today

| | |
|---|---|
| data | 77,831 consecutive pairs at Δ = 10 s, **on disk now** |
| cost | zero download, ~2 GPU-hours |
| status | **already executed** — this is what P3.2 ran |
| result | negative, decisive |

The regime is valid on its own terms: a 10 s window at a 10 s non-overlapping stride is exactly
Δ = 10 s with **zero input overlap**. Nothing about the Ego4D cache limited this result.

### Option B — wait for Epic-Kitchens, then run Arm B

| | |
|---|---|
| data | 87.5 h, 648 videos, ~308k windows at 1 s stride |
| cost | ~16 h extraction (in flight), ~1 TB source deleted, ~610 GB features retained |
| gain over A | more pairs, kitchen-scene diversity, contiguous sequences |
| **gain on the actual question** | **none** |

Epic-Kitchens is worth extracting — it is the contiguous-sequence corpus the RUN-5 pilot needs,
and the extraction is recovering ~430 GB net on md1. But it does not rescue Arm B. A bigger
dataset for a probe whose learned maps lose to the identity function buys a more precise
measurement of zero.

### Option C — reformulate as RUN-5 (**recommended**)

Arm B's defect is not its corpus. It is that **it probes a frozen representation for structure
that nothing ever trained into it.** `lam_pred = 0.0` in every run measured; no term ever asked
`W` to carry information about the future.

P3.2 is therefore best read not as Arm B's failure but as **its baseline**: ordinary multimodal
alignment was tested for spontaneous predictive structure and did not show any. The live
question is whether predictive structure must be **explicitly learned** — which requires
training, not a post-hoc regression.

## 3. Recommendation

**Run Option C. Do not run Arm B on either corpus.**

If forced to choose between A and B, the answer is **A** — because A is free, already done, and
its answer would not change. Waiting for Epic-Kitchens to re-run a probe whose learned maps lose
to the identity function is spending ~16 hours and ~1 TB of deletion to measure the same zero
more precisely.

**This does not slow Epic-Kitchens down.** The extraction proceeds on its own merits: it is the
source of the contiguous sequences RUN-5's pilot needs (P4.11), and it recovers disk. It is
simply no longer blocking, or blocked by, Arm B.

## 4. What is explicitly *not* concluded

* **Not** that predictive objectives fail. Nothing here trained one.
* **Not** that `W` is temporally uninformative. Δ = 10 s sits far above chance (9.1–14.8 vs
  2.198) and the shuffle control confirms that is real temporal structure. It is **symmetric**.
* **Not** that Epic-Kitchens was a wasted acquisition. It was mis-justified — it was argued for
  as the enabler of Δ < 10 s and of Arm B, and it is neither (see the correction banner in
  `docs/CORPUS_OPTIONS.md`). Its actual value is scale, scene diversity, and contiguity.

## 5. References

| | |
|---|---|
| the probe | `docs/FORWARD_INFORMATION_PROBE.md` |
| earlier phases | `docs/TEMPORAL_STRUCTURE_PROBE.md` |
| the spec being closed | `docs/P2_8_PREDICTIVE_WORLDSTATE_SPEC.md` (superseded banner) |
| corpus reasoning + corrections | `docs/CORPUS_OPTIONS.md` |
| artifacts | `docs/artifacts/temporal_probe/p32_abcd.json`, `p32e_falsifier.json` |
