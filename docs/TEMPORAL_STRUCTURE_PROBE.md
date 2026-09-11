# TEMPORAL_STRUCTURE_PROBE — is the M2 fused latent a "world-state"?

**Date:** 2026-09-11. **Status: all three phases run**, with Phases 2 and 3 capped at
Δ ≥ 10 s (see §7). Phase 1 = within-window temporal structure; Phase 2 = persistence across
windows; Phase 3 = forward predictability of the space.

**Checkpoint under test:** `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`,
sha256 `e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8`, verified before
every run (the scripts abort on mismatch).
**Gallery:** `data/vggsound_eval_1545.txt`, sha256 `89307c6d4104…`, n = 1,545, every run
asserts `dataset_len == clips_seen == 1545`.
**Raw artifacts:** `docs/artifacts/temporal_probe/p03c_*.json`.
**Reproduce:** `python scripts/temporal_probe/bin_scramble_eval.py --arms A0,<ARM> --seeds 0,1,2,3,4 --fix-padding --out <path>`

---

## 1. The reviewer's question, stated fairly

> A world-state should be persistent and predictive over time. Your fusion bridge appears
> to recompute a fresh vector from scratch on every window with no carry-over, so calling
> it a "world-state" overclaims.

The architectural half of that is simply **correct, and confirmed by reading the code**:
`AVJepaPredictor` has no recurrence, no hidden state, and no term connecting one window to
the next. `encode_world_state` embeds the tokens of one window, runs 8 self-attention
blocks, and attentive-pools to a single 1024-d vector. Nothing carries over. The reviewer
did not need our measurements to establish that.

What measurement can add is the question *behind* the objection: **given that it is
computed per window, does the vector at least use the temporal structure inside that
window?** If it does not, "world-state" is indefensible on a second, independent ground.

## 2. One finding that reframes everything below

**`lam_pred = 0.0` in RUN-2** (`logs/m2_run2_final.log:42`). The masked cross-modal latent
prediction loss — the JEPA-style predictive term this architecture was designed around —
had **weight zero**. RUN-2's actual objective was `1.0 × InfoNCE + 0.03 × SIGReg +
1.0 × fusion-matching BCE`. The `pred=` column in the training log is a logged diagnostic,
not a gradient contributor.

So **this checkpoint has no predictive training term of any kind** — not over future
windows, and not over masked regions either. The reviewer's objection is, if anything,
understated.

## 3. Method

Eval-time only. We intervene on the **integer bin indices** fed to the predictor's learned
temporal embedding (`temporal_emb`, a free 512×1024 lookup table, `trunc_normal_` init —
not sinusoidal, so nearby bins carry no built-in similarity prior). We do **not** reorder
tokens, touch features, or touch the modality embedding.

Two metrics per arm:

* **retrieval R@k** through the contrastive head — the *secondary* evidence. Note this head
  never sees the fused vector: `pool_and_project` → `encode_source_tokens` runs one backbone
  pass per modality **with the other modality fully masked**. It measures a per-modality
  representation.
* **world-state cosine to the un-intervened `W`** — the *primary* evidence, because it is
  the object the naming dispute is actually about.

All arms run with the **padding fix** applied and batch order held fixed, so the only thing
varying between an arm and A0 is the bin indices. Under this path the null range is
**≤ 0.06 R@1** (5 batch-order seeds), which makes a 2.0-point interpretation threshold very
conservative. Permutations are drawn **per clip**, not once gallery-wide: a single global
permutation merely relabels the bin axis self-consistently, which a model could be
invariant to for a trivial reason.

### 3.1 Gate, and proof the hook reaches the world-state

Phase 0's identity-permutation arm (A1) matched A0 **bit-exactly** on all six R@k *and* on
the world-state tensor sha256 (`034db8ae…`) — the hook is a no-op when it should be.

Because A2/A3/A4 return cosine `1.0000` at four decimals, we verified separately that the
intervention actually propagates into `encode_world_state` rather than only into the cached
retrieval path (these are separate assembly code). On the real checkpoint:

| intervention | cos to un-intervened `W` |
|---|---|
| a **single** vision token's bin 0 → 511 | 0.99999976 |
| A2 permute the 32 group values | 0.99978733 |
| A8 all vision bins → 256 | 0.99923229 |
| A5 all vision bins → 0 | 0.94643891 |

Every value is `< 1.0`: the hook reaches it. The `1.0000` in earlier drafts was **four-decimal
rounding**, not an inert intervention. All world-state cosines below are reported to 8 dp,
with a relative-L2 column beside them. `temporal_emb` rows are genuinely distinct
(pairwise cos 0.03–0.05), so this is not a degenerate embedding table.

## 4. Results — all arms, corrected path, 5 seeds, n = 1,545

Deltas vs A0 in parentheses. A0: v→a 29.90 / a→v 28.28, world-state effective rank 37.72/1024.

| arm | intervention | v→a R@1 | a→v R@1 | **WS cos → A0** | WS rel. L2 |
|---|---|---|---|---|---|
| A2 | permute the 32 vision **group** values | 29.94 ±0.13 (**+0.04**) | 28.22 ±0.08 (**−0.06**) | **0.99998774** | 0.0047 |
| A3 | permute all 512 vision values | 30.01 ±0.03 (**+0.11**) | 28.23 ±0.07 (**−0.05**) | **0.99998935** | 0.0044 |
| A4 | cyclic shift of vision values | 30.03 ±0.05 (**+0.13**) | 28.26 ±0.09 (**−0.02**) | **0.99998742** | 0.0048 |
| A6 | permute **ambient** values | 25.90 ±0.39 (−4.00) | 25.68 ±0.36 (−2.60) | 0.99747354 | 0.0571 |
| A7 | A2 + A6 | 26.28 ±0.30 (−3.62) | 25.49 ±0.41 (−2.79) | 0.99743410 | 0.0574 |
| **A10** | **time reversal, multiset-preserving** | 24.60 (**−5.30**) | 24.72 (**−3.56**) | 0.99827564 | 0.0511 |
| A5 | all vision bins → 0 — **confounded** | 15.73 (−14.17) | 11.97 (−16.31) | 0.99071634 | 0.1186 |
| A8 | all vision bins → 256 — **confounded** | 17.61 (−12.29) | 13.20 (−15.08) | 0.99981588 | 0.0181 |
| A9 | reflection `511−b` — **confounded** | 17.28 (−12.62) | 17.35 (−10.93) | 0.99820173 | 0.0528 |

R@5/R@10 for every arm are in `p03c_phase1_fixed_summary.json`; they track R@1 in sign and
rough magnitude throughout.

### 4.1 Three arms are confounded and must not be read as temporal evidence

**A5, A8 and A9 all move tokens onto embedding rows the model never used for vision**, so
they measure distribution shift, not temporal structure.

Vision occupies rows `{0, 16, 32, …, 496}`. A5 and A8 collapse all 512 tokens onto a single
row, replacing 32 distinct embeddings with 32 copies of one. A9 — the arithmetic reflection
`b → max_bin−1−b` originally specified — lands on `{15, 31, …, 511}`, which overlaps the
trained set in **0 of 32 rows** (verified directly). A9 therefore cannot isolate direction.

That A5 and A8 agree closely despite using *different* rows (bin 0 vs bin 256) is itself
the evidence: their effect is about **losing 32 distinct embeddings**, not about absolute
temporal position.

**A10 is the corrected direction test.** Group *i* takes group *(n−1−i)*'s value; ambient
token *t* takes token *(T−1−t)*'s value. The multiset of embedding rows is preserved
**exactly** in both modalities; only direction changes. This is the arm to cite.

## 5. What the numbers say

**(a) Vision's temporal axis is entirely unused.** A2, A3 and A4 destroy within-window
vision order — including A3, which destroys the 16-token grouping as well — and change
nothing: |Δ| ≤ 0.13 R@1 against a null range of 0.06, and a world-state cosine of
**0.99998–0.99999**, i.e. a relative L2 change of **0.5%**. By the pre-registered rule
(<2.0 points = no measurable effect) this is not a small effect; it is **no effect**.

**(b) Ambient's temporal axis is used.** A6 alone costs 2.60–4.00 R@1 and moves the
world-state ten times as far as A2/A3/A4 (rel. L2 0.057 vs 0.005). This asymmetry is the
headline, and it was **invisible under the leaked evaluation path**, where A6 read
−5.89/−9.50 and A2/A3/A4 read −0.18/−0.28 — the same qualitative shape, but with the
contrast inflated by an artifact.

The asymmetry has an obvious mechanical explanation: ambient contributes ~990 tokens spread
across 512 bins at ~2 tokens/bin, while vision contributes a 32-step staircase with all 16
spatial tokens in a group sharing one value. Vision's temporal resolution is 32 steps over
10 s and the model discards it.

**(c) Direction matters, but only through ambient.** A10 costs 5.30/3.56 R@1 — more than
A6's random ambient permutation — so the representation is not merely order-sensitive but
mildly *direction*-sensitive. The world-state still only moves to cosine 0.9983.

**(d) The fused world-state is near-invariant to all of it.** Across **every** arm,
including the three confounded ones that push inputs out of distribution, the worst
world-state cosine is **0.9907** (A5). For the clean arms it is 0.9975–0.99999. Whatever
the temporal bins do, they are not what this vector is made of.

## 6. Verdict on the naming

**"World-state" is not defensible for this object. "Audio-visual scene representation" is
the accurate term.** Three independent grounds, in descending order of strength:

1. **Architectural, and not in dispute.** No recurrence, no carry-over, no state. Confirmed
   by code inspection, not inferred.
2. **No predictive term was ever trained** (`lam_pred = 0.0`). "Predictive" cannot be
   claimed at all for this checkpoint.
3. **Measured.** The vision half of the temporal axis is provably unused, and the fused
   vector is near-invariant (≥0.9975 cosine on every clean arm) to interventions that
   destroy within-window temporal structure outright.

The honest positive claim the evidence *does* support is narrower and worth keeping: the
representation is **not a pure bag of co-occurring content**. Ambient temporal structure is
used, and it is used directionally (A10 > A6). That is real within-window temporal
sensitivity, carried by one modality. It is not persistence, and it is not prediction.

**Recommendation:** rename to "audio-visual scene representation" throughout, and state the
ambient-only temporal sensitivity as a measured property rather than implying the fused
vector integrates time across both streams.

---

## 9. Phase 2 — persistence across windows

**Question:** how long does information survive, and does the fusion add persistence beyond
the window its input already spans?

50 Ego4D files with runs of ≥10 **consecutive** windows, ~4,400 pairs per Δ, 2,000
different-file floor pairs, **batch size 1** (a single clip cannot be padded, so E-1 cannot
touch these numbers). `docs/artifacts/temporal_probe/phase2_persistence.json`.

| stream | Δ=0 | Δ=10 s | Δ=20 s | Δ=30 s | Δ=60 s | floor (c) | half-life |
|---|---|---|---|---|---|---|---|
| **world-state** | 1.0000 | 0.8699 | 0.8353 | 0.8217 | 0.8020 | **0.0937** | beyond 60 s |
| vision (reference a) | 1.0000 | 0.9612 | 0.9489 | 0.9450 | 0.9387 | 0.7814 | beyond 60 s |
| ambient (reference b) | 1.0000 | 0.9474 | 0.9351 | 0.9295 | 0.9202 | 0.7319 | beyond 60 s |

**Answer: the fusion adds no persistence of its own.** Three readings:

1. **It is a plateau, not a decay.** From Δ=20 s to Δ=60 s the world-state moves
   0.835 → 0.802 — essentially flat over 40 s. Windows are *non-overlapping* 10 s, so
   Δ=10 s already means wholly disjoint content. A flat residual similarity between
   disjoint windows of the same file is **scene identity** — same room, same wearer, same
   microphone — not a decaying memory trace. Nothing is carried across windows.
2. **Raw cosine flatters the references and must not be read directly.** Their floors are
   **0.78 and 0.73**: two windows from *different files* are already that similar, the
   anisotropy/cone effect. The world-state's floor is **0.094**, so SIGReg did make it
   near-isotropic. Normalised by each stream's own dynamic range, retention at Δ=10 s is
   world-state **0.857**, vision 0.823, ambient 0.804 — the same regime, no meaningful
   advantage to the fused vector.
3. **"Half-life beyond 60 s" for all three sounds impressive and is not** — it follows from
   the plateau in (1), not from memory.

**Methodological note, stated rather than hidden:** the raw and L2-normalised columns come
out identical to four decimals, **necessarily** — cosine is scale-invariant, so normalising
before taking cosine is a no-op. They are not two independent measurements. The distinction
would matter only under a Euclidean metric.

## 10. Phase 3 — forward prediction

**What this tests: whether the SPACE is predictable, not whether the MODEL predicts.**
RUN-2 trained with `lam_pred = 0.0` and contains no term referencing any future window
(§2). A positive result says the latent space admits a learnable forward map; it says
nothing about the architecture performing prediction.

120 files, **file-disjoint** 84 train / 36 test, seed 0.
`docs/artifacts/temporal_probe/phase3_forward.json`.

| Δ | method | cosine | **R@1** | R@5 | R@1 gallery |
|---|---|---|---|---|---|
| **10 s** | IDENTITY (copy `W(t)`) | **0.8569** | 0.43 | **29.37** | 2,533 |
| | corpus mean | 0.2796 | 0.04 | 0.20 | 2,533 |
| | per-dim rescaled copy | 0.8569 | 0.43 | 29.41 | 2,533 |
| | ridge | 0.7733 | 3.16 | 16.98 | 2,533 |
| | 2-layer MLP | 0.7858 | **3.43** | 16.70 | 2,533 |
| **20 s** | IDENTITY | **0.8161** | 0.20 | **15.38** | 2,497 |
| | ridge | 0.7078 | **2.04** | 9.05 | 2,497 |
| | 2-layer MLP | 0.7211 | 1.64 | 8.77 | 2,497 |
| **30 s** | IDENTITY | **0.7990** | 0.45 | **12.27** | 2,461 |
| | ridge | 0.6808 | **1.42** | 7.27 | 2,461 |
| | 2-layer MLP | 0.6909 | 1.30 | 6.46 | 2,461 |

**The result splits by metric, and the split is the finding.**

* **By cosine, IDENTITY wins at every Δ.** The learned maps are strictly *worse* at landing
  near `W(t+Δ)` in absolute terms.
* **By R@1, the learned maps beat IDENTITY by ~8×** (3.43 vs 0.43 at Δ=10 s; chance on a
  2,533 gallery is 0.04%).
* **By R@5, IDENTITY wins again** (29.37 vs 16.70).

The reconciling mechanism: copying `W(t)` produces a vector nearest to `W(t)` itself — and
`W(t)` is *in the gallery*, as the future of the window d steps earlier. IDENTITY therefore
retrieves a systematically **off-by-one** neighbour: close enough for R@5, wrong at rank 1.
The learned maps shift the prediction forward along the trajectory, which buys rank-1
discrimination at the cost of absolute proximity.

**So there is real forward structure beyond slow change — and it is small.** 3.43% R@1 is
~85× chance and still 96.6% wrong. Note also that the per-dimension rescaled copy is
identical to IDENTITY to four decimals, i.e. the world-state is already well-scaled
per-dimension — SIGReg doing its job, and one more baseline that the learned maps must and
do beat at rank 1.

**This does not rescue "world-state".** A space in which a ridge regression recovers a
little forward information is not a model that maintains state. The model was never trained
to predict anything, and Phase 2 shows it carries nothing between windows.

---

## 7. What this does NOT show

* **Phases 2 and 3 are capped at Δ ≥ 10 s and cannot be finer.** The Ego4D feature cache was
  extracted at a 10 s non-overlapping stride, and the raw video is gone from this machine
  (0 mp4 files survive), so Δ ∈ {1,2,5} s is unobtainable without re-acquiring the corpus.
  See `docs/ERRATA_PROPOSED.md` E-8/E-9 and `docs/CORPUS_OPTIONS.md`. Sub-10-second
  persistence and prediction are therefore **unmeasured**, and a faster-decaying component
  below 10 s would not be visible to either phase.
* **Phase 2/3 run on Ego4D only**, whose windows are long-form egocentric; the Phase 1
  arms run on VGGSound. The two phases are not on the same corpus.
* **The retrieval column does not measure the fused vector.** `encode_source_tokens` masks
  the other modality, so R@k describes a per-modality representation. A6/A7 act on ambient's
  real bins but reach the *vision* pass only through position-only mask tokens — a thinner
  channel than the arm names suggest. The world-state column is the one that speaks to the
  naming question.
* **One checkpoint, one gallery, one corpus.** VGGSound, 1,545 clips. Not replicated on
  Ego4D (E-8 makes that impossible today) and not tested on another checkpoint.
* **A2–A4's null result is a null result, not proof of impossibility.** It shows this
  trained model ignores vision's temporal bins. A different training objective — in
  particular one with `lam_pred > 0` — might not.
* **Absolute R@1 values here (~30/28) are the corrected, padding-free ones** and are ~24
  points below previously published figures for reasons unrelated to this probe (E-1).

## 8. Plain-language paragraph for the reviewer

> You are right, and we have measured how right. The fused vector is computed fresh from
> each window with nothing carried over — there is no recurrence in the architecture — and
> we can add that the predictive loss this model was designed around was switched off in
> the run that produced the released checkpoint, so nothing predictive was ever trained.
> We also tested whether the vector at least uses the ordering of events *inside* a window.
> If we randomly shuffle the timestamps attached to the visual tokens — destroying their
> order completely — the vector is unchanged to five decimal places and retrieval accuracy
> does not move. Shuffling the audio timestamps does have an effect, and playing the window
> backwards has a slightly larger one, so the representation is not entirely blind to time;
> but that sensitivity comes from the audio stream alone, and even then the fused vector
> barely moves. On this evidence "world-state" overclaims, and we are renaming it an
> audio-visual scene representation. We also measured how long information survives across
> windows and whether one window's vector predicts the next. It does not persist: the
> similarity between windows ten seconds apart and sixty seconds apart is almost the same,
> which is what you see when two clips merely come from the same room rather than when a
> system is remembering anything. A simple regression can recover a little information
> about the next window beyond just copying the current one, so the space is not entirely
> static — but the effect is small, and the model was never trained to predict, so this is
> a property of the representation rather than a capability of the architecture.
