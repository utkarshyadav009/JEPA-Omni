# RUN-5 SPECIFICATION — explicit temporal prediction

**Status: specification only. No training code has been written or modified.**

**Hypothesis.** Training the representation to retain information useful for predicting a
*future* representation may produce the directional temporal structure RUN-4 does not contain.

**The baseline this rests on.** P3.2 measured RUN-4 and found no arrow of time: forward−backward
gap −0.04 ± 0.15 against a pre-registered ≥2.0, and `IDENTITY` beating every learned map. That is
a result about a representation trained by *ordinary multimodal alignment*, with `lam_pred = 0.0`.
It says predictive structure did not emerge **spontaneously**. RUN-5 asks whether it emerges when
**explicitly trained**.

---

## 0. Code inspection (P4.10 precondition) — done, and it settles one question

`train_m2.py:1218` computes `pred_loss, metrics = model(feats, tbins, mask, key_padding_mask=kpm)`.
`AVJepaPredictor.forward` (`models/av_jepa_predictor.py:145`) documents itself as:

> "Cross-modal masked latent prediction to frozen targets."

`mask` is `{modality: (B, T_m) bool}` — a per-modality mask over the tokens of **one window**.
Every token comes from a single `feats`. **There is no second window, no future target, and no Δ
anywhere in that path.**

**Verdict: `lam_pred` is within-window masked cross-modal prediction, NOT temporal prediction.**
Per decision 2 it stays out of scope and is **not** reused, renamed, or turned on. RUN-5's
temporal loss is a new, separately named term (`lam_future`) so the two can never be conflated
in a config, a log line, or a results table.

**Where the temporal branch attaches.** `train_m2.py:1226` already computes a grad-carrying
representation for SIGReg:

```python
ws = raw.world_state(feats, tbins, key_padding_mask=kpm)   # (B, d) with grad
```

That is exactly `W_t`. The temporal branch takes a **second** such call on the future window's
features and attaches there. Nothing in the existing loss graph needs restructuring.

---

## 1. The specification

```
RUN-5 SPECIFICATION

Input:                  W_t = f_theta(V_t, A_t), the fused representation of a 10 s window.
                        Pilot: precomputed from frozen RUN-4 step18000 (1 s stride ws output).
                        Full:  recomputed with grad from cached token features (2 s stride).

Target:                 W_{t+D} = f_theta(V_{t+D}, A_{t+D}), SAME encoder, STOP-GRADIENT.
                        Detached so the target branch does not move toward the predictor during
                        the predictive update. NOTE: stop-gradient does NOT by itself prevent
                        representation collapse -- it only removes the trivial path where both
                        branches drift together. The representation can still collapse on its
                        own. The anti-collapse constraints are SIGReg's distributional pressure
                        and the explicit collapse diagnostics below, and those are what a
                        collapse would have to defeat.

Delta:                  10 s PRIMARY. 20 s secondary (generalisation check).
                        NOT 1/2/5 s -- see "Encoder receptive field".

Temporal stride:        1 s for the pilot (world-states exist at 1 s).
                        2 s for the full run (token features exist at 2 s; D=10 s is 5 windows).

Encoder receptive       MEASURED, not assumed (p4_receptive_field.json):
field:                    V-JEPA2 ViT-L : FULL 10 s. No zero cell anywhere in its influence
                                          matrix; earliest output group responds to the last
                                          input slice at 0.36 of self-response.
                          WavJEPA base  : ~2.25 s median, 4.25 s max. 76% of cells EXACTLY 0.
                          WavJEPA nat   : ~2.25 s median, 4.25 s max.
                        Because W fuses both and vision spans the whole window, a FUSED target
                        needs D >= 10 s. D < 10 s would share raw video input with the query.

Prediction              P_phi: a 2-layer MLP, 1024 -> 2048 -> 1024, GELU, LayerNorm on input.
architecture:           Deliberately small and NOT conditioned on Delta in the pilot: if a
                        larger or Delta-conditioned predictor is needed to see an effect, that
                        is a finding about effect size, not a reason to grow the model
                        mid-experiment. Any change here is a new experiment (section 5).

Prediction loss:        InfoNCE between P_phi(W_t) and stopgrad(W_{t+D}), temperature 0.05,
                        negatives = the other in-batch futures (section "Negative construction").
                        CONTRASTIVE, NOT L2/cosine, for one reason: under L2 or cosine the
                        identity map is already near-optimal (P3.2: IDENTITY beats ridge by 4.5
                        R@1), so a regression loss would be minimised by learning to copy and
                        would report success for a model that has learned nothing.

Contrastive loss:       L_AV unchanged from RUN-4 -- InfoNCE, temperature 0.05, 192 negatives.
                        Retained so RUN-5 remains comparable to RUN-4 on VGGSound retrieval,
                        which is a REQUIRED non-regression check, not a bonus.

SIGReg:                 lambda = 0.03, unchanged, on W_t. Its N(0,I) target supplies the
                        DISTRIBUTIONAL pressure against collapse -- this, not the stop-gradient,
                        is the actual anti-collapse mechanism. If SIGReg and the diagnostics both
                        pass and retrieval still degrades, the cause is not collapse and must be
                        diagnosed separately rather than assumed.

Total objective:        L = L_AV + lambda_future * L_future + lambda_sigreg * L_SIGReg
                        lambda_future = 1.0 primary; {0.3, 3.0} only if the primary is
                        ambiguous, and reported as a sweep, never as the pre-registered result.

Trainable parameters:   Pilot: P_phi ONLY (~4.2M). Everything else frozen.
                        Full:  P_phi + the AVJepaPredictor trunk + projections (155.9M + 4.2M),
                        exactly RUN-4's trainable set plus the predictor head.

Frozen parameters:      V-JEPA2 ViT-L and both WavJEPA towers, always. Never fine-tuned in any
                        arm. The cached features exist precisely so this stays true.

Negative construction:  In-batch futures from DIFFERENT source videos only. A negative drawn
                        from the same video at a nearby offset is a near-duplicate of the
                        positive (persistence cosine 0.78 at 10 s) and would make the loss
                        dominated by an impossible discrimination. Batch sampling therefore
                        enforces one pair per video per batch.

Overlap exclusion:      Pairs must satisfy |t_a - t_b| >= 10 s, i.e. zero shared video input.
                        At 1 s stride this excludes offsets -9..+9; at the 10 s window this is
                        exactly the receptive-field constraint, not a safety margin.
                        Evaluation additionally excludes the query's own window from its gallery.

Identity baseline:      THE PERSISTENCE BASELINE, and THE BAR TO BEAT. W_t used unchanged.
                        This is NOT a rival prediction model -- it is the null hypothesis stated
                        as a predictor: it asserts the scene does not change, and it scores well
                        only because RUN-4's representation is strongly persistent (rescaled
                        cosine 0.783 at 10 s). Beating it therefore means one specific thing:
                        the predictor has extracted CHANGE information rather than merely
                        preserving scene identity. In P3.2 it beat ridge by 4.5 and an MLP by
                        6.4 R@1, so it is the only baseline whose failure would be informative.

Ridge baseline:         Closed-form ridge W_t -> W_{t+D} fitted on the training split.
                        RUN-4 value at D=10 s: 9.78 micro R@1.

MLP baseline:           Same architecture as P_phi, trained post-hoc on FROZEN RUN-4
                        representations. RUN-4 value at D=10 s: 8.38 micro R@1.
                        This is the control that isolates "trained jointly" from "trained at
                        all" -- if RUN-5 only matches this, joint training bought nothing.

Shuffle control:        Windows shuffled within each file, everything else identical.
                        MUST land at chance (2.198%). RUN-4: 9.78 -> 1.99.

Backward control:       Identical pipeline predicting W_{t-D}. The forward MINUS backward gap is
                        the headline number, exactly as in P3.2.

Falsifier:              Mismatched-file pairs must score at chance. Run FIRST, before any other
                        evaluation, so a failure invalidates nothing downstream.

Collapse metrics:       Reported every eval, all four, any one of which fails the run:
                          effective rank of W over the eval gallery  >= 37 (half of RUN-4's
                                                                      73.53; below this the
                                                                      representation is worse
                                                                      than RUN-2's 37.72)
                          mean cos(W_t, W_{t+10s})                    <= 0.90 (RUN-4: 0.79)
                          mean cos(W_i, W_j) across different videos  <= 0.30 (RUN-4 floor 0.036)
                          std of W per dimension                      > 0 on every dimension

Success criterion:      See section 2. FROZEN BEFORE ANY RESULT IS INSPECTED.
```

---

## 2. Pre-registered success criteria

Fixed now, before any RUN-5 number exists. P3.2's threshold is reused verbatim so the two
experiments are directly comparable.

### 2.1 Pilot gate (P4.11) — decides whether the full run happens at all

All four must hold, at Δ = 10 s, within-file micro R@1, ≥3 seeds:

1. **Beats PERSISTENCE (IDENTITY)** by ≥ 2.0 points absolute and ≥ 3× SE.
   *Interpretation, stated precisely: the predictive model must outperform persistence, not
   merely learn to preserve the current scene.* IDENTITY is the null hypothesis written as a
   predictor — "nothing changes" — and at Δ=10 s RUN-4 is persistent enough that it scores
   14.81 micro R@1 against 2.198 chance. Clearing it requires extracting change information.
   *(RUN-4's ridge fails this by −5.0 points. This is the hard criterion and it is meant to be;
   it is not to be weakened if the pilot struggles.)*
2. **Forward − backward ≥ 2.0** points and ≥ 3× SE, same sign at Δ = 20 s.
3. **Shuffle control at chance** and **falsifier at chance**.
4. **No collapse** on all four metrics above.

> **If the pilot fails, stop and document.** Do not proceed to joint training, do not enlarge
> the predictor, do not re-tune Δ. A frozen-representation predictor that cannot beat copying
> is a valid, publishable result, and it is the honest end of this line if it happens.

### 2.2 Full-run criteria

1. **Forward − backward ≥ 2.0** points, ≥ 3× SE, sign held at Δ = 20 s.
2. **Beats RUN-4's forward R@1** at Δ = 10 s (9.78 ridge / 14.81 IDENTITY) by ≥ 2.0 and ≥ 3× SE.
3. **Beats the post-hoc MLP baseline** — otherwise joint training bought nothing over fitting a
   head to a frozen RUN-4.
4. **Retains multimodal retrieval:** VGGSound n=1,545 R@1 within **2.0 points** of RUN-4's
   **41.77 (v→a) / 41.88 (a→v)**. A predictive gain paid for with retrieval collapse is not a
   result.

   > **This number was contested and is now resolved** — see `CANONICAL_NUMBERS.md` §1.2.
   > Three RUN-4 figures were in circulation and one document contradicted itself, because the
   > **direction order was never written down**: `41.68/41.35` and `41.35/41.68` are the same
   > measurement in opposite orders, and neither is `step18000`.
   >
   > | value (v→a / a→v) | checkpoint | source | seeds |
   > |---|---|---|---|
   > | **41.77 / 41.88** | **`step18000`** | P3.0 sweep, `p30_shard2.json` | 3 |
   > | 41.34 / 41.68 | `step20000` | P3.0 sweep, `p30_shard0.json` | 3 |
   > | 41.35 / 41.68 | `step20000` | P2.3 matched grid | 5 |
   >
   > **The criterion above uses 41.77 / 41.88** — `step18000`, n=1,545, `T_a=896`, corrected
   > harness, 3 batch-order seeds, gallery `data/vggsound_eval_1545.txt`, checkpoint sha256
   > `27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4`. The `42.27 / 41.68` in
   > `EVIDENCE_LEDGER.md` is a *different July run* and must never be cited here.
5. **Controls clean:** shuffle and falsifier at chance, all four collapse metrics passing.

**Higher cosine alone is not a result** and will not be reported as one. Cosine rises under
collapse, which is why criterion 5 gates criteria 1–4 rather than sitting beside them.

### 2.3 Frozen protocol

Δ values, exclusion rules, temporal split, gallery construction, negative construction,
baselines and thresholds above are **frozen before results are inspected**. Any change after
seeing a number is a new exploratory experiment, reported as such, and never presented as the
pre-registered outcome.

---

## 3. Data — audited, not assumed (P4.11 step 2)

At 288 of 648 Epic-Kitchens videos extracted (41.8% by duration):

| | |
|---|---|
| windows | 129,043, **0.00% dropped, 0 unreadable, 0 non-finite** |
| pairs at Δ=10 s | **126,200** across 280 files — already 1.6× the Ego4D cache |
| pairs at Δ=20 s | 123,451 across 268 files |
| official train/val video split | **CLEAN, 0 overlap** |

> **Caveat, live:** only **9 participants** are present so far, because extraction runs in sorted
> video order. The participant-held-out split it computed (7 train / 2 eval) is **not
> representative** and must not size anything. **Re-run `p411_pair_audit.py` at completion**
> (~37 participants expected) before the pilot's split is fixed.

### Split: participant-held-out — **PROVISIONAL, not frozen**

**Rationale.** Two videos from one kitchen share lighting, utensils and layout, so a video-level
split leaks scene identity — and scene identity is exactly what a persistence-driven model
exploits. The official EK train/val split is video-level and is therefore not used as-is.

**But this is a consequential departure from the existing evaluation regime, and it is held
provisional until the extraction and the full pair audit finish.** The specific risk is real and
named here so it cannot be rationalised away later:

> A participant-held-out split may make the pilot **unnecessarily hard**. If it fails under a
> split that is harsher than anything RUN-4 was ever measured on, "prediction does not work" is
> **not** a licensed conclusion — "prediction does not work across kitchens" might be, and even
> that only with the video-level comparison in hand.

**Therefore, before the split is frozen:**

1. Re-run `p411_pair_audit.py` on all 648 videos (~37 participants expected, versus the 9 the
   provisional numbers were computed from).
2. Report pair counts under **both** a participant-held-out and a video-held-out split.
3. If the two differ materially in usable pairs or in per-participant scene diversity, **run the
   pilot under both** and report both. The difference between them is itself a result about how
   much of the signal is scene identity.
4. Only then fix the split — and fix it **before** any RUN-5 result is inspected (§2.3).

A failure under the harsher split with a pass under the easier one is a finding about
generalisation, not about prediction, and must be written up as such.

**Ego4D remains available** as a second corpus: 77,831 pairs at Δ = 10 s, already on disk.
Use it as an out-of-corpus generalisation check, not as training data, so that "works on
kitchens" and "works on egocentric video generally" stay separable.

---

## 4. The Δ = 5 s audio-only branch — recorded and RULED OUT of RUN-5

The receptive-field measurement opened an option that did not exist before: WavJEPA spans only
~4.25 s, so an **audio-only** predictive objective could train at Δ ≈ 5 s with genuinely no
input overlap.

**Not taken, for this run.** It changes the object under study from the fused audio-visual
representation to an audio-only one, which is a different research question and not comparable
to RUN-4 on any existing metric. Taking it would also mean the headline RUN-5 number could not
be placed beside P3.2's.

It is recorded here so the option is not lost, and it is the natural follow-up if the fused
pilot fails **specifically because of vision leakage** — a diagnosis that would require showing
the audio pathway carries forward information the fused one does not.

---

## 5. Execution order and isolation

1. Extraction completes → **re-run the pair audit** → fix the participant split.
2. Freeze RUN-4 `step18000` as the immutable baseline (sha256 `27b33c8c…`, backed up to md0 and
   GitHub, verified).
3. Train the pilot predictor on frozen representations. Monitor attached at launch.
4. Evaluate against §2.1 automatically on completion.
5. **Gate.** Pass → joint training. Fail → document and stop.
6. Full RUN-5, monitored, evaluated against §2.2.
7. Compare against the immutable RUN-4 baseline; preserve and back up.

**RUN-4 is never overwritten.** Separate namespaces `RUN-4/` and `RUN-5/`. Every RUN-5 result
records: RUN-4 checkpoint sha256, RUN-5 code commit, training config, dataset version, temporal
stride, Δ, train/eval sample counts, seed, RUN-5 checkpoint sha256, evaluation harness version.

**No training code is written until this specification is approved.**
