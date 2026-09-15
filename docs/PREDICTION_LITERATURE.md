# How the source papers actually do prediction — and what it changes for RUN-5

Read to ground RUN-5's design rather than invent it. Two papers matter: **LeJEPA**
(Balestriero & LeCun, arXiv:2511.08544), which is where our SIGReg comes from, and
**V-JEPA 2** (arXiv:2506.09985), which is where our vision encoder comes from.

---

## 1. LeJEPA does **not** do world modelling — this needs saying plainly

It is easy to assume the paper SIGReg comes from is a world-model paper. It is not.

| | LeJEPA |
|---|---|
| what "prediction" means | **multi-view augmentation invariance**, DINO-style: V_g=2 global 224² views, V_l=8 local 96² views |
| target | the **mean of the global-view embeddings** of the *same image*, `μ_n = (1/V_g) Σ z_{n,v}` |
| loss | **L2**: `L_pred = (1/V) Σ_v' ‖μ_n − z_{n,v'}‖²` |
| temporal structure | **none.** "No temporal structure is required; views come from data augmentations." |

**LeJEPA contains no temporal prediction, no future state, and no world model.** Its contribution
is a *geometry* result: the isotropic Gaussian is proved to be the unique embedding distribution
minimising worst-case downstream prediction risk, and SIGReg is the estimator that enforces it.

**SIGReg itself:** an Epps–Pulley test on the empirical characteristic function, applied to `M`
random 1-D projections and averaged:
`SIGReg = (1/|A|) Σ_{a∈A} EP({aᵀf_θ(x_n)})`, with `EP = N ∫|φ̂_X(t) − φ(t)|² e^{−t²/σ²} dt`.

**Full objective:** `L = (λ/V) Σ_v SIGReg({z_{n,v}}) + (1/B) Σ_n L_pred`.

## 2. LeJEPA removes stop-gradient entirely — and this both supports and complicates our spec

> "Thanks to LeJEPA's SIGReg loss, we can remove both the predictor and teacher-student
> architecture without suffering from collapse."

No stop-gradient, no EMA teacher, no predictor, no schedulers.

**This confirms the review correction.** SIGReg's distributional pressure *is* the anti-collapse
mechanism; stop-gradient is not. The spec's revised wording is right and now has a citation.

**But it does not licence dropping stop-gradient from RUN-5**, for a reason specific to *temporal*
prediction that does not arise in LeJEPA. LeJEPA's two branches are two augmentations of the
**same** sample, so pulling them together is exactly the intended invariance. Our two branches are
**different points in time**, and pulling them together means **increasing persistence** — the very
thing P3.2 showed already dominates and which RUN-5 must beat. Gradient into the target branch
therefore creates an incentive to solve the task by making the representation *more* temporally
smeared.

**Consequence, and it is convenient:** the question is **moot for the pilot**, where the
representation is frozen and `W_t`, `W_{t+Δ}` are both precomputed — stop-gradient is vacuous
there. It is a live decision only for joint RUN-5, and it is deferred to that point with the
trade-off recorded rather than guessed now.

## 3. SIGReg hyper-parameters: ours are below the paper's defaults, and RUN-5 changes what that costs

| | LeJEPA default | ours (RUN-4) |
|---|---|---|
| λ | **0.05** | 0.03 |
| slices \|A\| | **1024** | 256 |

In RUN-4 this did not matter: targets were frozen, so collapse was impossible by construction and
SIGReg was only shaping geometry — our own `models/sigreg.py` docstring says exactly this, that
SIGReg there is "NOT the load-bearing anti-collapse mechanism."

**In joint RUN-5 it becomes load-bearing**, because the target is produced by the same trainable
encoder. Running the load-bearing term *below* the paper's defaults is then a real risk.
**Action for joint RUN-5 (not the pilot):** raise to λ=0.05 and 1024 slices, or justify not doing
so. The paper notes slices are resampled per step, so even small `|A|` works — verify our
implementation resamples before relying on that.

## 4. V-JEPA 2: this is what an actual latent predictor looks like

| | V-JEPA 2 |
|---|---|
| predictor | ViT with 3D-RoPE, consumes encoder outputs + learnable mask tokens |
| target | `sg(E_θ̄(y))` — **EMA teacher with stop-gradient** |
| loss | **L1 in representation space**, on masked patches only |
| collapse prevention | stop-gradient **+** EMA — i.e. the heuristics LeJEPA removes |

So the two papers disagree on collapse prevention, and we are inheriting components from both.
That is worth stating in the paper rather than hiding: V-JEPA 2 uses the teacher/stop-grad
machinery; LeJEPA proves it is unnecessary given SIGReg.

## 5. V-JEPA 2's Epic-Kitchens-100 result is on **our corpus**, and its protocol validates our Δ floor

**39.7 mean-class recall@5, state of the art on EK-100 action anticipation**, with:

* **anticipation time τ_a = 1 second**
* input clip **ends 1 second before action onset**
* predictor takes encoder output + mask tokens for the future frame
* encoder **frozen**; encoder and predictor outputs concatenated into an attentive probe with
  three query tokens (verb / noun / action)

**τ_a = 1 s does not contradict our Δ ≥ 10 s floor.** The governing constraint is
**input/target disjointness**, not a fixed number. V-JEPA 2 gets disjointness by construction —
the clip *ends* before the target begins. We get it by requiring Δ ≥ window length, because our
query and target are both 10 s windows. Same principle, different bookkeeping. Our floor is
correct *for our window design*.

**The live consequence: a shorter query window would permit a smaller Δ.** And this is reachable
without re-decoding — the cached features hold 32 vision token-groups spanning 10 s (≈0.3125 s
each) and ~996 ambient tokens, so a 2 s query window is a **temporal slice of tokens we already
have**. This retroactively justifies keeping the 610 GB of features: the source video is deleted
and can never be re-decoded, but window length is still a free variable *because* the tokens
were kept.

Also worth noting: **the encoder is frozen and only a probe is trained** — structurally the same
choice as our pilot.

## 6. What this changed in the RUN-5 spec

| finding | effect |
|---|---|
| SIGReg is the anti-collapse mechanism, not stop-grad | confirms the revised wording; citation added |
| LeJEPA drops stop-grad entirely | **not adopted** — temporal branches differ from augmentation branches; moot for the pilot, deferred for joint |
| LeJEPA λ=0.05 / 1024 slices | flagged for joint RUN-5; ours are 0.03 / 256 and SIGReg becomes load-bearing there |
| V-JEPA 2 uses L1 to an EMA teacher | our InfoNCE choice stands — under L1/L2 the identity map is near-optimal here, which is exactly what we must beat |
| V-JEPA 2 EK-100 τ_a = 1 s | our Δ ≥ 10 s floor is confirmed *for 10 s windows*; shorter windows are a reachable future option |
| V-JEPA 2 freezes the encoder and trains a probe | our pilot is structurally the same, which is a point in its favour |

## 7. Sources

* LeJEPA: <https://arxiv.org/abs/2511.08544>
* V-JEPA 2: <https://arxiv.org/abs/2506.09985>
