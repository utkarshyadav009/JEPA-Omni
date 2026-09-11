# PREDICTIVE_OBJECTIVE_SURVEY — P0.6

**Survey only. Nothing here is implemented.** Compiled 2026-09-11 for the RUN-5 spec.

Scope: how published systems combine a predictive / regression latent loss with a
contrastive one without collapse; what weightings they report; what anti-collapse
mechanisms they rely on; what is standard for predicting a *future* latent; and why this
project's own seven regression-to-frozen-target runs produced chance-level retrieval.

Where a number is not stated in a source I could read, it is marked **NOT VERIFIED**
rather than guessed.

---

## 1. How published systems combine the two objectives

| system | year | predictive term | contrastive term | combination | anti-collapse mechanism | collapse reported? |
|---|---|---|---|---|---|---|
| **CPC** | 2018 | predicts future latents `z_{t+k}` from context `c_t` via a per-step linear map `W_k` | **the prediction IS the contrastive term** — InfoNCE with the true future as positive, other timesteps/sequences as negatives | single loss | negatives make the trivial solution unscorable | no |
| **BYOL** | 2020 | online→target regression (normalised MSE ≡ cosine) | none | single loss | EMA target + stop-grad + **predictor asymmetry** | no (that is the paper's claim) |
| **SimSiam** | 2021 | same, no EMA | none | single loss | stop-grad + predictor asymmetry alone | collapse occurs when either is removed |
| **VICReg** | 2022 | invariance (MSE) between views | none | `λ·inv + μ·var + ν·cov` | explicit **variance** floor + **covariance** decorrelation | collapse only without the var term |
| **I-JEPA / V-JEPA / V-JEPA 2** | 2023–25 | masked latent prediction to an **EMA teacher**; V-JEPA uses **L1** in embedding space | **none** | single loss | EMA teacher + stop-grad | no |
| **data2vec / data2vec 2.0** | 2022–23 | L2 to a contextualised target = **average of the teacher's top-K blocks**, with **target normalisation** | none | single loss | EMA teacher + stop-grad + target normalisation | no |
| **CAV-MAE** | ICLR 2023 | **reconstruction** (MSE on masked patches), not latent regression | InfoNCE on single-modal pooled streams | `L = L_r + λ_c · L_c` | reconstruction anchors the representation; contrastive supplies discriminability | not discussed in the paper |

**The single most important structural fact for us:** every system in that table that uses
a pure predictive loss uses an **EMA teacher**, not a frozen one — and every system that
uses a frozen target pairs it with something that rewards discriminability. No published
system regresses to a fixed frozen target with no contrastive and no variance term and
reports success at retrieval.

## 2. Reported weightings

| system | reported value | notes |
|---|---|---|
| **CAV-MAE** | **λ_c = 0.01**, loss `L = L_r + λ_c·L_c` | verified. The paper notes the weight "requires careful tuning, with both excessive and insufficient values degrading performance"; the per-value ablation table is **NOT VERIFIED** (not in the sources I could read) |
| **VICReg** | ν = 1; λ and μ grid-searched under the constraint **λ = μ > 1**. The commonly cited setting is λ = μ = 25, ν = 1 | the λ=μ>1 constraint is verified; the specific 25/25/1 triple is widely reported but I did **NOT VERIFY** it against the paper text |
| data2vec / I-JEPA / V-JEPA | n/a — single-objective | |

**Note the direction.** CAV-MAE puts the *contrastive* term at 0.01 beside a dominant
reconstruction loss. RUN-5 Arm A proposes the mirror image: a small **predictive** term
beside a dominant InfoNCE. CAV-MAE's 0.01 therefore is **not** directly transferable as
"use λ_pred = 0.01" — it tells us the two objectives coexist at roughly a 1:100 scale
ratio, not which one takes the small coefficient.

The defensible starting point is not a borrowed constant but a **gradient-magnitude
match**: pick λ_pred so the predictive term's gradient norm is ~1–10% of InfoNCE's at
step 0, and log both. That is measurable in one short run and does not depend on an
unverified number. Our own history supports caution about borrowed constants: the
**SIGReg λ was scaled 0.03 → 0.00375** to match an 8× negative increase and the run
**collapsed anyway** (`METHODOLOGY_FORENSICS.md` row 10, avg R@1 24.76 → 19.19, killed at
step 2000).

## 3. Anti-collapse mechanisms and compatibility with our SIGReg

| mechanism | what it does | compatible with our setup? |
|---|---|---|
| **EMA target** | teacher is a slow copy of the student; gradients cannot reach it | **Yes, but we do not have one.** Our targets are *frozen pretrained encoders* — a stronger condition than EMA. Collapse in the BYOL sense is impossible by construction (the teacher cannot move), which `av_jepa_predictor.py`'s docstring already argues correctly |
| **stop-gradient** | already present — `tgt = x[mm].detach()` | yes, in place |
| **predictor asymmetry** | prediction error stays non-zero even at collapse, giving a residual gradient | partially present: `out_head[m]` is a per-modality linear head |
| **target normalisation** | removes the constant/anisotropic component of the target | **already in place and this matters** — `av_jepa_predictor.py:176` applies `F.layer_norm(tgt)` with the comment *"removes near-zero-constant shortcut"*. This is exactly data2vec's recipe, and it was added **after** the seven failed runs in §5 |
| **variance floor (VICReg)** | forces per-dimension std above a threshold | compatible in principle, but **redundant with SIGReg**: SIGReg already shapes the unnormalised world-state toward N(0,I), which pins both variance and covariance. Adding VICReg would double-regularise the same object |
| **L2 normalisation of the world-state** | — | **incompatible.** SIGReg targets N(0,I), which is geometrically inconsistent with a unit sphere. This is recorded as a lesson already paid for across the M1 SIGReg sweeps |

**Conclusion for RUN-5 Arm A:** we already hold three of the four mechanisms (frozen
target, stop-grad, target layer-norm) plus SIGReg. Nothing new is structurally required.
The open question is only the weight.

## 4. Temporal prediction: InfoNCE over futures vs regression to an EMA target

Both are published and both work; they fail differently.

* **CPC-style InfoNCE over futures** — predict `z_{t+k}` and score it against negatives
  drawn from other timesteps and other sequences. The negatives make "predict the corpus
  mean" unscorable, so the objective *cannot* be satisfied by a degenerate predictor. This
  is the safer choice at small scale and it is also **the objective we already know works
  in this codebase** (every successful M2 run is InfoNCE; every regression run is not).
* **EMA-target regression** (data2vec / V-JEPA style) — needs an EMA teacher we do not
  have, plus target normalisation. It scales better and avoids negative-sampling cost, but
  it has no built-in discriminability pressure, which is precisely the failure mode of §5.

I found **no source** that directly benchmarks the two against each other at our scale
(~300k clips, ~150M trainable parameters). Claims of relative stability at a specific
scale would be unsupported, so none is made here.

**Recommendation for Arm B: CPC-style InfoNCE over futures.** It reuses the loss the
project has repeatedly made work, it needs no new EMA machinery, and its failure mode is
visible in the same retrieval metrics we already track. Note the hard data prerequisite
from `docs/CORPUS_OPTIONS.md` §1.3: the finest measurable Δ from surviving data is **10 s**.

## 5. Why our seven regression runs collapsed — and the minimum fix

`docs/METHODOLOGY_FORENSICS.md:107` records seven configurations of cosine regression to a
frozen ambient target (`m2_step1_calib_lam001/003/010`, `m2_step1_hf075/085/095`,
`m2_step1_hinge090`) at R@1(cos) **0.07–0.20%**.

**First, name the number.** On a 1,545-clip gallery chance is 1/1545 = **0.065%**. Those
runs scored **at chance**, not "near-zero". That is diagnostic, not rhetorical.

**Second, it was not collapse in the BYOL sense.** With a frozen, non-degenerate teacher,
representation collapse is prevented by construction — the target cannot move to meet the
student. The forensics doc's own hedge ("suggests the metric/objective mismatch") is the
right instinct; here is the mechanism.

**The actual failure: the objective contained no term rewarding inter-clip
discriminability.** Frozen encoder outputs are strongly anisotropic — they share a large
common component, the well-documented "cone effect". A predictor that emits approximately
the corpus mean therefore achieves *high cosine against every target simultaneously*,
while separating no two clips at all. The loss is minimised; retrieval is at chance. The
objective and the metric were optimising different things, and nothing in the loss
penalised that.

**Was the configuration collapse-guaranteed? On the literature, essentially yes** — with
one qualification. Regression to a fixed target, with no negatives, no variance/covariance
term, and no target normalisation, has no mechanism in it that can produce
discriminability; every published system in §1 supplies at least one. The qualification is
that the outcome is *chance retrieval*, not divergence — the model was learning something
(low loss), just nothing the metric could see.

**Minimum change that prevents it**, in increasing order of intervention:

1. **Normalise the targets** — strips the shared constant/anisotropic component that makes
   the mean a good answer. This is data2vec's recipe and it is **already in the code**
   (`av_jepa_predictor.py:176`), added after those runs. On its own it removes the
   shortcut but still supplies no positive pressure toward separation.
2. **Keep a dominant InfoNCE term** and add prediction as an auxiliary — negatives make the
   mean-predictor unscorable. This is exactly RUN-5 Arm A's shape and is the
   lowest-risk option.
3. **Add a variance floor** (VICReg-style) if 1+2 still show low effective rank — but
   see §3: SIGReg already covers this ground, and stacking them risks over-constraining
   the world-state.

**The practical implication for RUN-5 Arm A:** the historical failure is *not* evidence
that a predictive term is harmful. It is evidence that a predictive term **alone**, with
un-normalised frozen targets, is inert for retrieval. Both defects are already fixed in
the current code. Arm A should be safe — but it must be gated on **effective rank and
held-out R@1**, not on the prediction loss, because the prediction loss went down happily
in all seven failed runs.

---

## Sources

- CAV-MAE, ICLR 2023 — https://sls.csail.mit.edu/publications/2023/YuanGong_ICLR2023.pdf · summary https://www.alphaxiv.org/abs/2210.07839 · code https://github.com/yuangongnd/cav-mae
- CAV-MAE Sync — https://arxiv.org/pdf/2505.01237
- VICReg — https://arxiv.org/abs/2105.04906
- data2vec / data2vec 2.0 — https://www.emergentmind.com/topics/data2vec
- How Does SimSiam Avoid Collapse Without Negative Samples? — https://ar5iv.labs.arxiv.org/html/2203.16262
- Contrastive Predictive Coding — https://www.emergentmind.com/topics/contrastive-predictive-coding
- JEPA family review (I-JEPA / V-JEPA, EMA target, L1 loss) — https://kodu.ut.ee/~hadachi/Lecture_Notes/JEPA.html
