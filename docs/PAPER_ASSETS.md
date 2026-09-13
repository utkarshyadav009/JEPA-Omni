# PAPER_ASSETS — the inventory, so writing is assembly and not archaeology

Every asset the paper needs: the claim in one sentence, the exact numbers, the artifact path,
and **the caveat that must travel with it**. Numbers here are pointers — `CANONICAL_NUMBERS.md`
is the source of truth, and if the two ever disagree, that file wins.

**Figure status: none built.** Each entry says what a figure would plot; nothing is drawn yet.

| # | asset | status | figure needed? |
|---|---|---|---|
| 1 | System result vs 8 baselines | numbers final | yes — bar chart |
| 2 | Gallery contamination | numbers final | yes — grouped bars |
| 3 | The padding leak | numbers final | yes — 2×2 + control |
| 4 | R@1 saturation | numbers final | yes — dual-axis curve |
| 5 | The naming probe (4 phases) | numbers final | yes — forward/backward + decay |
| 6 | Ego4D transfer | numbers final, **unreproducible** | no |
| 7 | Query predictor | numbers final, RUN-2-based | no |
| 8 | Errata | 14 entries, **none applied** | no |

---

## 1. System result — RUN-4 against eight baselines

**Claim.** A 155.9M-trainable audio-visual predictor reaches 41.77 / 41.88 R@1 on a held-out,
contamination-free 1,545-clip VGGSound gallery.

| | v→a R@1 | a→v R@1 | v→a R@5 | a→v R@5 |
|---|---:|---:|---:|---:|
| **RUN-4 `step18000`** | **41.77** | **41.88** | 71.97 | 72.10 |
| RUN-2 `step19000` (corrected) | 29.90 | 28.28 | 56.83 | 56.25 |
| ImageBind (1200.8M) | 29.64 | 29.45 | 58.06 | 55.99 |
| EquiAV (211.8M) | 21.81 | 24.66 | 45.11 | 47.31 |

Full 8-row table: `CANONICAL_NUMBERS.md` §2. Artifacts: `docs/artifacts/temporal_probe/p30_shard2.json`,
`data/{model}_retrieval_results.json`.

> **CAVEAT THAT MUST TRAVEL.** **This is not a win over ImageBind and must not be written as one.**
> We train on 197k VGGSound clips; ImageBind and EquiAV have never seen VGGSound. The gallery is
> held out at the *clip* level, not the *distribution* level, so the ~12-point margin is
> in-domain versus zero-shot transfer and confounds representation quality with domain
> adaptation. No parameter-efficiency claim may be built on it. Full audit: §2.1.

**Figure.** Grouped bars, 9 models × {v→a, a→v} R@1, with R@5 as a lighter overlay so the
saturation point (asset 4) is visible in the same panel. Bars for models trained on VGGSound
must be visually distinguished from zero-shot ones — the confound belongs *in* the figure, not
only in the caption.

## 2. Gallery contamination

**Claim.** Evaluating on a gallery overlapping the training corpus inflates R@1 by ~9.5 points
and R@5 by ~18.

| pair | contamination | v→a Δ R@1 | v→a Δ R@5 | a→v Δ R@1 | a→v Δ R@5 |
|---|---|---:|---:|---:|---:|
| **E-13** (primary) | 0% vs **100%** | **+9.58** | **+17.80** | **+9.26** | **+17.60** |
| E-5 (corroborating) | 0% vs 90.9% | +9.36 ± 0.08 | +18.43 ± 0.04 | +9.97 ± 0.00 | +16.95 ± 0.00 |

Galleries: `data/vggsound_eval_1545.txt` (sha256 `89307c6d4104…`),
`data/vggsound_testsplit_contaminated_1545.txt` (md5 `dbd405117674715acda08dd08a65f3c2`),
`data/vggsound_eval_1545_balanced.txt` (md5 `2eeaceef6866895cf02bb4204fb63835`).
Artifacts: `p03a_contamination_summary.json`, `p13_contaminated_gallery.json`.

> **CAVEAT.** **"The effect shrank 40%" is a false summary and must not appear.** R@1 shrinks
> versus the published ≥15.4 claim only because the leaked contaminated gallery was near ceiling
> (94.85 / 97.86). On R@5 the effect is *larger* than published. Report R@1 **and** R@5 together
> or the direction of the finding inverts.

**Figure.** Grouped bars, clean vs contaminated, R@1 and R@5 side by side — the point is that
the two metrics move in opposite directions relative to the published claim.

## 3. The padding leak

**Claim.** Unmasked, batch-dependent padding created a shared-nuisance shortcut worth ≈24 R@1,
and it was the mechanism behind our previously published number.

| | v→a R@1 | a→v R@1 |
|---|---:|---:|
| published (leaked) | 53.27 | 53.72 |
| **corrected** | **29.90 ± 0.04** | **28.28 ± 0.00** |

Established three independent ways: batch-size-1 agreement; a **uniform**-pad control that
*hurts*; a **random** per-clip pad control that **recovers 20 of the 23 points**.
Artifacts: `p02_summary.json`, `pad_sweep.json`, `pad_sweep_random.json`, `diag_bs1_{leak,fix}.json`.

**All eight baselines are structurally immune** (E-11): each pads to a fixed config-level length
then stacks equal-shaped tensors, so batch-dependent padding cannot occur. Confirmed empirically
at batch size 1 for ImageBind, AVSIAM and AudioCLIP (bit-identical) and EquiAV (±0.67).

> **CAVEAT.** The leaked path had **no error bar**; its run-to-run range was ≈2.5 R@1 purely from
> batch order. The corrected path is nearly seed-invariant. Quote the ± for the corrected number
> and state that the published one had none.
>
> **CAVEAT 2.** The fix's mechanism is **length normalisation, not masking** — see asset 4 /
> §7.2. Anywhere the text says "we fixed it by masking padding", that is wrong.

**Figure.** 2×2 (leaked/fixed × shuffled/deterministic) with the two pad controls as a third
panel. The uniform-pad bar going *down* is the most persuasive single element.

## 4. R@1 saturation — and the mechanism of the RUN-4 gain

**Claim (a).** Over RUN-4's last 5,000 steps R@1 is flat (41.34–41.77, seed range 0.06–0.13)
while R@5 gains 2.00 and effective rank gains 4.52, both monotonically.

**Claim (b).** The RUN-4 gain comes from **fixing the ambient length to 896 tokens**, not from
masking. The P2.2 control (mask OFF, everything else identical) is within noise of RUN-4:
32.56 / 34.89 vs 33.01 / 33.01 R@1, Δ eff_rank −0.38.

**Canonical effective rank: 74.26 (RUN-4) vs 37.72 (RUN-2)**, full-gallery, P2.3 matched grid —
roughly **2×**. The previously circulated ~12.5 in-training figure is **withdrawn**: it was never
measured on the same footing as anything it was compared against, and the "~6×" it implied is wrong.

Write-up: `docs/R1_SATURATION.md`. Artifacts: `p30_shard*.json`, `p23_matched_length_grid.json`.

> **CAVEAT.** **"RUN-4 was still improving at step 20,000" is RETRACTED** — stated from partial
> data through step 16,000. R@1 is flat from 16,000; R@5 and effective rank were still rising,
> which is a different and weaker claim.
>
> **CAVEAT 2.** RUN-2 and RUN-4 see different amounts of audio (~996 vs 896 tokens). The diagonal
> (each model at its own training length) is the defensible comparison, and the control that
> makes it defensible is RUN-2 at 896 scoring *worse* (22.21 / 19.42), not better.

**Figure.** Dual-axis: R@1 and R@5 (left) and effective rank (right) against step, 14k–20k, with
the seed-noise band shaded on R@1 so "flat" is visibly flat rather than asserted.

## 5. The naming probe — a negative result in four phases

**Claim.** The M2 fused latent is an **audio-visual scene representation, not a world-state**.

| phase | finding |
|---|---|
| 0–1, bin scramble | vision's temporal axis is provably unused: \|Δ\| ≤ 0.13 R@1 against a 0.06 null, cosine 0.99998 |
| 2, persistence | a plateau, not a decay — rescaled 0.783 (10 s) → 0.680 (60 s) for RUN-4 |
| 3, forward map | a learned map beats copying on R@1 but loses on cosine and R@5 |
| **3.2, forward information** | **forward ≈ backward. No arrow of time.** |

**The headline number**, within-file micro R@1 at Δ=10 s, ridge, 3 seeds, chance 2.198:

| model | forward | backward | gap | ±SE |
|---|---:|---:|---:|---:|
| RUN-2 `step19000` | 9.13 | 8.52 | +0.61 | 0.40 |
| **RUN-4 `step18000`** | 9.78 | 9.82 | **−0.04** | 0.15 |

Pre-registered threshold (fixed before inspection): ≥2.0 R@1, ≥3× SE, sign held at Δ=20 s.
**Both models fail it.** `IDENTITY`, which is direction-blind by construction, shows gaps up to
+0.61 — so ≈0.6 is the artifact floor and every learned gap sits at or below it.

Write-ups: `docs/TEMPORAL_STRUCTURE_PROBE.md`, `docs/FORWARD_INFORMATION_PROBE.md`.
Artifacts: `p32_abcd.json`, `p32e_falsifier.json`, `phase1_summary.json`, `phase2_persistence.json`.

> **CAVEAT — SCOPE, and it is the one most likely to be misread.** This measures a representation
> trained with `lam_pred = 0.0`, by ordinary multimodal alignment. It shows predictive structure
> **did not emerge spontaneously**. It is **not** evidence that an explicitly predictive objective
> would fail — that is RUN-5's question, and this probe is its baseline. Any sentence implying
> "predictive training doesn't work" is unsupported by this experiment.
>
> **CAVEAT 2.** Δ < 10 s is not reported at any stride, because both encoders' receptive field
> spans the full 10 s window, so a nearer target shares raw input with the query.
>
> **CAVEAT 3.** Self-retrieval calibration is **undefined by construction** (the ±10 s exclusion
> removes offset 0, which is the query itself). Report it as undefined; do not substitute a
> proxy without saying so.

**Figure.** Two panels: (a) forward vs backward R@1 against Δ, with the IDENTITY artifact floor
drawn as a shaded band — the whole result is that the bars sit inside the band; (b) decay to
chance at Δ=60 s with the shuffle control pinned at chance.

## 6. Ego4D transfer

**Claim.** The representation transfers to egocentric video: 27.60 / 27.00 R@1 on 674
sibling-excluded windows across 350 files, seed 43, `file_disjoint_verified: true`.

> **CAVEAT — MANDATORY, E-8.** **PERMANENTLY UNREPRODUCIBLE.** The raw Ego4D video is gone from
> this machine and all 674 windows are absent from the surviving 134,491-window cache (file
> overlap 0 of 350 — which independently confirms the split really was file-disjoint). No future
> checkpoint can be compared against these numbers. If the paper cannot carry that caveat in the
> table itself, **cut the asset** rather than present it as a live result.
>
> Unaffected by the padding leak (E-4): that harness decodes one window at a time at batch size 1.

**Figure.** None. A single unreproducible row does not earn one.

## 7. Query predictor

**Claim.** A query-style predictor reaches 0.7372 cross-clip R@1 (`sig_runD_proj768`) and 0.4888
(`sig_runA_matched3stream`), n=624; ears-following drops 0.650 → 0.070 under the congruence mask.

Confirmed clean (E-6): batch-size-1 re-scoring moved these by ≤0.016, and `within_clip` /
`swapped_query` were bit-identical for `sig_runA`. Structurally immune — one side of the
retrieval is text, which has no access to a clip's pad count.

> **CAVEAT.** **These are RUN-2-based, and no downstream retrain is planned for this submission.**
> State that explicitly; a reader will otherwise assume they sit on the headline RUN-4 system.
>
> **CAVEAT 2.** `sig_runD`'s SigLIP2 scene features were lost with `/dev/shm`. Re-scoring used a
> re-extracted 900-clip subset, so the bs48-vs-bs1 *delta* is exact but the *absolute* value
> (0.8181 / 0.7356 vs logged 0.8114 / 0.7372) is a near-match, **not a reproduction**.

## 8. Errata — corrections that must be visible, not silently swapped

14 entries in `docs/ERRATA_PROPOSED.md`. **None has been applied to a published table.**
These are corrections to numbers that have already been circulated, so they must appear in the
paper as corrections:

| entry | correction |
|---|---|
| **E-1** | 53.27 / 53.72 → **29.90 / 28.28** (padding leak) |
| **E-5 / E-13** | contamination ≥15.4 → **+9.58 R@1 / +17.80 R@5**, with the "shrank 40%" framing rejected |
| **E-8** | Ego4D held-out is unreproducible |
| **E-11** | the ImageBind comparison is a tie at RUN-2 — and, after the P4.3 audit, **non-equivalent** at RUN-4 |
| **E-12** | `ICLR_RESULTS.md` §2 is stale for 7 of 8 baseline rows |
| **E-14** | `best.pt` is selected on training `loss_ema`; **a live bug**, worked around post hoc for RUN-4 |

> **CAVEAT.** Each requires human approval before it touches a published table. Two disputes are
> still open and must not be quietly resolved: Wav2CLIP's contamination flag (§13.2) and the
> RUN-2 `step20000` direction ordering (§13.3).

---

## Open items before submission

1. **Decide the ImageBind framing.** Asset 1 currently cannot claim a win. Either accept the
   non-equivalent framing or run the zero-shot AVE evaluation (§8, 3,230 clips, 0 overlap) that
   would settle it.
2. **Approve or reject the 14 errata.** The paper cannot be written around numbers whose status
   is undecided.
3. **Resolve §13.2 and §13.3.**
4. **Decide whether asset 6 (Ego4D) ships at all** given E-8.
