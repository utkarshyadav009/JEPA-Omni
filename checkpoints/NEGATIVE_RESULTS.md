# NEGATIVE_RESULTS.md

Results that did not pass their pre-registered gate, or that reversed an
earlier positive finding. Kept alongside `RESULTS_TABLE.md` deliberately —
falsifier discipline in this project means a negative result is reported
with the same rigor as a positive one, not omitted.

## RUN-3: AudioSet addition — both gates FAILED, and the result is CONFOUNDED

`checkpoints/m2_run3_vggsound197k_ego4d134k_audioset21k` — VGGSound197k +
Ego4D134k + 8,588 AudioSet-Strong clips (after the P1-P5 preflight filter).

| Eval | Locked M2 (RUN-2) | RUN-3 (+AudioSet) | Gate |
|---|---|---|---|
| VGGSound R@1 (v→a / a→v) | 53.27% / 53.72% | 23.04% / 12.56% | FAIL |
| Ego4D sibling-excl R@1 (v→a / a→v) | 27.60% / 27.00% | 8.75% / 12.91% | FAIL |
| Ego4D within-modality cosine, ambient side | 0.3893 | 0.3085 | improved (best in study) |
| Ego4D within-modality cosine, vision side | 0.4358 | 0.5299 | worse |

**Why this is not attributable to AudioSet alone**: two variables changed
simultaneously between RUN-2 and RUN-3, not one:
1. In-batch negatives: 200×200 → 176×176 (forced down by AudioSet's longer,
   more memory-expensive clips at the same batch size).
2. Ambient token cap (`max_ambient_t`): 1024 → 768 (same cause — AudioSet
   clips run up to 244s vs VGGSound/Ego4D's fixed ~10s, and the cap was
   lowered to keep peak memory bounded before the `_cap_ambient_len`
   ordering bug was found and fixed).

Both changes independently reduce retrieval quality (fewer negatives = an
easier contrastive task with weaker gradient signal; a lower ambient cap
throws away information on long clips). The AudioSet-Strong preflight
(P1-P5) itself passed cleanly (see below) — the corpus that was ADDED was
judged sound. What broke was how RUN-3 accommodated it, not necessarily
the data itself. **This result should not be read as "AudioSet hurts
performance."** It should be read as "adding AudioSet at RUN-3's settings,
with negatives and ambient cap both reduced at the same time, hurt
performance" — a materially weaker and different claim.

**Root cause found, not fixed within the freeze**: `train_m2.py`'s
`_cap_ambient_len` truncation was applied AFTER `.to(device)`, letting
outlier-length AudioSet clips spike GPU memory before truncation — this
was RUN-3's proximate OOM cause and the reason negatives/cap were both
cut. Fixed post-hoc (moved the cap before device transfer, added a
`--max-ambient-t` CLI override), but per the FREEZE directive **no
retrain was run with the fix** — Action100M and AudioSet are logged as
future work, not built on further.

**Standing methodological note carried since RUN-2's own launch**: "Keep
192 negatives fixed across all runs — changing negative count at the same
time as corpus size is two variables" was the explicit instruction this
project was operating under. RUN-3 violated it under memory pressure; the
violation is the reason this result is unattributable, not a surprise.

## Ego4D gains partially lost under corpus scaling (RUN-1)

RUN-1 (199,007 VGGSound + 17.1k Ego4D, single scaling step before Ego4D
was itself expanded) passed its VGGSound gate decisively (55.15%/55.53%)
but FAILED its pre-registered Ego4D gate (11.57%/10.68% vs required
18.40%/18.40%; shuffle-sanity gap 0.2140 vs required 0.2453). Mechanism
confirmed, not assumed: Ego4D's in-batch share fell as VGGSound grew to
full scale, diluting its gradient signal per batch. This directly
motivated RUN-2's Ego4D expansion (17.1k → 134k), which fixed it — this
negative result is the reason the locked M2 exists in its current form,
not a dead end.

## Ego4D within-modality cosine gate — never met

Across every M2 checkpoint in this project's history, including the
locked one, the within-modality cosine gate (target ≤0.25, meaning
same-modality embeddings for different clips should be relatively
spread out, not collapsed) has not been met. Locked M2: 0.4358 (vision) /
0.3893 (ambient) — closest-yet, not passing. This is disclosed as an open
gap, not resolved by the freeze.

## M3/M4b joint-exposure fine-tune cost

The first (and to date only) joint-exposure fine-tune of the M3 connector
+ M4b projector cost −9% M3 word-overlap F1 and −19% M4b semantic cosine
relative to their frozen-LLM baselines. This was accepted as a one-time,
bounded cost (the standing falsifier-tracking rule exists specifically to
catch *compounding* drift across further stages, which did not occur —
only one joint-training stage has been run).
