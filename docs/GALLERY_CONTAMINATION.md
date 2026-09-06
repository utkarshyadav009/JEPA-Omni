# Gallery Contamination: VGGSound retrieval inflates ~15 points when the gallery overlaps training

**Status:** measured 2026-09-06. Instrument: `data/vggsound_eval_1545_balanced.txt`
(+ `.provenance.json`). Checkpoint under test: `m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`
(sha256 `e1a8231e…`), unchanged and unretrained.

## What was measured

Building a protocol-matched retrieval gallery (balanced 5-per-class, AV-JEPA's construction)
produced a number 15 points higher than our reported figure. The cause is not the gallery
construction — it is that the new gallery overlaps the training corpus.

| gallery | construction | clips seen in training | a→v R@1/5/10 | v→a R@1/5/10 |
|---|---|---|---|---|
| `vggsound_eval_1545.txt` (reported) | unstratified random draw, 307/309 classes, 1–12 per class | **0 / 1,545** | **53.14 / 81.81 / 88.28** | **53.46 / 80.39 / 87.83** |
| `vggsound_eval_1545_balanced.txt` | balanced 5-per-class, 309 classes × 5 | **1,404 / 1,545 (90.9%)** | 68.61 / 94.69 / 97.28 | 68.41 / 94.17 / 97.73 |
| **delta** | | | **+15.47** | **+14.95** |

Both rows: same checkpoint, same script (`scripts/eval_checkpoint_gallery.py`), same cache,
same N=1,545, `clips_seen` asserted, full-gallery cosine similarity. The first row is a
re-run control against the documented 53.27 / 53.72 and reproduces it to within 0.13–0.26 pts.

## Why the delta is not explained by gallery construction

Exact-clip retrieval is hardest when same-class clips compete. A balanced 5-per-class gallery
gives every clip exactly **4.00** same-class distractors. Our random draw averages **4.91**
(51.1% of clips sit above 4, 28.8% below). So the balanced gallery is marginally *easier* on
distractor density — worth perhaps 1–2 points, an order of magnitude short of +15.4.

The remainder is memorisation:

```
feature cache (VGGSound)      199,007 clips
held out (eval list)            1,545
M2 training corpus            197,462     <- matches the documented figure exactly
```

`train_m2.py:979` passes `exclude_ids=eval_ids_set`, and that set is read from
`data/vggsound_eval_1545.txt` **alone**. Nothing else was withheld, so 1,404 of the balanced
gallery's 1,545 clips were training data.

## The consequence for VGGSound benchmarking generally

**Of the official VGGSound test split's 15,446 clips, 13,894 (90.0%) are in this training
corpus.** Only 1,552 lie outside it. Any gallery drawn from the official test split without an
explicit exclusion list will be ~90% contaminated against a model trained on full VGGSound —
and will report roughly 15 points too high.

This is a reproducibility observation, not an accusation. Most published VGGSound retrieval work
states the gallery *size* but not its held-out construction, and the two are independent: N=1,545
is satisfied identically by a clean gallery and a 90.9%-contaminated one, and the difference
between them here is larger than the gap between most published methods. Absent a stated
exclusion list, a VGGSound retrieval figure cannot be assumed to be held-out.

## What this does and does not change for our results

- **Unchanged.** 53.27 / 53.72 is measured on genuinely held-out clips and stands. Everything
  downstream of the locked checkpoint is unaffected.
- **Changed.** We do not claim a protocol match with AV-JEPA. Our gallery is a random draw, not
  balanced 5-per-class.
- **Not available without retraining.** A matched balanced gallery would need M2 retrained with
  the full official test split excluded (~15.4k clips withheld rather than 1,545). We did not do
  this: it would invalidate the provenance of every downstream result for one comparison.
  Recorded as future work.

## Files

- `data/vggsound_eval_1545_balanced.txt` — the instrument (N=1,545, 309×5, seed 0, md5
  `2eeaceef6866895cf02bb4204fb63835`). Retained deliberately: it produced this measurement, and
  it is the gallery a future retrained checkpoint should be evaluated on.
- `data/vggsound_eval_1545_balanced.provenance.json`
- `scripts/build_balanced_gallery.py`
