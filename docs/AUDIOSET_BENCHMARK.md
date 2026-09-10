# AudioSet-527 frozen probe — audio-only (partial replication)

**Date:** 2026-09-09. **Checkpoint:** `m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`,
sha256 verified `e1a8231ec9fbae6c`. **Artifacts:** `docs/artifacts/audioset_probe_ours.json`,
features at `/mnt/Raid-Storage-2/utkarsh-data/audioset_feats/{bal_train,eval}.pt`.

> ## THIS IS A PARTIAL REPLICATION — the A column only
> MJEPA's Table 1 reports three settings: audio-only (A), video-only (V) and audio-video (A-V).
> **We can report only A.** No public mirror carries AudioSet *video*: Google releases 128-d
> VGGish features only, and every audio mirror checked (`agkphysics/AudioSet`,
> `yangwang825/audioset`, `confit/audioset-qiuqiangkong`) is audio-only. Obtaining V and A-V
> means re-scraping 42,543 YouTube clips; at our measured yield that is days of work for
> availability-biased coverage, so it was not attempted. **MJEPA's headline is the A-V column,
> which we cannot fill.**

> ## THE ASYMMETRY — read before any comparison
> **M2 was trained on VGGSound + Ego4D and has never seen AudioSet.** Every number in Table 1
> below is therefore **zero-shot cross-dataset transfer**. MJEPA, CAV-MAE, CAV-MAE Sync and
> EquiAV all **pretrain on AudioSet**, so for them the same evaluation is **in-domain**. This
> disadvantages our raw number and is the point of reporting it.

> ## LEAKAGE FLAG — 38.3% of the eval split was removed
> The AudioSet eval split overlaps VGGSound, which M2 *did* train on. Of the 17,141 eval clips
> in the mirror, **6,569 (38.3%) share a YouTube ID with a VGGSound clip in M2's training
> corpus** and were excluded; **10,572 remain**. Verified two independent ways (feature-cache
> listing and the VGGSound CSV label files) — both give 7,415/20,371 = 36.40% against the full
> official eval split. Exclusion is at **video level** (shared YouTube ID): an AudioSet eval clip
> and a VGGSound training clip may be different 10 s windows of the same video, so this is the
> conservative choice. AudioSet-Strong (RUN-3) and Ego4D contribute **zero** overlap.
> After exclusion, 498 of 527 classes have at least one positive and are scored; 29 do not.

## Table 1 — what WE measured (frozen features, audio-only, A column)

Identical protocol for every row: frozen precomputed features, BCE-with-logits, 527-class
multi-label, AdamW + CosineAnnealingLR, **lr 1e-3, weight decay 1e-4, 30 epochs, batch 256,
seed 0**, no per-row tuning. n_train = 18,683 (balanced train), n_eval = 10,572.
Linear probe = `nn.Linear(d, 527)` on the mean over the full token sequence. Attentive probe =
one learned query → `MultiheadAttention(8 heads)` → LayerNorm → `Linear(d, 527)` over a
32-step sequence.

| variant | dim | probe | mAP | mAUC | d-prime |
|---|---|---|---|---|---|
| ambient — WavJEPA-base | 768 | linear | 8.69 | 87.60 | 1.634 |
| ambient — WavJEPA-base | 768 | **attentive** | **18.31** | 92.01 | 1.988 |
| world-state — vision ZEROED | 1024 | linear | 18.59 | 92.64 | 2.049 |
| world-state — vision ZEROED | 1024 | **attentive** | **20.00** | **93.93** | **2.190** |
| *CONTROL — label-shuffled* | *1024* | *linear* | *0.59* | *50.12* | *0.004* |
| *CONTROL — label-shuffled* | *1024* | *attentive* | *0.62* | *49.42* | *−0.020* |
| *CONTROL — matched-stats random* | *1024* | *linear* | *0.56* | *50.42* | *0.015* |

**`world-state (vision ZEROED)` is NOT an A-V number.** It is an audio-only forward pass through
a fusion model: the vision slot carries zeros of the training geometry (32×16 tokens, staircase
binning gate asserted), because AudioSet video does not exist locally. Reporting it as A-V would
be false.

**Controls.** Both falsifiers land at chance — mAUC 49.4–50.4, d′ ≈ 0, mAP 0.56–0.62 against a
2.33-labels-per-clip multi-label prior. The signal in the rows above is not an artifact of
dimensionality, probe capacity, or the metric.

## Result 1 — the attentive-vs-linear gap, independently verified

MJEPA claims linear probing materially understates frozen encoders and re-evaluates baselines
with an attentive probe (they report CAV-MAE Sync moving 8.7 → 21.66 mAP). **We independently
confirm the effect on our own features, and it is large:**

| features | linear mAP | attentive mAP | gap |
|---|---|---|---|
| WavJEPA-base ambient tokens | 8.69 | 18.31 | **+9.62 (2.11×)** |
| M2 world-state | 18.59 | 20.00 | +1.41 (1.08×) |

The gap is **7× larger on the raw WavJEPA token sequence than on the World-State**, and the
mechanism is visible: M2's `encode_world_state` already applies a learned single-query attentive
pool, so the attentive probe largely duplicates work M2 has done. The World-State's linear score
(18.59) nearly matches the ambient stream's *attentive* score (18.31) — i.e. **M2's fusion pool
delivers, to a linear classifier, most of what an attentive probe extracts from the raw tokens.**
This is a claim about the pooling, not about audio-visual fusion: the vision stream was zeroed.

## Table 2 — published frozen figures (NOT re-measured here, NOT comparable row-to-row)

Quoted from MJEPA's Table 1 for context only. **Different protocol from Table 1 above** (their
attentive-probe architecture is not stated in the material available to us; ours is documented
above), and all of these pretrain on AudioSet whereas ours does not.

| method | A | V | A-V | pretrain includes AudioSet |
|---|---|---|---|---|
| CAV-MAE | 19.38 | 18.14 | 34.59 | yes |
| CAV-MAE Sync | 21.66 | 16.20 | 28.50 | yes |
| EquiAV | 34.25 | 18.60 | 38.60 | yes |
| MJEPA ViT-L | 38.89 | 25.38 | 42.90 | yes |

## Table 3 — published FINE-TUNED figures (never to be mixed with frozen numbers)

End-to-end fine-tuned backbones. Not comparable to any frozen probe, including ours.

| method | AudioSet mAP | protocol |
|---|---|---|
| AV-JEPA | 32.7 / 29.6 | end-to-end fine-tuned |
| CAV-MAE | 51.2 / 42.0 | end-to-end fine-tuned |

## What did NOT complete

- **B5 — baseline audio towers through our probe.** Not run. All five checkpoints are present
  locally (CAV-MAE, CAV-MAE Sync, EquiAV, ImageBind, AudioCLIP, Wav2CLIP), so this is executable;
  it did not run tonight. Until it does, **Table 1 has no baseline rows measured by us**, and
  Table 2 cannot be compared to it.
- **Part C — AudioSet retrieval gallery.** Target was balanced 5-per-class from the clean eval
  split (442 classes × 5 = 2,210). Scrape achieved **478 clips from 676 attempts (70.7% yield)**
  before stalling, and only **57 classes reached ≥5 clips**. That is not a usable balanced
  gallery, so **no AudioSet retrieval row is reported.** Not padded. Failure modes: 179 other,
  17 unavailable, 2 HTTP 403.
- **V and A-V columns.** Structurally unavailable — see the banner above.

## Reproduction

```
python scripts/audioset_extract_features.py --split bal_train --out .../bal_train.pt
python scripts/audioset_extract_features.py --split eval      --out .../eval.pt
python scripts/audioset_probe.py --train .../bal_train.pt --eval .../eval.pt \
    --clean-eval-ids /tmp/audioset_eval_clean_ids.json --out docs/artifacts/audioset_probe_ours.json
```

---

## Table 1b — all six baselines through the IDENTICAL probe (measured by us)

`scripts/audioset_probe_baseline.py` imports `SEED/EPOCHS/BATCH/LR/WD` and `run()` directly from
`scripts/audioset_probe.py`, so a baseline row cannot differ from ours in optimiser, LR, schedule,
epochs, batch, seed or metric. Same n_train 18,683 / n_eval 10,572, same 498 scored classes, same
leakage exclusion.

> **`linear` is not one thing across models.** Each baseline's own retrieval script returns an
> **L2-normalised** embedding (measured norm 1.0000 for CAV-MAE and ImageBind) — a *retrieval
> head*, not a pooled encoder feature. Ours is a **raw mean over tokens** (norm 3.1925);
> cos(feat_mean, tokens.mean) = 0.5689 for CAV-MAE. Probing the native embedding gives baselines
> 1.05–2.48 mAP, barely above our 0.56 matched-stats control — an artifact of the mismatch, not a
> property of the models. **`linear (matched)` = mean over tokens is the comparable column**, and
> it exists only for towers exposing a real token sequence.

| model | dim | tokens? | linear (native, NOT comparable) | linear (matched) | attentive |
|---|---|---|---|---|---|
| AudioCLIP | 1024 | **no** | 1.05 | — | — |
| CAV-MAE | 768 | yes | 1.27 | 12.48 | 18.30 |
| CAV-MAE Sync | 768 | yes | 1.69 | 18.87 | 24.54 |
| EquiAV | 512 | **no** | 1.16 | — | — |
| ImageBind | 1024 | yes | 2.48 | 10.81 | 15.96 |
| Wav2CLIP | 512 | **no** | 1.55 | — | — |
| **Ours — ambient (WavJEPA-base)** | 768 | yes | n/a (raw pooled) | **8.69** | **18.31** |
| **Ours — world-state (vision ZEROED)** | 1024 | yes | n/a (raw pooled) | **18.59** | **20.00** |

### We do not win this column

**CAV-MAE Sync beats our best variant on both comparable columns** — 24.54 vs 20.00 attentive,
18.87 vs 18.59 linear (matched). Reported because it is what we measured. Two things frame it
without softening it: CAV-MAE Sync **pretrains on AudioSet**, so this is in-domain for it and
zero-shot cross-dataset transfer for us; and our variant is an **audio-only forward pass with the
vision stream zeroed**, i.e. a fusion model run on half its inputs. Neither makes 24.54 < 20.00.

**AudioCLIP, Wav2CLIP and EquiAV expose no token sequence at all**, so no comparable column exists
for them. Their native-embedding numbers (1.05–1.55) are shown for completeness and must not be
read as A-column scores.

### Harness validation

| model | ours (attentive) | MJEPA published | delta |
|---|---|---|---|
| CAV-MAE | 18.30 | 19.38 | −1.08 |
| CAV-MAE Sync | 24.54 | 21.66 | +2.88 |

Both land within ~3 mAP of the published figure for the same model under an attentive probe,
which is reasonable agreement given that MJEPA's probe architecture is not stated in the material
available to us and our eval subset differs after leakage exclusion. Published values live in
Table 2 and are **not** re-measured here.

**Cross-task inconsistency.** ImageBind is the *strongest* baseline on our VGGSound retrieval
gallery (29.45 / 29.64 R@1, `docs/BASELINE_1545.md`) but mid-pack here (10.81 / 15.96). Retrieval
quality and frozen-probe classification do not rank models the same way.

Artifact: `docs/artifacts/audioset_probe_baselines.json` — every row carries
`comparable_to_ours` and `preprocessing_source` with the exact file:line range reused.

