# Baseline Comparison Table — Fixed 1,545-clip VGGSound Gallery

**Purpose.** The dissertation's Table 1 is "contextual" because published baseline numbers are
quoted from their original papers, each on a different (and mostly unstated) gallery size. This
table replaces that with a *directly comparable* measurement: every model below — including our
own `m2_run2` checkpoint — is run through our own code on the exact same fixed 1,545-clip
VGGSound eval gallery (`data/vggsound_eval_1545.txt`), under the same retrieval protocol
(full-gallery cosine similarity, R@1/R@5/R@10, both directions). No number in this table is
copied from a paper.

**Protocol.** For each clip, the eval list gives a VGGSound `clip_id`. The corresponding audio
and video are decoded from the same source file
(`/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video/<clip_id>.mp4`). Each model
computes one audio embedding and one visual embedding per clip using **its own** published
preprocessing (read from its own inference code — not ours). Retrieval is scored as the full
N×N cosine-similarity matrix over all clips actually evaluated; R@k is the fraction of queries
whose ground-truth match appears in the top k. `clips_seen` is asserted to equal 1,545 for every
row unless stated otherwise (in which case the exact evaluated count and the reason for the
shortfall are given). **13 of the 1,545 clip IDs have no `.mp4` in the extracted video source
directory** (a pre-existing gap in this machine's local VGGSound extraction, confirmed
independently, not specific to any one model) — every baseline below that needed raw video hits
this same gap and is evaluated on the resulting 1,532-clip intersection, stated as `1532/1545` in
its row. Our own `m2_run2` row is unaffected (it reads pre-extracted feature tensors, not the raw
mp4s) and reports the full 1,545.

**Contamination flag.** Several public checkpoints were pretrained on VGGSound itself, so our
gallery clips may be in-distribution for them, while our own `m2_run2` training run explicitly
excluded these exact 1,545 clips. This asymmetry favours the baselines; flagging it is the point
of this column, not an oversight.

**Precise statement of what we held out.** M2's training corpus excluded the 1,545 clips of
`data/vggsound_eval_1545.txt` specifically. It did **not** exclude the official VGGSound test
split: of that split's 15,446 clips, **13,894 (90.0%) are in the 197,462-clip training corpus**
(`train_m2.py:979` passes only this eval list as `exclude_ids`). The 53.27 / 53.72 figures below
are measured on genuinely held-out clips and are unaffected by this. What it does mean is that a
different draw from the official test split would *not* be held out — measured directly in
`docs/GALLERY_CONTAMINATION.md`, where a balanced 5-per-class gallery that is 90.9% training
clips scores 68.61 / 68.41, i.e. **+15.4 points** on the same checkpoint and script.

**Gallery construction.** `data/vggsound_eval_1545.txt` is an **unstratified random draw** from
the official test split: 307 of 309 classes, 1–12 clips each, mean same-class distractors
**4.91**. AV-JEPA's retrieval gallery is **balanced 5-per-class** (309×5 = 1,545, mean same-class
distractors exactly **4.00**). The N matches by coincidence, not construction. Ours is therefore
marginally *harder* on distractor density, but it is **not protocol-matched and we do not claim it
is**. A matched comparison would require retraining M2 with the full official test split excluded;
recorded as future work rather than performed, because it would invalidate the provenance of every
downstream result for a single comparison.

- **IN-DISTRIBUTION** — model's pretraining corpus includes VGGSound.
- **HELD-OUT** — model's pretraining corpus does not include VGGSound (per its own paper/README).
- **UNKNOWN** — pretraining corpus for this checkpoint variant could not be verified.

Chance-level R@1 on a 1,545-clip gallery is 1/1545 ≈ **0.065%**.

---

## Results

| Model | Params | Training corpus | Contamination | a→v R@1 | a→v R@5 | a→v R@10 | v→a R@1 | v→a R@5 | v→a R@10 | N clips | Preprocessing | Checkpoint | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Chance level** | — | — | — | 0.065% | 0.32% | 0.65% | 0.065% | 0.32% | 0.65% | 1545 | — | — | — |
| **Ours (`m2_run2`, step19000, LOCKED)** | 155.9M trainable (predictor+heads) + 326.0M frozen V-JEPA2 ViT-L + 196.3M frozen WavJEPA-base = 678.2M total pipeline | 197,462 VGGSound + 134,491 Ego4D (training corpus excludes these 1,545 eval clips specifically; it does NOT exclude the official VGGSound test split — see "Precise statement of what we held out") | HELD-OUT for this gallery (these 1,545 clips excluded from training; 90.0% of the official test split was not) | 53.27% | 81.62% | 88.67% | 53.72% | 80.32% | 88.09% | 1545 | See `models/vision_encoder.py` (V-JEPA2 ViT-L, 16 frames, 256px) + `models/audio_encoder.py` (WavJEPA-base, raw 16kHz PCM) | `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` (sha256 `e1a8231e...`) | 2026-07-28 (training log); table compiled 2026-09-02 |
| CAV-MAE | 190.7M (both towers) | AudioSet + VGGSound | **IN-DISTRIBUTION** | 12.23% | 28.22% | 36.50% | 14.24% | 27.64% | 36.83% | 1545 | Log-mel (n_fft=1024, hop=160, n_mels=128, sr=16k, mean=-4.2677393/std=4.5689974), center frame 224×224 ImageNet norm — `scripts/cavmae_retrieval.py` | `~/models/cav-mae/audio_model.25.pth` (sha256 `a035b04c...`) | 2026-07-10 |
| CAV-MAE Sync‡ | 190.7M | AudioSet + VGGSound (same weights as CAV-MAE row — see caveat) | **IN-DISTRIBUTION** | 2.02% | 7.44% | 11.42% | 4.90% | 13.45% | 19.58% | 1532/1545 | Kaldi fbank (128 mel, htk_compat, mean=-5.081/std=4.4849, 1024→416-frame sync-window crop); 16 evenly-spaced frames, 224×224 bicubic, ImageNet norm; `CAVMAESync` CLS-token embedding, `diagonal_mean` aggregation across 16 sync-frame pairs — `scripts/cavmae_sync_retrieval.py` | `.../cavmae_sync/cav_mae_sync.pth` (sha256 `a035b04c...`) | 2026-09-02 |
| AVSiam (Base)† | 333.2M | AudioSet-2M | HELD-OUT | 2.48% | 7.90% | 12.21% | 3.00% | 9.86% | 15.08% | 1532/1545 | Kaldi fbank (128 mel, htk_compat, mean=-5.081/std=4.4849, 1024 frames); single frame @50% duration, 224×224 CLIP-style resize/crop, ImageNet norm; mean-pooled patch tokens, L2-norm — `scripts/avsiam_retrieval.py` | `.../avsiam/as2m_pretrained.20.pth` (sha256 `ec968abe...`) | 2026-09-02 |
| EquiAV | 211.8M | AudioSet-2M | HELD-OUT | 24.80% | 47.13% | 57.57% | 21.80% | 45.56% | 55.22% | 1532/1545 | Kaldi fbank (128 mel, htk_compat, norm mean=-4.346/std=4.332, 1024 frames, reshaped ×3 channels); middle frame, 224×224 bicubic resize/crop, ImageNet norm; `MainModel.forward_feat()`, L2-norm — `scripts/equiav_retrieval.py` | `.../equiav/equiav_pretrained.pth` (sha256 `a006fdb8...`) | 2026-09-02 |
| ImageBind | 1200.8M | Audio: AudioSet. Vision: web-scale image-text + video-audio pairs | HELD-OUT | 29.70% | 56.01% | 66.58% | 29.70% | 58.09% | 68.67% | 1532/1545 | ImageBind's own `data.py` defaults, unmodified: audio 2s×3-clip kaldi fbank (128 mel, target_len=204, norm mean=-4.268/std=9.138) mean-pooled; video 2s×5-clip × 2-frame subsample × 3-crop (15 views), CLIP norm, mean-pooled — `scripts/imagebind_retrieval.py` | `.../imagebind/imagebind_huge.pth` (sha256 `d6f6c22b...`) | 2026-09-02 |
| LanguageBind | 709.3M (video+audio towers) | VIDAL-10M (self-collected, language-anchored binding) | HELD-OUT | 7.57% | 20.82% | 30.94% | 10.31% | 26.83% | 38.45% | 1532/1545 | Video: decord, 8 uniform frames, 224×224 CLIP norm (train-time h-flip disabled for eval). Audio: 16kHz Kaldi fbank (112 mel, htk_compat, target_len=1036, ×3 tile, norm mean=-4.2677393/std=4.5689974×2) — both from each checkpoint's own `vision_config`. Direct audio↔video similarity (both independently CLIP-aligned to language), per repo's own `inference.py` demo — `scripts/languagebind_retrieval.py` | `LanguageBind_Video_FT` + `LanguageBind_Audio_FT` (HF hub; sha256 `9c848bd4...` / `b0fbb6a2...`) | 2026-09-02 |
| AudioCLIP | 134.1M (32.1M audio tower) | Audio: AudioSet. Image/text: frozen/fine-tuned OpenAI CLIP RN50 (~400M web pairs) | HELD-OUT | 0.20% | 0.59% | 1.17% | 0.91% | 2.74% | 3.98% | 1532/1545 | 44.1kHz raw waveform, center-crop/pad to 5.0s, AudioCLIP's internal FBSP frontend (n_fft=2048, hop=561, blackmanharris); center frame, 224×224 bicubic, CLIP norm; both towers L2-norm — `scripts/audioclip_retrieval.py` | `.../audioclip/repo/assets/AudioCLIP-Full-Training.pt` (sha256 `2441d35b...`) | 2026-09-02 |
| Wav2CLIP | 163.0M (11.7M audio tower + 151.3M CLIP) | **VGGSound** (audio tower distilled from frozen CLIP ViT-B/32) | **IN-DISTRIBUTION** (verified 0/1545 eval IDs in VGGSound's own 183,730-clip train split; caveat: not de-duplicated against near-identical re-uploads) | 5.35% | 12.99% | 19.19% | 6.46% | 16.51% | 23.37% | 1532/1545 | Raw 16kHz mono waveform → Wav2CLIP's own ResNet18 (internal Spectrogram n_fft=512/hop=353, log, global norm) → distilled MLP head; 10 evenly-sampled frames, open_clip ViT-B/32 'openai' preprocessing, frozen CLIP image tower, mean-pooled — `scripts/wav2clip_retrieval.py` | `.../wav2clip/hub/checkpoints/Wav2CLIP.pt` (sha256 `264f8796...`) | 2026-09-02 |
| MaViL | — | — | — | **NOT AVAILABLE** | | | | | | | — | — |
| CrossMAE (AV, Guo et al.) | — | — | — | **NOT AVAILABLE** | | | | | | | — | — |
| AV-JEPA / MJEPA | — | — | — | **NOT AVAILABLE** | | | | | | | — | — |

†**AVSiam caveat.** Checkpoint loaded 670/963 state-dict keys (293 missing, all under an
`ast_base.*` prefix — an unused/legacy branch in the released class, or a genuine
partial-checkpoint artifact; not fully resolved). Numbers are reported as measured but should be
read with more caution than the fully-loaded rows above.

‡**CAV-MAE Sync caveat — checkpoint identity, not a measurement discrepancy.** The file downloaded
by the CAV-MAE Sync repo's own `pretrained_models/get_pretrained_model.sh` (a Google Drive link,
independently re-downloaded and byte-verified here) is **sha256-identical** to the original
CAV-MAE `audio_model.25.pth` checkpoint — same size (762,886,272 bytes), same hash
(`a035b04cf615...`), and the same internal pickle path (`audio_model.25/data.pkl`,
`module.modality_a` / `module.modality_v` key names). This is the authors' own public release, not
a download error on our end. The `CAVMAESync` model class loads it with a strict, complete key
match, so the eval below ran successfully — but it is **not evaluating a distinct "Sync"
fine-tuned model**; it is the plain CAV-MAE weights run through CAV-MAE Sync's different
16-sync-frame preprocessing and `diagonal_mean` aggregation code, which is presumably why its
numbers differ substantially from the CAV-MAE row above despite identical weights. Until the
authors fix this release, **CAV-MAE Sync does not have a genuinely distinct public checkpoint** —
this row is included for transparency, not as a valid independent baseline.

Both real-world caveats above were caught during this table's construction, not assumed away —
consistent with the "measure it or mark it NOT AVAILABLE" rule this table exists to enforce.

---

## Availability Survey (Step 1)

11 candidates surveyed 2026-09-02. 8 had a real downloadable checkpoint; 3 did not.

| # | Model | Status | Checkpoint source |
|---|---|---|---|
| 1 | CAV-MAE | FOUND (evaluated) | github.com/YuanGongND/cav-mae |
| 2 | CAV-MAE Sync | FOUND (but see ‡ caveat above — release is sha256-identical to CAV-MAE) | github.com/edsonroteia/cav-mae-sync |
| 3 | AVSiam | FOUND | github.com/GenjiB/AVSiam (HF-hosted) |
| 4 | EquiAV | FOUND | github.com/JongSuk1/EquiAV (Google Drive) |
| 5 | MaViL | NOT AVAILABLE | repo archived, no weights ever released |
| 6 | CrossMAE (AV) | NOT AVAILABLE | no code repository exists |
| 7 | ImageBind | FOUND | github.com/facebookresearch/ImageBind |
| 8 | LanguageBind | FOUND | github.com/PKU-YuanGroup/LanguageBind (HF hub) |
| 9 | AudioCLIP | FOUND | github.com/AndreyGuzhov/AudioCLIP (release asset) |
| 10 | Wav2CLIP | FOUND | github.com/descriptinc/lyrebird-wav2clip (pip) |
| 11 | AV-JEPA / MJEPA | NOT AVAILABLE | paper-only (arXiv 2606.25225), no code/weights |

AVSiam also has a "Base+" variant (AudioSet-2M+VGGSound+ACAV2.4M, IN-DISTRIBUTION) not evaluated
here for time; only the cleaner "Base" (AudioSet-2M only, HELD-OUT) variant is in the table above.

---

## Artifacts

- **This table**: `docs/BASELINE_1545.md`
- **Figure**: `figures/baseline_1545.png` / `.pdf` — grouped R@1 bar chart, both directions, our
  system highlighted, chance line, IN-DISTRIBUTION bars hatched. 300dpi, IEEE-style. Source:
  `figures/make_baseline_1545_figure.py`.
- **Provenance JSON**: `docs/BASELINE_1545_PROVENANCE.json` — every number in this table with its
  source checkpoint sha256 and exact eval command.
- **Per-model raw results**: `data/{cavmae,cavmae_sync,avsiam_base,equiav,imagebind,languagebind,
  audioclip,wav2clip}_retrieval_results.json`
- **Per-model eval scripts**: `scripts/{cavmae,cavmae_sync,avsiam,equiav,imagebind,languagebind,
  audioclip,wav2clip}_retrieval.py`
- **Per-model run logs**: `logs/{cavmae_1545,cavmae_sync_1545,avsiam_base_1545,equiav_1545,
  imagebind_1545,languagebind_1545,audioclip_1545,wav2clip_1545}.log`
