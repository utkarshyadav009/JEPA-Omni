# ICLR_RESULTS.md — every measured number the PERCEPTION paper needs

**Scope.** M2 audio-visual congruency, the query predictor, and the SigLIP2 interface. Nothing
about the robot, turn-taking, TTS/STT, deployment policy or the Jetson *as a product* is in
scope; on-device latency appears only where it is the measurement being reported.

**This file is a gathering pass, not a claim pass.** Every row carries value / n / protocol /
gallery / source file / key-or-line / `measured` vs `published`. `measured` = measured by us with
our code. `published` = a number from someone else's paper. **Rule enforced: no table below mixes
the two.** Where a number could not be found it says `MISSING` and names what was searched.

**Provenance of this file.** Assembled 2026-09-08 from artifacts on disk; §9 (AudioSet-527 frozen
probe) added 2026-09-09 once `docs/AUDIOSET_BENCHMARK.md` and its artifacts were committed, which
also shifted every section from the original §9–§14 up by one (now §10–§15) — cross-references
throughout this file were mechanically remapped along with the shift, not hand-edited. Numbers
marked `RE-DERIVED` in §10 (and, for AudioSet, §9.2) were recomputed in this pass from raw data,
not transcribed. Where two documents disagree the disagreement is reported in §13, not silently
resolved.

---

> # ⚠ STATUS 2026-09-17 — READ BEFORE QUOTING ANY NUMBER FROM THIS FILE
>
> This file was assembled **2026-09-08/09**. It therefore predates the padding-leak discovery,
> the contamination re-measurement, the baseline reconciliation, **RUN-4 in its entirety**, the
> naming retirement, and RUN-5. Several of its headline numbers are **superseded**.
>
> **For drafting the paper or the abstract, use these instead:**
>
> | need | file |
> |---|---|
> | any single number, with provenance | **`docs/CANONICAL_NUMBERS.md`** — the single source |
> | per-claim inventory + the caveat that must travel | **`docs/PAPER_ASSETS.md`** |
> | the abstract itself | **`docs/ABSTRACT.md`** (two framings + claim ledger) |
> | what changed and why | **`docs/ERRATA_PROPOSED.md`** (15 entries) |
>
> **Superseded numbers are NOT deleted below.** They are marked `SUPERSEDED` in place with the
> corrected value and the errata ID, because they were previously circulated and the paper must
> present them as corrections rather than silently swap them (`PAPER_ASSETS.md` asset 8).
>
> **Marking is section-level, not line-level — this matters.** Corrected in this pass: **§1.1,
> §2, §5, §15**. But `53.27` / `53.72` and the `≥15.4` contamination figure recur **21 further
> times** in discussion prose that carries no local marker:
>
> | section | unmarked mentions | what they are |
> |---|---|---|
> | §12 (protocol-mismatch register) | 7 | prose *about* the published numbers |
> | §13 (conflicts between sources) | 7 | prose *about* the published numbers |
> | §5 (contamination) | 3 | body text under the corrected banner |
> | §4 (corpus-scale / batch-share) | 2 | **in-training evals on the leaked path — never re-measured** |
> | §1, §14 | 2 | cross-references |
>
> **Rule: any `53.27`, `53.72` or `≥15.4` anywhere in this file is superseded, whether or not the
> nearest heading says so.** §4's retrieval figures are the ones to watch — they were never
> re-measured through the corrected harness, so do not quote them at all.
>
> Sections **not** re-verified against the corrected harness: **§4, §6–§8, §11**.
>
> **The system result is now RUN-4 `step18000`: 41.77 (v→a) / 41.88 (a→v) R@1**, n=1,545,
> sha256 `27b33c8c…`. Everything below describing "our system" as 53.27/53.72 is RUN-2 on the
> leaked path.

---

## 1. M2 audio-visual congruency — retrieval on the 1,545-clip held-out VGGSound gallery

**Protocol (all rows).** Full N×N cosine-similarity retrieval over the fixed gallery
`data/vggsound_eval_1545.txt` (sha256 `89307c6d4104…`), R@1/R@5/R@10 in both directions.
`clips_seen` asserted equal to `dataset_len` before scoring. Checkpoint
`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt`
(sha256 `e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8`), frozen, no fine-tuning
on the gallery. Direction labels are taken verbatim from the log's own key names
(`ambient→vision` = a→v, `vision→ambient` = v→a).

### 1.1 Primary (training-time eval, the number every other doc cites)

> **⚠ SUPERSEDED — ERRATA E-1. Do not quote the R@k rows below as a result.**
>
> These were measured on the **leaked** eval path: `av_collate_fn` pads ambient to the longest
> clip *in the batch* and no consumer passed the resulting mask, so a per-clip pad count entered
> both embeddings as a shared nuisance variable that retrieval could match on.
>
> | | published below | corrected (RUN-2, same checkpoint) | **system result (RUN-4 `step18000`)** |
> |---|---|---|---|
> | v→a R@1 | 53.72 | **29.90** | **41.77** |
> | a→v R@1 | 53.27 | **28.28** | **41.88** |
>
> Established three ways: batch-size-1 agreement, a uniform-pad control that *hurts*, and a
> random-pad control that reproduces the gain. The cure is **length normalisation, not masking**
> (`CANONICAL_NUMBERS.md` §7.2). The rows below are retained as the record of what was
> circulated, not as a claim.

| metric | value | n | protocol | gallery | source file | key / line | measured/published |
|---|---|---|---|---|---|---|---|
| a→v R@1 | 53.27% | 1545 | full-gallery cosine, training-time eval @ step19000 | 1,545 held-out VGGSound | `logs/m2_run2_final.log` | line 1327, `ambient→vision_R@1=53.27%` | measured |
| a→v R@5 | 81.62% | 1545 | same | 1,545 | `logs/m2_run2_final.log` | line 1329, `ambient→vision_R@5=81.62%` | measured |
| a→v R@10 | 88.67% | 1545 | same | 1,545 | `logs/m2_run2_final.log` | line 1328, `ambient→vision_R@10=88.67%` | measured |
| v→a R@1 | 53.72% | 1545 | same | 1,545 | `logs/m2_run2_final.log` | line 1333, `vision→ambient_R@1=53.72%` | measured |
| v→a R@5 | 80.32% | 1545 | same | 1,545 | `logs/m2_run2_final.log` | line 1335, `vision→ambient_R@5=80.32%` | measured |
| v→a R@10 | 88.09% | 1545 | same | 1,545 | `logs/m2_run2_final.log` | line 1334, `vision→ambient_R@10=88.09%` | measured |
| matched cosine | 0.7231 | 1545 | mean cosine, matched pairs | 1,545 | `logs/m2_run2_final.log` | line 1330, `matched_cos_sim=0.7231` | measured |
| shuffled cosine | 0.0280 | 1545 | mean cosine, shuffled pairs (sanity control) | 1,545 | `logs/m2_run2_final.log` | line 1332, `shuffled_cos_sim=0.0280` | measured |
| shuffle sanity gap | 0.6951 | 1545 | matched − shuffled | 1,545 | `logs/m2_run2_final.log` | line 1331, `shuffle_sanity_gap=0.6951` | measured |
| gallery assertion | `dataset_len=1545 clips_seen=1545 (full-gallery OK)` | 1545 | — | 1,545 | `logs/m2_run2_final.log` | line 1326 | measured |

Same six R@k values are mirrored, byte-consistent, in `docs/BASELINE_1545_PROVENANCE.json`
(`our_system.audio_to_visual.{R@1,R@5,R@10}` = 53.27/81.62/88.67;
`our_system.visual_to_audio.*` = 53.72/80.32/88.09) and in `docs/BASELINE_1545.md` (Results table,
"Ours (`m2_run2`, step19000, LOCKED)" row).

### 1.2 Chance level on this gallery

| metric | value | n | protocol | gallery | source file | key | measured/published |
|---|---|---|---|---|---|---|---|
| chance R@1 | 0.0647% (1/1545) | 1545 | uniform random rank | 1,545 | `docs/BASELINE_1545_PROVENANCE.json` | `eval_gallery.chance_level_R@1_percent` = 0.0647 | measured (arithmetic) |
| chance R@1 (rounded, as printed in the table) | 0.065% | 1545 | same | 1,545 | `docs/BASELINE_1545.md` | "Chance level" row / "Chance-level R@1 … ≈ **0.065%**" | measured (arithmetic) |
| chance R@5 | 0.32% (5/1545 = 0.3236%) | 1545 | same | 1,545 | `docs/BASELINE_1545.md` | "Chance level" row | measured (arithmetic) |
| chance R@10 | 0.65% (10/1545 = 0.6472%) | 1545 | same | 1,545 | `docs/BASELINE_1545.md` | "Chance level" row | measured (arithmetic) |

R@1 is **823×** chance (53.27 / 0.0647). **⚠ SUPERSEDED (E-1/E-15): arithmetic on the leaked
number.** Corrected: RUN-2 **437×**, RUN-4 **647× (a→v) / 645× (v→a)**. Not stated in any source doc; computed here, flagged as
derived arithmetic.

### 1.3 Two independent re-runs of the SAME checkpoint on the SAME gallery (run-to-run spread)

Reported as a group because they bound reproducibility, and because §13 needs them.

| run | date | a→v R@1/5/10 | v→a R@1/5/10 | matched cos | n | source file | key/line | measured |
|---|---|---|---|---|---|---|---|---|
| training-time eval (primary, §1.1) | 2026-07-28 | 53.27 / 81.62 / 88.67 | 53.72 / 80.32 / 88.09 | 0.7231 | 1545 | `logs/m2_run2_final.log` | 1325–1334 | measured |
| independent re-run, `scripts/eval_checkpoint_gallery.py` | 2026-08-23 | 53.59 / 81.10 / 88.03 | 52.75 / 80.00 / 87.12 | 0.7206 | 1545 | `docs/EVIDENCE_LEDGER_V2.md` | lines 619–625, "Independent reproduction (2026-08-23)" | measured |
| re-run control inside the contamination study | 2026-09-06 | 53.14 / 81.81 / 88.28 | 53.46 / 80.39 / 87.83 | — | 1545 | `docs/GALLERY_CONTAMINATION.md` | Results table, row `vggsound_eval_1545.txt (reported)` | measured |

**Spread across three runs of one frozen checkpoint on one fixed gallery: a→v R@1 52.75–53.59
(0.84 pt), v→a R@1 52.75–53.72.** Stated cause (`docs/EVIDENCE_LEDGER_V2.md:623-625`): bf16
autocast non-determinism plus a changed feature-cache path (`/dev/shm/jepa_m2_cache` gone, re-runs
read `/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k`). **Any claimed improvement smaller
than ~0.9 pt on this gallery is inside our own measurement noise.**

### 1.4 Gallery construction (needed for §12)

| property | value | source file | key | measured |
|---|---|---|---|---|
| N | 1,545 | `docs/BASELINE_1545_PROVENANCE.json` | `eval_gallery.n_clips_in_list` | measured |
| construction | unstratified random draw from the official VGGSound test split | `docs/BASELINE_1545.md` | "Gallery construction" ¶ | measured |
| classes covered | 307 of 309 | `docs/BASELINE_1545.md` / re-derived §10 | "Gallery construction" ¶ | measured |
| clips per class | 1–12 | `docs/BASELINE_1545.md` / re-derived §10 | same | measured |
| mean same-class distractors | 4.91 (exactly 4.9061) | `docs/BASELINE_1545.md`, `docs/GALLERY_CONTAMINATION.md` | "averages **4.91**" | measured |
| % of clips with >4 same-class distractors | 51.1% (51.13%) | `docs/GALLERY_CONTAMINATION.md` | "51.1% of clips sit above 4, 28.8% below" | measured |
| % below 4 | 28.8% | same | same | measured |
| M2 training-corpus overlap with this gallery | 0 / 1,545 | `docs/GALLERY_CONTAMINATION.md` | Results table, `clips seen in training` | measured |

---

## 2. Eight-baseline comparison — same gallery, our code, our measurement

> **⚠ SUPERSEDED for 7 of 8 rows — ERRATA E-12.** This section cites source files that no longer
> contain its numbers. **Canonical baselines are the per-model JSONs** (`data/{model}_retrieval_results.json`),
> tabulated in `CANONICAL_NUMBERS.md` §2. Notably ImageBind is **29.45 a→v @ n=1,545**, not
> 29.70 @ n=1,532.
>
> **And the head-to-head must NOT be written as a win (`CANONICAL_NUMBERS.md` §2.1).** We train
> on 197k VGGSound clips; ImageBind and EquiAV have never seen VGGSound. The gallery is held out
> at the *clip* level, not the *distribution* level, so the ~12-point margin is **in-domain vs
> zero-shot transfer** and confounds representation quality with domain adaptation. No
> parameter-efficiency claim may be built on it.

**Every number in this section is `measured` by us.** No number here is copied from a paper; that
is the entire point of the table (`docs/BASELINE_1545.md`, Purpose ¶). Each baseline uses **its
own** published preprocessing, read from its own inference code.

**The N column matters.** 13 of the 1,545 clip IDs have no `.mp4` in
`/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video` (independently re-verified in this
pass, §10). Every baseline that needs raw video is therefore scored on the 1,532-clip intersection
and reads `1532/1545`. Our own row reads pre-extracted feature tensors and is unaffected (1,545).
**One exception, and it is a conflict: the CAV-MAE row reads 1545 — see §13.1.**

Source for every row below: `docs/BASELINE_1545.md` "Results" table; per-model raw JSON at
`data/<model>_retrieval_results.json` keys `audio_to_visual.{R@1,R@5,R@10}` /
`visual_to_audio.{R@1,R@5,R@10}` / `n_clips`; per-model run log at `logs/<model>_1545.log`;
per-model script at `scripts/<model>_retrieval.py`. Contamination flags from
`docs/BASELINE_1545.md` and the `contamination_flag` key of each JSON.

| model | contamination flag (as printed in `docs/BASELINE_1545.md`) | a→v R@1 | a→v R@5 | a→v R@10 | v→a R@1 | v→a R@5 | v→a R@10 | N (as printed) | params (M) | training corpus | raw JSON | measured/published |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ours (`m2_run2` step19000, LOCKED)** | HELD-OUT for this gallery | **53.27%** | **81.62%** | **88.67%** | **53.72%** | **80.32%** | **88.09%** | **1545** | 155.9 trainable + 326.0 frozen V-JEPA2 ViT-L + 196.3 frozen WavJEPA-base = 678.2 total | 197,462 VGGSound + 134,491 Ego4D | (own log, §1.1) | measured |
| ImageBind | HELD-OUT | 29.70% | 56.01% | 66.58% | 29.70% | 58.09% | 68.67% | 1532/1545 | 1200.8 | Audio: AudioSet. Vision: web-scale image-text + video-audio pairs | `data/imagebind_retrieval_results.json` | measured |
| EquiAV | HELD-OUT | 24.80% | 47.13% | 57.57% | 21.80% | 45.56% | 55.22% | 1532/1545 | 211.8 | AudioSet-2M | `data/equiav_retrieval_results.json` | measured |
| CAV-MAE | **IN-DISTRIBUTION** | 12.23% | 28.22% | 36.50% | 14.24% | 27.64% | 36.83% | **1545** (see §13.1) | 190.7 | AudioSet + VGGSound | `data/cavmae_retrieval_results.json` | measured |
| LanguageBind | HELD-OUT | 7.57% | 20.82% | 30.94% | 10.31% | 26.83% | 38.45% | 1532/1545 | 709.3 | VIDAL-10M | `data/languagebind_retrieval_results.json` | measured |
| Wav2CLIP | **IN-DISTRIBUTION** in the table / **HELD-OUT** in the JSON (see §13.2) | 5.35% | 12.99% | 19.19% | 6.46% | 16.51% | 23.37% | 1532/1545 | 163.0 (11.7 audio tower + 151.3 CLIP) | **VGGSound** (audio tower distilled from frozen CLIP ViT-B/32) | `data/wav2clip_retrieval_results.json` | measured |
| AVSiam (Base)† | HELD-OUT | 2.48% | 7.90% | 12.21% | 3.00% | 9.86% | 15.08% | 1532/1545 | 333.2 | AudioSet-2M | `data/avsiam_base_retrieval_results.json` | measured |
| CAV-MAE Sync‡ | **IN-DISTRIBUTION** | 2.02% | 7.44% | 11.42% | 4.90% | 13.45% | 19.58% | 1532/1545 | 190.7 | AudioSet + VGGSound (same weights as CAV-MAE — see ‡) | `data/cavmae_sync_retrieval_results.json` | measured |
| AudioCLIP | HELD-OUT | 0.20% | 0.59% | 1.17% | 0.91% | 2.74% | 3.98% | 1532/1545 | 134.1 (32.1 audio tower) | Audio: AudioSet. Image/text: OpenAI CLIP RN50 | `data/audioclip_retrieval_results.json` | measured |
| Chance | — | 0.065% | 0.32% | 0.65% | 0.065% | 0.32% | 0.65% | 1545 | — | — | — | measured (arithmetic) |

Rows are ordered by a→v R@1 descending; `docs/BASELINE_1545.md` orders them by survey number.

**Rows reading `1545`: 2 of 10** — ours and CAV-MAE. **Rows reading `1532/1545`: 7.** Chance is
arithmetic on 1545.

**Not evaluated / not available** (`docs/BASELINE_1545.md`, Availability Survey; all three are
`NOT AVAILABLE`, no number of ours and no number of theirs is recorded):

| model | status | reason | source |
|---|---|---|---|
| MaViL | NOT AVAILABLE | repo archived, weights never released | `docs/BASELINE_1545.md` Availability Survey #5 |
| CrossMAE (AV, Guo et al.) | NOT AVAILABLE | no code repository exists | same, #6 |
| AV-JEPA / MJEPA | NOT AVAILABLE | paper-only (arXiv 2606.25225), no code or weights | same, #11 |
| AVSiam "Base+" (AudioSet-2M+VGGSound+ACAV2.4M) | not evaluated | skipped for time; only the cleaner AudioSet-2M-only "Base" variant was run | same, closing ¶ |

**Two caveats that must travel with these rows** (both from `docs/BASELINE_1545.md`):

- †**AVSiam**: checkpoint loaded 670/963 state-dict keys; 293 missing, all under an `ast_base.*`
  prefix. Unresolved whether that is a legacy branch or a partial checkpoint. Read with more
  caution than the fully-loaded rows.
- ‡**CAV-MAE Sync**: the file the Sync repo's own `get_pretrained_model.sh` downloads is
  **sha256-identical** to CAV-MAE's `audio_model.25.pth` (both `a035b04c…`, both 762,886,272
  bytes, same internal pickle path). It is the authors' own release, not a download error. The row
  is therefore **plain CAV-MAE weights run through Sync's 16-sync-frame preprocessing and
  `diagonal_mean` aggregation** — included for transparency, **not a valid independent baseline**.

---

## 3. Ego4D sibling-excluded transfer

**Protocol.** File-disjoint held-out split, seed 43, cap ≤2 windows per source file: 350 held-out
files → **674 windows**; 946 train files → 17,140 windows
(`checkpoints/vjepa21_shelved/EGO4D_HELDOUT_SPLIT_SUMMARY_V2.json`, `file_disjoint_verified: true`).
**Sibling exclusion**: windows from the same source file as the query are removed from the ranking
before scoring, so a query is never beaten by a near-duplicate of itself; this lowers the
effective gallery from 674 to a **mean 673.04**. Direction labels follow the JSON key order:
`sibling_excluded.vision_to_ambient` = v→a, `.ambient_to_vision` = a→v.

### 3.1 The headline transfer number (RUN-2 step19000, the LOCKED checkpoint)

| metric | value | n | protocol | gallery | source file | key | measured/published |
|---|---|---|---|---|---|---|---|
| v→a R@1 | 27.60% | 674 | sibling-excluded instance retrieval | 674 windows / 350 files, mean effective 673.04 | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN2_STEP19000_RESULT.json` | `sibling_excluded.vision_to_ambient.R@1` = 27.6 | measured |
| v→a R@5 | 58.01% | 674 | same | same | same | `sibling_excluded.vision_to_ambient.R@5` | measured |
| v→a R@10 | 73.59% | 674 | same | same | same | `sibling_excluded.vision_to_ambient.R@10` | measured |
| a→v R@1 | 27.00% | 674 | same | same | same | `sibling_excluded.ambient_to_vision.R@1` = 27.0 | measured |
| a→v R@5 | 58.16% | 674 | same | same | same | `sibling_excluded.ambient_to_vision.R@5` | measured |
| a→v R@10 | 74.04% | 674 | same | same | same | `sibling_excluded.ambient_to_vision.R@10` | measured |
| mean effective gallery | 673.04 | 674 | after sibling removal | — | same | `sibling_excluded.*.mean_effective_gallery_size` | measured |
| raw (NOT sibling-excluded) v→a R@1 | 25.52% | 674 | full 674-window ranking | 674 | same | `raw_vision_to_ambient_R@1` | measured |
| raw a→v R@1 | 24.63% | 674 | same | 674 | same | `raw_ambient_to_vision_R@1` | measured |
| file-level v→a R@1 | 31.75% | 674 | correct *source file* retrieved | 350 files | same | `file_level.vision_to_ambient.R@1` | measured |
| file-level a→v R@1 | 30.71% | 674 | same | 350 files | same | `file_level.ambient_to_vision.R@1` | measured |
| file-level chance R@1/5/10 | 0.22 / 1.64 / 2.88% | 674 | uniform | 350 files | same | `file_level.chance.*` | measured |
| matched cosine | 0.8052 | 674 | mean cosine, matched | 674 | same | `matched_cos_sim` | measured |
| shuffled cosine | 0.5505 | 674 | shuffled control | 674 | same | `shuffled_cos_sim` | measured |
| shuffle sanity gap | 0.2546 | 674 | matched − shuffled | 674 | same | `shuffle_sanity_gap` | measured |
| within-modality off-diag cosine, vision | 0.4358 | 674 | mean off-diagonal | 674 | same | `vision_within_modality_mean_offdiag_cosine` | measured |
| within-modality off-diag cosine, ambient | 0.3893 | 674 | same | 674 | same | `ambient_within_modality_mean_offdiag_cosine` | measured |
| **instance-level chance R@1** | **0.1486% (1/673.04)** | 674 | uniform on the effective gallery | 673.04 | **not in any artifact** | derived here — the JSON stores `chance` only under `file_level` | measured (arithmetic, DERIVED) |

**27.60% is 186× the instance-level chance of 0.1486%.** Derived here; flagged.

### 3.2 The pre-trained (untrained-on-Ego4D) baseline this transfer is measured against

Separate table because it is a different checkpoint, not a different metric.

| metric | value | n | protocol | gallery | source file | key | measured/published |
|---|---|---|---|---|---|---|---|
| baseline v2, sibling-excl v→a R@1 | 2.82% | 674 | identical protocol, pre-retrain checkpoint | 674 windows / 350 files | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_V2.json` | `sibling_excluded.vision_to_ambient.R@1` | measured |
| baseline v2, sibling-excl a→v R@1 | 1.78% | 674 | same | same | same | `sibling_excluded.ambient_to_vision.R@1` | measured |
| baseline v2, v→a R@5 / R@10 | 8.01% / 10.53% | 674 | same | same | same | `sibling_excluded.vision_to_ambient.R@{5,10}` | measured |
| baseline v2, a→v R@5 / R@10 | 6.23% / 10.98% | 674 | same | same | same | `sibling_excluded.ambient_to_vision.R@{5,10}` | measured |
| baseline v2, matched / shuffled / gap cosine | 0.5962 / 0.5028 / 0.0934 | 674 | same | same | same | `matched_cos_sim`, `shuffled_cos_sim`, `shuffle_sanity_gap` | measured |
| baseline **v1** (SUPERSEDED) | 0.84% / 1.04% R@1 | 1542 | sibling-excluded, **v1 gallery: 1,542 windows from only 81 files** | 1,542 / 81 | `EGO4D_HELDOUT_BASELINE_SIBLINGEXCL.json` (at `checkpoints/vjepa21_shelved/`) | `docs/EVIDENCE_LEDGER.md:143` | measured — **do not use** |

**Why v1 was retired** (`docs/EVIDENCE_LEDGER.md:322`): the v1 gallery packed 1,542 windows into 81
files, so dozens of near-duplicate siblings competed per query; file-level R@k was 1.7–3.8×
chance (real signal) while instance-level R@1 sat at 0.84%, an ambiguity artifact. The gallery was
rebuilt as v2 (674 windows, 350 files, cap ≤2/file) and v2 adopted as authoritative. **The paper
must not cite v1's 0.84/1.04 as "the baseline."**

**Note the filename in the task brief.** `EGO4D_HELDOUT_BASELINE_V2.json` does **not** exist at
the repo root or under `data/`. Its only location is
`checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_V2.json`. Cite the full path.

### 3.3 Ego4D across the whole M2 run series (the transfer story)

| run | sibling-excl R@1 (v→a / a→v) | n | gallery | source file | key | measured |
|---|---|---|---|---|---|---|
| pre-retrain baseline v2 | 2.82% / 1.78% | 674 | 674 / 350 files | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_V2.json` | `sibling_excluded.*` | measured |
| VGGSound-60k + Ego4D-17.1k (1st scaling datapoint) | 18.40% / 18.40% | 674 | same | `checkpoints/falsifier_tracking.md` via `docs/EVIDENCE_LEDGER.md:36` | Table 1 row | measured |
| RUN-1 (197,462 VGG + 17,140 Ego4D) | 11.57% / 10.68% | 674 | same | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN1_RESULT.json` | `sibling_excluded.vision_to_ambient.R@1` = 11.57, `.ambient_to_vision.R@1` = 10.68 | measured |
| **RUN-2 step19000 (LOCKED, best in study)** | **27.60% / 27.00%** | 674 | same | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN2_STEP19000_RESULT.json` | `sibling_excluded.*` | measured |
| RUN-2 step20000 (final, unused) | 27.30% / 26.56% | 674 | same | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN2_STEP20000_RESULT.json`; `docs/METHODOLOGY_FORENSICS.md:52` (row 16b) | — | measured |
| RUN-2 `best.pt` (step13960, wrong selection) | 25.82% / 26.41% | 674 | same | `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN2_RESULT.json`; `docs/EVIDENCE_LEDGER.md:43` | — | measured — rejected |
| RUN-3 (+AudioSet, CONFOUNDED) | 8.75% / 12.91% | 674 | same | `checkpoints/NEGATIVE_RESULTS.md` via `docs/EVIDENCE_LEDGER.md:45` | — | measured — confounded |

Pre-registered Ego4D gate was **≥10%** sibling-excluded R@1, with a later "hold-gain" reference of
18.40/18.40 (`docs/EVIDENCE_LEDGER.md:36,38,40`).

---

## 4. Corpus-scale and batch-share ablations (M2)

**Protocol (all rows).** InfoNCE, temperature 0.03, 4-GPU DDP, `train_m2.py`. VGGSound column is
a→v/v→a R@1 on the 1,545-clip gallery; Ego4D column is sibling-excluded v→a/a→v R@1 on the
674-window v2 gallery. Primary table: `docs/EVIDENCE_LEDGER.md` TABLE 3 (lines 260–270), with
per-run detail in `docs/METHODOLOGY_FORENSICS.md` §1.1 rows 12–17.

### 4.1 Corpus scale at matched steps (the clean scale check)

| run | corpus | Ego4D batch share | negatives | steps | VGGSound R@1 | n | gallery | source file | key | measured |
|---|---|---|---|---|---|---|---|---|---|---|
| matched-step check A | 51,508 VGGSound, 0 Ego4D | 0% | 192×192 | 6,000 | 33.46% / 34.24% | 1545 | 1,545 | `docs/EVIDENCE_LEDGER.md` | TABLE 3 row 1; `docs/METHODOLOGY_FORENSICS.md:42` (row 12), log `logs/m2_fusion_bridge.log` | measured |
| matched-step check B | 199,007 VGGSound, 0 Ego4D | 0% | 192×192 | 6,000 | 44.27% / 43.95% | 1545 | 1,545 | `docs/EVIDENCE_LEDGER.md` | TABLE 3 row 2; `docs/METHODOLOGY_FORENSICS.md:43` (row 13), log `logs/m2_fusion_fullscale.log` | measured |

**Corpus-scale effect: +10.81 / +9.71 pts for 3.86× corpus at identical step count.** Both are
marked "diagnostic only, no gate" and are retrospective re-reads of existing logs, not new runs
(`docs/METHODOLOGY_FORENSICS.md` rows 12–13: "re-analysis, not a new run"). Direction labels for
this pair are **not** stated in either source — the pair is printed bare. Flagged in §12.

### 4.2 Ego4D batch share (RUN-1 vs RUN-2 vs the 60k datapoint)

| run | corpus (exact) | Ego4D batch share | negatives | steps | VGGSound R@1 (a→v/v→a) | Ego4D sibling-excl R@1 (v→a/a→v) | shuffle gap | within-modality cosine | gates | confounds |
|---|---|---|---|---|---|---|---|---|---|---|
| VGGSound-60k + Ego4D-17.1k | 60,000 + 17,140 = 77,140 | **22.2%** | 192×192 | 20,000 | 42.27% / 41.68% | 18.40% / 18.40% | 0.6049 (VGG) | not reported | VGGSound **FAIL** (<52%); Ego4D **PASS** | VGGSound deliberately shrunk to 60k to inflate Ego4D's share — flagged mid-run by the source doc |
| RUN-1 | 197,462 + 17,140 = 214,602 | **8.0%** | 192×192 | 20,000 | **55.15% / 55.53%** | 11.57% / 10.68% | 0.7081 (VGG) | 0.483 / 0.429 | VGGSound **PASS**; Ego4D **FAIL** (hold-gain) | Ego4D absolute volume unchanged, share diluted 22.2%→8.0% — **isolates the batch-share mechanism cleanly** |
| **RUN-2 (LOCKED, step19000)** | 197,462 + 134,491 = 331,953 | **40.5%** | 200×200 | 19,000 of 20,000 | **53.27% / 53.72%** | **27.60% / 27.00%** | 0.6990 (VGG) / 0.2546 (Ego4D) | 0.4358 / 0.3893 | VGGSound **PASS**; Ego4D **PASS**; cosine **NOT MET** (vs ≤0.25) | none identified — **but two levers moved at once**: Ego4D 17.1k→134.5k AND negatives 192→200 |
| RUN-2 step20000 (unused) | same | 40.5% | 200×200 | 20,000 | 51.20% / 52.69% | 27.30% / 26.56% | ~0.6990 | 0.4400 / 0.3910 | VGGSound near-miss/split; Ego4D PASS | plateau, LR annealed near-zero |
| RUN-2 `best.pt` (step13960) | same | 40.5% | 200×200 | 13,960 | ~46.5–49.2% (**interpolated, never directly evaluated**) | 25.82% / 26.41% | 0.2487 | 0.4524 / 0.3932 | worse on every measured metric | selected by lowest **training** `loss_ema`, not held-out eval — a selection bug, corrected in commit `0eb3337` |
| RUN-3 (+AudioSet-Strong) | 197,462 + 134,491 + 8,588 = 340,541 | AudioSet ~2.5% | **176×176** | 20,000(?) | 23.04% / 12.56% | 8.75% / 12.91% | n/a | 0.5299 / 0.3085 | both **FAIL** | **CONFOUNDED**: negatives 200→176 AND ambient token cap 1024→768 changed simultaneously with the AudioSet addition; root cause `_cap_ambient_len` applied after `.to(device)` |

Source for the whole table: `docs/EVIDENCE_LEDGER.md` TABLE 3, lines 260–270 (all `measured`);
per-run cross-check `docs/METHODOLOGY_FORENSICS.md` §1.1 rows 14–17 and 16b/16c. RUN-2's
step20000 VGGSound pair is stated a→v/v→a in `docs/METHODOLOGY_FORENSICS.md:53` and in the
**opposite** order in `docs/EVIDENCE_LEDGER.md` TABLE 3 — see §13.3.

**Batch-share finding as the artifacts state it:** at fixed Ego4D volume (17,140), dropping share
22.2%→8.0% cost Ego4D transfer 18.40→11.57 while VGGSound rose 42.27→55.15. Restoring share to
40.5% by growing volume 7.8× recovered Ego4D to 27.60 at a 1.88-pt VGGSound cost (55.15→53.27).

**Confound to carry:** RUN-1→RUN-2 also changed negatives 192→200. The source doc itself calls
RUN-2 "a clean two-lever result" — i.e. it explicitly is **not** a single-variable ablation. §12.

---

## 5. Gallery contamination — a LOWER BOUND of ≥15.4 points

> **⚠ SUPERSEDED — ERRATA E-13. The "lower bound" framing now actively misleads.**
>
> Re-measured on the padding-corrected path, against a better-controlled pair (clean vs the
> **official test split**, 0% vs 100% overlap, matched on source split, size and class structure):
>
> | | v→a Δ R@1 | v→a Δ R@5 | a→v Δ R@1 | a→v Δ R@5 |
> |---|---|---|---|---|
> | **E-13 (primary)** | **+9.58** | **+17.80** | **+9.26** | **+17.60** |
> | E-5 (corroborating) | +9.36 ± 0.08 | +18.43 ± 0.04 | +9.97 ± 0.00 | +16.95 ± 0.00 |
>
> **R@1 shrinks and R@5 grows.** Under the leak the contaminated gallery ran near ceiling
> (94.85 / 97.86), which compressed the R@1 gap. **Quoting R@1 alone inverts the finding**, and
> "the effect shrank 40%" is a false summary. Report R@1 **and** R@5 together.

**This must be stated as a lower bound, not an estimate.** The measurement is a single pair of
galleries; it establishes that ≥15.4 pts of a VGGSound retrieval figure can come from
training-set overlap alone. It does not upper-bound the effect.

**Protocol.** Same checkpoint (`…/step19000.pt`, sha256 `e1a8231e…`, **unchanged and
unretrained**), same script `scripts/eval_checkpoint_gallery.py`, same feature cache, same
N=1,545, `clips_seen` asserted, full-gallery cosine. The only thing that changed is which 1,545
clips are in the gallery.

| gallery | construction | clips in training corpus | a→v R@1/5/10 | v→a R@1/5/10 | n | source file | key | measured |
|---|---|---|---|---|---|---|---|---|
| `data/vggsound_eval_1545.txt` (reported) | unstratified random draw, 307/309 classes, 1–12 per class | **0 / 1,545** | **53.14 / 81.81 / 88.28** | **53.46 / 80.39 / 87.83** | 1545 | `docs/GALLERY_CONTAMINATION.md` | "What was measured" table, row 1 | measured |
| `data/vggsound_eval_1545_balanced.txt` | balanced 5-per-class, 309 × 5 (AV-JEPA's construction), seed 0, md5 `2eeaceef6866895cf02bb4204fb63835` | **1,404 / 1,545 (90.9%)** | 68.61 / 94.69 / 97.28 | 68.41 / 94.17 / 97.73 | 1545 | same | table row 2 | measured |
| **delta** | — | — | **+15.47** | **+14.95** | 1545 | same | table row 3 | measured |

**Statement to use:** *on the same checkpoint, same script and same N, moving from a gallery with
0/1,545 training overlap to one with 1,404/1,545 (90.9%) overlap raises a→v R@1 by **15.47 points**
(53.14 → 68.61) and v→a R@1 by **14.95 points**. This is a **lower bound of ≥15.4 points** on the
inflation attributable to gallery contamination.*

### 5.1 Why the delta is not explained by gallery construction

| quantity | value | source file | key | measured |
|---|---|---|---|---|
| mean same-class distractors, balanced gallery | **4.00** exactly (309 × 5) | `docs/GALLERY_CONTAMINATION.md` | "gives every clip exactly **4.00**" | measured |
| mean same-class distractors, our random draw | **4.91** | same | "Our random draw averages **4.91**" | measured |
| direction of the construction effect | balanced is marginally **easier** | same | "the balanced gallery is marginally *easier* on distractor density" | measured |
| magnitude the source doc allows for construction | "perhaps 1–2 points" | same | same ¶ | **judgment, not a measurement** — flagged |

The residual is attributed to memorisation, on this arithmetic (all three re-derived in §10):

| quantity | value | source file | key | measured |
|---|---|---|---|---|
| VGGSound feature cache | 199,007 clips | `docs/GALLERY_CONTAMINATION.md` | code block under "The remainder is memorisation" | measured |
| held out (eval list) | 1,545 | same | same | measured |
| M2 training corpus | **197,462** (= 199,007 − 1,545) | same | same; matches the documented corpus size exactly | measured |
| exclusion mechanism | `exclude_ids=eval_ids_set`, read from `data/vggsound_eval_1545.txt` **alone** | `train_m2.py` | line 979 (`exclude_ids=eval_ids_set`) — verified in this pass | measured |

### 5.2 The consequence for VGGSound retrieval benchmarking generally

| quantity | value | n | source file | key | measured |
|---|---|---|---|---|---|
| official VGGSound test split size | 15,446 clips, 309 classes | — | `docs/GALLERY_CONTAMINATION.md`; `data/test.csv` | "15,446 clips" | measured |
| test-split clips inside our training corpus | **13,894 (90.0%)** | 15,446 | same | "13,894 (90.0%) are in this training corpus" | measured |
| test-split clips outside it | **1,552** | 15,446 | same | "Only 1,552 lie outside it" | measured |
| balanced-gallery clips with video on disk (selection pool) | 15,341 of 15,446 | — | `data/vggsound_eval_1545_balanced.provenance.json` | `test_clips_with_video` | measured |

The source doc's framing, quoted so the paper does not overstate it: *"This is a reproducibility
observation, not an accusation. … Absent a stated exclusion list, a VGGSound retrieval figure
cannot be assumed to be held-out."*

### 5.3 What this does and does not change

- **Unchanged.** 53.27 / 53.72 is measured on genuinely held-out clips (0/1,545 overlap) and stands.
- **Changed.** **We do not claim a protocol match with AV-JEPA.** Ours is a random draw; theirs is
  balanced 5-per-class.
- **Not available without retraining.** A matched balanced gallery needs M2 retrained with the
  **full** official test split excluded (~15.4k clips withheld, not 1,545). Not done: it would
  invalidate the provenance of every downstream result for one comparison. Recorded as future work.

---

## 6. Perception → language interface: the three attempts

Three architecturally different interfaces were built. **Their quality metrics are not
comparable to each other** — that is a finding, not a gap, and §12 lists why for each pair. The
comparability verdicts below are copied from `docs/METHODOLOGY_FORENSICS.md` §2.4.

### 6.1 Quality, per attempt (each row's protocol differs — do not read down the column)

| attempt | metric | value | n | protocol | gallery / bank | source file | key / line | measured |
|---|---|---|---|---|---|---|---|---|
| 1 — M3 soft-prompt (Perceiver-IO, 32 latents → Qwen2.5-1.5B) | word-overlap F1, normal / swapped / zeroed | 0.471 / 0.268 / 0.274 | 200 | standalone M3 falsifier, **generative** | n/a (generation) | `checkpoints/falsifier_tracking.md` | line 17, via `docs/METHODOLOGY_FORENSICS.md` §2.1 table | measured |
| 1 — same, as quoted in the architecture doc | F1 | 0.317 | — | generative | n/a | `ARCHITECTURE.md` | line 80 / §1b table — **conflicts with 0.471, see §13.4** | measured |
| 1b — perception-prefix (16 soft tokens → thinker) | F1 | 0.269 | — | generative | n/a | `ARCHITECTURE.md` | line 80 / §1b table | measured |
| 2 — EmbeddingGemma + caption bank, **small bank** | F1 correct-clip | **0.4417** | 240 queries | nearest-neighbour retrieval against a pre-encoded caption bank | **6,000 captions / 1,000 clips** | `checkpoints/PERCEPTION_QUERY_E2E.json` | `F1_correct_clip` = 0.44166666… | measured |
| 2 — same, **large bank** | F1 correct-clip | **0.1889** | 360 queries | same | **48,000 captions / 8,000 clips** | `checkpoints/PERCEPTION_QUERY_E2E_bank48k.json` | `F1_correct_clip` = 0.18888888… | measured |
| 2 — chance, small bank | F1 chance | 0.001 | 240 | — | 6,000 / 1,000 clips | `checkpoints/PERCEPTION_QUERY_E2E.json` | `F1_chance` | measured |
| 2 — chance, large bank | F1 chance | 0.000125 | 360 | — | 48,000 / 8,000 clips | `checkpoints/PERCEPTION_QUERY_E2E_bank48k.json` | `F1_chance` | measured |
| 2 — field discrimination, small bank | F2 correct-field | 0.9458 (chance 0.1667, swapped 0.000) | 240 | same retrieval, scored on field/category not clip | 6,000 | `checkpoints/PERCEPTION_QUERY_E2E.json` | `F2_correct_field`, `F2_chance`, `F2_swapped_field` | measured |
| 2 — field discrimination, large bank | F2 correct-field | 0.9528 (chance 0.1667, swapped 0.0028) | 360 | same | 48,000 | `checkpoints/PERCEPTION_QUERY_E2E_bank48k.json` | same keys | measured |
| 2 — clip AND field | 0.4208 / 0.1806 | 240 / 360 | conjunction | 6,000 / 48,000 | both JSONs | `correct_clip_AND_field` | measured |
| 3 — SigLIP2 + trained proj768 (`sig_runD`, DEPLOYED) | cross-clip R@1 | **0.7372** (final) / **0.7356** (deployed `best.pt`) | 624 clips | query-predictor retrieval at a fixed field (`gpt_action_detailed`) | **624-clip held-out eval gallery** | `checkpoints/sig_runD_proj768/train_log.json` | last entry `vggsound.cross_clip_r1`; step-1249 entry for `best.pt` | measured |

**The bank-size collapse, stated exactly:** correct-clip F1 **0.4417 → 0.1889** as the bank grew
**exactly 8×** (6,000 → 48,000 captions; 1,000 → 8,000 clips) — a **57.2% relative drop** /
**2.34× relative degradation**. Field-level discrimination did **not** degrade (0.9458 → 0.9528,
mildly higher). Source: the two JSONs above, read directly; framing from
`docs/METHODOLOGY_FORENSICS.md:294,297`.

### 6.2 Latency, per attempt — read the hardware and the bank size, they are not the same

| what | value | protocol / hardware | bank | source file | key / line | measured |
|---|---|---|---|---|---|---|
| Attempt 1, M3-connector → Qwen autoregressive generation | **1–6 s** per generation | on-Jetson, HF-`transformers` Qwen2.5-1.5B (int8/unquantized, **not** the GGUF path) | n/a | `checkpoints/falsifier_tracking.md` | lines 2296–2298, quoted: "the old M3-connector-to-Qwen autoregressive path's 1-6s generation" | measured |
| Attempt 1, 60-token generation through locked M2+M3+Qwen2.5-1.5B-int8 | **12.2 s** | on-Jetson | n/a | `checkpoints/falsifier_tracking.md` | lines 1737–1738 | measured |
| Attempt 1, same, raw log | **12.40 s** (60 tokens), perception 3.48 s | on-Jetson | n/a | `jetson_artifacts/benchmarks/home/jetson_phase4_v2_withqwen_full60.log` | tail: `[phase4-v2] perception=3.48s generation=12.40s (60 tokens)` | measured |
| Attempt 1, per-round `generate=` across four live-caption logs | **360 ms – 10,430 ms** (length-dependent) | on-Jetson | n/a | `jetson_artifacts/benchmarks/home/jetson_m2m3_live_caption_v{1,2,3,4}.log` | per-round `generate=` | measured |
| **"1–12 s" as a range** | 1–12+ s | **not a point measurement** — it is `docs/METHODOLOGY_FORENSICS.md:259`'s own characterisation of the combined soft-prompt-family cost, spanning the 1–6 s and 12.2/12.40 s figures above | n/a | `docs/METHODOLOGY_FORENSICS.md` | line 259 | measured (composite range) |
| **Attempt 2, EmbeddingGemma query engine, total per question** | **403.8 ms** | on-Jetson, after `reboot` + `jetson_preflight.sh` PASS | **24,000 captions** | `jetson_artifacts/benchmarks/home/jetson_perception_query_results.json` | `total_ask_ms_median` = 403.8020372390747 | measured |
| Attempt 2, query encode (EmbeddingGemma-300M forward) | **263.5 ms** = **65.3%** of total | same | 24,000 | same | `query_encode_ms_median` = 263.5190486907959 | measured |
| Attempt 2, QueryPredictor forward | 138.5 ms | same | 24,000 | same | `predictor_ms_median` = 138.518 | measured |
| Attempt 2, bank lookup | **2.6 ms** (0.6% of total) | same | **24,000 candidates** | same | `bank_lookup_ms_median` = 2.608 | measured |
| Attempt 2, EmbeddingGemma load | 8.126 s | same | — | same | `embeddinggemma_load_s` | measured |
| Attempt 2, engine load | 0.919 s | same | — | same | `engine_load_s` | measured |
| Attempt 2, memory available after load | 4,237.14 MiB | same | — | same | `mem_avail_after_MiB` | measured |
| Attempt 2, streams | `["m2","vision","ambient"]` (3) | same | — | same | `streams` | measured |
| Attempt 2, offline retrieval latency, 6k bank | 17.17 ms median | **dev machine, not Jetson** | 6,000 | `checkpoints/PERCEPTION_QUERY_E2E.json` | `latency_ms_median` | measured |
| Attempt 2, offline retrieval latency, 48k bank | 17.68 ms median | same | 48,000 | `checkpoints/PERCEPTION_QUERY_E2E_bank48k.json` | `latency_ms_median` | measured |
| **Attempt 3, SigLIP2 4-stream pre-encoded retrieval, on-device** | **24.9 / 25.3 ms** (steady-state rounds 2–3) | on-Jetson, `qp_runD.pt`, `text_mode="preencoded"`, 4 streams | `candidates_siglip2.pt` (v1 tag bank) | `jetson_artifacts/benchmarks/fit_2026-08-15/fit_final.json` | `rounds[1].latency.query_ms` = 24.909…, `rounds[2].latency.query_ms` = 25.269… | measured |
| Attempt 3, same, second run | **32.1 / 33.1 ms** (steady-state rounds 2–3) | same config, different run | same | `jetson_artifacts/benchmarks/fit_2026-08-15/fit_nonat.json` | `rounds[1].latency.query_ms` = 32.087…, `rounds[2].latency.query_ms` = 33.105… | measured |
| Attempt 3, warm-up round (round 1) | 227.5 ms / 229.6 ms | same | same | both JSONs | `rounds[0].latency.query_ms` | measured |
| Attempt 3, SigLIP2 scene-encode leg | 86.0–104.7 ms (steady) | on-Jetson, part of the same rounds | — | both JSONs | `rounds[*].latency.siglip_ms` | measured |
| Attempt 3, SigLIP2-base image encode, isolated | **18.61 ms** | on-Jetson, first-ever `from_pretrained` bench | — | `jetson_artifacts/benchmarks/home/siglip2_jetson_bench.json` | `google/siglip2-base-patch16-224.img_encode_ms` | measured |
| Attempt 3, SigLIP2-large image encode, isolated | 37.88 ms | same | — | same | `google/siglip2-large-patch16-256.img_encode_ms` | measured |

#### 6.2.1 ⚠ THE 8 ms FIGURE IS NOT SigLIP2 — do not put it in the interface progression

The task brief asks for "SigLIP2 pre-encoded retrieval 8 ms". **That figure does not describe the
SigLIP2 interface.** Traced to its only source:

| what the 8 ms actually is | value | model | bank | hardware | source file | line | measured |
|---|---|---|---|---|---|---|---|
| predictor + nearest-neighbour lookup | **8 ms = 5 ms predictor + 3 ms NN** | `checkpoints/m2_embed_predictor_mlp_ddp_gradcache_bs16384/best.pt` (step 799) — **an MLP embedding predictor, not the SigLIP2 query predictor** | **30 real `gpt_action_detailed` captions** | Jetson `bmo-desktop`, JetPack R36.4.7 Orin, 2026-08-04 | `checkpoints/falsifier_tracking.md` | 2286–2298 | measured |
| total pipeline in that same test | 2,534 ms | same | 30 | same | same | 2299–2300 (`AV_encode=2366ms, M2=160ms, predictor=5ms, nn_lookup=3ms`) | measured |

So the 8 ms is a **different model** with a **30-caption bank**, measured **10 days before**
`sig_runD` existed. The comparable SigLIP2 number on the same hardware is **24.9–33.1 ms** with a
1,372-tag pre-encoded bank (table above). `figures/FIGURE_MANIFEST.md:126-131` already treats the
8 ms as a "historical figure that motivated the pivot", plotted as a marker distinct from the
deployed 25 ms — **follow that treatment, and do not label 8 ms as SigLIP2.**

The corresponding Jetson EmbeddingGemma predictor forward with a 24,000-caption bank was **138.5
ms**, not 5 ms — further evidence the 8 ms is not transferable across model/bank/hardware.

### 6.3 The frozen-target failure (0.489) and the text-space ablation that root-caused it

**Protocol for all four `sig_run*` rows.** `train_query_predictor_ddp.py`, effective global batch
1,024 clips, negatives 2,048 (Action100M) / 6,144 (VGGSound), λ_within 0.3, 1,500 steps.
`cross_clip_r1` = correct clip among all eval clips at the fixed field `gpt_action_detailed`,
scored on **N = 624 held-out clips** (`train_query_predictor.py:235`, key `n_clips`; the eval
`max_clips` cap is 600, batched to 624). `within_clip_acc` = pick the correct field among the same
clip's K=6 captions, chance 1/6 = 0.1667. Every query uses a **held-out phrasing never seen in
training** (`train_query_predictor.py:198`).

| run | target space | trainable text params | streams | VGG within-clip | VGG cross-clip R@1 | swapped-query | A100M R@1 | n | gallery | source file | key | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `query_predictor_ddp_lw0.3` (EmbeddingGemma reference) | EmbeddingGemma + proj 1536 | — | 3 (`m2,vision,ambient`) | 0.8830 | **0.6811** | 0.0037 | 0.0897 | 624 | 624 | `checkpoints/query_predictor_ddp_lw0.3/train_log.json` | last entry (step 1499) | measured |
| `sig_runA_matched3stream` | **SigLIP2 frozen (Identity)** | **0** | 3 | 0.6541 | **0.4888** | 0.0059 | 0.0513 | 624 | 624 | `checkpoints/sig_runA_matched3stream/train_log.json` | last entry (step 1499); `sig_runA.log:27` `trainable_text_params=0` | measured |
| `sig_runB_scene4stream` | SigLIP2 frozen (Identity) | 0 | 4 (`+scene`) | 0.6878 | 0.6266 | 0.0059 | 0.0689 | 624 | 624 | `checkpoints/sig_runB_scene4stream/train_log.json` | last entry (step 1499) | measured |
| `sig_runC_proj1536` | SigLIP2 + trained proj 1536 | 1,181,184 | 4 | 0.7471 | **0.7388** | 0.0032 | 0.0913 | 624 | 624 | `checkpoints/sig_runC_proj1536/train_log.json` | last entry (step 1499) | measured |
| **`sig_runD_proj768`** (deploy) | SigLIP2 + trained proj **768** | 590,592 | 4 | **0.8114** | **0.7372** | 0.0056 | 0.0849 | 624 | 624 | `checkpoints/sig_runD_proj768/train_log.json` | last entry (step 1499); `sig_runD.log:27-28` | measured |

**The frozen-target failure, stated exactly:** run A is the strict same-stream, same-pool,
same-audio-mode comparison against the EmbeddingGemma reference and lost: **cross-clip R@1 0.4888
vs 0.6811**. The within-clip curve was **flat from step 249** — 0.6512 / 0.6453 / 0.6421 / 0.6554 /
0.6576 / 0.6541 across the six evals (re-derived from `train_log.json`; matches
`JEPA_MEMORY_PLAN.md:1838`'s "0.651 / 0.645 / 0.642 / 0.655 / 0.658 / 0.654") — a ceiling, not slow
convergence.

**The first root-cause hypothesis was wrong**, and the ablation that replaced it:

| text space | within-clip cos | cross-clip cos | n | protocol | source file | key | measured |
|---|---|---|---|---|---|---|---|
| SigLIP2 raw | 0.7556 | 0.6648 | 400 clips × 6 caption fields | caption-vs-caption cosine in each text space | `JEPA_MEMORY_PLAN.md` | lines 1843–1847 | measured (**prose-only, see §11**) |
| EmbeddingGemma raw | 0.7306 | 0.6075 | same | same | same | same | measured (prose-only) |
| **EmbeddingGemma + trained proj** | **0.4340** | **0.1533** | same | same | same | same | measured (prose-only) |

Root cause, quoted (`JEPA_MEMORY_PLAN.md:1849-1851`): *"SigLIP2's raw space is barely worse than
EmbeddingGemma's raw space. The trainable projection is where the representation learning happens
— it spreads a space crammed between 0.61–0.73 out to 0.15–0.43. The frozen design deleted the
load-bearing component."*

**The fix and its measured effect** (B → D, `JEPA_MEMORY_PLAN.md:1855-1857`): restoring the Linear
moved within-clip **0.688 → 0.811** and cross-clip R@1 **0.627 → 0.737**. *(Caveat: B and D differ
only in the projection **within** the SigLIP2 arm — same 4 streams, same pool, same
`audio_mode=base` — so this pair IS a clean single-variable comparison. Verified from the logs in
this pass.)*

**Why 768 over 1536** (`JEPA_MEMORY_PLAN.md:1858-1859`, quoted): *"proj-768 beats proj-1536: same
R@1 (0.737 vs 0.739), much better within-clip (0.811 vs 0.747), and half the bank. Cheaper and
better."* The 0.2-pt R@1 difference is treated as noise by the source. On-device text machinery
933 MiB → 177 MiB (`JEPA_MEMORY_PLAN.md:1861`).

### 6.4 Candidate sets: tags vs captions (run D, 512 held-out clips)

| path | space | metric | value | n | source file | key | measured |
|---|---|---|---|---|---|---|---|
| captions via predictor | learned | caption R@1 | **0.7051** | 512 | `checkpoints/CANDIDATE_SET_EVAL_runD.json` | `captions_via_predictor.caption_r1` | measured |
| captions via zero-shot | raw | caption R@1 | 0.6191 | 512 | same | `captions_via_zeroshot.caption_r1` | measured |
| tags via predictor | learned | tag precision@5 | **0.4176** (shuffled 0.0211, gap **+0.3965**) | 512 | same | `tags_via_predictor.{tag_precision@5, tag_precision_shuffled@5, gap}` | measured |
| tags via zero-shot | raw | tag precision@5 | 0.3867 (shuffled 0.0480, gap +0.3387) | 512 | same | `tags_via_zeroshot.*` | measured |

**The load-bearing claim this supports** (`JEPA_MEMORY_PLAN.md:1874-1876`): predictor 0.705 >
zero-shot SigLIP2 0.619 — *"SigLIP2 alone is NOT sufficient — V-JEPA2 + WavJEPA + M2 + query
conditioning add real signal."* Caveat from the same source: tag p@5 is a word-overlap proxy, not
comparable to caption R@1, meaningful only by its gap to the shuffled control.

---

## 7. Four-way stream ablation

> **STEP-SELECTION CAVEAT — verified by the parent against `checkpoints/abl_*/best.pt['metrics']`.**
> The four values below are faithful to each arm's `best.pt`, but the arms' best checkpoints fall
> at **different training steps**: A and C at step 2999, **B and D at step 2499**. Arm A happens to
> peak at its final step; arm B does not (its step-2999 value is 0.5540, its step-2499 best is
> 0.5641). So the headline A→B gain depends on the selection rule:
>
> | comparison | A | B | delta |
> |---|---|---|---|
> | best checkpoint per arm (what is reported below) | 0.4407 | 0.5641 | **+0.1234** |
> | matched at final step 2999 | 0.4410 | 0.5540 | **+0.1130** |
>
> Both support the conclusion that the scene stream is the largest single contributor, but the
> paper must state which rule it uses. Best-per-arm is defensible; quoting A at its final step
> against B at its best step without saying so is not. n_clips = 624 for every arm (the query
> predictor's `evaluate()` caps at `max_clips=600` and the 48-clip loader overshoots), NOT 1,545.


**Protocol.** `train_query_predictor.py`, **3,000 steps per arm**, identical otherwise, EmbeddingGemma
target geometry. All four arms share **exactly the same pool**, asserted by the logs themselves
(line 5 of each: `restricted to scene-covered clips: VGGSound 172593->171430 Action100M
345754->69339 (same pool for EVERY ablation arm)`), pool = 171,430 VGGSound + 69,339 Action100M =
240,769 train clips; eval loaders 13,579 VGGSound / 7,728 Action100M. Metric = `cross_clip_r1` on
**N = 624 held-out clips** at the fixed field `gpt_action_detailed`; chance = 1/624 = **0.160%**.
`within_clip_acc` chance = 0.1667 (K=6).

**Reported-value protocol (recovered in this pass, previously undocumented).** Each arm's headline
number is its **best-scoring checkpoint**, where score = `within_clip_acc + cross_clip_r1` — the
same criterion `best.pt` uses. Verified: each log's `DONE best_score=` equals that sum at the step
listed. This is a consistent protocol across arms; it is **not** "final step".

| arm | streams (`token sources`) | audio_mode | best-score step | within-clip | swapped-query | **cross-clip R@1** | A100M R@1 | n | gallery | source file | key | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** `abl_A_m2_vision_ambient` | `m2, vision, ambient` (3) | mean | 2999 | 0.9057 | 0.0027 | **0.4407** | 0.0609 | 624 | 624 | `checkpoints/abl_A_m2_vision_ambient/train_log.json`; `abl_A_m2_vision_ambient.log:79,80` | last entry; `DONE best_score=1.3464` | measured |
| **B** `abl_B_plus_scene` (+SigLIP2 scene) | `m2, vision, ambient, scene` (4) | mean | 2499 | 0.9111 | 0.0016 | **0.5641** | 0.0737 | 624 | 624 | `checkpoints/abl_B_plus_scene/train_log.json`; `abl_B_plus_scene.log:69,81` | step-2499 entry; `DONE best_score=1.4752` | measured |
| **C** `abl_C_scene_baseonly` (4 streams, WavJEPA base only, no nat) | `m2, vision, ambient, scene` (4) | **base** | 2999 | 0.8865 | 0.0029 | **0.5657** | 0.0737 | 624 | 624 | `checkpoints/abl_C_scene_baseonly/train_log.json`; `abl_C_scene_baseonly.log:80,81` | last entry; `DONE best_score=1.4522` | measured |
| **D** `abl_D_scene_vision_only` (audio dropped entirely) | `scene, vision` (2) | mean (no audio stream) | 2499 | 0.8908 | 0.0067 | **0.5465** | 0.0673 | 624 | 624 | `checkpoints/abl_D_scene_vision_only/train_log.json`; `abl_D_scene_vision_only.log:69,81` | step-2499 entry; `DONE best_score=1.4372` | measured |
| chance | — | — | — | 0.1667 | 0.1667 | 0.00160 | 0.5 (K=2) | 624 | 624 | `train_log.json` | `within_clip_chance`; 1/624 derived | measured |

Ledger transcription of the same four numbers, rounded: **A 0.441 / B 0.564 / C 0.566 / D 0.546**
(`docs/EVIDENCE_LEDGER_V2.md:121` and `:394`). These match the best-score rows above to rounding.
**The brief's "m2+vision+ambient = 0.441, +scene = 0.564" is therefore correct and internally
consistent** — both are best-score-checkpoint values, not a final-vs-peak mix. Trainable params:
A 32.3M, B 32.9M, C 32.9M, D 31.5M (+ `TextTarget.proj` 1.2M each), from each log's line 13/14.

**Findings the arms support, with the confound noted for each:**

| comparison | delta | clean single variable? | note |
|---|---|---|---|
| A → B (+SigLIP2 scene stream) | 0.4407 → 0.5641 = **+12.34 pts, +28.0% relative** | **YES** — same pool, same `audio_mode=mean`, same steps, only `scene` added | the largest single gain in the ablation |
| B → C (WavJEPA nat removed; `audio_mode` mean → base) | 0.5641 → 0.5657 = **+0.16 pts** | YES (that is the only difference) | **WavJEPA-nat buys nothing**; B ≈ C |
| B → D (audio streams `m2`+`ambient` removed, scene+vision only) | 0.5641 → 0.5465 = **−1.76 pts** | YES | audio contributes little to *this* text-retrieval metric — but see §7.1, it is decisive for congruency |
| A → D | 0.4407 → 0.5465 = +10.58 pts | no — two streams swapped for one | not a controlled pair |

**Full per-step matrix**, so a different selection protocol can be re-derived without re-reading
the logs (VGGSound `cross_clip_r1`):

| step | A | B | C | D |
|---|---|---|---|---|
| 499 | 0.293 | 0.402 | 0.431 | 0.415 |
| 999 | 0.348 | 0.468 | 0.489 | 0.465 |
| 1499 | 0.362 | 0.530 | 0.543 | 0.495 |
| 1999 | 0.405 | 0.558 | 0.550 | 0.516 |
| 2499 | 0.425 | **0.564** | 0.559 | **0.546** |
| 2999 | **0.441** | 0.554 | **0.566** | 0.548 |

(3-dp values as printed in each `abl_*.log`'s `EVAL` lines; 4-dp values in the `train_log.json`s.)
**A→B holds at every step**: +0.109 / +0.120 / +0.168 / +0.153 / +0.139 / +0.113. It is not an
artifact of checkpoint selection.

### 7.1 AV congruency of the same four arms (the audio-visual claim itself)

**Protocol.** Audio is swapped to a different clip's audio, then the model is asked what it
**hears**. `audio_following_rate` = fraction of trials where the answer follows the (swapped)
ears rather than the eyes; a matched-audio control measures whether the arm can answer at all.
n = 640 trials per arm.

| arm | streams | audio_mode | has_audio | follows EARS | follows EYES | matched control acc | n | source file | key | measured |
|---|---|---|---|---|---|---|---|---|---|---|
| A `m2,vision,ambient` | 3 | mean | true | **0.6500** | 0.3500 | 0.9563 | 640 | `checkpoints/AV_CONGRUENCE_EVAL.json` | `A_m2_vision_ambient.*` | measured |
| B `+scene` | 4 | mean | true | 0.6094 | 0.3906 | 0.9578 | 640 | same | `B_plus_scene.*` | measured |
| C `+scene`, base-only audio | 4 | base | true | 0.5625 | 0.4375 | 0.9531 | 640 | same | `C_scene_baseonly.*` | measured |
| D `scene,vision` (no audio) | 2 | — | **false** | **0.0703** | **0.9297** | 0.9234 | 640 | same | `D_scene_vision_only.*` | measured |

**This is the arm that makes the congruency claim falsifiable:** drop the audio streams and the
ears-following rate collapses **0.650 → 0.070** while the matched control stays high (0.956 →
0.923). The model is genuinely reading audio, not inferring sound from vision.

Same measurement re-run in the SigLIP2 target space:

| run | streams | audio_mode | follows EARS | matched control acc | n | source file | key | measured |
|---|---|---|---|---|---|---|---|---|
| **`sig_runD_proj768`** (deployed) | 4 | base | **0.6078** | **0.9672** | 640 | `checkpoints/AV_CONGRUENCE_runD.json` | `sig_runD_proj768.{audio_following_rate, matched_control_acc}` | measured |
| `sig_runB_scene4stream` (frozen target) | 4 | base | 0.5016 | **0.5016** | 640 | same | `sig_runB_scene4stream.*` | measured |
| `sig_runA_matched3stream` (frozen target) | 3 | mean | 0.4672 | 0.5406 | 640 | same | `sig_runA_matched3stream.*` | measured |

**The frozen-target runs are at chance on the matched control (0.502 / 0.541)** — they cannot
answer the question at all, so their ears-following rates are uninterpretable. Only `sig_runD`'s
0.6078 sits on a working control (0.9672). The old EmbeddingGemma base+nat arm scored 0.609
(`JEPA_MEMORY_PLAN.md:1897`, `ARCHITECTURE.md` §6) — i.e. **nat bought nothing here either**
(0.608 base-only vs 0.609 base+nat).

---

## 8. Query-predictor swapped-query control

**Protocol** (`train_query_predictor.py:185-235`, read in full this pass). Three falsifiers in one
pass on N = 624 held-out clips:
- `within_clip_acc` — pick the correct **field** among the same clip's K captions. Chance 1/K.
  Can only be beaten by using the query, since all K captions describe the same clip.
- `swapped_query_acc` — identical, asked with a **genuinely different field's** question
  (`wrong = fields[(fi + K//2) % K]`). Must collapse toward chance; if it does not,
  `within_clip_acc` is meaningless.
- `cross_clip_r1` — correct clip among all 624 eval clips at the fixed field
  `gpt_action_detailed`. Guards against buying query-sensitivity by wrecking scene grounding.

Every query uses a **held-out phrasing never seen in training** (`get_query(f, rng, train=False)`).
K = 6 for VGGSound (chance 0.1667), K = 2 for Action100M (chance 0.5).

### 8.1 `qp_runD` — the DEPLOYED checkpoint

`qp_runD.pt` on the Jetson is byte-identical to `checkpoints/sig_runD_proj768/best.pt`
(131,615,595 bytes both, `docs/EVIDENCE_LEDGER_V2.md:185,209`). **`best.pt` is step 1249**, not the
final step — recovered in this pass: `sig_runD.log:71` records `DONE best=1.5534`, and step 1249 is
the only step where `within_clip_acc + cross_clip_r1` = 0.8178 + 0.7356 = 1.5534.

| metric | value | n | protocol | gallery | source file | key | measured |
|---|---|---|---|---|---|---|---|
| VGG `within_clip_acc` | **0.8178** | 624 | correct field among the clip's 6 captions, held-out phrasing | 6 captions/clip, 624 clips | `checkpoints/sig_runD_proj768/train_log.json` | step-1249 entry, `vggsound.within_clip_acc` = 0.8178… | measured |
| VGG `swapped_query_acc` | **0.0045** | 624 | same, asked with a different field's question | same | same | step-1249, `vggsound.swapped_query_acc` | measured |
| VGG `within_clip_chance` | 0.1667 | 624 | 1/K, K=6 | same | same | `vggsound.within_clip_chance` | measured |
| VGG `cross_clip_r1` | **0.7356** | 624 | correct clip at fixed field `gpt_action_detailed` | **624-clip gallery** | same | step-1249, `vggsound.cross_clip_r1` | measured |
| A100M `within_clip_acc` | 0.9431 | 624 | K=2 | 2 captions/clip | same | step-1249, `action100m.within_clip_acc` | measured |
| A100M `swapped_query_acc` | 0.0569 | 624 | same | same | same | `action100m.swapped_query_acc` | measured |
| A100M `within_clip_chance` | 0.5 | 624 | 1/K, K=2 | same | same | `action100m.within_clip_chance` | measured |
| A100M `cross_clip_r1` | 0.0833 | 624 | fixed field | 624 | same | step-1249, `action100m.cross_clip_r1` | measured |
| log line for the same step | `VGG within=0.818 (chance 0.167 swap 0.005) R@1=0.736 \| A100M within=0.943 R@1=0.083` | 624 | — | 624 | `sig_runD.log` | line 64, `[ddp] EVAL 1249:` | measured |

**The brief's "0.818 / 0.736 for qp_runD" is exactly this row** — the deployed `best.pt` at step
1249. **It is NOT the 0.811 / 0.737 quoted in `JEPA_MEMORY_PLAN.md` §A.3, `ARCHITECTURE.md` §6 and
`docs/EVIDENCE_LEDGER_V2.md:402,`** which are the **final** step 1499. Both are real; they describe
different weights. See §13.5 — **the deployed checkpoint's numbers are 0.818/0.736, and the widely
quoted 0.811/0.737 belongs to weights that were not shipped.**

### 8.2 The control as a falsifier, stated exactly

| condition | VGG accuracy | chance | n | verdict |
|---|---|---|---|---|
| correct query | **0.8178** | 0.1667 | 624 | 4.9× chance |
| swapped (wrong-field) query | **0.0045** | 0.1667 | 624 | **37× BELOW chance** |

Collapsing to 0.0045 against a 0.1667 chance level is the strong form of the control: the model
does not merely use the query, it **actively answers the question it was asked** — a wrong question
drives it to a systematically wrong field, not to a random one. Same pattern in every run measured
(swapped 0.0016–0.0067 across all `abl_*` and `sig_run*` arms, VGGSound).

The ledger's independent statement of this falsifier (`docs/EVIDENCE_LEDGER.md`, TABLE 4,
"Query-predictor swapped-query control"): *within-clip acc 0.897 (correct) vs 0.002–0.006
(swapped), chance = 0.167 for 6-way, verdict PASS* — a **different checkpoint** (0.897 within-clip
matches none of the `sig_run*` or `abl_*` arms; the closest is `query_predictor_v1` at 0.8948
final). Recorded, and flagged in §13.6.

### 8.3 The one GradCache correctness check that backs all of the above

| check | result | n | source file | key | measured |
|---|---|---|---|---|---|
| GradCache 3-phase vs direct single-batch backward | loss diff **0.00e+00**; grad rel-L2 error **1.37e-7**; cosine **1.0** (λ=0 and λ=0.5 both checked) | 1 controlled comparison | `docs/EVIDENCE_LEDGER.md` | TABLE 4, "Query-predictor GradCache gradient-equivalence check" | measured |

Rules out a silent GradCache implementation bug — relevant because every query-predictor number
above was trained through GradCache.

### 8.4 The earlier stream ablation (2026-08-11), for completeness

Not the four-way of §7. Different arms, different date, 3,000 steps, final-step values.

| arm | checkpoint | VGG within-clip | VGG cross-clip R@1 | n | source file | measured |
|---|---|---|---|---|---|---|
| `m2` only | `checkpoints/query_predictor_v1/train_log.json` | 0.8948 | **0.3846** | 624 | last entry (step 2999) | measured |
| `vision` only | `checkpoints/query_predictor_vision/train_log.json` | 0.8355 | **0.4471** | 624 | last entry | measured |
| `m2+vision` | `checkpoints/query_predictor_m2vision/train_log.json` | 0.8438 | **0.4776** | 624 | last entry | measured |
| `unified` | `checkpoints/query_predictor_unified/train_log.json` | 0.8972 | **0.4583** | 624 | last entry | measured |

**This RESOLVES an unreconciled conflict the ledger flags** (`docs/EVIDENCE_LEDGER_V2.md:678`,
item 4, and `:121`, `:394`): the v1 ledger's `0.385/0.447/0.478/0.458` and this ledger's
`0.441/0.564/0.566/0.546` were called "numerically incompatible … not resolved". They are **two
different experiments**, not two readings of one. The v1 four numbers reproduce **exactly** from
`query_predictor_{v1,vision,m2vision,unified}/train_log.json` final steps (0.3846/0.4471/0.4776/
0.4583); the v2 four reproduce from `abl_{A,B,C,D}` best-score steps. Nothing to reconcile.

### 8.5 The one comparison the repo calls protocol-matched — and why it is weaker than stated

`docs/METHODOLOGY_FORENSICS.md` §2.4 asserts: *"The ONLY protocol-matched comparison in this repo
is EmbeddingGemma R@1 0.681 vs SigLIP2 R@1 0.737"*, both on an "identical 518k-clip pool",
yielding **+8.2% relative** (`JEPA_MEMORY_PLAN.md:1861`). Verified against the logs in this pass —
**three things differ between those two runs, not zero**:

| property | `query_predictor_ddp_lw0.3` (0.681) | `sig_runD_proj768` (0.737) | matched? | evidence |
|---|---|---|---|---|
| batch / negatives / steps / λ_within | 1024 / 2048–6144 / 1500 / 0.3 | 1024 / 2048–6144 / 1500 / 0.3 | **YES** | `qp_ddp_lw0.3.log:38`; `sig_runD.log:28`; `cands=` lines in both |
| eval gallery | 624 clips | 624 clips | **YES** | both `train_log.json`, `n_clips` = 624.0 |
| streams | **3** (`m2,vision,ambient`) | **4** (`+scene`) | **NO** | `qp_ddp_lw0.3.log:38`; `sig_runD.log:28` |
| `audio_mode` | **mean** | **base** | **NO** | `qp_ddp_lw0.3.log:13-20`; `sig_runD.log:12-17` |
| train pool | **172,593 VGG + 345,754 A100M = 518,347** (unrestricted) | **171,430 VGG + 345,751 A100M = 517,181** (scene-restricted) | **NO** (0.2% in size; different membership) | same log lines |
| eval loader size | 13,679 VGG | 13,579 VGG | NO | same log lines |

So **+8.2% relative is a system-level improvement, not a controlled target-space comparison.**
The genuinely controlled comparison in this family is `sig_runA` (0.4888) vs the reference
(0.6811) — matched on streams (3), `audio_mode` (mean) and pool (unrestricted 518,347), verified
from `sig_runA.log:10-13` — which is the **frozen-target FAILURE**, not a win. And `sig_runB` →
`sig_runD` (0.6266 → 0.7372) is controlled *within* the SigLIP2 arm. §12 records this.

Note also: `checkpoints/query_predictor_ddp_b1024/train_log.json` — an EmbeddingGemma run trained
**3,000** steps instead of 1,500 — reaches `cross_clip_r1` **0.7147** at step 1999 and 0.7083 at
2999. Compared like-for-like on training budget, the EmbeddingGemma reference is 0.715, not 0.681,
which shrinks the SigLIP2 margin from +8.2% to **+3.1% relative**. Flagged in §12; the source docs
do not mention this run in the head-to-head.

---

## 9. AudioSet-527 frozen probe — audio-only (A column), partial replication

**Source.** `docs/AUDIOSET_BENCHMARK.md` (dated 2026-09-09) and `docs/artifacts/audioset_probe_ours.json`,
both already committed. Script: `scripts/audioset_probe.py`. Checkpoint: same locked
`m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` (sha256 `e1a8231ec9fbae6c`) used everywhere
else in this file — **unretrained, no AudioSet exposure**.

> ⚠ **`world_state (vision ZEROED)` is NOT an A-V number.** It is an audio-only forward pass
> through the fusion model with the vision slot filled by zeros of the training geometry
> (32×16 tokens), because AudioSet video is not available locally (see "What did NOT complete"
> below). Reporting it as an audio-video fusion result would be false. Both `world_state` rows in
> Table 9.1 below are audio-only, exactly like the `ambient` rows — they differ only in whether
> the frozen fusion pool (world-state) or the raw WavJEPA tokens (ambient) feed the probe.

> ⚠ **M2 has never seen AudioSet.** Every row in Table 9.1 is **zero-shot cross-dataset
> transfer** (trained on VGGSound + Ego4D only). MJEPA, CAV-MAE, CAV-MAE Sync and EquiAV in
> Table 9.3 all **pretrain on AudioSet**, so the identical evaluation is **in-domain** for them.
> This asymmetry disadvantages our number and is the reason for reporting it — not a caveat to
> bury.

### 9.1 What we measured — audio-only, A column (four real rows + three controls, all `measured`)

**Protocol (all rows).** Frozen precomputed features (no fine-tuning), linear or attentive probe
trained on top, 527-class multi-label AudioSet ontology, BCE-with-logits, AdamW +
CosineAnnealingLR, lr 1e-3, weight decay 1e-4, 30 epochs, batch 256, seed 0 — identical
hyperparameters for every row, no per-row tuning. Linear probe = `nn.Linear(d, 527)` on the mean
over the full token sequence. Attentive probe = 1 learned query → `MultiheadAttention(8 heads)` →
LayerNorm → `Linear(d, 527)` over the 32-step token sequence. n_train = 18,683 (AudioSet balanced
train, verified by direct load in this pass — see §9.2). n_eval = 10,572 (leakage-filtered, §9.2).
498 of 527 classes have ≥1 positive in the filtered eval set and are scored (`n_classes_scored`);
29 are dropped from mAP/mAUC (undefined with zero positives).

| metric | value | n | protocol | gallery | source file | key | measured/published |
|---|---|---|---|---|---|---|---|
| ambient (WavJEPA-base), linear, mAP | 8.69 | 10,572 | linear probe, mean-pooled tokens | 498/527 classes scored | `docs/artifacts/audioset_probe_ours.json` | `results[0]` (`variant="ambient (WavJEPA-base)"`, `probe="linear"`).`mAP` = 8.69253… | measured |
| ambient (WavJEPA-base), linear, mAUC | 87.60 | 10,572 | same | same | same | `results[0].mAUC` = 87.59702… | measured |
| ambient (WavJEPA-base), linear, d′ | 1.634 | 10,572 | same | same | same | `results[0].dprime` = 1.63352… | measured |
| ambient (WavJEPA-base), attentive, mAP | **18.31** | 10,572 | attentive probe, 32-token sequence | same | same | `results[1]` (`probe="attentive"`).`mAP` = 18.31491… | measured |
| ambient (WavJEPA-base), attentive, mAUC | 92.01 | 10,572 | same | same | same | `results[1].mAUC` = 92.00522… | measured |
| ambient (WavJEPA-base), attentive, d′ | 1.988 | 10,572 | same | same | same | `results[1].dprime` = 1.98757… | measured |
| world-state (vision ZEROED), linear, mAP | 18.59 | 10,572 | linear probe on M2's fused World-State, vision slot zeroed | same | same | `results[2]` (`variant="world_state (vision ZEROED)"`, `probe="linear"`).`mAP` = 18.59126… | measured |
| world-state (vision ZEROED), linear, mAUC | 92.64 | 10,572 | same | same | same | `results[2].mAUC` = 92.63531… | measured |
| world-state (vision ZEROED), linear, d′ | 2.049 | 10,572 | same | same | same | `results[2].dprime` = 2.04942… | measured |
| world-state (vision ZEROED), attentive, mAP | **20.00** | 10,572 | attentive probe on World-State tokens, vision zeroed | same | same | `results[3]` (`probe="attentive"`).`mAP` = 19.99717… | measured |
| world-state (vision ZEROED), attentive, mAUC | **93.93** | 10,572 | same | same | same | `results[3].mAUC` = 93.92710… | measured |
| world-state (vision ZEROED), attentive, d′ | **2.190** | 10,572 | same | same | same | `results[3].dprime` = 2.19017… | measured |
| CONTROL label-shuffled, linear, mAP | 0.59 | 10,572 | World-State features, train labels permuted (seed 0) | same | same | `results[4]` (`variant="CONTROL label-shuffled"`, `probe="linear"`).`mAP` = 0.58547… | measured |
| CONTROL label-shuffled, linear, mAUC | 50.12 | 10,572 | same | same | same | `results[4].mAUC` = 50.11613… | measured |
| CONTROL label-shuffled, attentive, mAP | 0.62 | 10,572 | same, attentive probe | same | same | `results[5]` (`probe="attentive"`).`mAP` = 0.61704… | measured |
| CONTROL label-shuffled, attentive, mAUC | 49.42 | 10,572 | same | same | same | `results[5].mAUC` = 49.42464… | measured |
| CONTROL label-shuffled, attentive, d′ | −0.020 | 10,572 | same | same | same | `results[5].dprime` = −0.02040… | measured |
| CONTROL matched-stats random, linear, mAP | 0.56 | 10,572 | Gaussian noise matched to World-State per-dim mean/std, real labels | same | same | `results[6]` (`variant="CONTROL matched-stats random"`).`mAP` = 0.55887… | measured |
| CONTROL matched-stats random, linear, mAUC | 50.42 | 10,572 | same | same | same | `results[6].mAUC` = 50.42153… | measured |

**Both falsifiers land at chance** (mAUC 49.4–50.4, d′ ≈ 0, mAP 0.56–0.62 against a
2.33-labels-per-clip multi-label prior) — the signal in the real rows is not an artifact of
dimensionality, probe capacity, or the metric itself.

**Attentive-vs-linear gap, independently confirmed on our own features**
(`docs/AUDIOSET_BENCHMARK.md` "Result 1"): ambient +9.62 mAP (8.69→18.31, **2.11×**); World-State
+1.41 mAP (18.59→20.00, **1.08×**) — a 7× smaller gap. Mechanism stated in the source doc: M2's
`encode_world_state` already applies a learned single-query attentive pool, so an external
attentive probe largely duplicates work the World-State has already done; its linear score
(18.59) nearly matches the raw ambient stream's *attentive* score (18.31). This is a claim about
the pooling, not about audio-visual fusion — the vision stream is zeroed throughout Table 9.1.

### 9.2 Leakage exclusion — re-verified independently in this pass (RE-DERIVED)

`docs/AUDIOSET_BENCHMARK.md` states the AudioSet eval mirror overlaps VGGSound (which M2 trained
on) at video level (shared YouTube ID) and that clips sharing an ID with a VGGSound training clip
were excluded before scoring. Re-derived here by loading the actual feature files directly
(`torch.load`, CPU, metadata only — no training, no GPU job) rather than trusting the prose:

| quantity | source doc's value | re-derived value | match? | how |
|---|---|---|---|---|
| eval clips in the AudioSet mirror | 17,141 | **17,141** | ✅ | `len(torch.load('/mnt/Raid-Storage-2/utkarsh-data/audioset_feats/eval.pt')['ids'])` |
| clean (non-VGGSound-overlapping) AudioSet ID allowlist | not stated as a count | **12,956** entries | new | `len(json.load(open('docs/artifacts/audioset_eval_clean_ids.json')))` |
| eval clips surviving the allowlist intersection | 10,572 | **10,572** | ✅ exact | `len([i for i,c in enumerate(eval_ids) if c in clean_set])` — the 12,956-entry allowlist is broader than the 17,141-clip eval mirror (covers train too), so only its intersection with the eval mirror matters, and that intersection is exactly 10,572 |
| excluded from eval | 6,569 (38.3%) | **6,569 (38.32%)** | ✅ | 17,141 − 10,572 |
| n_train (balanced train) | 18,683 | **18,683** | ✅ | `len(torch.load('.../bal_train.pt')['ids'])` |
| feature dims | ambient 768, world-state 1024, 32 tokens | **confirmed**: `ambient_mean` (17141,768), `world_state` (17141,1024), `*_tokens` (17141,32,d) | ✅ | direct tensor `.shape` read |
| checkpoint recorded in the feature file | `step19000.pt`, `vision_zeroed=True` | **confirmed** | ✅ | `eval.pt['m2_ckpt']`, `eval.pt['vision_zeroed']` |
| unique classes across train+eval | 527 | **527** | ✅ | union of `labels` sets in both files |
| `n_classes_scored` = 498 (29 dropped) | stated in prose ("498 of 527 … 29 do not") | **confirmed**: JSON `n_classes_scored=498` on every row | ✅ | direct JSON read |

**A second overlap figure in the source doc is NOT independently re-verifiable from what's in this
repo:** *"Verified two independent ways (feature-cache listing and the VGGSound CSV label files) —
both give 7,415/20,371 = 36.40% against the full official eval split."* This is a **different**
measurement from the 6,569/17,141 = 38.3% above — it uses the full official AudioSet eval split
(20,371 clips) as the denominator, not our 17,141-clip mirror, and the two independent checks that
produced it are not saved as separate artifacts (no JSON key, no log). It is consistent with (not
contradictory to) the 38.3% figure — both describe roughly the same overlap rate on different base
sets — but it is **prose-only** in this repo and could not be re-run in this pass without
re-scraping the full official split. Flagged in §14.

### 9.3 Published numbers this invites comparison with — kept in a SEPARATE table, per the no-mixing rule

**Table 9.3a — published frozen-probe figures, quoted for context only (`published`).** Different
protocol from Table 9.1 (their attentive-probe architecture is not stated in any source available
here; ours is fully specified in §9.1), and every method below pretrains on AudioSet, unlike ours.

| method | A (mAP) | V (mAP) | A-V (mAP) | pretrain includes AudioSet | source file | key | measured/published |
|---|---|---|---|---|---|---|---|
| CAV-MAE | 19.38 | 18.14 | 34.59 | yes | `docs/AUDIOSET_BENCHMARK.md` | Table 2 | published |
| CAV-MAE Sync | 21.66 | 16.20 | 28.50 | yes | same | Table 2 | published |
| EquiAV | 34.25 | 18.60 | 38.60 | yes | same | Table 2 | published |
| MJEPA ViT-L | 38.89 | 25.38 | 42.90 | yes | same | Table 2 | published |

**Table 9.3b — published fine-tuned figures, never to be mixed with any frozen-probe number
(`published`).** End-to-end fine-tuned backbones — not comparable to Table 9.1 or Table 9.3a.

| method | AudioSet mAP | protocol | source file | key | measured/published |
|---|---|---|---|---|---|
| AV-JEPA | 32.7 / 29.6 | end-to-end fine-tuned | `docs/AUDIOSET_BENCHMARK.md` | Table 3 | published |
| CAV-MAE | 51.2 / 42.0 | end-to-end fine-tuned | same | Table 3 | published |

**No baseline row measured by us exists yet.** `docs/AUDIOSET_BENCHMARK.md` "What did NOT
complete" — B5 (running CAV-MAE/CAV-MAE Sync/EquiAV/ImageBind/AudioCLIP/Wav2CLIP checkpoints
through our own probe harness) did not run; all five checkpoints are present locally and it is
executable. Until it runs, **Table 9.3a/9.3b cannot be compared to Table 9.1 on matched probe
code** — only on the published numbers' own terms.

### 9.4 What did NOT complete (carried over verbatim, all affect what the paper can claim)

| gap | status | detail | source file |
|---|---|---|---|
| V (video-only) column | **structurally unavailable** | no public AudioSet mirror carries video; Google releases 128-d VGGish audio features only, every checked audio mirror is audio-only | `docs/AUDIOSET_BENCHMARK.md` banner |
| A-V (audio-video) column | **structurally unavailable** | same reason; this is MJEPA's headline column and it cannot be filled without re-scraping ~42,543 YouTube clips | same |
| B5 — baseline audio towers through our probe | **not run** | executable, all 6 checkpoints present locally, did not run in the session that produced this doc | same, "What did NOT complete" |
| Part C — AudioSet retrieval gallery (balanced 5-per-class, target 442×5=2,210) | **not usable, not reported** | scrape reached 478/676 attempts (70.7% yield) before stalling; only 57 classes reached ≥5 clips; not padded | same |

### 9.5 One-line summary for the crib

Zero-shot (no AudioSet in training) audio-only World-State probe: **linear mAP 18.59, attentive
mAP 20.00** on 10,572 leakage-filtered eval clips (498/527 classes scored), against
label-shuffled/matched-noise controls at mAP 0.56–0.62 (chance). Not comparable to any published
number in Table 9.3a/9.3b (frozen vs fine-tuned, in-domain vs zero-shot, and — separately — the
attentive-probe architecture is not matched). No V or A-V column exists. §12.11 carries the full
mismatch register for this section.

---

## 10. Numbers RE-DERIVED from raw data in this pass (not transcribed)

Recomputed independently so the parent can trust the artifact over this file.

| claim | source doc's value | re-derived value | match? | how |
|---|---|---|---|---|
| eval list size | 1,545 | 1,545 | ✅ | `wc -l data/vggsound_eval_1545.txt` |
| classes covered by the gallery | 307 of 309 | **307 of 309** | ✅ | joined `data/vggsound_eval_1545.txt` against `data/test.csv` labels |
| clips per class | 1–12 | **1–12** | ✅ | same |
| mean same-class distractors | 4.91 | **4.9061** | ✅ | same |
| % clips >4 / <4 distractors | 51.1% / 28.8% | **51.13% / 28.80%** | ✅ | same |
| balanced gallery | 309 × 5 = 1,545 | **1,545 clips, 309 classes, min=max=5** | ✅ | `data/vggsound_eval_1545_balanced.txt` vs `data/test.csv` |
| VGGSound feature cache | 199,007 | **199,007** | ✅ | `find /mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k -name '*.pt'` |
| M2 training corpus | 197,462 | **197,462** (= cache − eval list) | ✅ | set difference |
| eval clips present in cache | (implied 1,545) | **1,545 / 1,545** | ✅ | set intersection |
| balanced-gallery clips in training corpus | 1,404 / 1,545 (90.9%) | **1,404 / 1,545** | ✅ | set intersection |
| official test split in training corpus | 13,894 / 15,446 (90.0%) | **13,894 / 15,446 (89.95%)** | ✅ | set intersection |
| test-split clips outside training corpus | 1,552 | **1,552** | ✅ | set difference |
| clips missing raw video (the 13) | 13 | **13**, all with IDs beginning `P` (a shard-shaped gap): `PEbtGD3Tgpo_000062, PSw5uqkTPzs_000314, PVSHN_arz2k_000021, PRriMdEO23I_000179, PSoOq28g2t0_000030, PNcdB7ptDis_000004, PVgL5wFOKMs_000030, PUcsQijslE4_000080, PEbtGD3Tgpo_000078, PWxH3l6Iri4_000000, PD9qAU6kazY_000270, PUzk3WmMteU_000292, PVGChCsrCuk_000020` | ✅ | `os.path.exists` over `/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video` |
| exclusion mechanism | `train_m2.py:979` | **confirmed**: `exclude_ids=eval_ids_set` at line 979 | ✅ | direct read |
| `cross_clip_r1` gallery size | "518k-clip pool" (`docs/METHODOLOGY_FORENSICS.md` §2.4) | **624 clips** — 518k is the **training** pool; the retrieval gallery is `N = n_clips = 624` | ❌ **CONFLICT, see §13.7** | `train_query_predictor.py:185-235`; `n_clips: 624.0` in every `train_log.json` |
| `sig_runD` best.pt step | not stated anywhere | **step 1249** (within 0.8178 + R@1 0.7356 = 1.5534 = the log's `DONE best=`) | new | arithmetic on `train_log.json` vs `sig_runD.log:71` |
| `abl_*` reported-value protocol | not stated anywhere | **best-score checkpoint** (`within + R@1`), matching each log's `DONE best_score=` | new | arithmetic on all four `train_log.json` vs `abl_*.log` |
| `sig_runA` uses a different pool/audio_mode than B/C/D | not stated | **confirmed**: A = mean + unrestricted (172,593/345,754); B/C/D = base + scene-restricted (171,430/345,751) — except B/D `audio_mode=mean` | new | `sig_run{A,B,C,D}.log` `AVCachedDataset` lines |
| CAV-MAE actually ran on 1,545 | table says 1545 | **confirmed**: `logs/cavmae_1545_v2.log:1` = `Loaded 1545 clips from data/vggsound_eval_1545.txt` | ✅ with a caveat, §13.1 | direct read |
| CAV-MAE's `eval_list` JSON field | `data/cavmae_eval_subset.txt` | **that file holds only 1,000 clips** (a strict subset of the 1,545). The field is a **reporting bug**: `scripts/cavmae_retrieval.py:421` writes `args.save_list`, not `args.eval_list`. The run itself used the correct 1,545 list. | bug found | direct read of script + list + log |

---

## 11. Published numbers (someone else's paper)

**This section is deliberately empty of values.** Exhaustively searched `docs/`, `figures/`,
`*.md` at root, and every `*.json` under `checkpoints/` and `data/`: **the repository contains no
published-paper retrieval numbers as data.** `docs/BASELINE_1545.md`'s own Purpose ¶ explains why —
the whole table exists to replace paper-quoted figures with our own measurements: *"No number in
this table is copied from a paper."*

The only published *facts* recorded anywhere are non-numeric or structural:

| published fact | value | protocol | source file | key | published |
|---|---|---|---|---|---|
| AV-JEPA's retrieval gallery construction | balanced 5-per-class, 309 × 5 = 1,545, mean same-class distractors exactly **4.00** | their stated gallery construction | `docs/BASELINE_1545.md` "Gallery construction" ¶; `docs/GALLERY_CONTAMINATION.md` | — | **published** (construction, not a result) |
| AV-JEPA availability | arXiv 2606.25225, **paper-only, no code or weights** | — | `docs/BASELINE_1545.md` Availability Survey #11 | — | published |
| each baseline's pretraining corpus | see §2 "training corpus" column | read from each paper/README | `docs/BASELINE_1545.md`; each `data/*_retrieval_results.json` `pretrain_corpus` | `pretrain_corpus` | **published** (corpus claim, not a result) |
| VGGSound official train split size | 183,730 clips | — | `data/wav2clip_retrieval_results.json` | `contamination_flag` text | published |
| **AV-JEPA's R@1 / R@5 / R@10** | **MISSING** | — | — | not recorded anywhere in this repo | — |

**Consequence for the paper:** there is currently **no published number available to compare
against**, so the §2 table is the only baseline comparison that exists, and it is entirely ours.
If the paper wants to cite AV-JEPA's reported R@1, that number must be pulled from the paper — it
is not in this repository, and any comparison to it inherits every mismatch in §12.

---

## 12. E2 — PROTOCOL-MISMATCH REGISTER

Every number above that is **not** protocol-matched to the figure it might be compared against,
with the specific reason. Ordered by how likely the mismatch is to mislead.

### 12.1 Our 1,545 gallery vs AV-JEPA's 1,545 gallery — **NOT protocol-matched**

| our figure | the figure it invites comparison with | why NOT matched |
|---|---|---|
| a→v 53.27% / v→a 53.72% R@1 on `data/vggsound_eval_1545.txt` | any AV-JEPA VGGSound retrieval R@1 on their 1,545-clip gallery | **The N matches by coincidence, not construction.** Ours: unstratified random draw from the official test split, 307/309 classes, 1–12 clips per class, mean same-class distractors **4.9061**. Theirs: balanced 5-per-class, 309×5, mean distractors exactly **4.00**. Ours is marginally *harder* on distractor density (~1–2 pts by the source doc's own estimate). **Separately and much larger:** the held-out construction differs and is unstated on their side. A matched comparison would need M2 retrained with the **full** official test split excluded (~15.4k clips, not 1,545) — recorded as future work in `docs/GALLERY_CONTAMINATION.md` because doing it would invalidate the provenance of every downstream result. **We do not claim a protocol match and must not imply one.** |

**The number that quantifies why this matters:** on the same checkpoint and script, a
balanced-5-per-class gallery that is 90.9% training clips scores **68.61 / 68.41** vs our clean
gallery's 53.14 / 53.46 — a **≥15.4-point** lower bound on contamination inflation (§5). Since
**90.0% of the official test split is inside our training corpus**, any balanced draw from that
split without an explicit exclusion list would be ~90% contaminated. This cuts both ways: it is
also why a published VGGSound figure with no stated exclusion list cannot be assumed held-out.

### 12.2 Fine-tuned / in-distribution vs held-out, inside our own §2 table

| rows | mismatch | reason |
|---|---|---|
| CAV-MAE (12.23%), CAV-MAE Sync (2.02%), Wav2CLIP (5.35%) vs our 53.27% | **IN-DISTRIBUTION vs HELD-OUT** | These three were pretrained on VGGSound itself, so our gallery clips may be in-distribution for them, while our own training run explicitly excluded these exact 1,545 clips. **The asymmetry favours the baselines**, and it is why the contamination column exists. Do not present our margin over them as clean. |
| Wav2CLIP specifically | flag disagreement | The table says **IN-DISTRIBUTION**; the artifact JSON says **HELD-OUT**. Same evidence, opposite reading. §13.2. |
| AVSiam Base (2.48%) vs any fully-loaded row | **incomplete checkpoint** | 670/963 state-dict keys loaded; 293 missing under `ast_base.*`. Not resolved whether legacy or partial. Its number is a lower bound on what the released model can do. |
| CAV-MAE Sync (2.02%) vs CAV-MAE (12.23%) | **identical weights** | sha256-identical checkpoints. The 10-pt gap is entirely a preprocessing/aggregation difference, not two models. Sync is **not a valid independent baseline**. |
| AVSiam "Base+" | **not evaluated** | AudioSet-2M+VGGSound+ACAV2.4M variant skipped for time. Only the cleaner AudioSet-2M-only Base is in the table. The stronger variant's number is unknown. |

### 12.3 Different pretraining corpora across the §2 rows

Every §2 row is protocol-matched on **gallery, retrieval code and metric**, and on nothing else.
Pretraining corpora span AudioSet (CAV-MAE, EquiAV, AVSiam, AudioCLIP audio tower), AudioSet +
VGGSound (CAV-MAE, Sync), VGGSound alone (Wav2CLIP), VIDAL-10M (LanguageBind), web-scale
image-text + video-audio (ImageBind), and 197,462 VGGSound + 134,491 Ego4D (ours). Parameter counts
span **134.1M to 1,200.8M** (ours 678.2M total / 155.9M trainable). **The table is a same-gallery
comparison, not a controlled one** — no claim about architecture can be read off it.

### 12.4 Different gallery sizes — the 1532 vs 1545 split

| rows | mismatch | reason |
|---|---|---|
| 7 baselines at `1532/1545` vs ours and CAV-MAE at `1545` | **different N** | 13 clip IDs have no `.mp4` in the extracted video dir (independently re-verified, §10). Every model needing raw video hits the same gap. A 1,532-clip gallery is very slightly *easier* (13 fewer distractors, 0.84% smaller), so the effect favours those rows by a negligible but nonzero amount. |
| CAV-MAE at `1545` | **different video source** | CAV-MAE ran 2026-07-10 with `--video-dir /home/utkarsh/data/vggsound`, a directory that **no longer exists**; the 2026-09-02 baselines used `/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video`, which is missing the 13. So CAV-MAE's row is on a **different, now-unavailable** video source. §13.1. |

### 12.5 In-domain vs zero-shot, and VGGSound vs Ego4D

| pair | mismatch | reason |
|---|---|---|
| VGGSound 53.27/53.72 (1,545 gallery) vs Ego4D 27.60/27.00 (674 gallery) | **different corpus, different gallery size, different exclusion protocol** | VGGSound: clip-level exclusion of exactly 1,545 clips. Ego4D: **file-disjoint** split (350 held-out files, no window from a held-out file ever seen), plus **sibling exclusion** at scoring time (effective gallery 673.04). The two numbers cannot be averaged, differenced, or read as "worse on Ego4D" — a 674-gallery R@1 and a 1,545-gallery R@1 are different tasks. Chance differs too: 0.0647% vs 0.1486%. |
| Ego4D 27.60% as "zero-shot transfer" | **it is not zero-shot** | 134,491 Ego4D windows are **in** RUN-2's training corpus. The held-out set is file-disjoint, so it is a held-out **in-domain** result, not zero-shot. The genuinely zero-shot figure is the 2.82%/1.78% pre-retrain baseline (§3.2). Do not describe 27.60% as zero-shot. |
| Ego4D sibling-excluded 27.60% vs Ego4D raw 25.52% | **different scoring rule** | Sibling exclusion removes same-source-file competitors before ranking, raising R@1 by ~2 pts. Any comparison to an external Ego4D retrieval number must confirm that number also excludes siblings — most will not. |
| Ego4D file-level 31.75% vs instance-level 27.60% | **different retrieval target** | File-level asks "correct source file"; instance-level asks "correct window". Chance differs (0.22% vs 0.1486%). Not interchangeable. |
| Ego4D v1 gallery (0.84%/1.04%, N=1542/81 files) vs v2 (2.82%/1.78%, N=674/350 files) | **different gallery, v1 retired** | v1 was ambiguity-bound: 1,542 windows from only 81 files meant dozens of near-duplicate siblings per query. v2 supersedes it. Never mix them. |

### 12.6 Interface progression — none of the three attempts is comparable to another

Verdicts from `docs/METHODOLOGY_FORENSICS.md` §2.4, all four "comparable to others? **NO**":

| pair | mismatch | reason |
|---|---|---|
| Attempt 1 F1 (0.471 or 0.317) vs Attempt 2/3 R@1 | **different metric family** | word-overlap F1 on free-form **generation** vs retrieval **R@1**. No shared protocol normalises them. No gallery exists for Attempt 1 at all. |
| Attempt 2 F1 0.4417/0.1889 vs Attempt 3 R@1 0.681/0.737 | **different metric, different bank, different target space** | F1 on a discrete correct/incorrect clip judgment vs R@1; banks 6,000/48,000 captions vs a 624-clip retrieval gallery; EmbeddingGemma raw-space retrieval vs trained-proj query-predictor. |
| Attempt 2 small bank vs Attempt 2 large bank | **not comparable even to itself** | 6,000 vs 48,000 captions, 240 vs 360 queries, chance 0.001 vs 0.000125. The whole point of the pair is that it is bank-size-dependent. |
| M2's 53.27% (1,545 gallery, unconditioned AV↔AV) vs any query-predictor R@1 | **different task, different gallery** | unconditioned audio↔visual retrieval on 1,545 clips vs query-conditioned text retrieval on 624 clips. Explicitly flagged in `docs/EVIDENCE_LEDGER_V2.md:402` and `docs/METHODOLOGY_FORENSICS.md` §2.4. |
| Attempt 3's Action100M R@1 (0.0513–0.0913) vs any VGGSound R@1 | **different corpus** | never averaged into any headline by the source docs; kept in separate columns. |

### 12.7 Interface latency — every figure is on different hardware, model or bank

| pair | mismatch | reason |
|---|---|---|
| **8 ms vs 1–12 s** (the pivot comparison) | **different model AND different bank** | The 8 ms is `m2_embed_predictor_mlp…bs16384/best.pt` with a **30-caption** bank (2026-08-04). It is **not** SigLIP2 and not `qp_runD`. The comparable SigLIP2 on-device figure is **24.9–33.1 ms** with a 1,372-tag bank. The same Jetson measured the EmbeddingGemma predictor forward at **138.5 ms** on a 24,000-caption bank. §6.2.1. **Never label 8 ms as SigLIP2.** |
| "1–12 s" as a soft-prompt figure | **a composite range, not a measurement** | Assembled from 1–6 s (`falsifier_tracking.md:2296`), 12.2 s (`:1737`) and 12.40 s (raw log). `figures/FIGURE_MANIFEST.md:132-137` already flags this substitution explicitly. State it as a range spanning multiple runs and output lengths, not a point. |
| 403.8 ms (Attempt 2) vs 17.17/17.68 ms (Attempt 2) | **Jetson vs dev machine, and a third bank size** | 403.8 ms is on-Jetson with a **24,000**-caption bank; 17.17/17.68 ms are off-device with **6,000/48,000**-caption banks. `docs/METHODOLOGY_FORENSICS.md:314` flags the 24k bank as "a third bank size … not directly comparable to the F1 table". |
| 403.8 ms (Attempt 2) vs 24.9–33.1 ms (Attempt 3) | **different query encoder, on purpose** | Attempt 2 runs EmbeddingGemma-300M live to encode the question (263.5 ms, 65.3% of the total). Attempt 3 pre-encodes the query vectors (`text_mode="preencoded"`), so no text tower runs on-device. The 16× speedup is real and is the architectural point — but it is **encoder-elimination, not a faster encoder**, and the two also differ in bank (24,000 captions vs 1,372 tags) and stream count (3 vs 4). |
| 24.9 ms vs 33.1 ms | **two runs, both real** | `fit_final.json` and `fit_nonat.json`, same config (`qp_runD.pt`, `candidates_siglip2.pt`, preencoded, 4 streams). Run-to-run spread is ~8 ms, i.e. ~30%. Quote a range. `ARCHITECTURE.md` quotes 25 ms in §1b and 33 ms in §8.2 — both file-backed, different runs. §13.8. |
| the ~855 MiB capability-cost figure | **prose-only** | `docs/METHODOLOGY_FORENSICS.md:314`: "not present in this specific JSON … **do not treat the 855 MiB figure as confirmed by this file alone**." The JSON has only `mem_avail_after_MiB=4237.14`, no delta. |

### 12.8 The query-predictor family — what is and is not a controlled comparison

| comparison | value | controlled? | what else changed |
|---|---|---|---|
| `sig_runA` 0.4888 vs reference 0.6811 (frozen-target FAILURE) | −19.2 pts | **YES** | streams (3), `audio_mode` (mean), pool (518,347 unrestricted) all match. **This is the clean one — and it is a negative result.** |
| `sig_runB` 0.6266 → `sig_runD` 0.7372 (restore the Linear) | +11.1 pts | **YES** | 4 streams, `audio_mode=base`, scene-restricted pool, 1,500 steps — all identical. Only the target projection changed. |
| `sig_runC` 0.7388 vs `sig_runD` 0.7372 (1536 vs 768) | −0.16 pts | **YES** | only proj width. Source treats as noise. |
| **reference 0.6811 vs `sig_runD` 0.7372 (+8.2% rel)** | +5.6 pts | **NO** — `docs/METHODOLOGY_FORENSICS.md` §2.4 calls it "the ONLY protocol-matched comparison"; it is not | **three** differences: streams 3→4, `audio_mode` mean→base, pool 518,347 unrestricted → 517,181 scene-restricted (0.2% in size, different membership). Also eval loader 13,679 → 13,579. **System-level, not controlled.** §8.5. |
| reference 0.6811 vs `query_predictor_ddp_b1024` 0.7147 | +3.4 pts to EmbeddingGemma | **NO** — 1,500 vs 3,000 steps | On a matched **3,000**-step budget the EmbeddingGemma baseline reaches 0.7147 (step 1999), shrinking the SigLIP2 margin from +8.2% to **+3.1% relative**. This run is not mentioned in any head-to-head table. **If the paper claims a SigLIP2 win, this is the number a reviewer will find.** |
| `abl_A` → `abl_B` (+scene, +28.0% rel) | +12.3 pts | **YES** | same pool, same `audio_mode=mean`, same 3,000 steps, only `scene` added; holds at all six eval steps. |
| `abl_B` → `abl_C` (nat removed) | +0.16 pts | **YES** | only `audio_mode`. |
| `abl_*` (3,000 steps, 240,769-clip pool) vs `sig_run*` (1,500 steps, 517,181-clip pool) | — | **NO** | different step budget, different pool (Action100M 69,339 vs 345,751 — the `abl_*` runs predate the Action100M scene-feature top-up that took coverage from 20% to ~100%). **Never compare an `abl_*` R@1 to a `sig_run*` R@1.** |
| within_clip_acc across corpora | 0.65–0.94 | **NO** | VGGSound K=6 (chance 0.1667), Action100M K=2 (chance 0.5). Different chance levels; the raw accuracies are not on one scale. |

### 12.9 Corpus-scale / batch-share ablations

| comparison | controlled? | what else changed |
|---|---|---|
| matched-step A (33.46/34.24) vs B (44.27/43.95) | **YES on step count and negatives** | 51,508 vs 199,007 clips at 6,000 steps, 192×192 both. Caveats: both are **retrospective log re-reads, not new runs** (`docs/METHODOLOGY_FORENSICS.md` rows 12–13), marked "diagnostic only, no gate", and **neither pair carries a direction label** in any source. |
| 60k+17.1k → RUN-1 | **NO** | VGGSound 60,000 → 197,462 **and** Ego4D share 22.2% → 8.0% moved together. |
| RUN-1 → RUN-2 | **NO** | Ego4D volume 17,140 → 134,491 **and** negatives 192×192 → 200×200. The source doc itself calls it "a clean two-lever result" — i.e. two levers. |
| RUN-2 → RUN-3 | **NO — explicitly CONFOUNDED** | AudioSet added **and** negatives 200→176 **and** ambient token cap 1024→768, all at once (root cause: `_cap_ambient_len` applied after `.to(device)`). Never cite RUN-3 as an AudioSet result. |
| RUN-2 `best.pt` VGGSound "~46.5–49.2%" | **not a measurement** | Interpolated, never directly evaluated (`docs/EVIDENCE_LEDGER.md` TABLE 3). Do not quote as measured. |
| RUN-2 step19000 vs step20000 | **YES** (same run, two checkpoints) | but note step19000 was selected **after** seeing held-out eval — a selection on the test gallery. With three-run noise at ~0.9 pt (§1.3), the 53.27 vs 51.20 a→v gap is ~2× noise. |

### 12.10 Metrics whose only backing is prose

| number | why flagged |
|---|---|
| text-space ablation 0.7556/0.6648, 0.7306/0.6075, 0.4340/0.1533 (n=400×6) | **No JSON or log artifact exists.** Present only in `JEPA_MEMORY_PLAN.md:1843-1847` and its copies (`ARCHITECTURE.md:311-313`, `docs/METHODOLOGY_FORENSICS.md:339-341`, `SESSION_LEDGER_2026-08.md:502-504`, `docs/EVIDENCE_LEDGER_V2.md:403`). Grepped for `0.7556`/`0.4340`/`0.1533`/`0.6648` across all `*.json` and `*.log`: **zero hits.** Prose-backed only — the four copies are transcriptions of one another, not independent measurements. |
| "worth perhaps 1–2 points" (construction effect in `docs/GALLERY_CONTAMINATION.md`) | An author judgment, not a measurement. The ≥15.4-pt lower bound does not depend on it. |
| ~855 MiB capability cost | Prose-only; explicitly disclaimed in `docs/METHODOLOGY_FORENSICS.md:314`. |
| M3Connector ≈39.4M params | **DERIVED by hand**, not executed (`docs/METHODOLOGY_FORENSICS.md` §2.1: "torch is not installed in this shell"). Re-run `python -m models.m3_connector` to confirm. |
| tag bank counts 1,372 / 1,482 / 110 appearance tags | Not re-derivable from the `.pt` binaries in the forensics pass; sourced from `PIPELINE_REMAINING.md:79` and `SESSION_LEDGER_2026-08.md`. File **sizes** are byte-verified; the **counts** are not. |
| training-corpus composition 197,462 / 134,491 | The 197,462 is re-derived here (cache − eval list, §10). No training-time manifest of the exact 197,462 IDs survives, so the identity of those clips is inferred from the code path, not read from a saved list. The 134,491 Ego4D count is prose-only in every source. |

### 12.11 AudioSet A column (§9) — five mismatches, none of them small

| pair | mismatch | reason |
|---|---|---|
| Our World-State A column (mAP 18.59/20.00) vs any Table 9.3a/9.3b figure | **frozen probe vs fine-tuned, and zero-shot vs in-domain, at once** | Table 9.3a is frozen but AudioSet-pretrained (in-domain); Table 9.3b is end-to-end fine-tuned. Ours is frozen AND AudioSet-zero-shot (M2 never saw AudioSet). Three-way mismatch, not one. Never place our number and either published table in one comparison table — §9.3 already keeps them physically separate. |
| Our attentive probe (1 query → 8-head MHA → LayerNorm → Linear) vs MJEPA's/CAV-MAE's attentive probe in Table 9.3a | **architecture not matched, and not matchable** | `docs/AUDIOSET_BENCHMARK.md` states MJEPA's attentive-probe architecture "is not stated in the material available to us." Our architecture is fully specified (§9.1); theirs is unknown. The attentive-vs-linear *effect* (§9.1 "Result 1") is independently confirmed on our own features, but the absolute attentive mAP numbers across the two tables are not on the same probe. |
| Our A column vs MJEPA's A-V column (their headline, 42.90) | **different task entirely** | We have no V or A-V column — structurally unavailable (§9.4). An A-only number can never stand in for an A-V headline; do not let a reader's eye slide from 42.90 to our 20.00 as if they answer the same question. |
| Our n_eval = 10,572 (leakage-filtered) vs any published AudioSet eval-set n | **different eval set, different size, and ours is post-hoc filtered** | Published AudioSet eval is commonly reported on ~17,000–20,000+ clips depending on which official/mirror split a paper uses; ours is a 17,141-clip mirror further cut to 10,572 by removing VGGSound-training overlap (§9.2). No published row states whether it applied an equivalent VGGSound-overlap exclusion — most will not, since VGGSound overlap is specific to our training corpus, not theirs. |
| Our checkpoint-selection / hyperparameters vs any baseline's own paper-reported protocol | **not controlled** | lr, epochs, batch, probe architecture in §9.1 are ours, chosen once and applied uniformly across our own rows — not the protocol each baseline paper used to produce its own Table 9.3a number. This table has never been run through one shared probe harness (§9.3, "No baseline row measured by us exists yet" — B5 did not complete). |

---

## 13. CONFLICTS BETWEEN SOURCES — findings, not nuisances

### 13.1 CAV-MAE's N: 1545 in the table, but from a video source that no longer exists
`docs/BASELINE_1545.md`'s protocol ¶ says every baseline needing raw video hits the 13-clip gap and
is scored on 1,532. **CAV-MAE's row reads 1545.** Resolved as far as the artifacts allow:
`logs/cavmae_1545_v2.log:1` = `Loaded 1545 clips from data/vggsound_eval_1545.txt` and the log's
progress counter runs to `1472/1545`, so the run genuinely scored 1,545. **Cause of the
discrepancy:** `scripts/cavmae_retrieval.py:319` defaults `--video-dir` to
`/home/utkarsh/data/vggsound`, which **does not exist on this machine any more**; the 2026-09-02
baselines read `/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video`, which is missing the
13. **CAV-MAE was therefore evaluated against a different, now-unavailable video source.** Its
12.23/14.24 is not re-derivable today, and it is not N-matched to the seven `1532/1545` rows.
Additionally its results JSON is much thinner than the others (no `clips_requested`,
`contamination_flag`, `preprocessing` or `eval_command` keys) because it predates the 2026-09-02
harness. **Recommendation: footnote CAV-MAE's N and video source, or re-run it on the raid path.**

### 13.2 Wav2CLIP contamination flag: the table and the artifact disagree
- `docs/BASELINE_1545.md`: **IN-DISTRIBUTION** — "(verified 0/1545 eval IDs in VGGSound's own
  183,730-clip train split; caveat: not de-duplicated against near-identical re-uploads)".
- `data/wav2clip_retrieval_results.json`, key `contamination_flag`: **"HELD-OUT (verified: 0/1545
  eval clip IDs present in data/train.csv … all 1545 are in data/test.csv instead) — caveat:
  same-dataset split, not de-duplicated against near-identical re-uploads, so treat as weaker than
  a cross-dataset held-out set"**.

Same evidence, opposite label. The table reads it at **corpus** level (Wav2CLIP was trained on
VGGSound, so this gallery is in-distribution); the JSON reads it at **clip** level (none of these
exact clips were in its train split). Both readings are defensible; they cannot both be printed.
**The parent must pick one and say which level the flag means.** Note that the table's own
"Precise statement of what we held out" ¶ shows why the corpus-level reading matters: our own row
is HELD-OUT at clip level too, yet 90.0% of the test split is in our training corpus.

### 13.3 Direction labels on the VGGSound R@1 pairs — one doc is swapped
Ground truth is the training log's own key names (`logs/m2_run2_final.log:1327,1333`):
`ambient→vision_R@1=53.27%` (**a→v**) and `vision→ambient_R@1=53.72%` (**v→a**).
- `docs/METHODOLOGY_FORENSICS.md` row 16 and `docs/EVIDENCE_LEDGER_V2.md:589,612-613`: correct
  (a→v/v→a), and `:612-613` documents fixing §2.4's previously swapped label.
- `docs/EVIDENCE_LEDGER.md:39` and its TABLE 3 column header both read **"VGGSound R@1 (v→a /
  a→v)"** with the values 53.27/53.72 — i.e. **still swapped** relative to the log. Same for the
  step20000 pair, printed `52.69/51.20` there vs the log's a→v 51.20 / v→a 52.69.

**`docs/EVIDENCE_LEDGER.md` (v1) has the directions reversed. Use the log or
`docs/EVIDENCE_LEDGER_V2.md`.** Also note: the matched-step A/B pair (33.46/34.24, 44.27/43.95) has
**no direction label in any source** — do not assign one.

### 13.4 Attempt-1 M3 F1: 0.471 vs 0.317
- `checkpoints/falsifier_tracking.md:17` → `docs/METHODOLOGY_FORENSICS.md` §2.1 table:
  frozen-LLM baseline `m3_multigran_best/connector.pt`, F1 normal **0.471** (swapped 0.268,
  zeroed 0.274, cos 0.724), n=200.
- `ARCHITECTURE.md:80` and its §1b table: "F1 **0.317**, 1–6 s".
- `figures/FIGURE_MANIFEST.md:138-139` uses **0.317** for panel (b), citing `ARCHITECTURE.md`.

Two different F1 values for "the M3 soft-prompt baseline", with no reconciling note anywhere. The
0.471 is checkpoint-identified and falsifier-controlled (it has swapped/zeroed arms); the 0.317 is
not attributed to a checkpoint. **Prefer 0.471 with its controls, and state the checkpoint** — but
the conflict is unresolved and the figure manifest currently uses the other one.

### 13.5 `sig_runD`: the quoted numbers are not the deployed weights
- Quoted everywhere as **within 0.811 / R@1 0.737** (`JEPA_MEMORY_PLAN.md` §A.3,
  `ARCHITECTURE.md` §6, `docs/EVIDENCE_LEDGER_V2.md:122,402`) = **final step 1499**.
- `best.pt` — the file that was renamed `qp_runD.pt` and shipped, byte-identity confirmed
  (131,615,595 bytes, `docs/EVIDENCE_LEDGER_V2.md:185,209`) — is **step 1249**, with
  **within 0.8178 / R@1 0.7356** (recovered in this pass from `DONE best=1.5534`).

Same for `sig_runB`, quoted at 0.688/0.627 (step 1499) while its `best.pt` is step 1249
(0.6918/0.6298). **The §A.3 head-to-head table uses FINAL-step values; the §7 four-way ablation
table uses BEST-score values. Two tables in the same document family, two different selection
rules, neither stated.** The differences are small (0.6–0.8 pt) but the paper should say which
weights it is reporting, and if it reports the deployed checkpoint the numbers are **0.818/0.736**.

### 13.6 The ledger's swapped-query control cites a checkpoint that cannot be identified
`docs/EVIDENCE_LEDGER.md` TABLE 4, "Query-predictor swapped-query control": *within-clip acc
**0.897** (correct) vs 0.002–0.006 (swapped)*, n="varies". **No `sig_run*` or `abl_*` arm has
within-clip 0.897.** Closest is `query_predictor_v1` at 0.8948 (final step 2999) or
`query_predictor_unified` at 0.8972 — the latter rounds to 0.897. The ledger row cites no
checkpoint or file. **Either attribute it to `query_predictor_unified` (0.8972) or drop the 0.897
in favour of the deployed `qp_runD` figure (0.8178, swapped 0.0045).** The swapped range
"0.002–0.006" is consistent with every run measured, so the falsifier verdict (PASS) is safe; only
the 0.897 is unattributed.

### 13.7 "518k-clip pool" is the TRAINING pool, not the retrieval gallery
`docs/METHODOLOGY_FORENSICS.md` §2.4 lists Attempt 3's gallery/bank size as **"518,347–518,461-clip
pool"** and **"518k-clip pool"**, in a column headed "gallery/bank size", and uses that identity to
call the EmbeddingGemma-vs-SigLIP2 comparison protocol-matched. **The retrieval gallery for
`cross_clip_r1` is 624 clips**, not 518k: `train_query_predictor.py:185-235` builds the eval set
with `max_clips=600` (batched to 624) and returns `"n_clips": float(N)`, and **every**
`train_log.json` in the family records `n_clips: 624.0`. The 518k figure is the training pool
(`172,593 + 345,754` VGGSound + Action100M).

**Consequence: any R@1 in §6.3/§8 is a 624-clip-gallery R@1.** Chance is 1/624 = 0.160%, not
1/518,347. Reporting 0.737 as a "518k-pool R@1" would overstate the task's difficulty by ~830×.
**This must be corrected before the number appears in the paper.**

### 13.8 Deployed retrieval latency: 25 ms and 33 ms both cited, both real
`ARCHITECTURE.md:80,88` (§1b) says **25 ms**; `ARCHITECTURE.md` §8.2's table says **33 ms** (no-nat)
/ 23 ms (with-nat) and asserts "The whole perception-query addition costs 33 ms". Resolved: they
are two runs of the same config. `fit_final.json` `rounds[1..2].latency.query_ms` = 24.91 / 25.27;
`fit_nonat.json` = 32.09 / 33.10. **Not a contradiction — a ~30% run-to-run spread. Quote
24.9–33.1 ms, or name the run.**

### 13.9 Two "4-way ablations" with different numbers — RESOLVED here
`docs/EVIDENCE_LEDGER_V2.md:678` item 4 records this as **"not resolved; flagged only"**:
v1's `0.385/0.447/0.478/0.458` vs v2's `0.441/0.564/0.566/0.546`, "same 4-arm shape …
numerically incompatible". **They are two different experiments.** The v1 four reproduce exactly
from `checkpoints/query_predictor_{v1,vision,m2vision,unified}/train_log.json` final steps
(0.3846 / 0.4471 / 0.4776 / 0.4583 — arms `m2` / `vision` / `m2+vision` / `unified`, 2026-08-11);
the v2 four from `checkpoints/abl_{A,B,C,D}/train_log.json` best-score steps (arms
`m2+vision+ambient` / `+scene` / `+scene,base-only` / `scene+vision`, 2026-08-14). **Nothing to
reconcile — different arms, different dates, both correct.** §8.4.

### 13.10 GALLERY_CONTAMINATION's control row vs the primary figure
`docs/GALLERY_CONTAMINATION.md`'s clean-gallery row reads **53.14 / 53.46** and is labelled
"(reported)", while the figure reported everywhere else is **53.27 / 53.72**. The doc states this
openly ("reproduces it to within 0.13–0.26 pts"). Combined with the 2026-08-23 re-run
(53.59 / 52.75), the three runs bracket a→v R@1 at 52.75–53.59. **Not a conflict, but the
contamination table's baseline row is a re-run, not the primary figure — and the +15.47 delta is
computed against 53.14, not 53.27.** Against the primary 53.27, the delta is +15.34 — still
≥15.4 only if rounded from the doc's own arithmetic. **Safest statement: "≥15 points", or "+15.47
pts within one re-run pair", or the doc's own "+15.4 points" quoted as the doc's number.**

---

## 14. MISSING — searched and not found

| number the paper may want | status | files and locations searched |
|---|---|---|
| AV-JEPA's published R@1/R@5/R@10 | **MISSING** | grepped `AV-JEPA`, `MJEPA`, `2606.25225` across all `*.md`; only `docs/BASELINE_1545.md` (Availability Survey: "paper-only, no code/weights"), `docs/GALLERY_CONTAMINATION.md` (gallery construction only), `M1 Achieved.md` (prose about the architectural bet). **No numeric result from that paper is recorded anywhere in this repo.** |
| any published baseline retrieval number | **MISSING by design** | grepped `published` across `docs/` and `figures/`; `docs/BASELINE_1545.md:3` states the table exists precisely to replace paper-quoted numbers. §11. |
| MaViL, CrossMAE (AV) numbers | **NOT AVAILABLE** (not MISSING) | `docs/BASELINE_1545.md` Availability Survey #5, #6 — no weights ever released / no repo exists. |
| AVSiam Base+ number | **not evaluated** | `docs/BASELINE_1545.md` closing ¶ of the survey. |
| Ego4D **instance-level** chance R@1 | **not in any artifact** — derived here as 1/673.04 = 0.1486% | every `EGO4D_HELDOUT_*.json` stores `chance` only under `file_level` (0.22/1.64/2.88). |
| `sig_runD` `best.pt` step, as recorded | **not stated in any doc** — recovered here as step 1249 | `sig_runD.log` prints only `DONE best=1.5534`; no doc names the step. |
| `abl_*` batch size | **NOT FILE-BACKED** | no `EFFECTIVE GLOBAL BATCH`, `cands=` or batch line in any of the four `abl_*.log`s (grepped). `JEPA_MEMORY_PLAN.md:866`'s "batch 96" describes the **2026-08-11** ablation (`query_predictor_{v1,vision,m2vision,unified}`), a different experiment — do not attribute it to `abl_A–D`. |
| a text-space-ablation artifact (JSON or log) | **MISSING** | grepped `0.7556`, `0.4340`, `0.1533`, `0.6648` across every `*.json` and `*.log`: zero hits. Prose-only in five `.md` files. §12.10. |
| an on-Jetson end-to-end latency for the **EmbeddingGemma vs SigLIP2** query path measured in one session | **MISSING** | 403.8 ms (`jetson_perception_query_results.json`, 2026-08-14, 24k bank, 3 streams) and 24.9–33.1 ms (`fit_2026-08-15/*.json`, 4 streams, 1,372-tag bank) are from different sessions with different banks and stream counts. No head-to-head exists. |
| `m2_ablation_audio_{mean,base,nat}` results | **MISSING — completely undocumented** | `docs/METHODOLOGY_FORENSICS.md` §1.7 rows 18–20: grepped `falsifier_tracking.md`, `RESULTS_TABLE.md`, `NEGATIVE_RESULTS.md`, both evidence ledgers — zero hits. Only `best.pt`/`last.pt` weight files survive. Relevant because the SigLIP2 runs' `audio_mode` (mean vs base) is an uncontrolled variable in §12.8. |
| the exact eval command that produced 53.27/53.72 | **NOT reconfirmable** | `docs/BASELINE_1545_PROVENANCE.json` `our_system.eval_command`: "NOT independently reconfirmed: no m2_run2-specific launch script or config file survives … the exact CLI invocation for this specific run is not reproducible from what's on disk." What *is* verified: the checkpoint, 4-GPU DDP, and the `clips_seen=1545` assertion. |
| a saved manifest of the 197,462 training clip IDs | **MISSING** | `data/` holds `vggsound_train_60k.txt`, `ego4d_train_clip_ids.txt`, `audioset_train_clip_ids.txt` — none is the 197,462 list. It is re-derivable as (feature cache − eval list), which reproduces 197,462 exactly (§10), but that is an inference from `train_m2.py:979`, not a read of a saved list. |
| statistical tests / confidence intervals on any retrieval number in §1–§8 | **MISSING** | no bootstrap, CI, or significance test exists for any R@1 in this file. `docs/EVIDENCE_LEDGER.md` TABLE 4 has bootstrap CIs only for the A1 World-State head (n=300/651), which is out of scope here. The only usable uncertainty signal is the three-run spread in §1.3 (~0.9 pt). **Any claim of a small margin needs this caveat.** |
| per-class or per-category breakdown of the 1,545-gallery result | **MISSING** | no artifact holds per-class R@k; `data/*_retrieval_results.json` store only aggregate R@1/5/10. |
| AudioSet V (video-only) and A-V (audio-video) columns | **structurally unavailable, not merely unmeasured** | `docs/AUDIOSET_BENCHMARK.md` banner: no public AudioSet mirror carries video (Google ships 128-d VGGish audio features only); obtaining it means re-scraping 42,543 YouTube clips. §9.4. |
| AudioSet baseline rows (CAV-MAE/Sync/EquiAV/ImageBind/AudioCLIP/Wav2CLIP) through our own §9 probe harness (task B5) | **not run** | all six checkpoints are present locally and the harness is executable; grepped for any `*audioset*probe*` output referencing them: none exists. `docs/AUDIOSET_BENCHMARK.md` "What did NOT complete" item B5. §9.3. |
| AudioSet retrieval gallery (balanced 5-per-class, target 442×5=2,210 clips) | **not usable, not reported** | scrape stalled at 478/676 attempts (70.7% yield), only 57/442 classes reached ≥5 clips; not padded to a usable gallery. `docs/AUDIOSET_BENCHMARK.md` "What did NOT complete", Part C. §9.4. |
| the "7,415/20,371 = 36.40%" AudioSet overlap cross-check, as a separate artifact | **prose-only** | `docs/AUDIOSET_BENCHMARK.md`'s two independent verification methods ("feature-cache listing" and "the VGGSound CSV label files") are described but not saved as JSON/log; only the headline 6,569/17,141=38.3% figure was independently re-derived in this pass (§9.2, from `eval.pt` + `docs/artifacts/audioset_eval_clean_ids.json` directly). |

---
## 15. One-page crib: the numbers most likely to go in the abstract

> **⚠ REWRITTEN 2026-09-17 — ERRATA E-15.** The previous version of this section was assembled
> 2026-09-09 and was stale in **six rows**, including the headline R@1 and the contamination
> framing. Because it is titled "the numbers most likely to go in the abstract", it was the most
> direct route for a retracted number to reach a submission.
>
> The superseded table is preserved in git (`git show 2cbff16:docs/ICLR_RESULTS.md`), not here —
> keeping a stale abstract crib next to a corrected one is how the wrong row gets copied.
>
> **The drafted abstract itself is `docs/ABSTRACT.md`** (two framings, a per-number claim ledger,
> and a list of sentences that must never appear). This table is its input.

**All `measured`. Every value traces to `docs/CANONICAL_NUMBERS.md` (CN), which is the single source.**

| claim | number | n | source | must-carry caveat |
|---|---|---|---|---|
| **system result** — held-out VGGSound retrieval | **41.77 (v→a) / 41.88 (a→v)** R@1; R@5 71.97 / 72.10 | 1,545 | CN §1.1 — RUN-4 `step18000`, sha256 `27b33c8c…`, 3 seeds, range 0.06 | **always write the direction.** `41.35/41.68` is `step20000`, a different checkpoint (CN §1.2) |
| vs chance | **647× (a→v) / 645× (v→a)** chance (0.0647%) | 1,545 | arithmetic on CN §1.1 | the published **823×** was arithmetic on the leaked number |
| **the padding leak** | 53.72 → **29.90** (v→a), 53.27 → **28.28** (a→v); ≈24 pts | 1,545 | CN §1, E-1 | this is **RUN-2 corrected**, not the system number — do not conflate with row 1 |
| leak mechanism | **length normalisation, not masking** | — | CN §7.2 (P2.2 control) | the mask-off control is within noise; "we fixed it by masking" is wrong |
| effective rank | **74.26** (RUN-4) vs **37.72** (RUN-2) | 1,545 | CN §7.1, full-gallery, corrected harness | ≈2×, **not 6×**. The in-training ~12.5 figure is withdrawn |
| **gallery contamination** | **+9.58 R@1 / +17.80 R@5** (v→a); +9.26 / +17.60 (a→v) | 1,545 | CN §3, E-13 primary | **quote R@1 AND R@5.** R@1 shrinks only because the contaminated gallery saturates; "shrank 40%" is a false summary |
| official test split inside our corpus | **13,894 / 15,446 (90.0%)** | 15,446 | CN §3.1 | reproducibility observation, not an accusation |
| best same-gallery baseline | ImageBind **29.45 (a→v) / 29.64 (v→a)** | 1,545 | CN §2 | **NON-EQUIVALENT (CN §2.1).** We train on VGGSound; ImageBind never has. In-domain vs zero-shot. **Never "beats ImageBind"; no parameter-efficiency claim** |
| R@1 saturation | R@1 flat 16k–20k (41.34–41.77) while R@5 +2.00 and eff_rank +4.52 rise monotonically | 1,545 | CN §9, `R1_SATURATION.md` | **never quote R@1 alone anywhere in the paper** |
| **naming verdict** | forward−backward gap **−0.04 ± 0.15** (RUN-4); IDENTITY beats every learned map | 5,101 q | `FORWARD_INFORMATION_PROBE.md` | pre-registered threshold ≥2.0 at ≥3×SE; IDENTITY's artifact floor is ≈0.6. Use **"audio-visual scene representation"**, never "world state" |
| **fusion bottleneck** | pre-fusion vision → ΔW: forward **0.0446**, backward **−0.0009** (exactly zero); fused `W`: **1.14×** fwd/bwd | 6,409 | `FUSION_BOTTLENECK.md`, capacity-matched 768d | the directional signal is **real but small** (R² 0.045). Observational — "localises to the fusion", not "the fusion destroys" |
| **predictive objective (RUN-5)** | recovers 70% of the deficit at 3k steps, decays to 9% by 20k; **r = −0.75** vs retrieval | 6 ckpts | `RUN5_DECAY_ANALYSIS.md` | **the mechanism gate FAILED.** `step1000` clears every mechanism criterion and retrieves at **9.26** vs RUN-4's 41.77 — a Pareto frontier, not a gain |
| Ego4D transfer | **27.60 / 27.00** R@1 | 674 | CN §4 | **PERMANENTLY UNREPRODUCIBLE (E-8).** If the caveat cannot travel in the table, **cut the row** |
| query predictor | cross-clip R@1 **0.7372** (`sig_runD`), **0.4888** (`sig_runA`) | 624 | CN §5, E-6 clean | **RUN-2-based**, no downstream retrain planned — say so, or a reader assumes they sit on RUN-4 |
| audio is genuinely read | ears-following **0.650 → 0.070** when audio is dropped | 640 | CN §5 | reproduced exactly at 0.6500; moved +0.0031 when the one unmasked op was masked |
| SigLIP2 scene stream earns its place | cross-clip R@1 **0.4407 → 0.5641** (+28.0% rel) | 624 | §7 `abl_A`/`abl_B` | §12.8 clean; never compare to `sig_run*` numbers |
| WavJEPA-nat buys nothing | **+0.16 pts** (0.5641 → 0.5657) | 624 / 640 | §7 `abl_B`/`abl_C` | §12.8 clean |
| AudioSet zero-shot audio-only probe | linear mAP **18.59**, attentive **20.00** | 10,572 | `audioset_probe_ours.json` | audio-only (vision zeroed), **not an A-V number**; §12.11 |
| interface latency | **1–6 s → 403.8 ms → 24.9–33.1 ms** | — | §6 | §12.7 — the 8 ms figure is a different model with a 30-caption bank |

### Rows deliberately NOT in the abstract

* **Ego4D 27.60 / 27.00** — real, but E-8 unreproducible; an abstract cannot carry that caveat.
* **Query predictor 0.7372 / 0.4888** — clean, but RUN-2-based; beside a RUN-4 headline they
  imply they sit on it.
* **RUN-5 retrieval 42.59 / 42.14** — higher than RUN-4, but from a run whose mechanism gate
  **failed**, and under 1 point against a 0.13 seed range.
* **Corpus-scale 33.46 → 44.27** — in-training evals on the leaked path, never re-measured.
