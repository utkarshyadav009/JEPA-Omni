# CORPUS_OPTIONS — P0.7 data audit and acquisition assessment

> ## CORRECTION (2026-09-13, P4.6) — read before using this document
>
> Two load-bearing claims below are **wrong** and are corrected here rather than edited out of
> the analysis, so the reasoning stays auditable.
>
> **1. "Δ ∈ {1, 2, 5} s becomes measurable" with a finer-stride corpus — WRONG.**
> Both encoders' receptive field spans the **full 10 s window**: V-JEPA2 (`fpc64`) samples 64
> frames uniformly across it and WavJEPA runs at 100 Hz over the same 10 s. A target less than
> 10 s away therefore **shares raw input** with the query, so Δ < 10 s is not a prediction
> horizon **at any stride, on any corpus**. Finer striding cannot unlock it; only a shorter
> window could. Every "Δ ∈ {1,2,5} s" argument below is void.
>
> **2. "Epic-Kitchens is the corpus that makes Arm B possible at all" — WRONG.**
> The surviving Ego4D cache is **already in the valid regime**: a 10 s window at a 10 s
> non-overlapping stride is exactly Δ = 10 s with **zero input overlap**, giving **77,831
> consecutive pairs on disk today**, no download. Epic-Kitchens buys **scale and scene
> diversity**, which are real but different goods. It was never an enabler.
>
> **3. Arm B as specified is CLOSED.** See `docs/ARM_B_CLOSURE.md`. The storage and stride
> recommendations below are still sound *as storage arithmetic*; the experiment they were sizing
> has been superseded by RUN-5.


**Date:** 2026-09-11. Every row in §1 was read **from disk**, not from documentation.
Raw artifacts: `docs/artifacts/temporal_probe/p07_disk_inventory.json`,
`p07_ego4d_ordering.json`.

Disk headroom: `/mnt/Raid-Storage-2` is **90% full — 716 GB free of 7.0 TB**.
`/mnt/Raid-Storage` has 1.5 TB free. `/` has 26 GB free (97% full).
`/dev/shm` is 756 GB and **empty** — everything that lived there is gone (see §1.6).

---

## 1. Inventory — what we actually hold

| corpus | features on disk | raw video on disk | clips / windows | size |
|---|---|---|---|---|
| VGGSound | `feature_cache_vgg51k` | `vggsound_raw/extracted/video` | 4,097 shards; **197,970** mp4 | 764 GB feats + 316 GB video |
| Ego4D | `feature_cache_ego4d_train_v1` | **GONE — 0 mp4 files** | 134,491 windows / 2,821 files | 518 GB feats |
| Action100M | `feature_cache_action100m` | `action100m_videos` | 2,054 shards; **134,989** videos | 1.5 TB feats + 2.3 TB video |
| AudioSet (audio-only eval mirror) | `audioset_feats/*.pt` | none (never obtainable) | eval 17,141 / bal_train 18,683 | 9.8 GB |
| AudioSet scrape residue | `audioset_scrape` | partial | — | 1.1 GB |
| EasyCom | `easycom` | yes | — | 145 GB |
| Rich captions | `scripts/qwen_omni_full_captions_v2.jsonl` | n/a | 6 fields/clip | 209 MB |
| SigLIP2 scene features | **GONE** (`/dev/shm/scene_all`) | rebuildable from VGGSound video | — | — |

### 1.1 VGGSound — healthy
Features and raw video both present. 1,545/1,545 eval-gallery clips cached; 13,674/13,679
query-predictor eval clips cached (99.96%). Ambient `T_a` spans 400–1002 tokens
(53 distinct values), `clip_duration_s` 4.02–10.08 s.

### 1.2 Ego4D — features survive, **video is gone**
`find` across `ego4d/` and `ego4d_probe/` returns **zero** `.mp4` files. Only 91 MB of
metadata and 6.1 GB of annotations remain. Consequences:

* The **674-window held-out gallery cannot be scored again by anything.** All 674 were
  checked against the 134,491 cached windows: **0 present**, file overlap **0 of 350**.
  (That independently confirms the split's `file_disjoint_verified: true` was honest.)
* RUN-2's 27.60 / 27.00 stand as published but are **permanently unverifiable**, and no
  future checkpoint can be compared against them. See `docs/ERRATA_PROPOSED.md` E-8.

### 1.3 Ego4D ordering — **recoverable, but only at 10-second granularity** (CRITICAL)

Cache ids are `ego4d_<file-uuid>_w<index>`, so per-file ordering is fully recoverable by
grouping on the uuid and sorting on the index. Contiguity, measured over all 134,491:

| quantity | value |
|---|---|
| files / windows | 2,821 / 134,491 (mean 47.7 windows per file) |
| adjacent index pairs that are consecutive | **59.1%** |
| files entirely contiguous | 258 (9.1%) |
| longest consecutive run per file | mean 12.6, median 5, max 149 |
| files with a run ≥ 5 / ≥ 10 / ≥ 30 | 1,539 / **1,042** / 357 |

**But the stride is 10 s, non-overlapping.** Every `start_sec` in
`EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json` is a multiple of 10, and index × 10 = start
second. So:

* Δ ∈ {10, 20, 30, 60} s **is** measurable today, from ~1,042 files with runs ≥10.
* Δ ∈ {1, 2, 5} s is **not obtainable at any price** from cached features, and
  re-extraction at a finer stride requires the raw video, which is gone (§1.2).

**Re-extraction cost, if Ego4D were re-downloaded:** the surviving cache is 518 GB for
134,491 windows at a 10 s stride = **3.85 MB/window**. A 1 s stride over the same footage
is 10× the windows ≈ **5.2 TB** — more than the 716 GB free, so it is a storage problem
before it is a compute problem. A 2 s stride over a 1,000-file subset is the largest
version that fits. GPU cost scales from the original extraction; that run is not
timed in any surviving log, so a GPU-hour figure here would be a guess and is **not given**.

### 1.4 Action100M — ordering is **not** recoverable
Ids are `<video-id>__<8-hex-hash>` (e.g. `qMVZ_805pP8__67abf3f6`), not an ordered index,
and `clip_duration_s` varies per segment (observed 23.69 s). Segment start times are not
in the cached tensors. Action100M is therefore unusable for temporal-sequence work
without re-deriving offsets from the source manifest.

### 1.5 AudioSet — audio-only, and **immune to the padding leak**
`audioset_feats/{eval,bal_train}.pt` hold `ids`, `labels`, `ambient_mean` (N,768),
`ambient_tokens` (N,32,768), `world_state` (N,1024), `world_state_tokens` (N,32,1024).
`scripts/audioset_extract_features.py:43-45` pads/crops every waveform to a **fixed**
`NSAMP = 160000` before batching, so every clip in a batch has identical length: no
`av_collate_fn`, no batch-dependent padding, no per-clip nuisance. **The AudioSet A-column
results are unaffected by E-1.** (The vision stream is deliberately zero-filled at fixed
geometry and is a constant across all clips, not a per-clip variable.)
AudioSet **video** remains unobtainable — the prior scrape stalled at 70.7% yield; not pursued.

### 1.6 What the reboot destroyed
`/dev/shm` is empty. Lost: the 199k-clip M2 training cache (`jepa_m2_cache`), the SigLIP2
scene features (`scene_all`), and the SigLIP2 text cache. The first two are rebuildable
(VGGSound video survives — I re-extracted 900 scene vectors in 117 s, 900/900, all
(8,768)); `scene_all`'s exact **membership** is not, which is why `sig_runD`'s absolute
figure can only be near-matched (E-6).

---

## 2. Epic-Kitchens-100 — the strongest acquisition candidate

| property | value |
|---|---|
| content | 100 h unscripted egocentric kitchen activity, 700 videos, 89,977 action segments |
| audio | yes, real ambient, 24 kHz (EPIC-Sounds provides audio-event annotations over the same footage) |
| download | **direct** from data.bris / Academic Torrents — no YouTube scraping, no per-clip failure rate |
| size | **741 GiB** (one source states 795.91 GB) |
| licence | **CC BY-NC 4.0 — non-commercial only** |

**Why it fits:** it is the only candidate that solves §1.3. Continuous long-form
recordings mean any stride is available, so Δ ∈ {1,2,5} s becomes measurable — the exact
thing Ego4D can no longer provide. It is egocentric with real ambient audio, matching
Ego4D's role in the corpus. EPIC-Sounds gives an audio event vocabulary that overlaps our
target classes far better than Ego4D's narration.

**Two blockers to decide, not for me to decide:**
1. **741 GiB against 716 GB free on Raid-Storage-2.** It does not fit. `/mnt/Raid-Storage`
   has 1.5 TB free and would hold the video, but features would then need a home too:
   at VGGSound's observed ~3.85 MB/window, a 2 s stride over 100 h ≈ 180,000 windows ≈
   **690 GB** of features. Video + features ≈ 1.4 TB. It fits on Raid-Storage **only if
   nothing else lands there**.
2. **CC BY-NC 4.0 is non-commercial.** Fine for an ICLR submission; a constraint on
   anything downstream. Flagging, not adjudicating.

## 3. External eval set with zero training overlap

**AVE (Audio-Visual Event)** — 4,143 ten-second videos, 28 event categories, split
3,339/402/402. Small, which is fine for an eval-only gallery. **Caveat that must be
checked before use: AVE is a subset of AudioSet**, and our corpus does not include
AudioSet video, so overlap with our *training* data is plausibly zero — but it overlaps
the AudioSet audio-only mirror in §1.5, so it is not automatically clean for the A-column.
Verify by id before adopting.

**VGGSound-Sync holdout** — would NOT be clean: VGGSound is 197k clips of our training
corpus, and only the 1,545 eval clips were excluded. Not recommended.

**Recommendation:** AVE, after an explicit id-level overlap check against both the
VGGSound corpus and `audioset_feats`. A 402-clip test gallery is small but it is the only
genuinely external option that does not require a scrape.

## 4. ACAV100M — available only as URLs, not usable without a scrape

ACAV100M is distributed as **public video URLs**, not media: 100M ten-second clips curated
from 140M source videos. Obtaining it means a YouTube scrape at exactly the scale that
already failed here (the Action100M scrape measured ~300 clips/hour at a 37% failure rate,
i.e. ~27 days for 199k clips, per `scripts/extract_siglip2_scene_vgg.py`'s own docstring).
Pre-extracted **features** are offered via Google Drive for the paper's own experiments,
but they are the authors' encoders, not ours, so they cannot feed our pipeline.

**Honest answer: not obtainable at useful scale here.** Do not plan around it.

## 5. AudioSet video — not pursued

Recorded per instruction. The prior scrape stalled at **70.7% yield**; the missing 29.3%
is not randomly distributed (deleted/private videos correlate with age and channel), so a
partial download is a biased sample, not a smaller one. The audio-only mirror in §1.5 is
the supported path and it is already extracted.

---

## 6. What this means for RUN-5 Arm B

Arm B (predict `W(t+Δ)`) has a hard data prerequisite that current holdings do not meet at
the proposed horizons. Three options, in order of cost:

1. **Retarget Arm B to Δ ∈ {10,20,30} s** on the surviving Ego4D cache (1,042 files with
   runs ≥10). Zero acquisition cost, runnable now. Tests persistence at a coarser horizon
   than intended.
2. **Acquire Epic-Kitchens-100** (§2). Solves it properly; costs ~1.4 TB and a storage
   decision.
3. **Re-download Ego4D** and re-extract at a finer stride. ~5.2 TB at 1 s stride — does
   not fit anywhere on this machine.

Option 1 is the only one that needs no decision from anyone. It is also the only one that
can start today.

---

## 8. Epic-Kitchens-100 — costed for BOTH consumers (P2.6)

Budget: **1.03 TB** on md1 after reclaiming the 315 GB of redundant VGGSound tarballs
(md1 is now 1.1 TB free; md0 is excluded — see the correction in the header).

Per-window costs: **3.94 MB** full token features (measured, 518 GB / 134,491 windows in
`feature_cache_ego4d_train_v1`) and **5.5 KB** scene representation-only (W + two reference means,
fp16). Ratio **734×**.

### Consumer A — probing, and a predictor fit on frozen `W(t)`

| stride | windows | storage |
|---|---|---|
| 1 s | 360,000 | **1.9 GB** |
| 2 s | 180,000 | **0.9 GB** |

Storage is a non-issue. This covers Phases 2–3 at fine Δ and RUN-5 Arm B **as its spec is
written** — "fit `g: W(t) → W(t+Δ)`", ridge and a 2-layer MLP over frozen scene representations.

### Consumer B — Arm B if it RETRAINS THE BRIDGE

Then scene representations are useless: you cannot backprop into the bridge through a stored `W`.
Full token features are required and the 734× reduction does not apply.

| footage | stride | windows | features | fits 1.03 TB? |
|---|---|---|---|---|
| 100 h | 1 s | 360,000 | **1.35 TB** | ✗ |
| **100 h** | **2 s** | **180,000** | **0.68 TB** | **✓** |
| 50 h | 1 s | 180,000 | 0.68 TB | ✓ |
| 25 h | 1 s | 90,000 | 0.34 TB | ✓ |

Maximum that fits: **76 h at 1 s stride**, or the whole 100 h at 2 s.

**Arm B is feasible on Epic-Kitchens — this is the headline.** The full 100 h at 2 s stride
yields **180,000 windows, more than the 134,491-window Ego4D cache the current model was
trained on**, for 0.68 TB. Ego4D could not do this at any useful scale (§1.3: 2,000 files at
2 s = 3.27 TB), so **Epic-Kitchens is not merely a nicer option, it is the difference
between Arm B being possible and impossible.**

The binding constraint is no longer features but the 741 GiB video download: 741 GB video +
680 GB features = 1.42 TB > 1.03 TB **if held simultaneously**. Stream-processing removes
that — `scripts/temporal_probe/stream_extract.py` downloads one file, extracts, writes, and
deletes the video, so peak usage is a working set.

### The one-shot constraint — decide the stride before downloading

The video is deleted after extraction, so **the stride cannot be revisited without
re-downloading the corpus**. A fine stride can always be decimated to a coarse one; the
reverse is a re-download. **Recommendation: 1 s if Arm B is scene representation-only (1.9 GB, no
reason not to), 2 s if Arm B retrains the bridge (0.68 TB, the whole corpus fits).**
If it is not yet settled which Arm B will be, extract **both** — scene representations at 1 s cost
1.9 GB alongside features at 2 s, which is free insurance against a re-download.

### Licence — FLAGGED FOR A HUMAN, NOT ACCEPTED

Epic-Kitchens-100 is **CC BY-NC 4.0: non-commercial use only**, with attribution. For an
academic ICLR submission that is ordinarily fine, but:

* it is a **restriction on downstream use** of anything trained on it, including the BMO
  deployment if that is ever commercial;
* it is **not a decision this session should make**. `stream_extract.py` refuses to run
  against `epic_kitchens` until the decision is recorded, and prints why.

Direct download from data.bris or Academic Torrents — **no YouTube scraping and no per-clip
yield risk**, unlike AVE or ACAV100M. That is a significant practical advantage.

---

## 9. Ego4D re-acquisition — prep (P2.7), and a conflict with §8

Everything below is ready to run the moment credentials arrive. The old AWS key is
**deleted server-side** (STS `InvalidClientTokenId`, S3 `InvalidAccessKeyId`), so nothing
can be fetched until the re-approval lands.

### 9.1 THE CONFLICT: Epic-Kitchens and an Ego4D feature subset cannot both fit

Both draw on the same **1.03 TB**. Epic-Kitchens at 100 h / 2 s stride takes **0.68 TB**,
leaving 0.35 TB.

Ego4D, full token features, from the measured 3.94 MB/window and a median source-file
duration of 870 s:

| budget | stride | files | windows |
|---|---|---|---|
| 1.03 TB (all of md1) | 1 s | **315** | 274,050 |
| 1.03 TB (all of md1) | 2 s | **630** | 274,050 |
| 0.35 TB (after Epic-K) | 1 s | **107** | 93,090 |
| 0.35 TB (after Epic-K) | 2 s | **214** | 93,090 |

**This is a choice, not an oversight.** Taking Epic-Kitchens for Arm B leaves room for only
~107–214 Ego4D files — well short of the 2,000–3,000 originally scoped.

**Recommended resolution: they serve different purposes, so split by output mode.**

* **Epic-Kitchens → full features (0.68 TB).** Continuous 100 h footage, any stride, direct
  download, no yield risk. It is the corpus that makes Arm B possible at all (§8).
* **Ego4D → scene representations only.** At 5.5 KB/window, **2,000 files at 1 s stride costs 9.1 GB**
  and 3,000 files costs 13.7 GB. Effectively free. That covers Phases 2–3 at fine Δ and any
  predictor fit on frozen `W(t)` — the things Ego4D is actually needed for, given the
  held-out gallery is unrecoverable (E-8) and Ego4D's value now is temporal structure rather
  than retraining data.

That combination fits comfortably: 0.68 TB + 13.7 GB against 1.03 TB, with ~0.33 TB spare.

### 9.2 Proposed acquisition

| | |
|---|---|
| files | **2,000** (of 2,821 in the surviving cache, so the selection is already scored) |
| stride | **1 s** |
| output | **scene representations** (`--mode world_state`) |
| cost | **9.1 GB**, 1,740,000 windows |
| download | ~0.5–0.7 GB/file ⇒ ~1.4 TB transferred, but **peak disk is a working set** via `stream_extract.py` |

1 s rather than 2 s because at scene representation cost the difference is 9.1 GB vs 4.6 GB — the
saving is meaningless and the resolution is not.

### 9.3 Filter adaptation for fine-stride selection

`scripts/ego4d_av_relevance_filter.py` scores candidate windows on
`top_nonspeech_event_prob × (1 − vad_speech_fraction) × energy_cov_norm`, deliberately
independent of M2. That scoring is **reusable unchanged**.

What must change is **candidate generation**. The current version spreads candidates
**evenly and non-consecutively** across each file (`--per-file-cap`, "EVENLY spread (not
consecutive)"). That is correct for building a diverse contrastive corpus and **wrong for
temporal work** — it is exactly why the surviving cache has only 59.1% adjacent-pair
contiguity and a median longest run of 5 windows (§1.3).

The change: select **contiguous SEGMENTS**, not spread singletons. Score each file's
candidate segments by the mean of the existing per-window score over the segment, then keep
the top-k segments per file at the target stride. A segment of N seconds at 1 s stride
yields N consecutive windows, which is what Phases 2–3 and any temporal objective need.

Recommended: **3 segments × 120 s per file**, giving 360 consecutive windows per file at 1 s
stride — 6× the current median run length of 5, and enough for Δ up to 60 s with many pairs.

### 9.4 ONE-SHOT CONSTRAINT — read before authorising

`stream_extract.py` **deletes each video after extraction**. The stride cannot be revisited
without re-downloading ~1.4 TB. A fine stride can always be decimated to a coarse one; the
reverse cannot be done. **Err finer.** At scene representation cost there is no reason not to use 1 s.

If there is any chance Arm B will retrain the bridge on Ego4D rather than Epic-Kitchens,
say so **before** the download — that decision changes the output mode from 9.1 GB to
~1.08 TB and does not fit alongside Epic-Kitchens.
