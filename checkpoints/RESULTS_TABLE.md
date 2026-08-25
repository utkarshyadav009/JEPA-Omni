# RESULTS_TABLE.md

Locked results as of the submission freeze (git tag `freeze-submission-v1`,
2026-07-26). Full provenance for every number here is in
`checkpoints/falsifier_tracking.md`; this table is a summary, not a
replacement for it.

## M0 — spine sanity gate

| Check | Result |
|---|---|
| `python -m models.spine_m1` | finite loss, correct shapes — PASS |

## M1 — offline vision-text spine

| Metric | Value | Gate |
|---|---|---|
| Video→text R@1 (MSR-VTT+VATEX, MLP predictor, batch 256, cached features) | 22.5 | within 5 pts of SigLIP2 baseline (32.5) — PASS |

## M2 — joint audio-visual predictor (LOCKED CHECKPOINT)

`checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` — VGGSound 197,007 clips + Ego4D 134k windows, 200×200 in-batch negatives, no AudioSet.

| Eval | vision→ambient R@1 | ambient→vision R@1 | Gate | Verdict |
|---|---|---|---|---|
| VGGSound retrieval | 53.27% | 53.72% | ≥52% | **PASS** |
| Ego4D held-out, sibling-excluded R@1 | 27.60% | 27.00% | ≥18.40%/18.40% (pre-registered) | **PASS**, best in the entire scaling study |
| Ego4D within-modality cosine (vision / ambient) | 0.4358 | 0.3893 | ≤0.25 | NOT MET (closest yet) |

### Scaling study, in run order (all VGGSound R@1, vision→ambient / ambient→vision unless noted)

| Run | Corpus | VGGSound R@1 | Ego4D sibling-excl R@1 | Notes |
|---|---|---|---|---|
| Matched-step check (step 6000) | 51,508 clips | 33.46% / 34.24% | — | apples-to-apples scale check |
| Matched-step check (step 6000) | 199,007 clips | 44.27% / 43.95% | — | confirms scale effect, same step count |
| VGGSound-60k + Ego4D-17.1k | 60k + 17.1k | 42.27% / 41.68% | — | first scaling-study datapoint |
| RUN-1 | 199,007 + 17.1k Ego4D | 55.15% / 55.53% (PASS) | 11.57% / 10.68% (FAIL, req. 18.40/18.40) | Ego4D loss traced to batch-share dilution as Ego4D's in-batch share fell with corpus growth |
| **RUN-2 / locked M2** | 197,007 VGGSound + 134k Ego4D, neg200×200 | **53.27% / 53.72% (PASS)** | **27.60% / 27.00% (PASS)** | decisive fix: Ego4D expanded 17.1k→134k restored batch share |
| RUN-3 (AudioSet added) | + 8,588 AudioSet-Strong clips | 23.04% / 12.56% (FAIL) | 8.75% / 12.91% (FAIL) | **CONFOUNDED — see NEGATIVE_RESULTS.md**, not attributable to AudioSet alone |

## M3 — vision-language connector (LOCKED CHECKPOINT)

`checkpoints/m3_multigran_richcaption_v2/last.pt` — trained on corrected rich captions (`qwen_omni_full_captions_v2.jsonl`), no alignment loss.

| Condition | Word-overlap F1 | Semantic cosine |
|---|---|---|
| Frozen-LLM baseline (pre any joint training) | normal 0.471 / swapped 0.268 / zeroed 0.274 | 0.724 |
| After first joint-exposure fine-tune (step 700) | −9% F1 relative to baseline | −19% relative (M4b side) |

## M4b — speech projector

| Condition | Semantic cosine (normal / swapped-vs-target / swapped-vs-donor) |
|---|---|
| Frozen-LLM baseline | 0.517 / 0.152 / 0.517 |
| Joint-exposure fine-tune (step 700), `m4_joint/best.pt` | 0.419 / 0.133 / 0.419 |

## M4c — turn-taking decision head (LOCKED CHECKPOINT)

`checkpoints/m4_decision_head_3class_speechonly_v2/best.pt` — speech-only, 3-class (silence/speak/backchannel), no World-State input.

| Condition | Accuracy | Macro F1 |
|---|---|---|
| World-State-consuming head (`m4_decision_head_3class_bothpresent`) | 93.67% | — |
| **Speech-only head (locked, deployed)** | **95.00%** | **94.98%** |

Six-condition falsifier (`A1_PROVENANCE.txt`) showed the World-State input functioned as a presence/scale slot, not an information channel — a constant dataset-mean vector matched or beat the real, correctly-paired World-State. Vision is not removed from the system by this finding; it still feeds generation via M3. Only turn-taking stops depending on it.

## M5 — Jetson deployment

| Measurement | Value | Source |
|---|---|---|
| Full-stack memory, directly on locked checkpoints (ViT-L + WavJEPA-base + WavJEPA-nat + M2 + Whisper-medium + Qwen2.5-1.5B-Instruct int8 + M3 + decision head, resident during generation) | **peak 6781MiB, headroom 839MiB / 7620MiB usable** | `checkpoints/m5_jetson/PHASE_A1_LOCKED_CHECKPOINTS_MEMORY_RESULTS.json` (2026-08-01, real run on-device, not architectural inference — confirms the prior 854MiB figure within 15MiB) |
| Real acoustic self-echo, `no_fix` false-interruption rate | **16.7% (2/12 windows)** | `checkpoints/m5_jetson/PHASE_B4_REAL_ECHO_TEST_RESULTS.json` (2026-08-01, real Piper playback + real mic recording through the ReSpeaker, with the speaker properly powered and audio correctly gain-normalized — see NEGATIVE_RESULTS.md-style note in falsifier_tracking.md for two earlier confounded runs, 91.7%/83.3%, discarded) |
| `MicGate` mechanism correctness during real playback | 100% (159/159 ticks) | same |
| Live control: quiet-room correctly labeled `silence`, live speech correctly labeled `speak` | PASS (both, with user physically present) | same |
| Streaming tick latency (10s window/stride, priority decision stream) | p95=786ms, mean=372ms, duty 37-41% | `JETSON_PHASE4_2_3_RESULTS.json` |
| VAD interruption latency (CPU) | 54.57ms | `JETSON_VAD_CPU_RESULTS.json` |
| TTS (Piper, `en_US-lessac-medium`, CPU/onnxruntime) — backchannel synthesis | mean 57.8ms (44-79ms range) | `JETSON_TTS_LATENCY_RESULTS.json` |
| TTS — generated-turn synthesis | mean 340.8ms (305-383ms range) | same |
| TTS — one-time voice load | 1.41s | same |

Hardware confirmed present and working (2026-08-01): Intel RealSense D435i camera + Seeed ReSpeaker 4 Mic Array (mic+speaker, one USB device), both via `lsusb`. C3 (live-vs-offline World-State cosine gate) is blocked on a `torchcodec`/libstdc++ incompatibility on this Jetson image, not yet resolved. Phase D (full end-to-end loop, sustained conversation, demo recording) needs a person physically at the Jetson and has not yet been run.
