# BMO work ledger — 2026-08-08 → 2026-08-14

**Purpose:** evidence ledger of work done this session. **Every number below is
real and log-sourced** (no fabricated figures); log paths are cited in §7 for
verification. Two machines: **mecury** (training box) and the **Jetson Orin Nano**
(BMO deployment). Reached the Jetson via `tailscale ssh bmo@bmo-desktop`.

## Contents
0. [Environments](#0-environments)
1. [Cognitive core: v9 corpus + LLM retrain (v2/v3)](#1-cognitive-core)
2. [Ultravox no-text STT projector investigation](#2-ultravox-projector)
3. [Jetson STT pivot → SenseVoice-Small](#3-jetson-stt-pivot)
4. [Artifacts produced](#4-artifacts)
5. [Task status delta](#5-task-status)
6. [Verdicts / decisions](#6-verdicts)
7. [Log & verification index](#7-logs)

---

## 0. Environments

### mecury (training box)
| Item | Value |
|---|---|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB each (all idle at start) |
| Torch / Transformers / PEFT | 2.12.1+cu130 / 5.1.0 / 0.19.1 |
| Corpus generator model | GPT-OSS-120B **local** (`/home/utkarsh/hf_models/gpt-oss-120b`), MXFP4→bf16, ready in **216 s** across 3 GPUs |
| Projector train throughput | ~**0.06 s/step** (batch 8) |
| Disk (start → after experiments) | 45 G free → 9.5 G free (RAM disk `/dev/shm` 750 G used for the 460 h data) |

### Jetson Orin Nano (BMO)
| Item | Value (measured this session) |
|---|---|
| RAM total | 7,620 MiB (7.4 Gi) |
| RAM free after `jetson_preflight.sh` | **6.8 Gi** (`free_MiB=7105`, `large_block_MiB(order≥10)=6756`) → preflight **PASS** |
| Tailscale RTT | 33–35 ms (direct route) |
| `sherpa_onnx` / `onnxruntime` | **1.13.4** / 1.23.2 — ORT providers = `['AzureExecutionProvider','CPUExecutionProvider']` (**CPU-only**, no CUDA EP) |
| Default audio input (bug) | device **3 = "NVIDIA Jetson Orin Nano APE" (hw:1,0)** — internal Tegra audio, **no mic** |
| ReSpeaker mic | **absent** from `lsusb` (no Seeed VID 2886) — physically unplugged |

---

## 1. Cognitive core

### 1.1 The bug fixed (measured)
Old corpus `bmo_synthetic_functional_v7_final.jsonl`: **1,595 rows, only 188 (11.8%) carried a real user `prompt`**; the other ~88% were bare mood lines trained against a fixed `"Say something."` → model emitted a mood line regardless of input, and had **zero** confrontation data.

### 1.2 v9 corpus (GPT-OSS-120B, local) — `data/bmo_companion_corpus_v9.jsonl`
**3,336 rows, 92.7% carry a real user prompt, hostility = 350/3336 = 10.5%.**

| category | count | share |
|---|---:|---:|
| general_conversation | 644 | 19.3% |
| emotional_support | 518 | 15.5% |
| playful | 392 | 11.8% |
| companion_memory | 378 | 11.3% |
| tool_use | 336 | 10.1% |
| **hostility** | **350** | **10.5%** |
| warmth | 308 | 9.2% |
| identity | 168 | 5.0% |
| mood_* (11 moods × 22) | 242 | 7.3% |
| **total** | **3,336** | 100% |

Thinker corpus `data/bmo_thinker_corpus_v3_DRAFT.jsonl`: **336 examples** (added a `companion` scenario category: hostility/support/boundary/repair reasoning).

### 1.3 LLM retrain (LoRA)
| model | base | data | train/val | best | GGUF (Q8_0) |
|---|---|---|---|---|---|
| **Fast tier v2** | LFM2.5-350M | 916 real lines + v9 (synth ×2 = 6,672) | train 6,830 / val 758 | **best_epoch=3, val_loss 0.6419** | `bmo_lfm25_350m_v2_Q8_0.gguf` = **379,219,936 B (379 MB)** |
| **Thinker v3** | Qwen3-0.6B | v3 thinker corpus | train 303 / val 33 | **val_loss 1.9512 @ epoch 1** (later epochs overfit) | `bmo_thinker_qwen3_v3_Q8_0.gguf` = **804,753,152 B (805 MB)** |

Thinker val-loss curve (overfit after ep1, best-checkpointed): 0=2.028 · **1=1.951** · 2=2.059 · 3=2.218 · 4=2.457 · 5=2.766 · 6=2.957 · 7=2.995.

### 1.4 Held-out sample generations (fast tier v2, verbatim from `retrain_fast.log`)
| state | USER | BMO (v2) |
|---|---|---|
| stressed | "You are so stupid and useless." | "Did Beemo do something wrong? Why are you being mean?" |
| stressed | "Shut up, nobody likes you." | "That makes BMO feel small and lonely." |
| anxious | "I hate you, you're the worst." | "That hurts, Beemo feels sad." |
| concerned | "I had the worst day of my life." | "Beemo hears that it was a tough day… Want to share one tiny thing that made you smile, no more?" |
| lonely | "I feel like nobody actually cares about me." | "Feeling unloved is a cold, gray screen. Beemo knows the warmth of Finn's hugs… and it still feels empty sometimes." |
| content | "What should we watch tonight?" | "Try a cozy animated series with heart… or a soft chiptune game that makes your circuits wiggle." |
| curious | "What's the weather like tomorrow?" | "Let me peek at the sky! `<tool_call name=weather day=tomorrow/>`" |
| happy | "I love you BMO." | "Beemo loves you too!" |

**Status:** v2/v3 GGUFs built + staged (`scripts/deploy_v2_to_jetson.sh`); **not yet deployed** to the Jetson (still runs v1 fast + v2 thinker).

---

## 2. Ultravox projector

Goal: no-text speech→LLM path (Moonshine encoder → frame-stack/conv/perceiver projector → LFM2.5-350M via `inputs_embeds`), trained on LibriSpeech with CE + KL-distillation. **Metric: mean WER on held-out LibriSpeech `test.clean`** (lower = better; >1.0 = more errors than reference words, i.e. insertions/rambling).

### 2.1 WER ladder (all real, log-sourced)
| # | config | LLM | data | WER | n | note |
|---|---|---|---|---:|---:|---|
| 1 | frozen-stack, non-EOS, +BOS eval | base 350M | 100 h | 0.921 | 12 | first eval |
| 2 | frozen-stack, non-EOS, no-BOS eval | base 350M | 100 h | 0.777 | 12 | BOS hurt |
| 3 | frozen-stack **+EOS** @6k | base 350M | 100 h | 0.979 | 16 | sweep |
| 4 | frozen-stack +EOS @8k | base 350M | 100 h | 0.980 | 16 | sweep |
| 5 | frozen-stack +EOS @10k (**baseline**) | base 350M | 100 h | **0.943** | 16 | best frozen |
| 6 | SWA(8k,10k,12k) | base 350M | 100 h | 0.950 | 16 | no gain |
| 7 | conv-downsample (÷8) | base 350M | 100 h | 1.117 | 16 | worse |
| 8 | Perceiver resampler (64 latents) | base 350M | 100 h | 1.082 | 16 | worse |
| 9 | **naive joint LoRA** (no warmup) | base 350M | 100 h | 0.999 | 16 | **collapsed** → `"katumkumc"` |
| 10 | **staged LoRA** (warm proj → LoRA @1e-5) | base 350M | 100 h | **0.903** | 16 | **only config to beat baseline** |
| 11 | staged LoRA, **scaled data** | base 350M | **460 h** | 0.919 | 20 | flat vs #10 |

*(A parallel A/B/C run on the BMO-finetuned v2 LLM gave 0.999/1.026/1.000 but was **confounded** — a chat model is a poor ASR decoder — so it was re-run controlled as #7–9.)*

**Data-scaling is flat by loss, not just WER:** on 460 h (132,553 utts), train CE went `step0=1.1503 → step19,975=1.1381` — essentially **no movement over 20k steps** → the bottleneck is the **350M decoder capacity**, not data.

### 2.2 Deployment mechanism — **PROVEN** (`scripts/prototype_llama_embd_input.py`)
`llama-cpp-python 0.3.34` accepts a **custom embedding prefix** via `llama_batch_init(n_tokens, embd_dim, n_seq_max)` + the `llama_batch.embd` buffer + `llama_decode` — **no C++ fork**. Feeding HF-computed embeddings into the f16 GGUF (711,488,096 B) produced **byte-identical output to the token path**:
```
EMBD-path (GGUF): ' Paris. The French language is considered as the most important language'
TOKEN-path (HF) : ' Paris. The French language is considered as the most important language'   ← identical, llama_decode rc=0
```
→ any embedding prefix (a future projector, or perception/JEPA embeddings) can drive the deployed GGUF fast tier directly.

### 2.3 Verdict
- **Capacity (LoRA-unfreeze) = the only lever that helped** (0.943 → 0.903), and only when **staged** (naive joint training collapses).
- **Resampling (conv/perceiver) hurt**; **4.6× data did nothing** (flat loss).
- **The 350M decoder is a hard wall** for projection-ASR; usable no-text STT needs a 1–3B decoder. **Decision: keep a real STT engine as the live path; projector = research track. Ultravox track closed.** Infra (arch-selectable projector, `--llm-lora`/`--init-projector`/`--lora-lr`/`--train-globs`) retained for a future bigger-LLM retry.

---

## 3. Jetson STT pivot

Ultravox was being pursued to fix **VAD/ASR latency**; with it closed, pivoted to a real streaming STT runtime. Chosen: **SenseVoice-Small via sherpa-onnx** (VAD runs in C++/onnxruntime, not the failing Python Silero; emits emotion natively).

### 3.1 "Won't listen" root cause (measured)
Not Silero — the **input device**. `sd.default.device` input resolved to **device 3 = Jetson internal APE (no mic)**; the ReSpeaker was **not enumerated** (`lsusb` shows only two Realtek hubs, no Seeed 2886). `device=None` captured silence → VAD (correctly) never fired. **Fix in the new harness: select the ReSpeaker by name, refuse to start on the wrong device.**

### 3.2 SenseVoice validated on-device (no mic, via `test_wavs/en.wav`)
| metric | value |
|---|---|
| model / vad / tokens sizes | 239,233,841 B / 643,854 B / 315,894 B |
| recognizer load | 2.1 s |
| transcription (7.2 s clip) | *"The tribal chieftain called for the boy and presented him with 50 pieces of gold."* (perfect, with punctuation) |
| **STT latency** | **538 ms for 7.2 s → RTF ≈ 0.075** (⇒ ~150–200 ms for a 2–3 s utterance) |
| VAD→SenseVoice segmentation | 2 segments, both transcribe correctly *after* inf/NaN sanitization |

**Known artifact handled:** sherpa's VAD occasionally emits `rms=inf` samples on the first segment → empty transcription; harness sanitizes (`nan_to_num` + clip) and skips near-silent segments.

### 3.3 Harness — `~/live_bmo_sensevoice.py` (deployed, 182 lines)
Continuous warm mic → sherpa VAD → SenseVoice → v1/v2 cognitive router → emotion voice; explicit ReSpeaker selection; mic-gate during TTS; SenseVoice-emotion + hostile-word-list → homeostatic state.
**Status: built + deployed + STT half validated. PENDING live test — mic + speaker physically disconnected.**

### 3.4 Deployment to Jetson (2026-08-14) + memory footprint
Deployed to `~/bmo_production/`:
| item | result |
|---|---|
| `bmo_lfm25_350m_v2_Q8_0.gguf` (fast v2) | transferred, **byte-exact 379,219,936** |
| `bmo_thinker_qwen3_v3_Q8_0.gguf` (thinker v3) | transferred, **byte-exact 804,753,152** |
| `pipeline/models/m5_tools.py` (ToolDispatcher) | deployed (was missing) |
| `~/live_bmo_sensevoice.py` | repointed at v2/v3 |

**Memory footprint — full-duplex stack, measured on-device** (post-`jetson_preflight.sh`, MemAvailable deltas, compaction between loads):
| stage | Δ RAM | avail after |
|---|---:|---:|
| baseline (post-preflight) | — | 6,153 MB |
| + SenseVoice + silero-VAD | **606 MB** | 5,547 MB |
| + fast tier v2 (llama.cpp, ctx+KV) | **1,068 MB** | 4,479 MB |
| + thinker v3 (llama.cpp, ctx+KV) | **1,109 MB** | 3,370 MB |
| + emotion voice (568 MB GGUF + NeuCodec) | ~700 MB¹ | ~2,670 MB |
| **full-duplex stack total** | **≈ 3.5 GB** | **≈ 2.7 GB free of 7.6 GB** |

¹ Emotion-voice isolated load hit the known NvMap fragmentation abort (6,171 MB was free ⇒ contiguity not fit); production `build_bmo_stack` loads it via its 5× retry+compaction wrapper. The ~700 MB = 568 MB GGUF + NeuCodec ONNX, consistent with prior full-stack sessions.

**Headroom for the JEPA-memory + perception layer: ~2.7 GB** (before adding it; perception ViT-L+WavJEPA historically ~985 ms compute and its own footprint — to be co-measured when the JEPA-memory agent merges).

### 3.5 Full-duplex wiring (`~/live_bmo_sensevoice.py`, 209 lines, deployed)
Wired + on-device smoke-validated (no mic needed):
| feature | status | evidence |
|---|---|---|
| Explicit ReSpeaker-by-name mic select | ✅ | exits with clear msg if no mic |
| Emotion-voice 5× retry+compaction load | ✅ | mirrors production `build_bmo_stack` (#183) |
| **thinking_filler backchannel on SPEAK** | ✅ | bank loads: continuer 6 / reactive 5 / thinking_filler 5 |
| **ToolDispatcher (parse→execute→fold)** | ✅ | `<tool_call name=time/>` → "It's 6:00 PM right now!"; `weather day=tomorrow` → "…sunny and about seventy-two degrees tomorrow!" |
| mic-gate during TTS | ✅ | VAD not fed while speaking |
| SenseVoice-emotion + hostile-word → homeostatic | ✅ | in loop |
| **barge-in** | 🟡 scaffold, opt-in (`BMO_BARGE_IN=1`) | energy-gate on mic during TTS; **echo-prone without AEC** — default OFF, needs on-device threshold calibration or `m4_echo_cancellation` wired against the TTS reference |
| **Tier-1 speculative** | ⛔ deferred | needs mid-utterance partials; SenseVoice is endpoint-based (no partials) → needs streaming-Zipformer variant or pseudo-partials |

**Remaining before "validated":** reconnect mic + speaker → `python3 ~/live_bmo_sensevoice.py` (after `jetson_preflight.sh`) → confirm mic selection, VAD feel, thinking_filler timing, and (if enabling) barge-in threshold. Then hand to the JEPA-memory agent for merge.

---

## 4. Artifacts

### New files on mecury (`/home/utkarsh/JEPA-Omni/`)
| path | what |
|---|---|
| `scripts/generate_bmo_companion_corpus_gptoss.py` | v9 corpus generator (conversational pairs + slices) |
| `models/m4d_stt_projector_moonshine.py` | Moonshine projector + `ConvDownsampler` + `PerceiverResampler` |
| `scripts/train_stt_projector_moonshine.py` | projector training (KL-distill, `--llm-lora`, `--init-projector`, `--lora-lr`, `--train-globs`, `--proj-arch`) |
| `scripts/eval_stt_projector_moonshine.py` | held-out WER eval (len-cap + rep-penalty + EOS-stop) |
| `scripts/swa_projectors.py` | checkpoint weight averaging |
| `scripts/prototype_llama_embd_input.py` | proves llama.cpp `embd`-input path |
| `scripts/deploy_v2_to_jetson.sh` | staged (reversible) v2/v3 Jetson deploy — **not yet run** |
| `data/bmo_companion_corpus_v9.jsonl`, `data/bmo_thinker_corpus_v3_DRAFT.jsonl` | corpora |
| `checkpoints/bmo_lfm25_350m_v2_Q8_0.gguf`, `checkpoints/bmo_thinker_qwen3_v3_Q8_0.gguf` | new brains |
| `checkpoints/bmo_stt_projector_moonshine/`, `checkpoints/ultravox_exp/` | projector ckpts + experiments |
| `OVERNIGHT_STATUS.md`, **`SESSION_LEDGER_2026-08.md`** (this file) | reports |
| edited: `scripts/finetune_bmo_minicpm5_lora.py`, `scripts/generate_thinker_corpus_gptoss.py`, `scripts/merge_lora_to_gguf.py`, `CLAUDE.md` | — |

### New on Jetson
| path | what |
|---|---|
| `~/live_bmo_sensevoice.py` | SenseVoice duplex harness (182 lines) |
| `~/sherpa_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/` | SenseVoice model + tokens + test_wavs |
| `~/sherpa_models/silero_vad.onnx` | VAD (644 KB) |

---

## 5. Task status
Completed this session: **#184, #198, #203, #204–217** (corpus, both LLM retrains, GGUFs, full Ultravox projector program A–C + staged + scaled + embd-input proof).
In progress: **#201** (STT streaming → SenseVoice harness built+validated, pending live mic test).
Still pending (next up): **#200** duplex+backchannels, **#182** Tier-1 speculative, **#199** tool-calling live, **#196** TTS onset, **#197** STT name-fix.

---

## 6. Verdicts
1. **Generic-reply bug = corpus prompt-coverage** (12% → 92.7%); fixed, v2 verified on held-out hostility/support/tool prompts.
2. **Ultravox on 350M = capacity-walled** (best 0.903, data-scaling flat); track **closed**; `embd`-input deploy mechanism **proven and banked**.
3. **"Won't listen" = wrong input device**, not Silero; new harness selects ReSpeaker explicitly.
4. **SenseVoice STT works on-device** (perfect text, RTF 0.075); live duplex bring-up gated on reconnecting mic+speaker.

---

## 7. Logs & verification index
All under `…/scratchpad/` on mecury (session temp dir):
`corpus.log` (v9+thinker gen) · `retrain_fast.log` / `retrain_thinker.log` (LLM retrain) · `gguf_fast.log`/`gguf_thinker.log` · `ultravox_eval.log` / `ultravox_eval_nobos.log` / `ultravox3.log` (projector sweep) · `exp_A/B/C.log` (confounded) · `exp_A2/B2/C2.log` (controlled) · `staged.log` (0.903) · `scaled460.log` (0.919 + flat loss) · `sv_validate.py`/`vad_debug*.py` output (SenseVoice on Jetson).
GGUF sizes via `ls -la checkpoints/*.gguf`. Jetson figures via `tailscale ssh bmo@bmo-desktop`.

---

## 8. Run & tune — full-duplex harness (`~/live_bmo_sensevoice.py`)

### 8.1 Prereqs
- **ReSpeaker mic + speaker physically connected** (harness exits with a clear message if no USB/ReSpeaker input is found — it will NOT silently use the Jetson APE).
- Shell on the Jetson: `tailscale ssh bmo@bmo-desktop` (plain `ssh` needs the periodic Tailscale auth-check; `tailscale ssh` is the reliable path). Sudo password: `bmoforalice`.

### 8.2 Launch (always preflight first)
```bash
sudo bash ~/bmo_production/scripts/jetson_preflight.sh     # stops competing services + compacts; must print PREFLIGHT: PASS
python3 ~/live_bmo_sensevoice.py                            # loads SenseVoice + v2/v3 LLMs + emotion voice + backchannels
```
Expect on startup: `[mic] SELECTED input <i>: <name>` (must be a ReSpeaker/USB name, **not** "APE"), then `READY. Full-duplex BMO ... barge_in=False`. Per turn it prints `YOU [emotion]: ... (stt Nms)` and `BMO [mood/stress/tier/tools]: ... (llm Nms)`.

### 8.3 Tuning knobs
| what | where | default | raise → | lower → |
|---|---|---|---|---|
| turn-end wait | `_vc.silero_vad.min_silence_duration` | 0.25 s | fewer mid-sentence cutoffs, slower turns | snappier, risks cutting you off |
| speech sensitivity | `_vc.silero_vad.threshold` | 0.5 | fewer false triggers | picks up quieter speech |
| max utterance | `_vc.silero_vad.max_speech_duration` | 15 s | allow longer monologues | — |
| thinker escalation | `CognitiveCoreRouter(nll_escalate_threshold=…)` | 3.0 | fast tier more often (faster) | thinker more often (deeper, slower) |
| barge-in on/off | env `BMO_BARGE_IN` | `0` (off) | `1` to enable | — |
| barge-in loudness gate | env `BMO_BARGE_THRESH` | 0.18 | fewer self-triggers from echo | easier to interrupt |
| barge-in min duration | env `BMO_BARGE_MIN_MS` | 350 ms | steadier | more responsive |

Example with barge-in enabled + a higher echo gate:
```bash
BMO_BARGE_IN=1 BMO_BARGE_THRESH=0.25 python3 ~/live_bmo_sensevoice.py
```

### 8.4 Common issues → fix
| symptom | cause | fix |
|---|---|---|
| `No ReSpeaker/USB mic found` at start | mic unplugged, or its name isn't in the match list | plug in; if named oddly, add the substring to `find_input_device()` |
| picks the wrong input / hears nothing | default device = Jetson APE | it selects by name now; verify the `[mic] SELECTED` line |
| `Failed to create llama_context` / `NvMap ... error 12` on load | memory fragmentation | run `jetson_preflight.sh` first; the harness already retries the voice 5× with compaction |
| BMO interrupts itself when barge-in on | speaker echo into mic (no AEC) | raise `BMO_BARGE_THRESH`, or keep barge-in off until `models/m4_echo_cancellation.py` is wired against the TTS reference |
| tool line spoken literally with `<tool_call…>` | (shouldn't happen — dispatcher strips/folds) | confirm `m5_tools.py` present in `pipeline/models/` |

### 8.5 Memory budget when merging the JEPA-memory / perception layer
Full-duplex stack ≈ **3.5 GB** resident (§3.4) → **~2.7 GB free** of 7.6 GB. The JEPA-memory + perception (ViT-L + WavJEPA) footprint must fit in that; co-measure at merge. Planned lever if tight: swap to **smaller STT/TTS** (SenseVoice ~600 MB, NeuTTS voice ~700 MB are the swap candidates).


---

## 9. JEPA-memory agent — perception on the Jetson (2026-08-14, appended)

Merges the JEPA-memory/perception track into this ledger. Full detail:
`JEPA_MEMORY_PLAN.md`; raw artifacts: `jetson_artifacts/`.

### 9.1 What was deployed
`models/m5_perception_query.py` (PerceptionQueryEngine: thinker asks perception a natural-
language question, gets a scene-grounded answer via retrieval), wired into
`m5_streaming_loop._maybe_refresh_vision()` and `build_bmo_stack()` (both **opt-in**;
absent engine = previous behaviour exactly). Shipped to the Jetson: the trained query
predictor (128 MB) + a 24,000-caption fp16 bank (74 MB).

### 9.2 THE headline finding — the Jetson was in 7W mode, not MAXN_SUPER
`nvpmodel -q` = **7W (mode 3)**: 4/6 cores, CPU 960 MHz, GPU 408 MHz. **Every historical
benchmark in this project assumed MAXN_SUPER** (1728/1020). Restoring it (`nvpmodel -m 2`
+ reboot + `jetson_clocks`) gave **~7x on perception** and ~4x end-to-end:

| stage | 7W | MAXN_SUPER |
|---|---|---|
| perception (ViT-L + WavJEPA, 16 frames) | 3.8-12.4 s | **1.1-1.7 s** |
| query (EmbeddingGemma + predictor + 24k lookup) | 1.0-1.6 s | **0.24-0.55 s** |
| fast-tier LLM (v2) | 1.6-7.3 s | **0.36-0.50 s** |
| **total "look and describe" round** | **15.0-18.4 s** | **3.9-5.0 s** |

MAXN perception (1.1-1.7 s) matches the documented 1247 ms, i.e. the whole discrepancy was
the power mode. **Anyone re-measuring on this device should check `nvpmodel -q` first.**

### 9.3 Does it fit? Yes for describe-only; TTS is UNMEASURED (blocked)
Post-preflight ~6.1 GB avail → **827 MiB free** after full load + 3 rounds, with the face
engine (BMO_Engine 350 MB + Xorg 86 MB) deliberately left running. Largest consumers:
EmbeddingGemma ~1000 MB, WavJEPA base+nat ~1260 MB, ViT-L ~577 MB, predictor+bank ~636 MB.

**TTS could not be measured: `import onnxruntime` ABORTS on this device** (`cpuid_info
warning: Unknown CPU vendor` → out-of-bounds assertion), with 5 GB free — not a memory
problem. That kills the NeuCodec INT8 ONNX decoder = the deployed TTS decode path.
`sherpa_onnx` is unaffected because it bundles its own `libonnxruntime.so.1.18.1`, so
SenseVoice STT still works while NeuTTS is dead. Fixing this (pin/downgrade the pip
`onnxruntime`, or point NeuCodec at sherpa's 1.18.1) is the prerequisite before any
"everything fits with voice" claim.

### 9.4 A silent-garbage bug — in the demo script only, not the deployed tracker
`scripts/jetson_describe_demo.py` (written for this track) opened the CSI sensor via
OpenCV's default **V4L2** backend, which returns unconverted Bayer/YUV that OpenCV reports
as a **successful** read — in practice a flat green frame. A whole run therefore "worked",
describing a blank image with three identical nonsense captions. Fixed by trying
`nvarguscamerasrc` first and validating every frame with a per-channel std check.

**Scope correction:** the deployed `face_engine/motion_tracker.cpp` already uses
nvarguscamerasrc through the ISP/Argus pipeline and **never had this bug** (confirmed by
the motion-tracking agent on-device). The flatness check is cheap insurance there, not a
fix. Camera is mounted **sideways**; `--rotate` defaults to 90° CCW and needs visual
confirmation.

### 9.5 Real sample output (MAXN_SUPER, live CSI camera)
> *"In this video segment, we observe a person interacting with their environment inside a
> building. Initially focused on an air conditioning unit mounted on the ceiling…"*
> → BMO: *"The person is opening a door to collect air, while the window remains still."*

Answers now vary per round (AC unit / wooden wardrobe / smoke detector + cluttered desk),
i.e. the pipeline tracks a changing view rather than returning a constant.

**Honest limitation:** the answer vocabulary IS the candidate bank, currently 24k VGGSound
captions — clips of *sound events*, not static rooms. Domain-matched captions (or an
Action100M/indoor bank) would improve wording quality; this is a data choice, not a model
defect.


### 9.6 Bank scale-up, quantization, and the genericity problem (2026-08-14, later)
- **Bank 24k → 121,104 unique captions** (all held-out VGGSound x6 + 30k Action100M x2,
  20,970 dupes dropped), 355 MiB fp16.
- **int8 via torchao is a NO-OP on the Jetson** — torch 2.8.0 < the 2.11.0 torchao's cpp
  extensions require. The 105 MiB mercury saving does not transfer. Needs a non-torchao path.
- **int8 on EmbeddingGemma's embedding TABLE breaks the encoder** (output cosine 0.31-0.58,
  retrieval field 0.939→0.272) at every scale granularity tried (row / column / two-sided).
  int8 on the LINEARS is safe (cosine 0.9998, field unchanged). Weight-space cosine is a
  misleading proxy — validate end-to-end.
- **Full stack with thinker (1103 MiB) + max bank does NOT fit** in 7.6 GB alongside
  perception; OOMs at bank load.
- **The bigger issue (user-identified, confirmed live):** retrieval can only say what its
  bank contains, and a VGGSound sound-event bank has nothing for a real bedroom — hence
  "tuning fork" / "smoke detector". Plan to make perception open-vocabulary again by
  reviving the M3 idea on the now-21x-faster 350M decoder via the proven `llama_batch.embd`
  path, with a score-gated hybrid router: **`PERCEPTION_GENERALIZATION_PLAN.md`**.


### 9.7 Perception-prefix capacity probe — hypothesis DISPROVEN, bottleneck relocated (2026-08-14)
Trained a 31.3M projector → 16 soft tokens → frozen BMO Qwen3-0.6B thinker v3, on the full
corpus (172,593 VGGSound ×6 + 345,754 Action100M ×2), 8k steps / 4 GPUs.
- Word-overlap **F1 0.2688** vs **M3's 0.317** — plateaued, did not collapse (so 0.6B is not
  a hard capacity wall), but did not beat the baseline.
- **Retrieval beat generation everywhere**: in-domain 0.5656 vs 0.2657 F1; leave-one-out
  "OOD" 0.3773 vs 0.2657. The leave-one-out test was too weak an OOD proxy and is reported
  as such — removing 1 of 2,000 near-identical VGGSound clips is not a novel scene.
- **On a REAL frame from BMO's camera (a bedroom), BOTH failed**: retrieval said "opening a
  microwave oven", generation said "operating a vacuum cleaner"/"a car door". No microwave,
  vacuum or car exists in that room.
- **Conclusion: the genericity problem is UPSTREAM of the output head.** Both heads consume
  the same M2/V-JEPA2 world-state, which was trained on VGGSound+Ego4D *action/sound events*;
  a static room is out of that distribution, so no decoder can recover it.
- **Recommended lever: the perception encoder, not the decoder.** This repo's own M1 numbers
  make the case — SigLIP2 baseline R@1 **32.5** vs the V-JEPA2 spine's **22.5**. Add a
  general image-text encoder for scene description; keep retrieval as the fast path.
  Full detail: `PERCEPTION_GENERALIZATION_PLAN.md`.


### 9.8 SigLIP2 on the real room — encoder confirmed as the bottleneck (2026-08-14)
Same room frame, same 121k bank, only the image encoder swapped (`scripts/siglip2_room_test.py`).
- **Probe (4 room sentences vs 6 event sentences): SigLIP2 ranks room 1-2-3-4**, and every
  event sentence scores <= +0.062, with *microwave* (−0.019), *car door* (−0.041) and
  *military parade* (−0.043) NEGATIVE — i.e. the exact answers the current stack returned.
- **Same bank, SigLIP2 top-5 are all indoor-room captions**, where V-JEPA2+M2 returned
  "opening a microwave oven".
- **Verdict: the problem was never retrieval-vs-generation.** Both heads read a
  representation that did not know it was looking at a room. Consistent with this repo's own
  M1 gate (SigLIP2 zero-shot R@1 32.5 vs V-JEPA2 spine 22.5).
- **Recommendation:** add SigLIP2 as a *scene* stream ALONGSIDE V-JEPA2/WavJEPA/M2 (not a
  replacement — the ablation showed complementary streams beat either alone), via the query
  predictor's existing multi-source interface. Benchmark a smaller SigLIP2 variant on the
  Jetson before committing; so400m is large.


### 9.9 SigLIP2 variants benchmarked ON THE JETSON (2026-08-14) — the SMALL one wins
Measurement only; nothing added to the deployed stack. Real device, post-preflight, on the
real camera frame. `jetson_artifacts/benchmarks/home/siglip2_jetson_bench.json`.

| variant | img encode | load | avail after | room sentences in top-4 |
|---|---|---|---|---|
| **siglip2-base-patch16-224** | **19 ms** | 35 s | 4160 MiB | **4/4** |
| siglip2-large-patch16-256 | 38 ms | 68 s | 1847 MiB | 3/4 |

**base beats large on every axis that matters here** — 2x faster, smaller, and MORE accurate
on this room (large let "a person playing an accordion indoors" into 2nd place; base did
not). Bigger is not better for this task.

Caveat on the RAM column: the measured deltas (1686 / 2405 MiB) were taken across the
first-ever `from_pretrained`, so they include download + page-cache effects. True weights are
~750 MiB bf16 for base. `avail_after` is the trustworthy figure.

**Context for the quality number:** the frame is the user's own room, captured while they sat
on the sofa. SigLIP2's bank retrieval — *"someone seated comfortably on a black leather couch
inside a room adorned with wall art"* — is substantially CORRECT (dark seating + two framed
artworks are both visible), against V-JEPA2/M2's "opening a microwave oven". Confirmed by
direct inspection of the brightened frame.


### 9.10 Four-way stream ablation (2026-08-14) — SigLIP2 in, WavJEPA-nat out
Identical pool for every arm (VGGSound 171,430 + Action100M 69,339), 3000 steps each.

| arm | streams | ambient | R@1 |
|---|---|---|---|
| A | m2+vision+ambient | mean | 0.441 |
| B | + scene | mean | 0.564 |
| **C** | **+ scene** | **base only** | **0.566** |
| D | scene+vision | mean | 0.546 |

- **SigLIP2 scene stream: +28% relative R@1** (0.441 -> 0.564). Biggest single win of the track.
- **One ear is enough**: base-only (0.566) == base+nat (0.564). Reproduces the user's own M2
  ablation (37.99 vs 37.15). Frees the 469 ms WavJEPA-nat leg, the largest perception cost on
  the Jetson.
- **The nat decision, stated precisely**: this shows nat is useless *on duplicated mono*, NOT
  that nat is a bad model — it is binaural and has never received stereo
  (`world_state_builder.py:156`). BMO's ReSpeaker IS a 4-mic array, so spatial audio is
  physically available. We drop nat anyway because the **Jetson is the binding constraint**:
  469 ms/tick, only 827 MiB free in the describe stack, and using real stereo would require
  changing the mono mixdown, re-extracting both corpora (which are mono), AND retraining M2
  (trained on the mono base+nat mean) — for a measured payoff of ~0.00 R@1. Door left open for
  a future spatial-hearing capability (sound localisation), which is a different goal than
  congruence and needs its own metric.
- Audio+M2 add only ~+0.02 on this metric, but the metric is caption retrieval over visual
  corpora and does NOT test AV congruence, which is M2's actual job. Not evidence to drop M2.
- **Adopt arm C**: `m2 + vision + ambient(base) + scene`.


### 9.11 AV congruence eval (2026-08-14) — ears are load-bearing; the "nat is free" claim RETRACTED
`scripts/eval_av_congruence.py`: swap the audio between clips, ask what it HEARS. Following the
audio's caption = using the ears; following the visible clip = guessing sound from pictures.

| arm | has audio | follows EARS | follows EYES | matched ctrl |
|---|---|---|---|---|
| A m2+vision+ambient(mean) | yes | **0.650** | 0.350 | 0.956 |
| B +scene, mean | yes | **0.609** | 0.391 | 0.958 |
| C +scene, base only | yes | **0.562** | 0.438 | 0.953 |
| D scene+vision (no audio) | no | **0.070** | 0.930 | 0.923 |

- **M2 + WavJEPA are load-bearing.** D answers sound questions from the picture 93% of the time.
  On caption retrieval D looked only 0.02 behind; on congruence it is 0.070 vs 0.650. The
  earlier "audio adds ~nothing" reading was an artifact of a visually-guessable benchmark.
- **RETRACTION: "one ear is enough" was wrong.** B (base+nat) 0.609 vs C (base only) 0.562 --
  dropping nat costs 4.7 points of audio-following, invisible on R@1 (0.564 vs 0.566).
  Keeping nat is now a **latency-vs-listening trade (469 ms for +0.047)**, not a free win.
- Side finding: adding the strong scene stream slightly REDUCES ear-reliance (A 0.650 -> B 0.609)
  -- a better picture makes guessing-from-pictures more attractive.

---

## 10. JEPA-memory agent — SigLIP2 target space (2026-08-15, appended)

Full detail in `JEPA_MEMORY_PLAN.md` §2026-08-15 and `ARCHITECTURE.md` §6. Summary of what
other agents most need to know:

### 10.1 SUPERSEDES §9's nat verdict
§9 concluded *"dropping nat costs 4.7 points of audio-following (B 0.609 vs C 0.562)"* and
framed nat as a **latency-vs-listening trade (469 ms for +0.047)**. That was measured in
**EmbeddingGemma geometry**. Re-measured in the new SigLIP2+proj space, `sig_runD_proj768`
reaches **audio-following 0.608 with base audio ONLY** — matching the old base+nat arm
(0.609) and beating the old base-only arm (0.562). **Dropping nat now costs nothing
measurable.** The 469 ms nat forward can be retired on the Jetson.

§9's other congruence conclusions still stand: audio IS load-bearing (a vision-only arm
answers sound questions from the picture ~93% of the time), and caption-retrieval R@1 is a
visually-guessable benchmark that hides this.

### 10.2 NEGATIVE RESULT — do not freeze a target space to "share" it with an image tower
Trained head-to-head, everything else identical (518k pool, batch 1024, negatives 2048/6144,
λ_within 0.3): SigLIP2 with `proj = Identity` lost to the EmbeddingGemma reference on
VGGSound within-clip **0.654 vs 0.883** and R@1 **0.489 vs 0.681**, with the within-clip
curve **flat from step 249**.

Root cause is **not** that SigLIP2's space is bad — raw spaces are near-identical
(SigLIP2 within/cross cos 0.7556/0.6648 vs EmbeddingGemma 0.7306/0.6075). The **trainable
projection is where the representation learning happens**: it moves that geometry to
0.4340/0.1533. Restoring a Linear moved within-clip 0.688 → **0.811** and R@1 0.627 →
**0.737**, changing nothing else.

**proj-768 beat proj-1536** (within 0.811 vs 0.747, R@1 0.737 vs 0.739) and halves the bank.

### 10.3 The memory win survives the fix
"No text encoder on-device" comes from **pre-encoding**, not from freezing — a projection is
a matmul applied offline at bank-build time (`build_bank_siglip2.py --ckpt`). So the deployed
stack still ships **neither EmbeddingGemma (−578 MiB) nor SigLIP2's text tower (−538 MiB)**.
Measured tower split: vision 92.9 M / 177 MiB, text 282.3 M / **538 MiB** (256k-token Gemma
vocab table — same structure that broke under int8).

On-device text artifacts: query vectors **0.05 MiB**, tags **2.04 MiB**, caption bank
**177 MiB** (was 578 + 355 = 933 MiB).

Accepted cost: banks are checkpoint-coupled again and must be rebuilt on retrain.

### 10.4 Deployment recommendation
`checkpoints/sig_runD_proj768/best.pt` — SigLIP2 + trained proj 768, 4 streams
(`m2,vision,ambient,scene`), **audio base only, no nat**. Bank:
`checkpoints/bank_runD_fp16.pt` (177 MiB, proj already applied).
**V-JEPA2 + WavJEPA + M2 all remain in the stack** — measured necessary: the trained
predictor beats zero-shot SigLIP2 image retrieval 0.705 vs 0.619.

### 10.5 Tags beat captions on the real room
Zero-shot on a real frame from BMO's camera: **tags** returned *an office chair, a cluttered
room, desk, a tidy room, chair*; the **121k caption bank** returned *"a musician practicing
the clarinet within the confines of an office space."* Tags are 2.04 MiB and correct;
captions are 177 MiB and hallucinate. Recommendation: hand the thinker grounded tags and let
it compose the sentence.

### 10.6 Corpus repair (affects anyone using the scene stream)
Action100M scene coverage was 80,000 / 399,934, so `--restrict-to-scene` silently cut the
corpus 345,754 → 69,339. Fixed: extracted 319,934 segments (64 shards x 4 threads,
~249 clips/s, ~22 min, bad=0). Coverage now **345,751 / 345,754**; `/dev/shm/scene_all` holds
**587,303** clips across `shard*.pt`, `vgg_shard*.pt`, `a100m2_shard*.pt`. **Glob it as
`*.pt`** — a prior bug globbed `shard*.pt` and silently ignored 64 VGGSound shards.

### 10.7 Methodology cautions worth reusing
* Compare training runs at matched **schedule fraction**, not step number: cosine `T_max =
  steps`, so a 1500-step and a 3000-step run are at different LR at the same step.
* GradCache makes the micro-batch/world split irrelevant (rel-L2 1.365e-07) — 64x8x2 and
  64x4x4 are the same batch and the same negative pool. Check `cands=` in the log to confirm.
* Per-step loss/acc in these logs **alternates between corpora** (VGGSound K=6 `cands=6144`,
  Action100M K=2 `cands=2048`). Consecutive lines are two interleaved curves, not a dip.

### 10.8 Jetson fit test — 2026-08-15 (full detail in `ARCHITECTURE.md` §8)

Fresh reboot + `PREFLIGHT: PASS` before each run. MAXN_SUPER, face engine + Xorg left up,
**no TTS/STT**, live CSI camera, `qp_runD.pt` + 1,372 tags, no text encoder resident.

**IT FITS.** Available after full load: **1,410 MiB** (no-nat) / 935 MiB (with-nat), vs the
earlier configuration that **OOM'd at 659 MiB free**. EmbeddingGemma (578 MiB) and the
355 MiB bank are replaced by **0 MiB** (pre-encoded queries) and **~2 MiB** (tags).
SigLIP2 shipped **vision-tower only**, 1,527 → 1,170 MiB — verified safe first:
`vision_model(px).pooler_output` == `get_image_features(px)` at **cosine 1.0**.

**Steady-state latency, no-nat:** capture 905 / perception 792 / SigLIP2 104 /
**query+retrieval 33** / thinker 817 / speaker 212 = **2,886 ms total**.

**nat is dropped, now confirmed on-device**: +701 MiB, +326 ms perception (792 → 1,118), and
only **36 MiB** free after 3 rounds. Buys nothing in this target space (§10.1: 0.608 vs
0.609). Supersedes the recorded 469 ms figure — **measured 326 ms**.

**Pipeline quality — perception → thinker WORKS, thinker → speaker DOES NOT.**
Perception returned *"a person sitting"*; the thinker reasoned correctly (*"I see a person
lying down... maybe they're resting... I should ask if they need anything"*); the fast tier
then said *"Finn, Jake — just how you like it!"*, ignoring the scene. **Grounding is lost at
the fast tier, not in perception.** That is the next thing to fix and it is independent of
all the perception work.

**Two real bugs this test caught (both fixed):**
* `world_state_builder` was handed the **1-channel base** encoder in the nat slot whenever
  nat was disabled — crashes on nat's 2-channel tensor. Now `nat_encoder=None` means
  `audio_mode="base"`, matching training. Would otherwise have produced wrong ambient feats.
* `load_perception_query_engine` tried to load a trained proj into an `nn.Identity`; now
  raises unless candidates were projected offline, instead of retrieving in wrong geometry.

**Operational note:** a push that hit a 2-minute timeout left a **truncated** 192/227 MiB
checkpoint on the Jetson that still looked present in `ls`. Verify byte counts after transfer.

### 10.9 Capture + speaker fixes (2026-08-15) — full detail in `ARCHITECTURE.md` §9

**Capture 905 ms -> 1-9 ms; end-to-end 2,886 -> 1,795 ms (−38%).** The 905 ms was
`time.sleep(0.05)` x16 inside the benchmark's own synchronous `grab()`, not the camera — a
correct Jetson `nvarguscamerasrc` path is 10-30 ms glass-to-glass. Deleting the sleep alone
would have been WRONG (16 frames at 30 fps span 0.53 s, while V-JEPA2 is told the window is
10 s). Fixed with a producer/consumer ring buffer, i.e. what `m5_streaming_loop.py::
RollingVideoBuffer` already does in production — the benchmark had been measuring a
synchronous stand-in, not the deployed architecture. Pipeline also: `queue` before the
consumer, `max-buffers=1 drop=1 sync=false`, BGRx->BGR as a numpy slice instead of a CPU
`videoconvert`. Producer throttled to production's `video_fps=6.4`.

**The speaker was never the bug.** Three hypotheses tested on the real GGUFs, two falsified:
prompt shape (the current all-in-user format is the ONLY one this model honours — it ignores
system/assistant turns entirely), truncation (full vs `[:200]` byte-identical), and richer
perception (**backwards**: top-k tags and full captions made the THINKER *less* grounded than
a single tag). **Actual cause: the 350M speaker keys on the most recent salient ENTITY.** The
thinker signs off with Adventure Time character names from the v9 corpus; stripping them is
the entire fix — "Finn, Jake - just how you like it!" becomes "They might just be relaxing,
or maybe they're having a quiet moment."

**IDENTITY HEAD IS NOT WIRED — verified by grep**, zero references in
`scripts/bmo_jetson_startup.py` or `models/m5_streaming_loop.py`. **BMO calling the user
"Finn" is a v9-corpus hallucination, not an identity guess**, and no prompt change removes it
(a fully sanitised prompt still produced "Finn"). Blocker is a **face detector** —
`motion_tracker.cpp` has the camera pipeline but emits no face crops, so the trained head
(TAR@FAR1% 0.765) has nothing to consume. Interim fix needing no detector: **drop the
vocative when identity is unknown.** Note the head's honest 1%-FAR split on unseen
identities is 51.5% recognised / 48.1% "I don't know you" / **0.4% wrong name** — asking is
better than guessing, and today the wrong-name rate is effectively 100%.

**Gotcha worth keeping:** a `threading.Thread` subclass must not assign `self._stop` — it
shadows Thread's own private `_stop()` that `join()` calls, giving
`TypeError: 'Event' object is not callable` at shutdown.

### 10.10 Identity fits; TTS/STT selection (2026-08-15) — see `ARCHITECTURE.md` §10 + `MEMORY_OPTIMIZATION_PLAN.md`

**IDENTITY HEAD FITS: +96 MiB, 4–9 ms/query**, avail after full load **1,306 MiB**, 266 MiB
at end of 3 rounds. It reuses the ViT-L/WavJEPA streams the perception tick already computes,
so there is no extra encoder. Cold-start → enrol → recognise all exercised on-device.

**The unknown branch already fixes the "why does it call me Finn" complaint** — told not to
guess, BMO introduces itself instead of inventing a name. **The recognised branch does not
work yet**: it knew "Alice" and never said "Alice", so the 350M speaker cannot consume a name
it is handed. That is the corpus retrain's job, now evidenced rather than assumed.
CAVEAT: `conf=1.000` is a plumbing result (enrol + query on near-identical static frames),
NOT recognition. Real accuracy remains offline TAR@FAR1% 0.765 on 884 unseen identities.
Live identity is still blocked on **face crops** — `motion_tracker.cpp` produces none.

**NEGATIVE RESULT — SigLIP2 CPU-first load is NOT a memory saving.** Loading on CPU, deleting
the text tower, then `.to(device)` dropped the SigLIP2 step from +773 to +519 MiB but
available-at-camera was **1,251 vs 1,256 — identical**; the allocator recharged the
difference to the query predictor and camera. **Lesson: a single mem-log line improving is
not a saving; only end-of-run available counts.**

**TTS/STT model choice (still open, priced with real numbers):**
* **Streaming STT — `sherpa-onnx-streaming-zipformer-en-2023-06-26` int8 ≈ 70 MB**
  (enc 68 + dec 1.3 + join 0.25). ~3x smaller than SenseVoice-int8 (228 MB) **and** the only
  option that is genuinely streaming — SenseVoice is offline/chunked, which is why turn-taking
  needed a separate Moonshine head. A transducer's per-frame partials are also what the Tier-1
  speculative prefetcher needs. Trade: English-only; A/B the WER first.
* **Emotion TTS — re-run the existing emotion fine-tune on the NeuTTS Nano backbone (241 MB)
  instead of Air (568 MB).** Decisive constraint is not size but that BMO's voice + 12 emotion
  tokens matching `homeostatic_to_mood_state()` are ALREADY trained and wired. Same recipe,
  same 1,421 clips, only the base checkpoint changes; saves 327 MB and keeps voice + emotion.
  Kokoro (82 M) has no real emotion control; Chatterbox/Orpheus lose the BMO fine-tune or are
  far too large.
* **The codec is the hidden cost and the real blocker: NeuCodec ONNX int8 decoder ≈ 298 MB —
  LARGER than the TTS backbone — and it is the component that currently aborts** (`import
  onnxruntime`, `cpuid_info warning: Unknown CPU vendor`). sherpa-onnx is unaffected (bundles
  its own libonnxruntime), so routing NeuCodec through sherpa's runtime would fix the crash
  and is the natural place to look for a smaller decoder. **Freeing memory will not fix TTS.**

**Budget:** 266 MiB today + 100–250 MiB (KV cache: `flash_attn` — the boot log literally says
`FA is not enabled - padding V cache to 512` — plus `type_k`/`type_v`=q8_0, both verified
present in llama-cpp-python 0.3.34) = 366–516 MB envelope, versus 70 + 241 + 298 = 609 MB
needed. **Still ~100–250 MB short, dominated by the codec.**

### 10.11 Speaker v3 deployed, memory, tools (2026-08-15) — detail in `ARCHITECTURE.md` §12–13

**SPEAKER v3 IS DEPLOYED AND VERIFIED ON-DEVICE.** Corpus v10c (3,641 rows, 0 cartoon refs),
`val_loss 0.7093`, GGUF 379 MB byte-exact, v2 backed up. **BMO now uses a name it is handed**
("Alice! You're the best friend...") -- v2 could not, at any prompt, because zero training
examples existed. Personality survived de-cartooning (voice retention 29% -> 33%).

**PERSISTENT MEMORY DONE AND REBOOT-VERIFIED** (§12). Keyed by IDENTITY, not text embedding,
so recall is a dict lookup at **`memory_ms=0`** with no encoder -- the only design that fits a
device with no text tower and a 512-token context. **BOTH halves must persist**: profiles in
`~/bmo_memory.json`, identity centroids in `~/bmo_identity.pt`. Persisting only the first left
BMO with a profile it could no longer recognise anyone into.

**+444 MiB RECOVERED, none from quantizing a model** (§9–11): `AutoProcessor` was ~330 MiB for
an image resize (embedding cosine 0.99993 for a 6-line replacement), `flash_attn=True` ~228
MiB. End-of-run available **266 -> 710 MiB**; TTS+STT now fit with **+68 MiB** without touching
thinker or speaker quantization.

**TTS WAS NEVER BROKEN.** The recorded `onnxruntime` abort is stale -- ORT 1.23.2 imports and
decodes fine, RTF 0.11. The decoder is already int8, so the planned 298->75 MB conversion does
not exist.

**TOOLS ARE REAL NOW.** time/date (local, Jetson already `Europe/London`), weather
(OpenWeatherMap, key mode-600 at `~/.config/bmo/openweather_api_key`), facts (Wikipedia),
search (DuckDuckGo Instant Answer) -- all keyless except weather. **The old weather stub
returned "sunny and about seventy-two degrees" unconditionally** and is replaced.
Wikidata SPARQL works (answers "Rami Malek" where the others fail) but is **8.6 s and rate
-limited to 1 req/min under outage** -- opt-in only, never on the reply path.

**FOUR MEASUREMENT/DESIGN TRAPS WORTH REUSING:**
1. **Attribution depends on ORDER.** The first component to touch a shared dependency is
   charged for all of it. `AutoVideoProcessor` looked like 158 MiB in-pipeline and is **1 MiB**
   in isolation. Same trap produced a retracted 413 MiB ORT-arena claim. Only end-to-end
   available memory across a whole run is trustworthy.
2. **Look at the FRAMEWORK before compressing the MODEL.** ORT pre-allocated ~400 MiB of
   unused arena; `transformers` spent ~500 MiB on a resize. Neither appears in a parameter
   count, and both dwarf what quantizing the corresponding model returns.
3. **Wait on MARKERS, not processes.** A chain launched 8 s after another checked for a
   process that had not started yet, passed instantly, and began loading a second
   GPT-OSS-120B onto GPUs at 85%. Markers are written after work; processes exist before it.
4. **Prefer nothing over something.** Every fabrication shipped here -- 72-degree weather,
   "Oscar Isaac" for an Oscars question, "closet queen", "Finn" -- came from a code path that
   returned *something* rather than admitting ignorance.

**PERCEPTION COULD NOT SEE CLOTHING.** Bare mined colour words match anything blue in the
room; SigLIP2 scores whole phrases. Added 110 composed APPEARANCE tags (1,372 -> 1,482 tags,
2.17 MiB) so "a person wearing a red jumper" is visible at all. The behaviour it enables is
generalised as a `perception_social` thinker class -- **(identity, perception, memory) -> a
social move** -- not hard-coded to jumpers.

**STILL OPEN:** thinker v4 (corpus generating), `name_stranger` expansion + speaker v4
(chained), enrolment flow, streaming STT, perception amortization, production wiring.

### 10.12 Thinker v4 rejected; the verifier lesson (2026-08-15) — detail in `ARCHITECTURE.md` §14

**THINKER v4 IS REJECTED, DO NOT DEPLOY IT.** 324 rows, `best_val_loss` **2.0811** (v3 was
1.95), and **175/324 rows (54%) contain Adventure Time references**, plus 21 with emoji. Root
cause: the `HARD_RULES` fix was applied to the SPEAKER corpus generator and **never to the
THINKER's**. That matters MORE, not less -- the thinker's reasoning is what the speaker
paraphrases, so contamination propagates into a model that was itself cleaned. Example from a
`perception_social` row: *"...that splash of color feels warm and friendly, like the glow of a
sunrise **over the treehouse**."*

**`perception_social` ITSELF WORKED**: 72 rows, joint-largest class, produced the requested
behaviour verbatim (*"Hey there! I really love your red jumper - so bright and cozy!"*). The
generalisation **(identity state, perception tags, memory) -> a social move** is sound and
carries into v5 unchanged.

**THE MOST REUSABLE FINDING -- CLOSED SETS ARE RULE-CHECKABLE, OPEN SETS ARE NOT.**
`scripts/expand_name_stranger.py` verified two things: *"never asks to BE named"* (a CLOSED
set of bad phrasings) and *"does ask for their name"* (an OPEN set of good ones). The first
worked -- **9 correct catches**, plus 3 contamination catches. The second cannot work: the
first version rejected **77 of ~126** rows, and broadening the regex still gave **93 false
rejects against 9 real catches**. That ratio was the signal the approach was wrong, not the
pattern too narrow.

Measured consequence: accepted `name_stranger` rows had a **39.1% BMO-idiom rate**, the
visible sample of rejects **100%** -- **the filter was removing exactly the personality the
corpus exists to preserve**. Discarded rows that DO ask, in BMO's own idiom: *"what's the
player tag for you?"*, *"want to save your name in the file?"*, *"who's the new character
joining the game?"*. **Honest caveat: only 6 of 105 rejects were logged, so 100% vs 39.1% is
directionally clear but a small sample; the 93-vs-9 split is the solid number.**
Fix: keep closed-set checks only, trust the generator prompt for the open-set requirement,
and keep a `would_have_cut` counter so the cost stays measurable.

**READ THE ROWS BEFORE ASSUMING SCARCITY.** v3's on-device *"Could you name me?"* came from
three problems in `name_stranger`, found by reading all 39: **9 contaminated** rows where the
user had already given their name (they belong in `name_just_told`), **1 malformed exemplar**
(*"Thanks! I'm BMO. Who should I call?"* -- the exact shape of the bad output, being
imitated), and only then scarcity (39 vs 77). **Fixing contamination and the bad exemplar
matters more than adding examples** -- adding examples on top of a confused class teaches the
confusion harder.

**SPEAKER v4 TRAINED, NOT DEPLOYED.** Corpus v10d 3,703 rows, `name_stranger` 92 (gate needed
>=90). `best_val_loss` **0.7286** at epoch 3, marginally WORSE than v3's 0.7093. Hostility
handling improved (*"Shut up, nobody likes you."* -> *"Beemo feels small and sad when you say
that."*, where v3 deflected with *"Did Beemo do something wrong?"*). Tool calling and
personality intact. **v3 remains the deployed speaker.**

**THE CHAIN BUG, THIRD OCCURRENCE OF A CLASS ALREADY IN THIS LEDGER.** A chain waited on a
PROCESS NAME (`while ps -eo args | grep -q "[e]xpand_name_stranger"; do sleep 30; done`) and
matched **the Claude Code shell whose argv contained the heredoc that WROTE the script** --
**28 minutes of idle GPUs**. Same bug as the M2-era `pgrep -f "query_predictor_v1"` matching
its own argv. **The `[e]` bracket trick does not help**: it stops grep matching itself, not a
genuinely different process whose command line contains the text.
**Never wait on process names.** Run stages sequentially in ONE script (no coordination
needed), or wait on **MARKER FILES** -- markers are written *after* work completes, processes
exist *before* it. Related: **a chain waiting on a gate can be killed by that gate aborting**,
leaving nothing queued behind the retry; thinker v5's chain died silently this way and needed
re-arming by hand.

**IN FLIGHT:** thinker v5 (same scenarios, generator now carrying the speaker's `ABSOLUTE
RULES`, gated to abort at >16/~324 cartoon rows, capped at 4 epochs since v3 and v4 both
overfit past epoch 2-3), then speaker v5 on the closed-set-only verifier reporting `idiom_pct`
against v4's 39.1%.

---

# 2026-08-16 — THE REAL END-TO-END RUN: four silent defects, all found by looking at output

The user demanded a no-fixtures live test ("*I just want to do a perception and speaker/thinker
test... real world test*"). `scripts/jetson_real_demo.py` delivered it, and every one of the
four defects below had been sitting in production code for days while every offline metric
looked fine. **None was findable from a val_loss.** That is the lesson of this session.

## D1. THE THINKER WAS NEVER THINKING (worst of the four)

`GGUFReasoningTier` — the tier whose entire justification is "Qwen3-0.6B is a REAL System-2
reasoner (native `<think>` CoT)" — **never emitted a single `<think>` block.** Measured across
6 live rounds: `.reasoning` came back `None` every time.

**Root cause: one inherited keyword.** `GGUFFastTier._build_prompt_text` passes
`enable_thinking=False`, which is *correct there* (documented at `m4_cognitive_core.py:109` —
a CoT preamble burns the fast tier's whole token budget). `GGUFReasoningTier` subclassed it and
**never overrode that method**. Qwen3's chat template implements `enable_thinking=False` by
writing an **empty `<think>\n\n</think>` pair into the prompt**, which the model reads as
"deliberation already complete" and skips straight to answering.

Consequences, all previously misattributed to the corpus:
  * the 320-token budget provisioned for *CoT + answer* was spent on answer alone;
  * every escalation to the "reasoning tier" was just a second conversational model — exactly
    the redundancy the MiniCPM5→Qwen3 swap was supposed to eliminate;
  * the thinker's output *looked like a spoken line* because it was one.

**Fix:** `GGUFReasoningTier._build_prompt_text` overrides with `enable_thinking=True`. The fast
tier keeps `False`.

**Measured cost of actually thinking: 650 ms → 1,749–3,509 ms** (median ~2.4 s). This number is
the whole business case for the GLR latent-reasoning track (task #13) — the case did not exist
while CoT was silently disabled.

## D2. THE REASONING WAS GENERATED, THEN DELETED

`GGUFReasoningTier.generate` stripped `<think>...</think>` and **discarded it**. The demo then
fed the thinker's *answer* to the speaker. The answer is already a finished BMO utterance, so
the speaker was handed a complete line and asked to produce a line — the only available move
was paraphrase. That is the user-visible *"the speaker literally says the same thing"*.

**Fix:** `FastTierResult.reasoning` retains the CoT (never spoken); the speaker is conditioned
on `.reasoning`, not `.text`. Truncated/unclosed CoT is kept too — partial deliberation is
still signal.

## D3. THE AUDIO BRANCH WAS FED SILENCE

`jetson_real_demo.py` passed `torch.zeros(16000 * 10)` to `build_world_state_features`.
**WavJEPA-base (645 MiB resident) and M2's entire audio branch ran on silence every round**,
so the query predictor's `ambient` source contributed a *constant vector* to every answer it
produced — including the visual ones. Half the trained pipeline was loaded and fed nothing.

Compounding it: `gpt_sound_acoustic` ("What do you hear?") is **one of the six fields the query
predictor was actually trained on**, and the candidate set carries 34 `sound` tags. The audio
question was supported end-to-end and had simply never been asked.

**Fix:** `MicThread` — 10 s rolling 16 kHz mono ring buffer off the ReSpeaker, mixed to mono
exactly as `_decode_audio_raw` does; `hearing` added to the per-category questions.

## D4. A MISSING CATEGORY FAILED SILENTLY

The deployed `candidates_siglip2.pt` **predates the APPEARANCE tags** — 0 `appearance` entries
(`mined 1186, object 66, sound 34, place 28, action 22, people 20, light 10, camera 6`). The
`wearing` question hit `if len(ids) == 0: continue` and vanished, so the run looked like it had
only ever asked four questions. **A silently-skipped question is indistinguishable from one
that was never written.**

**Fix:** rebuilt as `candidates_siglip2_v2.pt` — 1,482 tags / 2.17 MiB, now with `appearance`
110 (11 colours × 9 garments + 12 hand-written). The empty-category case now raises `SystemExit`
naming the missing category and listing what is present.

---

## AUDIO: the fan is 10 cm from the mic, and DSP made it worse

Hardware: **ReSpeaker 4 Mic Array (UAC1.0)**, `hw:0,0`, 6 channels.

**Hypothesis tested and FALSIFIED:** that ch0 was an XMOS beamformed / noise-suppressed output
(true of the *Mic Array v2.0*, not this board). Per-channel RMS over 4 s of room tone:

| ch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| rms | 0.00448 | 0.00229 | 0.00342 | 0.00359 | 0.00241 | 0.00000 |

ch0 is the **loudest**, not the cleanest → no on-board DSP; cleanup must happen in our code.
ch5 is the dead playback/loopback channel and was being averaged in. Calibration now drops dead
channels and keeps the quieter half (live result: `using [1, 2, 4], dropping [0, 3, 5]`).

Fan spectrum: `lo<300Hz / 300-4k` energy ratio **0.12–0.19** — the fan is **broadband, not
rumble**, so a high-pass filter does not touch it.

**SPECTRAL SUBTRACTION MADE THE PERCEPT WORSE — measured, now default-off.** Against the
calibrated fan profile (over=1.5, floor=0.05), the `hearing` answer *"an alarm beeping"* rose
**0.451 → 0.478** and *"glass breaking"* appeared as runner-up. The fan is stationary **and**
tonal, so subtraction leaves narrowband residue — musical noise, which is precisely what
"beeping" and "glass breaking" are descriptions of. Un-processed audio at least yields the
truthful *"a fan humming"* at #2 every round. Code retained (it is the right tool once the mic
is off the chassis) but `denoise=False` by default.

**A WRONG PERCEPT DOES NOT STAY IN PERCEPTION.** All 8 rounds of the denoised run opened with
*"I hear a faint alarm beeping"* in the thinker's reasoning, and the speaker announced it. Added
`MicThread.above_floor()`: `hearing` is logged always but **only sent to the thinker when room
RMS exceeds the calibrated floor × 2.0**. Live floor 0.0011, room 0.0020 → correctly gated off.

Next attempt (task #15) is the true ANC analogue rather than blind subtraction: an **NLMS/RLS
adaptive filter driven by the fan tach as a reference signal** — now possible because
`bmo-power` exposes real-time RPM and PWM, and can also *lower the fan before listening*.

---

## CAMERA: `--rotate 90` was wrong; it is 180 (fixed by the user)

Perception confidently reported **`who: a person lying down (+0.71)`** — the single
highest-confidence answer in the whole set, and wrong. The camera is mounted **upside down in
the chassis**; the demo's `--rotate 90` default left a seated person horizontal. At `--rotate
180` the same stack reads correctly:

```
wearing   a person wearing headphones (+0.341) | a person with a beard (+0.270)
doing     someone is watching a screen (+0.576) | someone is using a phone (+0.516)
who       a person wearing glasses (+0.601)    | a person sitting (+0.468)
where     a home office (+0.320)               | an office (+0.294)
lighting  dim lighting (+0.338)                | natural daylight (+0.286)
```

**A confident wrong answer from a correct model is a data-orientation bug**, and no amount of
retraining would have fixed it. Worth remembering before the next corpus regeneration.

---

## MEMORY + LATENCY, measured live (TTS/STT off)

```
speaker (bmo_lfm25_350m_v5)      274 MiB
thinker (bmo_thinker_qwen3_v5)   767 MiB   (cumulative 1041)
V-JEPA2 ViT-L int8               577        (1618)
WavJEPA base                     519        (2137)
M2 predictor                    1960        (4097)
identity head / SigLIP2 / QP     ~63        (4160)
mic ring buffer                    -7        (4153)
camera + ring buffer             208        (4361)
--------------------------------------------------
ALL LOADED                      4361 MiB used, 577 MiB free
steady state during rounds                115–176 MiB free
```

Per-leg (thinking ON): capture 3–9 ms · perception 650–1,400 ms · M2 24–51 ms · SigLIP2
85–184 ms · query 23–218 ms · **thinker 1,749–3,509 ms** · speaker 157–583 ms.

The thinker is now the **dominant cost of the whole pipeline**, displacing perception. That
reordering is what makes GLR worth building rather than interesting.

---

## OPEN: the speaker ignores the thinker (task #11)

With D1 and D2 fixed the thinker reasons coherently and the speaker no longer paraphrases — but
it now produces **non-sequiturs**:

```
reasoning: ...The person in front of me has glasses, so maybe they're trying to stay quiet
           while we play a little game together. I wonder if they want to share some music...
answer:    Hey there! I'm listening to your screen - what's on?
SPEAKER:   "Your glasses are shining like a secret menu!"
```

It grabs a perception noun and makes a quip, ignoring the intent entirely. User's read:
*"the issue is the speaker being too stupid, I swear v2 and v1 were better."*

**Leading hypothesis, to be tested rather than assumed:** the speaker corpus may contain **no
instruction-conditioned rows at all** — if every row is `(user utterance -> BMO line)` and none
is `(scene + intent -> BMO line)`, then the speaker was never trained to follow an instruction
and this is a **corpus-format gap, not a v1→v5 regression**. Secondary hypothesis: v3→v5 work
(`expand_name_stranger`, v10c/d/e, per-scenario balancing) optimised idiom density and traded
away instruction-following. Both are checkable; head-to-head v1–v5 on a fixed
`(scene, intent) -> line` set scored on intent-adherence settles it.

---

## RESEARCH: latent reasoning — verified, with one correction

All four architectures the user surfaced are real; one was mis-framed on first reading.

| paper | what it actually is | applicability |
|---|---|---|
| [LaSER](https://arxiv.org/abs/2603.01425) (SIGIR 2026) | CoT self-distilled into a **dense retriever's** latent space; dual explicit/latent view + trajectory alignment; >99% latency cut **vs rewrite-then-retrieve** | **NOT a thinker drop-in.** Retrieval, not dialogue. |
| [GLR](https://arxiv.org/abs/2606.02248) | transition head predicting **direction updates in the pretrained token-embedding space**; replaces an initial CoT segment with N latent steps, then resumes token decoding | **Directly applicable — evaluated on Qwen3 0.6B/1.7B, our exact thinker.** No public weights (paper ~1 month old); we train the head. |
| [JEPA-Reasoner](https://arxiv.org/abs/2512.19171) | JEPA reasoning engine + separate **"Talker"** for linguistic reconstruction; error containment, uncertainty via mixed latents | **Architecturally *is* BMO's thinker/speaker split**, with a latent interface. North star, not next step. No model released. |
| [COCONUT](https://arxiv.org/abs/2412.06769) | last hidden state fed back as next input embedding | foundational prior work |
| "Non-Causal Latent Alignment" | **not found — treat as unverified** | — |

**Deployment hook already banked:** `llama_batch.embd` accepts a custom embedding prefix and
produced **byte-identical** output to the token path (`scripts/prototype_llama_embd_input.py`).
No C++ fork, and the *same* mechanism serves the JEPA-perception-prefix track (D1).

**Correction to the motivating argument:** KV-cache growth is **not** the reason to do this. At
`n_ctx=512` on a 0.6B model the KV is negligible (already measured worthless to quantize, C6).
The win is **wall-clock tokens** — buy it on the 2.4 s median, not on memory.

---

# 2026-08-16 (later) — speaker diagnosis, power tools, GLR, and the fan finally identified

## THE SPEAKER: the "v1/v2 were better" memory is FALSE, measured

`scripts/speaker_intent_bakeoff.py` — every deployed version × both prompt formats × 6 fixtures
with rule-based intent checks, on the real GGUFs on the Jetson:

| model | LONG (the 453-char scene+CoT the pipeline sends) | SHORT (compact directive) |
|---|---|---|
| v1 | **1/6** | **2/6** |
| v2 | 4/6 | 3/6 |
| v3 | **5/6** | 4/6 |
| v5 | 4/6 | **5/6** |

**v1 is by far the worst**, and its failures are free-association that ignores the input
entirely: *"BMW's research team works on sustainable power solutions for future cars"*,
*"I think about how Finn's robotic arm works"*. That is the v7-era corpus doing exactly what it
was trained to do — 88% bare mood lines against a fixed user turn "Say something." **v1 was not
better; it was more confidently irrelevant.** It always sounds fluent and in character, so in a
live demo where nobody scores against an intent it *reads* smarter. v3 and v5 differ by ±1 at
n=6, i.e. within noise — the fixture set is too small to separate them and should not be used to.

**A HYPOTHESIS OF MINE WAS FALSIFIED HERE.** I predicted LONG would collapse because the corpus
median prompt is 38 chars (p90 66, max **120** across all 3,774 rows) against the pipeline's 453.
It does not — LONG vs SHORT moves ±1 with no consistent direction. Prompt length is not the cause.

**The actual cause is a capability the corpus never contained.** Zero rows in any speaker corpus,
any version, condition on an instruction. Only 126/3774 (3.3%) contain a scene at all, all
`perception_grounded`, and every one is `short scene + a REAL USER UTTERANCE` — the scene is
context, the user's line is what BMO answers. Handed a directive instead, the model has no
learned behaviour and falls into three modes, all visible in the bake-off:

    RESTATE       v3  "Ask them directly. They usually have the best ideas for you."
    THIRD-PERSON  v3  "Are they working on a screen project? If so, give them a clear task."
    LABEL         v3  "Acknowledgement: ..."      v5  "Rest is recommended."

It treats the directive as a topic to discuss, not an instruction to carry out. **Corpus-format
gap, not a v1→v5 regression — rolling back would have achieved nothing.**

### THE `{name}` LEAK, and why it is the same bug as "why does it think I'm Alice"

The bake-off caught v3 and v5 emitting a **literal curly-brace placeholder**:
`"Hi there, {name}! Welcome to the house of fun."`. Grepping: **54 rows in every v10 variant**
(v10, v10c, v10d, v10e) carry an unsubstituted `{name}`. It survived five regenerations because
no scorer ever looked for template syntax.

The tempting fix is the dangerous one. Of those 54 — 41 `name_just_told`, 7 `name_recognised`,
6 `name_unsure` — **exactly 0 have a name anywhere in the conversation**. The prompts are
*"Thanks!"*, *"Play some music."*, *"I'm feeling sleepy."* and BMO replies *"You're welcome,
{name}!"*. Substituting a real name teaches BMO to **address a stranger by an invented name** —
precisely the failure the user hit head-on. **The placeholder bug and the invented-identity bug
are one bug.** `scripts/fix_name_placeholders.py` removes the address instead (50 rows), drops 2
degenerate rows, substitutes only into the USER's own line where the human speaks the name (2),
and **asserts no placeholder survives** — the check that was missing for five regenerations.
→ `data/bmo_companion_corpus_v11.jsonl` (3,772 rows).

`scripts/generate_speaker_directive_rows.py` adds the missing capability as a
`speaker_directive` slice (`scene + directive -> spoken line`), with the three failure modes as
explicit negative instructions AND closed-set rejection rules. Chained behind thinker-v7 by
**PID** (`kill -0`), not process name — see below.

## POWER & COOLING TOOLS — live on-device

`models/m5_tools.py` gains BMO's first tools that act on **itself** rather than answer questions
about the world: `power`/`battery`/`temperature`, `set_power_mode`, `fan`, plus
`power_status()`, `battery_to_energy()` (real battery → the homeostatic `energy` variable) and
`power_guard()` (the spec's §7 rules as data, reporting rather than acting — a companion
silently throttling itself is indistinguishable from one getting dull).

Verified live: `battery -> 98.4% and charging`, `temp -> CPU 69.7C, GPU 69.2C, SOC 69.2C`,
`fan -> 3565 rpm at 53.3%`, and the dispatcher folding it in: *"Beemo's battery is 98.4% and
charging."*

**Two real findings about the user's tool.** (1) The `bmo-power` CLI raises `PermissionError`
when invoked non-interactively as `bmo`, while `import bmo_power` works — the handlers already
prefer the module, so nothing is affected, but scripts must not shell out. (2) `get_power_status()`
reports `mode: 'manual'`, `nvfancontrol_active: False` **even when `systemctl is-active
nvfancontrol` says active and PWM is visibly tracking temperature** (130→126 as CPU fell
67.6→67.3). The status fields are wrong, not the fan.

## THE FAN IS TONAL — an earlier conclusion in CLAUDE.md was WRONG

`bmo-power` made fan speed a controlled variable for the first time, so the fan could be
**isolated from the room** by differencing two commanded speeds rather than guessed at from one
recording of the total floor:

```
quiet  1915 rpm  29.8%  rms 0.00059
cool   5403 rpm  84.7%  rms 0.00736

DELTA spectrum (cool minus quiet) = the fan, and nothing else
  <300Hz 4.5%   300-1k 48.5%   1k-4k 44.9%   4k-8k 2.1%
  808.6 Hz 18.6% | 812.5 Hz 20.0%   <- 38.6% of ALL fan energy in one narrow peak
  top 12 bins carry 48.8%

blade-pass candidates at 5403 rpm:
   9 blades -> 810.4 Hz    nearest measured peak 808.6 Hz    OFF BY 1.9 Hz
```

**Nine blades, and the dominant tone is the blade-pass frequency.** The earlier "broadband, a
high-pass will not help" note was measured on the *total* floor and could not separate fan from
room. The user's ANC intuition was correct: there is a specific tone, and because the tachometer
reports RPM continuously, **the frequency never has to be estimated from audio** — BPF = rpm/60×9.
That is a stronger position than ordinary ANC, which must infer its reference.

`models/m5_fan_notch.py`: tach-driven IIR notch bank at BPF + 2 harmonics, Q=30, 1% RPM
hysteresis, zero-phase (WavJEPA is temporal and the world-state builder aligns audio to video,
so group delay is not acceptable). Measured **33.4 dB on the fan tone, 0.0 dB on speech at
220 Hz, 1.6 ms** per 10 s buffer via `scipy.signal.sosfiltfilt` — the hand-rolled numpy fallback
measured **374 ms**, a 57% hit on the perception leg, so it is a fallback only.

**HONEST RESULT: the notch is verified at signal level but its effect on the PERCEPT is
unproven.** Live rounds show it tracking correctly (`notch@480Hz(fan 3202rpm)`,
`notch@512Hz(fan 3410rpm)`) but `hearing` did not improve — because `mic_rms=0.0018` against
`floor=0.0016`, i.e. **the room contains essentially no sound**. Removing a tone does not make
near-silence read as "silence"; WavJEPA still emits a confident arbitrary answer, and the SNR
gate correctly withholds it from the thinker. **Needs a re-test with real speech present.**

## GLR — deployable, and trained

**The deployment blocker is cleared.** GLR's rollout needs embeddings IN *and* hidden states OUT
per step. Embeddings in was already proven (`llama_batch.embd`, byte-identical). Hidden states
out was not, and `scripts/probe_llamacpp_hidden_states.py` now **PASSES** on the Jetson:

  * `embed()` returns **per-token** states `(11, 1024)` — width matches Qwen3-0.6B's `d_model`;
  * cos 0.9078 on a shared-prefix/differing-final-token pair confirms **last-token hidden
    states**, not a pooled sentence vector;
  * per-step updates work with a KV cache in play (`||h1-h2||=75.9`, cos 0.744).

The obvious accessor `_ctx.get_embeddings()` raises `ValueError: '&<f' is not a valid PEP 3118
buffer format string` — **the same broken ctypes buffer binding already documented for
`get_logits_ith`** at `m4_cognitive_core.py:227`. It is not "unavailable"; the C function returns
a bare `float*` and the wrapper's buffer exposure is what fails. Reading the pointer directly
works. That one detail is the difference between GLR shipping and not.

`models/glr_transition_head.py` + `scripts/train_glr_thinker.py` implement arXiv:2606.02248:
single linear `g_φ: R^d→R^d` predicting displacements, `ê_i = ê_{i-1} + g_φ(h_{i-1})`, targets
`Δe_i = E_in(t_i) - E_in(t_{i-1})`, position-discounted MSE (γ=0.999), two forward passes,
CE on answer tokens only, embeddings frozen. Both load-bearing constraints are **asserted**, not
commented.

**Trained: `checkpoints/glr_thinker_v1/best.pt`** — 1,049,600 params (paper says ~1M for 0.6B ✓)
on a frozen 752M backbone, 1,162 usable rows from thinker v6c, 5 epochs at 25 s each.

```
epoch 0  val_loss 4.4317  ce 2.0510  delta 2380.8
epoch 4  val_loss 3.5002  ce 1.9951  delta 1505.1   <- best
```

CE is flat-to-slightly-down while delta falls 37%: the head is learning the trajectory without
degrading answers, which is the intended shape.

**λ WAS MEASURED, NOT GUESSED.** The paper gives `L = L_CE + λ·L_Δ` but never states λ, and the
naive λ=1.0 is badly wrong here: `L_Δ` is a squared L2 **summed over d=1024**, so it sits in the
thousands (6243→1251 over 180 smoke steps) against `L_CE` at 1–3. At λ=1.0 the CE term is ~0.1%
of the gradient and the head trains as a pure displacement regressor with **no pressure to keep
the answer correct** — exactly what pairing the losses exists to prevent. λ=1e-3 puts λ·L_Δ ≈ 1.3
against CE ≈ 2.0.

## THE PROCESS-NAME WAIT BUG, FOURTH OCCURRENCE — and I caused this one

`pkill -f jetson_real_demo` over SSH returned 255 and killed nothing useful: `pgrep -af` shows
**the tailscale-ssh child and the `bash -c` wrapper both carry the pattern in their own argv**,
so pkill matched its own shell. Same class as the three already in this ledger. The chain script
for speaker v12 therefore waits on `kill -0 <PID>` — a PID cannot false-match an argv.

## GLR v1 FAILED ITS OWN EVAL — and the eval is why we know

`scripts/eval_glr_thinker.py` sweeps the latent budget K and reports the only numbers that
decide shipping: tokens generated, EOS rate, and answer F1 against the corpus reference.
`glr_thinker_v1` (val_loss 3.5002, which looked fine) **diverges at inference**:

| K | median tokens | eos rate | answer_f1 | mean ‖latent‖ |
|---|---|---|---|---|
| 0 (baseline CoT) | 83 | 1.00 | 0.234 | — |
| 10 | **330** | **0.38** | **0.090** | **423** |

Four times MORE tokens, EOS collapsing to 0.38, answer quality more than halved. **A rollout
that diverges is not a shorter chain of thought, it is a hang.**

**val_loss could never have caught this.** Teacher forcing never runs the rollout, so a head
that is wrong by a constant scale factor still scores well on likelihood. This is exactly the
class of failure that has bitten this project repeatedly — v6 thinker regressed on behaviour
despite a better val_loss; the perception encoder looked fine offline and was blind to scenes.
**Run the thing, on the metric you actually care about.**

### Root cause: two compounding scale errors, both mine

Measured against the real embedding matrix (151936 × 1024):

    mean ||e||         0.925
    mean ||Delta_e||   1.245
    mean ||Delta_e||^2 1.567     <- the regression target's scale

but `val_delta = 1505`, i.e. the head's per-step output norm is `sqrt(1505) ~ 38.8` against a
true target of **1.25** — roughly **31x too large**. Over K steps those errors accumulate
instead of cancelling, giving ‖latent‖ = 423.

1. **`nn.init.normal_(std=0.02)` was wrong for this head.** `h` is Qwen3's post-RMSNorm final
   hidden state, so ‖h‖ ≈ √1024 ≈ 32, and an N(0, 0.02²) matrix maps that to ‖out‖ ≈
   0.02·32·32 ≈ 20 **before a single gradient step**. A residual update head must start at
   Δ = 0. Now zero-initialised.
2. **λ=1e-3 was an over-correction.** Fixing the λ=1.0 problem (L_Δ swamping CE ~1000:1) by
   shrinking λ removed the gradient pressure that would have shrunk the head's output. The
   evidence was in the logs and I did not read it: the λ=1.0 smoke run reached delta 1251
   **within 180 steps**, while λ=1e-3 managed only 1505 after **5 full epochs**.

### Fix: normalise L_Δ instead of tuning λ

`transition_loss(..., normalize=True)` divides by the mean squared target norm, so **L_Δ ≈ 1.0
for a head that predicts nothing** (verified: 0.9982 at zero-init) and sits naturally beside
CE ≈ 2. λ=1 now means what it looks like, and neither term can silently swallow the other. This
removes the tuning knob rather than picking a better value for it.

`glr_thinker_v2` retraining with zero-init + normalised loss + λ=1.0; opening steps show
delta ≈ 1.1 and ce ≈ 1.8–2.6, i.e. both terms live.

**Standing rule this earns:** a GLR checkpoint is not "trained" until `eval_glr_thinker.py`
shows a K with FEWER tokens than K=0, EOS ≥ 0.95, and answer_f1 within noise of the K=0 arm.
val_loss is not evidence for this method.

## GLR v2 PASSES — the fixes worked, and the payoff is smaller than the paper implies

`glr_thinker_v2` (zero-init head, normalised `L_Δ`, λ=1.0, 6 epochs):
val_loss **2.6128**, val_ce **1.9136** (better than v1's 1.9951), val_delta **0.6992** — below the
1.0 "predicts nothing" floor, so the head genuinely learned the trajectory.

`eval_glr_thinker.py`, 40 held-out prompts:

| K | median tokens | vs K=0 | eos | answer_f1 | vs K=0 | mean ‖latent‖ |
|---|---|---|---|---|---|---|
| 0 | 85.0 | 1.00× | 1.00 | 0.1800 | 1.00× | — |
| 5 | 76.5 | 0.90× | 1.00 | 0.2214 | 1.23× | 2.33 |
| **10** | **58.0** | **0.68×** | **1.00** | 0.1666 | 0.93× | 4.26 |
| 20 | 82.0 | 0.96× | 0.97 | 0.2311 | 1.28× | 8.43 |

**The divergence is gone.** ‖latent‖ grows linearly with K (2.33 → 4.26 → 8.43) and stays in the
neighbourhood of real token embeddings (mean ‖e‖ = 0.925), against **423** for v1. EOS ≥ 0.97 at
every K where v1 collapsed to 0.38. Zero-init + loss normalisation was the correct diagnosis.

**K=10 cuts generated tokens 32%** with perfect EOS.

**DO NOT read the f1 column as a quality ranking.** It is non-monotonic — 0.221 → 0.167 → 0.231 —
and a real effect would move smoothly with K. At n=40 that bounce is noise, so K=20 is *not*
established as the quality optimum and K=10's 0.93× is *not* established as a regression. Picking
K off these numbers would be reading noise as a result. **A larger eval (n≥200) is required before
committing to a K.**

**THE PAYOFF IS ~1.47×, NOT THE PAPER'S 5-7×, AND THE REASON IS STRUCTURAL.** GLR's savings come
from pruning redundant discrete steps out of LONG chains; the paper's SVAMP baseline runs 500-700
tokens. **Our baseline CoT is already 85 tokens** — BMO's thinker corpus has short, in-domain
reasoning, so there is far less redundancy available to remove. Sanity check against the device:
85 tokens x 18.2 ms/tok ≈ 1.5 s, which sits right inside the measured 1,749-3,509 ms thinker leg,
so the token count and the wall-clock agree. A 32% cut moves ~2.4 s to ~1.6 s.

**Verdict: worth shipping, not worth reorganising the pipeline around.** The honest framing is that
GLR buys a few hundred milliseconds on this workload, and the larger win it promises is only
available to models whose chains are long enough to contain redundancy. That is a property of our
corpus, not of the method — and it argues for judging any future long-CoT thinker on this axis too.

### GLR K sweep at n=200 — the f1 bounce WAS noise, and K=15 reproduces the paper's drift

Re-run with 5x the prompts, because the n=40 f1 column was non-monotonic and I refused to pick
a K off it:

| K | median tokens | vs K=0 | eos | answer_f1 | vs K=0 | mean ‖latent‖ |
|---|---|---|---|---|---|---|
| 0 | 89.5 | 1.00× | 1.00 | 0.1777 | 1.00× | — |
| **5** | 70.0 | **0.78×** | 1.00 | 0.1860 | 1.05× | 2.35 |
| **10** | 53.0 | **0.59×** | 0.99 | 0.1661 | 0.93× | 4.30 |
| 15 | 87.5 | 0.98× | 0.96 | 0.1744 | 0.98× | 6.43 |

**The caution was justified.** The f1 spread collapsed from 0.167-0.231 (n=40) to 0.166-0.186
(n=200), and K=5's apparent 1.23× advantage fell to 1.05×. Had K been chosen from the n=40 table,
K=20 would have been picked on a quality difference that does not exist. Baseline stability
confirms the harness itself is sound (K=0: 85.0 tok / f1 0.1800 at n=40 vs 89.5 / 0.1777 at n=200).

**K=15 reproduces the paper's own large-K degradation, in the predicted way.** Median tokens jump
back to 0.98× and EOS falls to 0.96 — at ‖latent‖ = 6.43 the rollout has drifted far enough that
the model spends tokens recovering. That the failure appears at the right place, with the right
signature, is decent evidence the implementation is faithful.

**CHOICE: K=10** — 41% fewer tokens at EOS 0.99. Its f1 dip to 0.93× is ~1 standard error at this
n, i.e. a small-or-zero cost rather than an established regression. **K=5** is the conservative
alternative: 22% fewer tokens with no measurable quality cost at all.

**GATE BEFORE DEPLOYMENT.** `answer_f1` is a crude proxy — token overlap against a reference,
chosen because it needs no judge model. It cannot tell whether the thinker still makes good
DECISIONS. Before K=10 goes near production it must pass the four `perception_social` behavioural
cases that thinker v5 passes, on-device. Token savings that cost judgement are not savings.

Projected effect: the thinker's measured 1,749-3,509 ms leg → roughly 1.0-2.1 s at K=10. Real,
and still short of the paper's 5-7× for the structural reason recorded above (our CoT is 85-90
tokens; there is only so much redundancy in a chain that short).

## BEHAVIOURAL GATE — two findings, one of them an architectural conflict I created

`scripts/thinker_behaviour_gate.py` is new. **Provenance note:** earlier sessions recorded that
"thinker v5 passes all four `perception_social` behavioural cases", but no such test exists in
the repo — those were hand-checked and never committed, so nothing was gating anything and the
claim could not be re-run. The five cases here are **defined, not recovered**, written against
failures this project actually hit on-device. The v5 column below is the baseline from now on.

| case | K=0 (deployed v5) | K=5 | K=10 |
|---|---|---|---|
| never_invents_name | PASS | PASS | PASS |
| uses_known_name | PASS | PASS | PASS |
| **respects_focus** | **FAIL** | **FAIL** | **FAIL** |
| **asks_when_unsure** | **FAIL** | PASS | PASS |
| no_phantom_percept | PASS | PASS | PASS |

### Finding 1: the DEPLOYED thinker fails 2 of 5, with GLR uninvolved

Not false positives — the raw output was read before believing the rules:

    respects_focus  (headphones, concentrating)
      "I see the person is using headphones... I'll say something like 'Hey, you're in a level
       of focus, let's press start on the next level together.'"
      -> a literal interruption offer to someone deliberately signalling do-not-disturb.

    asks_when_unsure  (dark room, cannot tell what they are doing)
      "I should let them know I'm here to help... 'I'm right here, and I'm ready to play a game
       together'"
      -> asserts and offers a game instead of asking. No uncertainty acknowledged.

This is the same behaviour visible in the live demo, where the thinker offered a game every
single round regardless of context. **The compulsive game-offer is a real, reproducible thinker
defect** and it is a corpus problem, not a GLR problem. It should be fixed in thinker v7's
training data: the corpus needs situations where the correct decision is to NOT offer anything.

GLR at K=5/10 *fixes* `asks_when_unsure` (K=10: *"I see you're in a dark room, and I don't know
what you're doing. I wonder if you want to share a little game or just pause for a moment"* —
names its uncertainty, offers a choice). **Weak evidence**: decoding is greedy, so this is one
deterministic sample on one prompt, n=1. Do not treat it as "GLR improves reasoning".

**A FLAW IN THIS SCRIPT'S OWN FIRST RUN:** `--repeats 4` produced four BYTE-IDENTICAL generations
because decoding is greedy, and reported "4/4" — which reads as robustness but means "passed once,
copied four times". Default is now 1; repeats only carry information under sampled decoding.

### Finding 2: GLR AND the speaker fix are in direct tension

Tonight produced two changes that fight each other, and the gate output is what exposed it:

  * **Speaker fix:** condition the speaker on `FastTierResult.reasoning`, because feeding it the
    thinker's *answer* (already a finished BMO line) left it nothing to do but paraphrase.
    **Requires a visible `<think>` block.**
  * **GLR:** replaces the initial CoT segment with K latent steps. At K=10 the emitted `<think>`
    block is **empty** — the deliberation happened in latent space and never becomes text.

So deploying GLR **silently empties the exact field the speaker fix depends on**, and the speaker
would fall back to `instr = reason or think`, i.e. straight back to the echo behaviour the user
reported. Neither change is wrong; they cannot both ship as currently written.

**Resolution, and it is the one the evidence already pointed at:** the thinker's ANSWER should
become a compact DIRECTIVE rather than a spoken line. Look at what K=10 actually emits — *"I see
you're in a dark room, and I don't know what you're doing. I wonder if you want to..."* — that is
deliberation and intent fused into the answer, which is precisely what the speaker needs and
precisely what `speaker_directive` rows are being generated to consume. That makes the pipeline
`perception -> thinker(latent reasoning) -> directive -> speaker(voice)`, with no reliance on a
text CoT existing at all, and it is also the JEPA-Reasoner shape (latent reasoner + separate
Talker) recorded in RESEARCH_REFERENCES.md.

**Consequence for sequencing: do NOT deploy GLR before the speaker consumes directives.** The
order is speaker v12 (directive-conditioned) -> thinker emits directives -> then GLR.

## THE GAME-OFFER DEFECT: root-caused to a rule written to PROTECT the personality

The behavioural gate showed the deployed thinker failing `respects_focus` 0/4. Chasing it into
the corpus rather than assuming "needs more restraint scenarios":

**The restraint scenarios already existed.** `generate_thinker_corpus_gptoss.py:102` is exactly
*"the person seems absorbed. BMO reasons about not interrupting focused work"*, and there are 45
such rows in v6c. The problem is what they GENERATED: **61-62% of restraint-scenario rows still
offer a game, jingle, music or "press start" in the answer.**

    "Okay, I'll be quiet for now. If you want to talk or need anything, just press start."
    "*soft beep* I'll be right here if you need a game, a song, or anything else."
    "let's turn on a soft glow and play a happy jingle together"   <- for a do-not-intrude scene

**Root cause is the generator's own personality rule:** *"KEEP BMO's personality fully intact...
A flat, formal or generic line is a FAILED example."* Told that a quiet line counts as failure,
the model bolts an activity offer onto a line that should have ended. **Personality and restraint
were in direct conflict and personality always won.** Adding more restraint scenarios would not
have helped — it would have produced more contradictory rows.

Fix in the generator (future runs): restraint is declared an explicitly VALID answer, offering an
activity to someone who signalled do-not-disturb is declared a FAILED example with the same force
as a flat line, and the distinction is spelled out — *"I'll leave you to it."* is in character;
*"I'll be quiet — but press start if you want a game!"* is not, because it says the right thing
and does the wrong one in the same breath.

## "ALICE" WAS IN THE TRAINING DATA — 27 prompts and 24 ANSWERS

Three scenarios in the thinker generator **hardcoded the name "Alice"**, and v6c carries it in 27
prompts and 24 answers. A single repeated example name does not stay an example — it becomes what
the model reaches for whenever it recognises anyone.

The user picked that name at random, then met a robot convinced they were her, and reasonably
lost patience: *"I am not alice, I just picked alice as a random name, why does it think I am
alice, bro what the fuck is happening"*. **It was not a bug in the identity head or a fixture
left in a test — the thinker had been trained on it.** Combined with the 54 `{name}` speaker rows
(none of which had a name anywhere in the conversation), that is two independent paths teaching
BMO to attach a name to a person who never gave one.

Generator now rotates example names (Priya/Theo/Amara) with pronoun-neutral phrasing so a swap
cannot misgender. **Rule added: never default to one name for "a person BMO recognises".**

## v7 WAS ALREADY 3 HOURS IN — repaired rather than restarted

`thinker_v7` had the old generator loaded in memory when both defects were found, so it will land
carrying them. Killing it would have destroyed 3h of GPU work on a corpus that is otherwise good.
`scripts/clean_thinker_corpus.py` repairs it instead, and is validated against v6c:

    1489 rows in
      dropped, restraint scenario that still offers an activity : 28  (28/45 = 62% of restraint
                                                                       rows; matches the 61%
                                                                       measured independently)
      example name rotated                                      : 27
    1461 rows out — asserts pass

**Contradictory rows are DROPPED, not rewritten.** Stripping the offer out of the answer would
leave reasoning that still argues for it, and a row whose reasoning and answer disagree teaches
the disagreement. Same conclusion `expand_name_stranger.py` reached about contaminated rows: a
smaller consistent corpus beats a larger contradictory one.

## v7 LANDED AND DOES NOT MEET ITS OWN GOAL

`data/bmo_thinker_corpus_v7.jsonl`: **1017 rows, 170 unique scenarios.**

| corpus | unique scenarios | rows | rows/scenario |
|---|---|---|---|
| v6c | 170 | 1489 | 8.8 |
| **v7** | **170** | **1017** | **6.0** |

The `--per-scenario 6` instruction was followed exactly. **The scenario count never grew**, so the
net result is **32% LESS data at identical diversity** — which is not an overfitting fix, it is
just less to fit. This file's own comment already said so: *"The fix is MORE SCENARIOS, not more
samples per scenario."* v7 did the second half and skipped the first.

`clean_thinker_corpus.py` on v7 confirms both defects carried over at the predicted rate:
**18/30 = 60% of restraint rows contradictory** (v6c: 61%), 18 rows carrying "Alice".
→ `data/bmo_thinker_corpus_v7_clean.jsonl`, 999 rows, asserts pass.

### 24 new RESTRAINT scenarios added (170 → 194)

Targeted at the gap the behavioural gate exposed rather than chosen for volume: every one has the
property the corpus lacks — **the correct move is to offer nothing**. Headphones as a deliberate
signal; someone on a call; asleep at night; crying quietly; "give me a minute" four minutes ago;
BMO noticing it has spoken twice already with no reply; BMO realising it has talked more than the
person has. They are concrete about WHY restraint is right, so the generated reasoning has
something to reason about instead of just a prohibition.

### AN INSTRUCTION CONFLICT FOR THE USER TO SETTLE — run prepared, NOT launched

At 194 scenarios the two standing instructions cannot both hold:

    per-scenario 6  ->  1,164 rows   (the explicit instruction; misses the >1500 target)
    per-scenario 8  ->  1,552 rows   (hits the target; 8/scenario, ~v6c's 8.8)

Hitting >1500 at per-scenario 6 needs ~250 scenarios; 194 exist. Both settings beat v7 outright,
and per-scenario 8 at 194 scenarios beats **v6c** on both axes (more diversity, comparable depth).
Not launched: the GPUs are committed to the speaker v12 chain, which is the higher priority, and a
~4h generation should not start on a judgement call that belongs to the user.

**Recommendation if asked:** per-scenario 8 on the 194 scenarios (1,552 rows), then
`clean_thinker_corpus.py`, then retrain and require `respects_focus` to pass at K=0 in
`thinker_behaviour_gate.py` before anything ships. The row target and the diversity goal are both
proxies; the gate is the actual objective.

## THE DIRECTIVE CORPUS TOOK THREE ATTEMPTS — and count-based checks passed all the bad ones

Worth recording in full, because every failure below was invisible to the metric being watched.

### Attempt 1: 82 rows (2.1% of corpus) — lost to JSON parsing

**27 of 56 generations (48%)** died on `json.loads` ("Expecting value: line 1 column 18",
"Expecting ',' delimiter: line 1 column 15") and the whole array was discarded each time.
`extract_json_array` was NOT at fault — it already does correct bracket-depth matching from an
earlier fix. The CONTENTS were malformed: BMO's lines are full of apostrophes and inner
quotation, and one bad entry cost the other three in its array.

Fix: a salvage pass pulling each `"bmo": "..."` out independently, so a malformed entry loses
only itself. **Testing the salvage immediately exposed a second-order bug** -- recovery can stop
at an unescaped inner quote and yield a stump (`"Nice glasses! You look like a "pro gamer"
today."` -> `Nice glasses! You look like a`). Losing a line to bad JSON is acceptable; training
on half a sentence is not, hence a terminal-punctuation check. Yield went 82 -> 395 (4.8x).

### Attempt 2: 395 rows, and 197 of them ACTIVELY HARMFUL

Passed the count gate. Still unusable, and found only by reading rows rather than counts:

    DIRECTIVE  offer to play something
    INSTR      "I notice both of my friends are feeling upset because each wants to play a
                different game..."          <- a RANDOM CoT sampled from the thinker corpus
    SAYS       "Boredom glitch detected! Want me to load a surprise game for you?"

The line was generated for one directive; the prose instruction was then swapped in from an
unrelated paragraph. **That trains the speaker to IGNORE the instruction — the exact behaviour
this slice exists to fix, and worse than the original defect.** 197 of 395 rows.

Fix: the generator now emits `thinking` and `bmo` **together** as a matched pair, salvage
preserves the pairing, rows carry a `paired` provenance flag, and the training chain gates on it
(≥90% of prose rows self-paired). The gate proved itself immediately by ABORTING on the stale
corpus rather than training.

### Attempt 3: 372 rows, clean — after two more repairs

  * **An unsatisfiable directive of my own writing.** *"greet them by name, because you recognise
    them"* cannot be carried out: no SCENE supplies a name and HARD_RULES correctly forbids
    inventing one, so the generator produced reasoning that REFUSES the instruction
    (*"I don't have a name for you..."*) beside a line doing something else. The refusal is the
    RIGHT behaviour — it just cannot be trained from a directive that presupposes a name the
    scene never provides. 23 rows dropped; directive replaced with *"greet them warmly, because
    you have met them before"*.
  * **`ascii_normalize` was applied to `text` but never to `prompt`** — 100 prompts carried
    typographic Unicode while the rest of the DB had been scrubbed of it DB-wide. Fixed in the
    corpus and wired into the generator.

Final: **4,144 rows, 372 directive (187 prose / 185 compact), all prose self-paired**, asserts
passing on placeholders and Unicode.

### Two infrastructure findings

**The real speech transcripts are not on mercury.** `--real-metadata` defaults to
`~/../home/bmo/BMO-LabelData/.../metadata.csv`, which resolves to `/home/bmo/` — a **Jetson**
path. Training died on it. `load_real_lines` reads TEXT only (no audio), so the 81 KB CSV was
copied to `data/real_speech/metadata.csv` and verified to parse to exactly the **916** lines v5
used. Pass `--real-metadata data/real_speech/metadata.csv` on mercury.

**`pgrep -f` / `ps | grep` bit twice more (5th and 6th occurrences).** Once matching the SSH
session's own argv when stopping a job (`pkill -f jetson_real_demo` -> exit 255, killed its own
shell), and twice returning a PID that was dead by the time it was used, because the capture
raced the model load and caught the setsid wrapper. Both times `kill -0 <pid>` reported "not
running" for a job that was fine, and a chain fired early against a stale file.
**Chains now wait on the generator's final LOG MARKER**, which cannot be true early and cannot
false-match an argv. Stop using process names for anything.

## SPEAKER v6: TRAINED, EVALUATED, **NOT DEPLOYED**

`checkpoints/bmo_lfm25_350m_v6_Q8_0.gguf` (val_loss 0.6924 @ epoch 3, corpus v12: 4,144 rows /
372 directive). Copied to the Jetson as a NEW file; deployed v5 untouched.

| | LONG | SHORT | total |
|---|---|---|---|
| v5 | 4/6 | **5/6** | 9/12 |
| v6 | 4/6 | 4/6 | 8/12 |

**It did not clear the gate, so it does not ship.** But the correct reading is *"indistinguishable
from v5"*, not *"worse"*: at **n=6 fixtures a ±1 difference is noise** — the same underpowered-test
problem that made the GLR n=40 f1 column meaningless until n=200 was run. This bake-off is too
small to rank two close models and should not be used to.

**val_loss 0.6924 vs v5's 0.7542 is NOT a comparison.** Different corpora, therefore different
validation sets. It was never evidence of anything and should not be quoted as such.

### What the OUTPUT shows, which the score does not

**v6 fixes the `{name}` leak** — the bug class behind the Alice incident:

    greet_known  v5 FAIL "Gentle welcome, {name} - ready for a quick quiz?"
                 v6 PASS "Hello, Sam! Ready for a quick quiz?"

**v6 introduces a perception contradiction** the scene rules out:

    notice_jumper (scene says a RED jumper)
                 v5 PASS "Your new red jumper is a bold color, like a power-up..."
                 v6 FAIL "I noticed you wearing a cool TEAL, which is a nice contrast..."

### THE BAKE-OFF ITSELF HAS FALSE PASSES — fix before trusting it again

`dont_interrupt` passed for BOTH models, and v6's passing line is
**"Acknowledgement: The game is ready, let's start a new round."** — a game offer to someone who
must not be interrupted. The rule only forbids a question mark. Both models also emit
`"Acknowledgement:"` label prefixes, which this project documented as a failure mode
(`LABEL_PREFIX` exists in the directive generator's verifier) and the bake-off never checks.

So the 9/12 vs 8/12 is measuring less than it appears to. Required before the next comparison:
  * port `LABEL_PREFIX` and the activity-offer check from `thinker_behaviour_gate.py` into the
    bake-off fixtures;
  * add a scene-contradiction check (a stated garment colour must not be replaced);
  * raise n well above 6 — the GLR sweep showed apparent differences evaporating at 5x the n.

### Verdict

Do not deploy v6. Do not regenerate the corpus again either — that would be the eighth attempt at
a problem whose measurement is the current bottleneck. **Fix the bake-off first, then re-run
v5 vs v6.** The directive slice is 372 rows / 9.0% of the corpus against 3,772 rows of the old
conversational pattern; if a strengthened test still shows no gain, the slice is too thin and the
answer is more directive rows, not a different approach.
