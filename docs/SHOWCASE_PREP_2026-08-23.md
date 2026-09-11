# Showcase prep — 2026-08-23

Everything below was **measured on the Jetson** (`bmo@bmo-desktop`, MAXN_SUPER) unless marked
otherwise. Training ran on mercury. Raw logs are listed at the bottom.

**Headline:** the showcase stack — speaker + Nano voice + full perception + SenseVoice + VAP —
**fits with the camera open and 681 MiB to spare**. The thinker does not fit alongside it.
The deployed Air voice was producing 40-second lockups on 17% of utterances; that is fixed.

---

## 1. TTS: Air → Nano. Measured, not assumed.

`bmo_neutts_nano_v1/v2_Q8_0.gguf` were already on the device, never benchmarked. Nano is a
LlamaForCausalLM (hidden 576, 24 layers, GQA 3 KV heads) vs Air's Qwen2-0.5B.

n=18 utterances per model (3 lines × 6 stochastic samples, temp 0.7):

| | size | TTFA | RTF | over-runs | worst case |
|---|---:|---:|---:|---:|---:|
| **Air v5** (was deployed) | 803 MB | 421–432 ms | 0.82–0.90 | **3/18** | **40.0 s audio, 35.9 s to synthesize** |
| **Nano v1** | 253 MB | **283–294 ms** | **0.59–0.64** | 1/18 | 4.3 s |
| Nano v2 | 253 MB | 290–314 ms | 0.61–0.66 | — | — |

Air hit the 2000-token cap **twice on a 23-character line**. A ~17% chance per utterance of a
half-minute lockup is disqualifying for conversation on its own, before considering speed.

RTF 0.62 < 1.0 is the number that matters: synthesis stays ahead of playback, so
sentence-level pipelining works. Nano is 1.4× faster than Air and 3.2× smaller.

## 2. The runaway is now bounded — and the first fix was wrong

`m5_streaming_voice.py` on the Jetson was **ahead** of mercury, not stale. It carries three
fixes mercury lacked, all kept:
- a `Llama.__del__` teardown guard (llama_cpp shutdown race)
- `n_ctx` 1024 → 2560 (MAX_TOK=2000 left only 48 tokens for prompts that measure 81–108)
- `kv_cache_clear()` — fixes a real `failed to find a memory slot for batch` crash

The length-aware cap had been **deliberately removed** (MAX_TOK 350 → 2000) to diagnose the EOS
problem. Reinstating it on top of those three fixes:

**First attempt was miscalibrated and clipped real speech.** A linear 7-tokens/char cap fired
7/36 times, pinning 3/6 of Nano's short line at exactly the cap. Cause: tokens/char is not
constant — this is a slow character voice with fixed per-utterance overhead, so it runs ~9
tok/char at 23 chars but ~4 at 73 chars. Corrected to intercept + slope:

```python
CAP_BASE, CAP_PER_CHAR = 200, 5
tok_cap = min(MAX_TOK, CAP_BASE + CAP_PER_CHAR * len(text.strip()))
```

| | cap fires | worst case |
|---|---:|---:|
| no cap (as deployed) | — | **40.0 s** |
| linear 7/char | 7/36 (clipping real speech) | 9.8 s |
| **intercept+slope (deployed)** | **1/36** | **9.8 s** |

`last_exit_reason` now distinguishes `EOS_TOKEN` / `LENGTH_CAP_FIRED` / `SAFETY_CAP_FIRED` so a
legitimate clip is visible rather than silent. **Re-verify `CAP_BASE`/`CAP_PER_CHAR` if the TTS
backbone is ever retrained.**

Deployed to both trees (Jetson backup: `m5_streaming_voice.py.bak_20260823`).

## 3. `causal_duplex_pipeline.py` could not run at all

`NeuTTSEngine.synthesize()` referenced `valid_chunks`, which is **never assigned anywhere in the
file** — the streaming loop that produced it had been dropped in an edit. Every turn with
non-empty text raised `NameError`. Restored, plus:
- voice preference order now Nano-emotion → Nano → Air-emotion → Air v5 (Air last, as fallback)
- ASR default Citrinet → **SenseVoice** (recognition quality outranks the 50 ms vs ~200–560 ms
  latency; the gap is maskable by firing the backchannel at the VAD endpoint before STT returns)

Verified end-to-end on-device, headless:

| turn | speaker | TTS (full) | visemes | audio | RTF |
|---|---:|---:|---:|---:|---:|
| 0 | 255 ms | 2929 ms | 29 ms | 4.46 s | 0.66 |
| 1 | 198 ms | 2801 ms | 27 ms | 4.44 s | 0.63 |
| 2 | 160 ms | 2156 ms | 20 ms | 3.52 s | 0.61 |

**Visemes cost 20–29 ms for an entire utterance.** The streaming/lip-sync conflict was never a
cost problem — `AudioFormantExtractor` just needs to run per NeuCodec chunk instead of over the
finished waveform. It is a 30 ms-frame/15 ms-hop FFT and already causal; one segment (~60 ms) of
lookahead covers the `MIN_DWELL` merge. This is cheap and unblocks streaming + lip-sync together.

## 4. Emotion voice retrained on the Nano backbone

`finetune_bmo_emotion_neutts.py --restore-from checkpoints/bmo_neutts_nano_finetune_v2/best`
— one flag, same 1,421 Fish clips, same recipe. 2,400 steps, ~21 min on one Blackwell.

- best `eval_loss` **0.5404** (Air emotion plateaued at 0.516 — Nano marginally behind, expected)
- overfits past epoch 1 (0.6379 by the end); `load_best_model_at_end=True` handles it
- → `checkpoints/bmo_neutts_emotion_nano_Q8_0.gguf`, **252 MB** vs Air-emotion's 595 MB
- deployed to `~/bmo_production/models_gguf/`

On-device verification — `has_emotion=True`, all 12 moods synthesize, TTFA 287–322 ms, RTF 0.59–0.67:

**8/12 moods are clean.** `neutral, excited, happy, content, stressed, concerned, tired, curious`
all exit on `EOS_TOKEN`.

**4/12 hit the length cap** — `surprised, anxious, lonely, bored` all ran to 7.50 s on a 35-char
line. This is the known thin-mood problem (`lonely` had 19 clips, `bored` similar) inherited from
the Air recipe, not something the backbone swap caused. The cap bounds them at 7.5 s instead of
40 s, but **these four are not demo-safe** — gate the mood set to the clean 8 for the showcase,
or do targeted augmentation for those moods.

WAVs for a listen test: `bmo@bmo-desktop:~/emotion_nano_wavs/*.wav` (12 files).

*(f0 was measured by crude autocorrelation and shows some octave errors — `content` 500 Hz and
`tired` 490 Hz are almost certainly doubled. Treat f0 as indicative of separation only.)*

---

## 5. THE MEMORY RESULT

### CORRECTION — the first per-component numbers in this file were wrong

An earlier revision reported **V-JEPA2 ViT-L int8 at 14 MiB**. That is wrong and the method was
the reason: `q_int8_cpu_then_move` loads the model on CPU (+960 MiB RSS), then frees the CPU copy
as it creates the CUDA copy (−299 RSS, +314 CUDA). Measuring only `MemAvailable` **after both
steps** made those cancel to noise. `MemAvailable` also nets against reclaimable page cache.

Re-measured with parameter counts, `VmRSS`, and `torch.cuda.memory_allocated`:

| component | params | CUDA resident | bytes/param | CPU transient at load |
|---|---:|---:|---:|---:|
| **V-JEPA2 ViT-L** | 326.0M | **314 MiB** | 0.96 (real int8) | +960 MiB |
| **WavJEPA-base** | 196.3M | **384 MiB** | **2.05** | +830 MiB |
| **M2 predictor** | 105.0M | **199 MiB** | **1.99** | +995 MiB |
| **SigLIP2 vision** | 92.9M | 192 MiB | 2.0 (bf16) | — |
| perception total | 720M | **~1090 MiB** | | **~1825 MiB transient** |

Two things this exposes that the wrong numbers hid:

1. **WavJEPA and M2 are not actually being int8-quantized** (~2.0 bytes/param, not ~1). V-JEPA2
   loads as **bfloat16** and reaches 0.96 B/param; the other two load as **float32**, so whatever
   `torchao`'s `Int8WeightOnlyConfig` leaves unquantized stays 4 bytes.
2. **The CPU transient peak at load (~1825 MiB) is far larger than the resident cost (~583 MiB
   for WavJEPA+M2).** That transient — not the steady state — is what OOMs this box at boot, and
   is why `malloc_trim()` in the quantization helper is load-bearing.

### O1 — bf16 before int8. Applied to M2, refused for WavJEPA.

| | baseline | bf16-first | saving |
|---|---:|---:|---:|
| M2 CUDA resident | 199 MiB | **134 MiB** | −64 |
| M2 CPU transient | 685 MiB | **454 MiB** | −231 |

Numerically safe: `encode_pre_pool_tokens` cosine vs the float32 path is **0.999969**.
**Applied** in `bmo_jetson_startup.py` (backup `.bak_20260823`), synced to mercury.

**WavJEPA must NOT get the same treatment** — tested, it raises
`TypeError: torch.bfloat16 is not supported in MaskedTensor`, exactly matching the warning in
`audio_encoder.py::encode`. Its 384 MiB stands. Do not retry this.

### TTS is still the largest single component
~791–1290 MiB against a 253 MB GGUF. **NeuCodec, not the backbone, dominates** — consistent with
the standing note that the int8 decoder is larger than the model it decodes for.

### The decisive test: same stack, thinker on vs off, camera actually opened

| config | free before camera | camera result |
|---|---:|---|
| thinker **OFF** | **971 MiB** | `CAMERA_OPEN=True`, **16/16 frames**, 681 MiB left |
| thinker **ON** | **0 MiB** | `CAMERA_OPEN=True`, **0 frames** — NVMM alloc fails |

The thinker-on case reproduces the documented failure exactly: GStreamer reports the camera open,
then delivers no buffers. **Qwen3-0.6B cannot be co-resident with perception + camera +
SenseVoice + TTS on 7.6 GB.**

### `LazyGGUFReasoningTier` is DEAD — measured, at three quantizations

Its docstring says *"do not switch without re-running the full-stack test."* Test run, with
perception + camera resident and the perception `look` tool wired in:

| thinker | size | free at attempt | result |
|---|---:|---:|---|
| Q8_0 | 805 MB | 781 MiB | **fail** ×5 with compaction retry |
| Q6_K | 623 MB | 445 MiB | **fail** ×3 |
| Q4_K_M | 484 MB | 675 MiB | **fail** ×3 |

**It is not a size problem** — 484 MB failed with 675 MiB free. This is the mandated load-order
rule biting: *llama.cpp models must load before the torch perception stack*, and lazy loading
does precisely the forbidden thing. Note `LazyGGUFReasoningTier` also constructs
`GGUFReasoningTier` directly, bypassing the `_load_gguf_retry` + `_compact_memory()` wrapper every
eager load uses; adding that wrapper did not save it either (10/10 failures).

**Close this option. Do not re-open it.**

### What DOES work: Q4_K_M loaded eagerly, in order

Built from the merged HF checkpoint (not requantized from Q8): Q8_0 805 MB → **Q6_K 623 MB** →
**Q4_K_M 484 MB**. With Q4_K_M + O1, the full stack including the thinker:

```
speaker 147 | thinker 717 | TTS 1141 | perception 1217 | qp+SigLIP2 776 | SenseVoice 732 | VAP 19
>>> FREE BEFORE CAMERA: 135 MiB
CAMERA_OPEN=True   CAMERA_FRAMES=16      <-- vs 0 frames at Q8_0
[percep] encode 1206 ms  streams=['m2','vision','ambient','scene']   avail=18 MiB
FINAL FREE: 49 MiB
```

**Everything fits, camera included — but with 18–49 MiB of margin.** That is not a safe live-demo
margin; a background service starting would tip it over. `jetson_preflight.sh` becomes mandatory.

**Reasoning survives the quantization.** Isolated A/B, all three emit real `<think>`:

| | CoT | latency | answer |
|---|---|---:|---|
| Q8_0 | yes | 1194 ms | "Hey there! What would you like tonight?" |
| Q6_K | yes | 1343 ms | "Hey there! What would you like this evening?" |
| **Q4_K_M** | yes | **1205 ms** | "Let's make some sweet music together. Shall we start with a gentle jingle?" |

Q4_K_M is the fastest of the three and its answers are no worse. Also note the thinker is
**1.2–1.4 s here, not the 2.4 s** recorded earlier — that figure came from a longer prompt.

### Recommendation
Two supportable configurations:
- **Safe:** no thinker — 681 MiB margin, proven, camera 16/16.
- **Full:** Q4_K_M thinker eager + O1 — everything works, 49 MiB margin, requires preflight.

Not viable: lazy thinker (any size), Q8_0 thinker with perception+camera.

### Perception reaches the thinker — verified
`ToolRegistry.register("look", pq.as_tool_handler())` → `reg.execute("look", {...})` returns
`'a shadowy room'` from live camera frames. The thinker→perception hookup works end to end.

## 6. Perception verified on real camera frames

Showcase config, camera open, all four trained streams live:

```
[percep] encode 1083 ms   streams=['m2','vision','ambient','scene']
  who is here          -> a shadowy room (0.593) | someone is leaving the room (0.579)
  where am i           -> a shadowy room (0.567) | a cluttered room (0.543)
  what do you hear     -> a fan humming (0.403) | a heater (0.399)
FINAL FREE: 310 MiB
```

Retrieval 11–80 ms. Answers are correct for the conditions (3 am, dark room, audible fan).
Note these used `ask_topk` over the whole bank; the production demo restricts each question to
its tag category, which is why "what are they doing" returned the same as "who is here" here —
a limitation of the test, not the model.

## 7. Camera hub — the unification works

`~/bmo_production/scripts/bmo_camera_hub.py` is a single master GStreamer pipeline with three
hardware branches: 160×120 GRAY8 → motion centroid → `/dev/shm/bmo_motion.txt`; 256×256 BGR →
`/tmp/bmo_cam_perception.sock`; 1280×720 → `/tmp/bmo_cam_display.sock`. This is what makes eye
tracking and perception coexist on an exclusive CSI sensor.

`bmo_camera_hub.service` exists but is **`inactive` and not enabled**. The tests above opened the
camera directly via `nvarguscamerasrc`, so the 290 MiB NVMM figure is for a direct client. If the
hub runs as the single owner and clients read the sockets, that allocation is paid **once** rather
than per-client — worth measuring, as it may be what buys room for the thinker.

---

## 7b. Pipeline walkthrough — what a turn actually does, and what can go

Reading `causal_duplex_pipeline.py` step by step. Four things fall out, one of them serious.

```
mic callback ── robot_speaking? -> RETURN (hard gate)          [S4]
             ── raw * MIC_GAIN(1.8)                            [S2]
             ── sosfilt(highpass 150 Hz)                       [S3]
             └─ vad.accept_waveform
VAD segment ─→ process_turn
             ── np.clip(seg * 1.5)   <-- SECOND gain stage     [S2]
             ── SenseVoice transcribe        ~200-560 ms
             ── _fix_name regex
             ── homeostatic_to_mood_state(self.state)          [S1]
             ── fast_llm.generate            ~160-255 ms
             ── voice.synthesize  <-- FULL utterance first     [S5]
             ── AudioFormantExtractor over the whole waveform  [S5]
             ── viseme_thread polls every 10 ms -> /dev/shm
             ── play_audio -> UnifiedAudioEngine
             ── busy-wait while is_playing()
             └─ vad.reset()   <-- discards buffered speech     [S4]
```

**S1 — `HomeostaticState.update()` is never called. This is the serious one.**
`self.state = HomeostaticState()` is constructed at line 412 and read at line 435 via
`homeostatic_to_mood_state(self.state)`, and **`update()` appears nowhere in the file**. The mood
is therefore permanently its initial value. Consequence: **the 12-mood emotion voice trained
today would always render exactly one mood.** We would have shipped a mood-conditioned voice
driven by a constant and never noticed, because every sample would have sounded consistent.
Fix: call `state.update(dt_s, user_speaking, user_present, ...)` each turn — `BmoDuplexTick`
already does this correctly and has no caller.

**S2 — two independent gain stages, both clipping.** `MIC_GAIN = 1.8` in the mic callback and
`np.clip(seg * 1.5, -1, 1)` in `process_turn` = **2.7× total** with clipping at both. ASR is
being fed clipped audio. This is a plausible contributor to Citrinet being judged "bad at
recognition" — it may have been judged on distorted input. Consolidate to one calibrated gain
and measure ASR accuracy again before ranking engines.

**S3 — the highpass targets the wrong frequency.** `butter(4, 150 Hz, 'highpass')` on the mic,
but the fan was measured **tonal, 38.6% of its energy in one peak at 808–812 Hz** (blade-pass
frequency, 1.9 Hz from prediction). A 150 Hz highpass does nothing to it.
`models/m5_fan_notch.py` — tach-driven notch, **33.4 dB on the tone, 0.0 dB on speech, 1.6 ms** —
is written, measured, and unused. Wire it; drop or keep the highpass on its own merits.

**S4 — barge-in is blocked twice over, before VAP even enters the picture.** The mic callback
returns immediately while `robot_speaking` is set, *and* `vad.reset()` after the turn discards
whatever was buffered. So user speech during BMO's reply is both gated out and then thrown away.
Wiring the VAP head requires removing both, which is only safe once the echo-augmented retrain
(§8) exists — otherwise BMO interrupts itself.

**S5 — synthesis waits for the whole utterance, and the lip-sync is why.**
`AudioFormantExtractor.analyze_waveform(audio)` needs the complete waveform, so `synthesize()`
must finish before anything plays. Measured cost of the viseme pass: **20–29 ms for an entire
utterance**. It is a 30 ms-frame / 15 ms-hop FFT and already causal. Running it per NeuCodec
chunk with ~60 ms lookahead for the `MIN_DWELL` merge makes streaming and lip-sync coexist and
takes first audio from **~2900 ms to ~290 ms**. This is the single biggest perceived-latency win
left and it costs almost nothing.

---

## 8. The backchannel / self-interruption bug — root cause

You do not need a new decision head. The VAP head is architecturally right: 2-channel
(user mic + robot reference) is the only formulation that *can* solve this — a mic-only
classifier has no information to separate "user talking" from "me talking" when the mic
contains BMO's voice. The bug is entirely in the training data, in `build_vap_dataset.py`:

1. **Zero echo augmentation.** ch0 = clean user, ch1 = clean robot, in every sample. The only
   augmentation is white noise added *identically to both channels* (lines 230–232), which
   teaches nothing. The head's learned definition of `barge_in` is "energy in ch0 while ch1 is
   active" — which is exactly what its own echo produces.
2. **The synthetic "user" pool is BMO's own voice** (`piper_bmo/dataset_bmo/wavs`,
   `bmo_speech_dataset/wavs`, `bmo_fishapi_synth/wavs`). User/robot discrimination is therefore
   100% channel-position, with no timbre cue to fall back on when channels leak.
3. **The `backchannel` scenario scales ch1 by 0.7**, so a quiet backchannel's echo looks *more*
   like genuine user speech than a loud line — which is why backchannels specifically trigger it.
4. **Nothing routes backchannel WAVs into the reference channel** — `PrebuiltVoiceBank` plays
   them directly, so even a correct head would be blind to them.

Fixes:
- `ch0 += α · simulate_echo_path(ch1)`, α ∈ [0.05, 0.6], delay ∈ [5,40] ms. That function already
  exists in `models/m4_echo_cancellation.py`. **Labels stay derived from the clean futures** —
  that is precisely the supervision that teaches echo rejection.
- Draw "user" audio from real humans. `~/data/librispeech_clean100` is on mercury; EasyCom
  participants are already in the pipeline.
- Add an explicit self-backchannel scenario: ch1 = short BMO backchannel, ch0 = continuing user
  speech + echo of it, label = user active.
- Tap the reference channel from the `UnifiedAudioEngine` **callback** (what actually left the
  speaker, sample-aligned), not from `play_audio()` enqueue time; route backchannels through it.
- Demo insurance needing no retrain: normalized cross-correlation between mic and reference at
  the expected echo delay; if the mic correlates with what BMO just emitted, suppress barge-in.

VAP head measured on-device: **6.43 ms/inference** on CPU (config claimed 1.661 ms), output
(1, 8, 2), 18–24 MiB resident. At a 20 ms hop that is ~32% of one core — affordable but not free.

---

## 9. Architecture: thinker + quick speaker

The old split failed for one reason: **both tiers emitted a BMO utterance**, so the speaker could
only paraphrase. Fix the contract, not the sizes — the thinker emits a directive that is never
spoken:

```
<directive>acknowledge they're tired; offer to dim the lights; do not ask a question</directive>
```

This removes the paraphrase failure by construction, collapses the speaker's job to one shape
(`scene tags + directive + user line → spoken line` — exactly the existing 372-row
`speaker_directive` slice), and cuts the thinker's token budget from 320 to ~100.

Measured response budget with Nano:

```
SenseVoice   ~200-560 ms  (utterance-length dependent)
speaker       160-255 ms
Nano TTFA     287-322 ms
             -------------
first audio  ~650-1100 ms
```

The thinker's 2.4 s median cannot fit in that, so it runs alongside: VAP predicts the shift up to
1.6 s ahead → speaker's first line starts at ~650 ms and plays for 2–5 s → the directive lands
inside that playback window → speaker delivers the substantive line. That is "reasoning in
parallel while speaking", achieved by pipelining rather than an omni model.

**Constraint:** llama.cpp contexts on one GPU serialize, so speaker/thinker/TTS cannot literally
overlap in compute. At RTF 0.62 there is ~38% idle GPU during playback. Combined with the lazy
thinker (§5), this is the path to test.

Note `bmo_fast_dialogue_Q8_0.gguf` (`Bmo_Dialogue_Merged`, **qwen2** arch, 494M, trained Aug 20)
has no native `<think>` — which is *correct* for a speaker that must never deliberate. Qwen3-0.6B
stays the thinker.

---

## 10. State and next steps

**Done and deployed:**
- `m5_streaming_voice.py` — length cap reinstated and calibrated, both trees
- `causal_duplex_pipeline.py` — `NameError` fixed, Nano preferred, SenseVoice default
- `bmo_neutts_emotion_nano_Q8_0.gguf` — trained, converted, deployed, verified

**Needs your ears:** `~/emotion_nano_wavs/*.wav` — 12 moods. Decide whether the clean 8 are
enough for the showcase.

**Also done:** O1 (M2 bf16-before-int8) applied to `bmo_jetson_startup.py`; thinker Q6_K and
Q4_K_M built from the merged HF checkpoint and deployed; `LazyGGUFReasoningTier` tested and
closed; perception→thinker `look` tool verified live.

**Next, in order:**
1. **S1 — call `HomeostaticState.update()`.** Without it the emotion voice is decorative. One
   call site, and `BmoDuplexTick` already shows the correct usage.
2. **S5 — per-chunk causal visemes.** Takes first audio ~2900 ms → ~290 ms for ~30 ms of work.
   Biggest perceived-latency win remaining.
3. **S2 — consolidate the double mic gain**, then re-rank ASR engines on undistorted input.
4. Measure the camera hub as single owner; may recover NVMM headroom and buy back the thinker's
   safety margin.
5. VAP dataset rebuild (echo + real-human users + self-backchannel), retrain, re-export.
6. Wire VAP + remove the two barge-in blockers (S4). **This is what makes backchanneling work.**
7. S3 — wire `m5_fan_notch.py`, retire the mis-targeted 150 Hz highpass.
8. Thinker directive contract + speaker retrain with the directive slice primary.

**Do not re-litigate:** Air v5 as the voice (17% lockup rate, measured); Piper (rejected — the
NeuTTS fine-tune is BMO's voice identity); `LazyGGUFReasoningTier` (fails at every quantization,
§5); WavJEPA in bf16 (MaskedTensor forbids it, §5); mini-omni / minimind-o / Ultravox connector /
NeuTTS hidden-state bridge (all research track, none shortens the path to a showcase).
**Provisionally re-open:** Citrinet's recognition quality — it may have been judged on
double-gain-clipped audio (S2).

---

## Logs

On the Jetson (`~/`): `bench_tts.log`, `bench_runaway.log` (uncapped), `bench_runaway2.log`
(linear cap), `bench_runaway3.log` (final cap), `smoke_cdp.log`, `fullstack_fit.log`,
`mem_breakdown2.log`, `mem_truth.log`, `opt_baseline.log`, `opt_bf16.log`, `opt_verify.log`,
`lazy_thinker.log` / `lazy_thinker2.log` / `lazy_thinker3.log`, `showcase_q4.log`, `thinker_ab.log`, `showcase_fit.log` (thinker off), `showcase_fit_thinker.log` (thinker on),
`showcase_fit2.log` (perception on real frames), `emotion_check.log`.

On mercury: `JEPA-Omni/emotion_nano_train.log`.

---

## 11. `bmo_showcase.py` — the hardened pipeline (BUILT, SELFTEST PASS)

`scripts/bmo_showcase.py` (also at `bmo@bmo-desktop:~/bmo_showcase.py`). Supersedes
`causal_duplex_pipeline.py`, which is kept as a fallback. Run
`python3 bmo_showcase.py --selftest [--perception]` before any live demo — it exercises the
whole chain with no mic or speaker and fails loudly.

**Load order is enforced in code, not comments:** speaker GGUF → thinker GGUF → TTS GGUF →
STT/VAD → *then* torch perception + camera. Provenance: `bmo_jetson_startup.py` docstring
("confirmed 2026-08-07, `jetson_full_stack_v4_reversed_order.py`"), `CLAUDE.md:124`,
`MEMORY_OPTIMIZATION_PLAN.md:103`. Re-confirmed 2026-08-23 by the lazy-thinker failures.

### Fixes carried in it
| | |
|---|---|
| **S1** | `HomeostaticState.update()` now actually called → `state_changed=True` every turn, mood varies (curious/happy/content/bored). Without this the 12-mood voice renders one mood forever. |
| **S5** | `StreamingVisemeExtractor` — per-chunk causal formant→viseme, scheduled against the audio device's **real playback position** (`engine.played_sec()`), so a generation stall cannot desync the mouth. **First audio ~2900 ms → ~300 ms.** |
| **S2** | One mic gain stage (was 1.8 × 1.5 = 2.7× with clipping at both). |
| placeholder | v6 default + retry-once + `strip_placeholder()` fallback that removes the address construct rather than leaving `"I'm , and you are...?"`. |
| robustness | every stage wrapped; the turn loop catches and continues; refuses to start if no input device (rather than capturing silence, the original "won't listen" bug). |

### Measured, full stack incl. perception + camera
```
[boot] speaker v6 | thinker Q4_K_M | Nano emotion voice | SenseVoice+VAD | perception(4) | camera
[boot] DONE  avail=688 MiB
t0 ttfa=464ms visemes=30 cov=0.98 | t1 ttfa=324ms cov=0.92 | t2 ttfa=331ms cov=0.92
[thinker] 1446 ms CoT=yes            SELFTEST PASS   avail=280 MiB
```
RTF rises 0.66 → 0.74–0.77 with perception resident (GPU contention). Still < 1.0, so
synthesis stays ahead of playback.

## 12. Are the speaker and thinker correctly fine-tuned? — No, and here is exactly how

**Speaker: v5 leaks template placeholders. v6 does not.** n=20 prompts:

| | placeholder leaks | label prefixes | latency |
|---|---:|---:|---:|
| v5 (was deployed) | **2/20** — `"I'm [name], and you are...?"`, `"...for you, {name}."` | 0/20 | 173 ms |
| **v6** | **0/20** | 0/20 | **147 ms** |

v6 was shelved on an 8/12-vs-9/12 bake-off that the ledger itself records as having false
passes and being underpowered at n=6. A visible placeholder on stage is the worse failure.
**v6 is now the default.** Neither version is meaningfully instruction-conditioned though —
v6's directive slice is 372 of 4,144 rows (9%).

**Thinker — see §17. The first reading here was WRONG** (it was an underspecified prompt, not
a training gap). The conclusion that the directive path must stay off still stands, but for a
different and more interesting reason.

So the directive path ships **off by default**, behind `--use-directive`, with
`_directive_guard()` rejecting anything that reads as a spoken line. v5 fails that guard,
which is the correct outcome. **Closing this needs a thinker corpus of
`(context → directive)` pairs — that is the real remaining fine-tuning work.**

## 13. Edge cases — what is and is not handled

| | state |
|---|---|
| turn boundaries | ✅ VAD, 0.55 s silence |
| self-echo / talking to itself | ✅ hard mic gate during playback — **safe, but no barge-in** |
| **barge-in** | ❌ **not implemented.** Blocked twice over (mic gate + `vad.reset()`), and the VAP head that would arbitrate has never seen its own echo in ch0 (§8) |
| backchannel during user speech | ❌ needs VAP; only `thinking_filler` at turn start works |
| TTS runaway | ✅ bounded 40.0 s → 9.8 s worst case |
| unstable moods | ✅ 4/12 auto-mapped to neutral (`--unsafe-moods`) |
| placeholder leak | ✅ v6 + retry + strip |
| ASR failure / empty transcript | ✅ skipped, loop continues |
| speaker/TTS/thinker exception | ✅ caught; speaker falls back to a spoken recovery line |
| no mic present | ✅ refuses to start rather than capturing silence |
| 7 W power mode | ⚠️ **onnxruntime hard-aborts** (offline cores). Keep MAXN_SUPER; `bmo-power --auto-guard` drops to 7 W at ≤20% battery and would reintroduce this |
| background services | ⚠️ 280 MiB end margin — run `jetson_preflight.sh` before the demo |

## 14. C++ / omni — asked and answered, do not spend demo time here

**C++ won't help the memory.** llama.cpp is already C++; the Python is a thin ctypes binding,
and the footprint is model weights, not runtime. The one real datapoint is your own audio.cpp
eval: `sense_asr` there is ~33% smaller (695 vs 1034 MiB isolated) and ~2× faster — a genuine
win, but it costs splitting one process into two over IPC. With 688 MiB of headroom after boot
that trade is not worth taking this week. Revisit after the showcase.

**Omni models don't solve the actual gap.** mini-omni and minimind-o are both half-duplex with
VAD barge-in — they would not have fixed turn-taking, and adopting either discards the BMO
voice, the corpus, and the perception integration. The thing standing between us and fluent
conversation is the VAP echo retrain (§8), not the model family.

---

## 15. Second audit — five more things, found by checking rather than assuming

### 15a. The emotion voice trained tonight is NOT demo-safe. Plain Nano is now the default.
n=32 utterances per voice, **neutral only** (so this is not a thin-mood problem):

| voice | non-EOS exits | same 52-char line |
|---|---:|---|
| **`bmo_neutts_nano_v1`** | **0/32** | 3.74–5.84 s |
| `bmo_neutts_emotion_nano` | **9/32 (28%)** | **2.64–9.20 s** |

A 30-char line hit the cap on *most* samples (median == max == 7.00 s). The emotion fine-tune
degraded EOS calibration at its own anchor mood. Caught live: a 52-char reply produced 460
speech tokens / 9.2 s. **Voice order is now nano_v1 → nano_v2 → emotion_nano → Air.** The
emotion model needs an EOS-calibrated retrain (fewer steps, or EOS-weighted loss) before it
goes on stage; `--voice bmo_neutts_emotion_nano_Q8_0.gguf` overrides.

### 15b. The backchannel never fired — silently.
`PrebuiltVoiceBank` exposes `play(category)`, **not** `pick()`. The `pick()` call raised
`AttributeError` into a bare `except`, so latency masking silently did nothing. Also `play()`
drives its own sounddevice output (device=24), bypassing the engine — and would keep the clip
out of the VAP reference channel later. Now the 5 `thinking_filler_*.wav` are loaded directly
and pushed through `UnifiedAudioEngine`.

### 15c. The CSI camera is exclusive, and that is why the hub matters.
`BMO_Engine` reads `/dev/shm/bmo_motion.txt`; `motion_tracker.cpp` opens
`nvarguscamerasrc sensor-id=0` to produce it. Perception opening the sensor directly therefore
makes **eye tracking and perception mutually exclusive**. `bmo_camera_hub.py` solves it (one
source, tee'd to a motion branch and a 256×256 `shmsink`) — and **nothing was consuming its
sockets.**

Measured: hub RSS **176 MiB**, client reads **24/24 frames at 31.4 fps**, real image data.
`bmo_showcase.py` now prefers the hub socket and falls back to direct capture with a warning.
**Start the hub before the demo** if you want a face with eye tracking *and* perception.
(Its docstring advertises a third 1280×720 display branch that is not in the pipeline string.)

### 15d. `BMO_Engine` does not currently start.
It is **not running**, is not a service, and launching it over SSH with `DISPLAY=:0` fails:
`Failed to compile fragment shader` (`mouth_es.fs`, `sdf_font_es.fs`), `Framebuffer object can
not be created`, `GLFW library is not initialized`. It *does* load its assets
(`[VisemeDatabase] Loaded 12 viseme presets (142 alias keys)`), so this reads as an X/GL
context problem from a non-console session (`XAUTHORITY`), not a broken build.
**Unverified — must be confirmed on the actual console.** Note the face costs ~350 MiB + 86 MiB
Xorg, which is *not* included in any margin in this document.

### 15e. The viseme write path had never been executed.
`selftest` used the extractor but never the scheduler, so nothing wrote
`/dev/shm/bmo_speech.txt`. The face engine parses it with
`fscanf("%d %63s %f %63s")` every 15 ms — format confirmed to match. Selftest now writes and
re-reads it, and asserts the mouth closes to `0` on stop. A broken write would otherwise have
shown up only as a still mouth on stage.

### Selftest now also gates on
* `last_exit_reason == EOS_TOKEN` (catches 15a-class regressions)
* utterance duration ≤ 1.3× expected at the measured 8.3 chars/sec

### Current verified state
```
speaker v6 | thinker Q4_K_M | nano_v1 voice | SenseVoice+VAD | perception(4 streams) | camera
[boot] DONE avail=511 MiB
t0/t1/t2 all exit=EOS_TOKEN, ttfa 314-481 ms, viseme coverage 0.94-0.97
[viseme] '1 AEI 1.00' parse_ok=True -> '0 mouth_phoneme_X 0.0' on stop
SELFTEST PASS  avail=239 MiB
```

### Still open, in priority order
1. **Start the camera hub** (and confirm `BMO_Engine` launches on the console) — otherwise it is
   a face *or* perception, not both. Neither is running today.
2. **Barge-in** — VAP echo retrain (§8) plus removing the two blockers (S4).
3. **Emotion voice EOS recalibration** — 28% non-EOS is why it is benched.
4. **Thinker directive corpus** — `(context → directive)` pairs (§12).
5. Face-engine memory is not in any budget here; with ~239 MiB end margin it likely does not
   fit alongside the thinker. Expect to choose: **thinker, or face+eye-tracking.**

---

## 16. The face engine — 16,489 crash-restarts, root-caused to an unplugged screen

`bmo_face_engine.service` reported `active` but no `BMO_Engine` process was ever visible in
`ps` — because it was **crash-looping every 2 seconds** (`Restart=always`, `RestartSec=2`).
`systemctl show -p NRestarts` reads **16,489**.

Every run failed the same way: all six `_es.fs` shaders fail to compile, `FBO: Framebuffer
object can not be created`, window closes. The tempting read is a shader bug. It is not:

```
INFO: Platform backend: NATIVE DRM
WARNING: DISPLAY: No suitable DRM connector found      <-- everything cascades from here
```

```
/sys/class/drm/card1-DP-1/status = disconnected     (the ONLY connector on the box)
```

**No screen was attached** (confirmed by the user). With no connector there is no GL context,
so the default texture and default shader fail first, and every subsequent shader load fails
as a consequence. Both builds fail identically with no panel — this was never a code defect.

`card1` is the `nv_platform` display controller exposing a single **DisplayPort** output;
`card0` exposes no connectors. So DP is the port to plug into.

### There ARE two builds, and production is pointed at the wrong one

| build | PLATFORM | GRAPHICS | X needed? |
|---|---|---|---|
| `face_engine/BMO Face Engine/build` (**what the service runs**) | `Desktop` | `OPENGL_ES2` | yes |
| `face_engine_drm/BMO Face Engine/build_drm` | **`DRM`** | `OPENGL_ES2` | **no** |

The deployed one is internally inconsistent — a **desktop** GL context asked to compile
**OpenGL ES** shaders. The DRM build is consistent and needs no X server at all, which also
frees Xorg. `start_engine.sh` in the DRM tree is stale: it still `cd`s to the *Xorg* build.

Note the two builds also load **different viseme databases**: the Xorg build reports
`viseme_database.txt` → 12 presets / 142 alias keys; the DRM build reports
`visemes_database.txt` → **29 phoneme visemes**. `bmo_showcase.py` emits the 12-symbol set
(`AEI, O, CDGKNST, QW, L, BMP, F, EE, TH, J, R, U, mouth_phoneme_X`). **Verify the DRM build
accepts those symbols once a screen is attached** — if it expects a 29-phoneme vocabulary the
mouth will sit still even though the file is being written correctly.

### `scripts/use_drm_face.sh` (also at `~/use_drm_face.sh`)
One command to switch, and it **refuses to run with no connected display** rather than
recreating the crash loop. It stops+disables `bmo_xorg`, repoints
`bmo_face_engine.service` at the DRM binary, and adds `StartLimitBurst=5` so an unplugged
panel can never burn a core again. Revert instructions are printed at the end.

### State left behind
* `bmo_xorg` — **restored to active** (as found).
* `bmo_face_engine` — **left stopped deliberately.** It is still `enabled`, so it will resume
  crash-looping on the next reboot until either a screen is attached or the service is
  repointed. Run `sudo systemctl disable bmo_face_engine` if you want that suppressed.

### Memory implication
The face has never actually rendered, so the "~350 MiB" figure quoted for it in earlier notes
is **unverified on this device**. Budget for it only after it runs with a panel attached.
Going DRM also removes Xorg from the budget.


---

## 17. CORRECTION + the real speaker/thinker coordination answer

Two earlier claims in this document were wrong. Both were tested against the **base** models,
which are still on the device, so this is measured rather than argued.

### The persona fine-tune did NOT damage instruction-following — it improved it

Same 8 rule-scored directive fixtures (`[scene] + [instruction] -> one line`):

| speaker | instruction-following |
|---|---:|
| **BASE `lfm25_350m_Q8_0`** (no BMO tuning) | **5/8** |
| BMO v5 | 7/8 |
| **BMO v6** | **8/8** |

The base model is not only worse, it is barely conversational — its outputs were `'"Red
Jumper."'`, `'Battery low.'`, `'Goodnight.'`. **The "we tuned it to sound like BMO and it
stopped understanding instructions" hypothesis is falsified.** v6 is the best
instruction-follower available, which is a second independent reason to prefer it over v5.

### The thinker CAN emit directives — §12's claim was a prompt bug, not a training gap

| thinker | directive rate |
|---|---:|
| BASE `Qwen3-0.6B` | 1/4 (3 exhausted the token budget mid-`<think>`) |
| **BMO thinker v5** | **4/4** |

The difference between the failing and passing prompt is one clause:
**"Start with a verb like 'Ask', 'Offer', 'Acknowledge'."** Without a verb anchor the thinker
returns a spoken line; with one it returns an instruction. §12 concluded "not fine-tuned for
directives" from a single underspecified prompt. That was wrong.

### But the directive path still must stay OFF — for a third reason

End-to-end, thinker → directive → speaker v6, 6 turns. The directives are **degenerate**:

```
"I've been debugging this for hours."  ->  "Ask me what you'd like to do next?"
"What should I do tonight?"            ->  "Ask me what you'd like to do tonight!"
"You're being annoying."               ->  "Ask me what you'd like to do next?"
"I don't feel like talking."           ->  "Ask me what you'd like to talk about next!"
```

Syntactically valid, **semantically empty** — near-identical regardless of input, so they carry
no information about what was actually said. They are also confused about who is who ("Ask me
what you'd like" instructs Beemo to ask, phrased as if Beemo is the one being asked).

And injecting them measurably **degrades the turns that matter most**:

| user | without directive | with directive |
|---|---|---|
| "You're being annoying." | *"That... that really hurts the artist's feelings."* ✅ | *"That jittery move makes me nervous."* ❌ |
| "I don't feel like talking." | — | *"Beemo wants to know what you want to hear next."* ❌ ignores them |
| "I finally finished my project!" | *"I'll send you a celebratory sound effect."* | *"Shall we review and add a bonus round?"* ✅ |

One helps, two hurt, and the two that hurt are hostility and withdrawal — exactly the cases the
companion corpus was built to handle.

**Root cause: the thinker's corpus teaches conversational answers, so when forced into
directive form it collapses onto a single template.** The fix is the same one §12 named, but
now for a precise reason: a thinker corpus of **diverse** `(context → directive)` pairs, where
the directive varies with the input. Prompt engineering gets the *form*; only training gets the
*content*.

`--use-directive` stays off by default and `_directive_guard()` stays in place.

### Also fixed here
v6 emitted `'BMU says hi!'`. `NAME_RE` covered `bmw` but not `bmu`; now folds
`bmj|bmw|bmu|bmp|bno|bmo0` as well.

---

## 18. Speaker/thinker sync — the complete diagnosis, and the one job that fixes it

### What is NOT broken
* **Speaker follows directives.** v6 scores **8/8** on rule-checked directive fixtures (base
  model: 5/8). Not over-tuned — see §17.
* **Perception reaches the thinker.** The engine works (`look` returns live tags) and the
  thinker's corpus has 237 `perception_social` rows that take perception **as text in the
  prompt**. `bmo_showcase.py` already does exactly that.

### What IS broken, precisely
**The thinker has no notion of a directive, and cannot acquire one by prompting.**

Its corpus (`bmo_thinker_corpus_v7c.jsonl`, 987 rows) has schema
`prompt → reasoning → answer`, where **`answer` is always a finished BMO utterance**. There is
no directive field anywhere. So asked for an instruction it produces a BMO line in imperative
clothing — *"Ask me what you'd like to do next?"* for four different inputs.

Every prompting route was tried and every one collapses to a mode:

| approach | result |
|---|---|
| free-form "give an instruction" | returns spoken lines (§12) |
| + verb anchor ("start with Ask/Offer/…") | correct *form*, near-identical *content* (§17) |
| closed-set selection, 13 options, with CoT | **3/8** — picked #12 five times |
| closed-set selection, without CoT | **1/8** — picked **#1 all eight times** |
| speaker v6 (350M) as the selector | **1/8** — picked #7 five times |

Neither a 0.6B nor a 350M model does 13-way classification here. **Prompting is exhausted.**

### The interface is already defined by the speaker's own training data
`bmo_companion_corpus_v12.jsonl`, `speaker_directive` slice — 372 rows, and critically only
**13 distinct directives** (~29 examples each):

```
prompt: "You can see: wearing: <..>; doing: <..>; who: <..>; where: <..>;
         lighting: <..>; hearing: <..>. Your private thinking: <DIRECTIVE>"
text:   "<the spoken line>"
```

Two shapes, 187 `prose_cot` / 185 `compact`. Directives are imperative + a `because` clause.
**Note `bmo_showcase.py` currently injects `[instruction] …`, which is NOT this format** — that
is fixed below regardless of whether the directive path is enabled.

### THE ONE JOB
**Add a `directive` field to the thinker corpus and fine-tune on it.** That is the whole fix,
and it is the same move that took the speaker from 0 to 8/8.

Spec, fully determined by the data above:
* **input**: the six-category perception string + the user's utterance + homeostatic state
* **output**: `reasoning` (CoT, keep it — it is the thinker's value) **+ `directive`**
* **directive form**: imperative + `because <rationale>`, matching the speaker's slice
* **coverage**: the existing 13 as a floor, then widen — 13 is too narrow for open conversation
* **pairing**: the directive MUST be generated together with its context and carry a `paired`
  flag. This project already paid for this lesson once: 197 of 395 directive rows were ruined
  by pairing a random CoT with a line written for a different directive.
* **size**: ≥1,000 rows to beat the 372-row precedent; balance across directives, and include
  hostility/withdrawal, which are where injected directives measurably hurt (§17).

Then widen the **speaker** slice past 13 distinct directives so it generalises rather than
memorises.

### Do NOT bother with
* Teaching the thinker the `look` tool. Its corpus has **zero** `look` rows (128 tool rows, all
  weather/date/timer/reminder/search/lights, in 85 inconsistent spellings). Perception as
  **push** is already trained and already works. Pull is a research track, not a fix.
* Any further prompt engineering on the current thinker — the table above closes it.

---

## 19. Thinker directive corpus — generator written, RUNNING on mercury

`scripts/generate_thinker_directive_rows.py` — the inverse of
`generate_speaker_directive_rows.py`, reusing its `SCENES`, `PERSONA`, `HARD_RULES`, salvage
path and closed-set verifier discipline.

**Launched 2026-08-23 22:17**, gpt-oss-120b across 4 Blackwells →
`data/bmo_thinker_directive_rows_v1.jsonl`, log `thinker_dir_gen.log`.
208 combos (26 directives × 8 scenes) × 6 = ~1,248 target. Measured **2.7 rows/min → ETA ~7.6 h**.

### The design choice that avoids the collapse
Asking the teacher "here is a situation, what should BMO do?" reproduces the exact collapse
being fixed — the teacher also gravitates to a few safe directives. So generation is
**inverted**: the directive is FIXED per request and the teacher invents user utterances for
which that directive is right. Coverage across directives is then balanced by construction,
and every row is correctly conditioned by design rather than by luck.

### Vocabulary widened 13 → 26
The speaker slice had only 13 distinct directives, and the gaps were exactly where injected
directives measurably hurt (§17): hostility, withdrawal, celebration, admitting ignorance,
owning a mistake. **The speaker slice must be widened to match** — the two halves share one
vocabulary or the interface breaks.

### Pairing is structural, not asserted
`said` and `thinking` come out of ONE JSON object and are never substituted afterwards. The
salvage path walks forward from each `"said"` to its own `"thinking"` and drops anything not
ending in terminal punctuation (a truncated half-sentence is worse than a dropped row). This is
the 197-of-395 bug designed out rather than checked for.

### Verifier rejects exactly the measured failure modes
```
"Ask me what you'd like to do next?"              -> first_person     (the degenerate directive)
"Alright! Let's press start on a jingle."         -> not_imperative   (a spoken line)
"offer to play something, because they look bored" -> accepted
```

### Gates are on coverage and collapse, not count
≥900 rows · ≥80% of the 26 directives present · no directive >25% of rows · <15% duplicate
utterances. The 395-row attempt passed a count gate while 197 rows were actively harmful.

### First 12 rows — quality signals
| check | value |
|---|---|
| distinct utterances | 12/12 |
| distinct reasoning | 12/12 |
| reasoning↔directive word overlap | mean **0.28**, max 0.43 |
| salvaged (malformed JSON) | 0 |

Low overlap is the one that matters: the reasoning reasons *toward* the directive instead of
restating it. Sample:
```
SAID     : "I don't know if I should keep the game saved or start a new level."
THINKING : "They're weighing two options... I can ask which direction feels better right now."
DIRECTIVE: ask a follow-up question, because they hinted at something they did not finish
```

### A fix applied mid-run
The first version buffered all rows and wrote only at the end — on a 7-hour run a late crash
loses everything, and nothing is inspectable until it is over, which contradicts this project's
own "read rows, not counts" rule. It now appends per combo (`buffering=1`) and prints real
SAID/THINKING/DIRECTIVE samples at combos 0, 3, 12, 40, 100. Cost: one 4-minute model reload.

### DO NOT train on this unfinished
Read rows first. Then: fine-tune the thinker with `directive` as the target, widen the speaker
slice past 13, and re-run the coordination test (§17/§18) before enabling `--use-directive`.

---

## 20. Speaker slice widening 13 → 26 (queued behind the thinker corpus)

### One vocabulary, one source of truth
`generate_speaker_directive_rows.py` now defines `EXTRA_DIRECTIVES` (12) and
`ALL_DIRECTIVES` (26 = 14 base + 12 new), and
`generate_thinker_directive_rows.py` **imports** that list rather than keeping its own copy.
Verified identical at import time.

This matters more than it looks. If the two lists drift, the thinker emits directives the
speaker was never trained to obey — the same class of train/deploy mismatch that caused the
original coordination failure. Previously the vocabulary was duplicated in two files.

### The 12 additions
Chosen from the measured failures in §17 — the shipped 13 had no directive for hostility,
withdrawal, celebration, admitting ignorance, or owning a mistake, and those are exactly the
turns where injecting a directive made the line *worse*:

```
celebrate ... | hold your ground gently ... | give them quiet space ...
acknowledge how long they have been working ... | answer their question directly ...
admit you do not know ... | reassure them ... | match their excitement ...
ask a follow-up question ... | suggest something calm ... | thank them ...
own the mistake plainly ...
```

### Run
`scripts/_chain_speaker_widen.sh`, launched 2026-08-23, **waiting on the log marker**
`THINKER_DIRECTIVE_DONE` (not on a process name — `pgrep`/`pkill -f` matched this session's own
argv six separate times in this project's history; a marker printed after the work completes
cannot be true early and cannot false-match an argv). Sleeps 45 s after the marker so the 120B
fully releases the GPUs — a partial release reproducibly causes an illegal memory access here.

`--only-new` generates just the 12 additions (the original 13 already carry ~29 rows each in
v12), 12 × 8 scenes × 5 = **~480 target rows** →
`data/bmo_companion_corpus_v13.jsonl` (= v12's 4,144 + the new slice).
Log: `speaker_widen.log`.

### Gate before training on v13
Confirm all 26 directives are present and that no directive dominates — the same collapse check
applied to the thinker corpus. Then retrain the speaker (v7) and re-run the §17/§18
coordination test against the newly fine-tuned thinker.

---

## 21. Can the thinker query perception? And how does identity work?

### 21a. "Describe the current view" — supported, but not the way you'd expect

The query predictor answers a **fixed set of pre-encoded queries** (`query_vectors_siglip2_v2.pt`),
and they do cover this:

```
"Describe the room and setting in detail."      "What do you hear?"
"Summarize the scene in one sentence."          "Tell me in detail what the person is doing."
"Explain everything that happens, in order."    "Give a one-line summary of the whole scene."
```

An arbitrary query is snapped to the nearest of these by word overlap, so "describe the view"
and "what is happening" both land somewhere sensible.

**The catch is the answer, not the question.** `ask()` performs *retrieval over 1,482 tags* and
returns ONE of them. "Describe the room in detail" comes back as `'a shadowy room'` — a single
phrase, not a description. **The interface cannot compose.** That is the 1,482-tag ceiling.

**Fix applied:** `_ask_perception` now asks **per category** and assembles, restricting each
question to the tags that could answer it (bank categories: `mined` 1186, `appearance` 110,
`object` 66, `sound` 34, `place` 28, `action` 22, `people` 20, `light` 10, `camera` 6):

```
who: <..>; wearing: <..>; doing: <..>; where: <..>; lighting: <..>; hearing: <..>
```

That is also **exactly the format the speaker's directive rows were trained on**, so it feeds
the prompt cleanly. An empty category now logs loudly rather than being skipped — a silently
vanishing question is how the `wearing` field disappeared for an entire run once before.

### 21b. Can the THINKER pull perception on demand? No.

The `look` tool exists and works (`pq.as_tool_handler()` → verified returning live tags), but
the thinker's corpus contains **zero** `look` rows — 128 tool rows, all
weather/date/timer/reminder/search/lights, across 85 inconsistent spellings. It will not call
a tool it has never seen.

What IS trained is **push**: 237 `perception_social` rows take perception as text already in the
prompt. So perception reaches the thinker by being handed to it, which is what
`bmo_showcase.py` does. Pull is a research track, not a gap to fix now.

### 21c. Identity — the machinery is complete, the FLOW does not exist

Two stores, both persisting across reboot:

| store | file | holds |
|---|---|---|
| `JepaMemory` | `bmo_identity.pt` | biometric: label → 256-d centroid, `n`, `spread`; cfg `threshold=0.5` |
| `BmoMemory` | `bmo_memory.json` | episodic: name → facts, episodes (summary+mood+timestamp), links, `n_encounters` |

Full API on both — `enroll/query/forget/save/load/calibrate_threshold`, and
`ensure/note_encounter/note_fact/rename/recall/to_prompt_line`.

**But nothing in the live path ever calls `enroll()`.** The only three call sites are
`jepa_memory.py`'s self-test, `jetson_core_pipeline_test.py:569`, and `jetson_real_demo.py` —
whose own docstring says it called `memory.enroll(emb, "Alice")` **to SIMULATE** someone
answering "I'm Alice". So the ask→hear→extract-name→enrol→save loop does not exist.

**⚠ LIVE DEMO RISK.** The device currently has `Alice` persisted (centroid `[256]`, **n=1**,
spread 0.0, 3 encounters) — a test artifact, and the origin of the "why does it think I'm
Alice" incident. With identity wired, BMO can greet a stranger as Alice. **Clear the store
before any demo**, or leave identity unwired (it is unwired in `bmo_showcase.py` today).

Two further caveats when it is wired:
* `threshold=0.5`, not the calibrated 0.691–0.765 operating point. `calibrate_threshold` exists
  and is never called.
* `n=1` enrolment is the weak end of the curve: 1 clip ≈ TAR 0.584 vs 8 clips ≈ 0.760.

**To make it real, the missing piece is small and well-defined:** on an unknown-face query
result, have the speaker ask for a name; parse the name from the next transcript; call
`memory.enroll(emb, name)` + `bmo_memory.ensure(name)`; `save()` both. Everything it needs
already exists and is reboot-verified.

---

## 22. Enrolment flow BUILT · Alice removed · perception reaches the speaker

### 22a. "Alice" is gone
Removed from both stores; backups at `bmo_identity.pt.bak_20260823` /
`bmo_memory.json.bak_20260823`.
```
identity entries: ['Alice'] -> []
memory people:    ['Alice'] -> []
```
Verified live: `[boot] identity ready (0 enrolled)`.

### 22b. The enrolment flow (`--identity`)
The machinery was complete and reboot-verified; only the flow was missing. Now:

1. Identity rides on the **same world-state features perception already built** — no second
   encode. Vision+voice pooled jointly (measured TAR@FAR1% 0.765 joint vs 0.694 voice-only,
   0.571 vision-only).
2. `idmem.query(emb)` → recognised / `below_threshold` / `ambiguous` / `empty_memory`.
3. **Recognised** → `bmem.note_encounter()`, and `bmem.to_prompt_line()` is injected as
   `"You remember: …"`.
4. **Stranger** → the speaker is given the trained directive
   *"ask what their name is, because you have never met them"* — **once per session**. It never
   nags; pestering a stranger for a name is worse than not knowing it.
5. Next turn, `extract_name()` runs. On a hit: `idmem.enroll` + `save`, `bmem.ensure` +
   `note_encounter` + `save`. Both persist.

**BMO can never invent a name.** A name is taken only from an explicit self-statement
(`"I'm X"`, `"My name is X"`, `"Call me X"`, `"It's X"`, or a bare capitalised reply to the
question). The prefix is case-flexible but the NAME must stay capitalised — that is what
separates `"I'm Utkarsh"` from `"I'm tired"`; using `re.I` on the whole pattern would make
`[A-Z]` case-insensitive and destroy exactly that signal. A `NOT_A_NAME` stoplist catches
`okay/tired/beemo/...`. If STT ever stops capitalising, enrolment simply does not fire — the
safe direction to fail. 12/12 extraction cases pass, including the negatives.

**Still uncalibrated:** `threshold=0.5`, not the measured 0.691–0.765 operating point, and
`calibrate_threshold()` is still never called. At 0.5 false-accepts are likelier than measured,
so keep enrolments few and distinct until it is calibrated on this camera and room.

### 22c. "Describe what you see" — fixed WITHOUT touching the thinker
The speaker already has **126 `perception_grounded` rows** trained as:
```
"You can see: a person typing on a keyboard, a computer monitor, a desk. I'm stuck on this code."
  -> "Your monitor shows a glitch, maybe a secret cheat. Need a hint?"
```
The capability existed; the scene was simply never being put in the speaker's prompt. Now a
`PERCEPTION_ASK` regex (*what do you see / describe / what's happening / what am I wearing /
who is here / what is that noise …*) pushes the composed six-category scene in exactly that
trained shape.

**So the thinker does not need to learn tool-calling before the showcase.** That was the
expensive fix; this is the cheap one, it uses a trained capability, and it needs no retrain.
Thinker-side `look` remains a research track.

### 22d. Verified live
```
[boot] identity ready (0 enrolled)   [boot] DONE avail=375 MiB
t0/t1/t2 all exit=EOS_TOKEN, ttfa 310-422 ms, viseme coverage 0.92-0.96
SELFTEST PASS  avail=110 MiB
```
Margin is tighter with the identity head resident (110 MiB vs 239). Preflight is mandatory,
and this is another reason the camera hub matters — it moves ~277 MiB out of this process.

---

## 23. Display connected — full stack measured, and ONE blocking defect found

### What now works
* **Display is connected.** `xrandr: DP-1 connected 1280x720+0+0` (470×260 mm panel).
  `/sys/class/drm/card1-DP-1` still reads `disconnected` — that is the **KMS** view, which the
  NVIDIA proprietary X driver does not populate. §16 relied on it and was therefore wrong about
  *why*, though right that no output existed. **xrandr is the authority here, not sysfs.**
* **Face engine renders.** `FBO created successfully`, textures loading, `NRestarts` frozen at
  16,489 — stable since the panel went in. The shader failures were entirely downstream of
  having no output; no code fix was needed.
* It runs the **Xorg** build, which loads `viseme_database.txt` — **12 presets / 142 aliases**,
  matching the 12-symbol set `bmo_showcase.py` emits. The 29-phoneme mismatch flagged in §16
  applies only to the DRM build, so it is not a problem on the current path.
* **The whole stack co-resides.** face + Xorg + hub + nvargus + speaker v6 + thinker Q4_K_M +
  Nano voice + SenseVoice + identity + perception: `avail=279 MiB` after boot, SELFTEST PASS.
  §5's "thinker OR face, not both" is **resolved** — the hub moving camera capture out of the
  perception process is what bought it.

### KV cache is not a constraint
Measured with the real prompts (`n_ctx=512`):

| prompt | speaker (+48 gen) | thinker (+200 gen) |
|---|---:|---:|
| bare turn | 76/512 | 226/512 |
| + perception scene | 120/512 | 269/512 |
| + memory recall | 142/512 | **291/512** |

~220 tokens spare. **History is what would consume it** (~100 tokens/turn), not perception —
directly relevant to any diary/RAG feature.

### ⛔ BLOCKING: the camera dies silently under full-stack memory pressure
`hub_live.log`, timestamped exactly when the perception stack loaded:
```
NvMapMemHandleAlloc: error 0
NvRmStream: Buffer allocation failed (err=6)
(Argus) Error InsufficientMemory ... IImageNativeBuffer not supported by Image
```
`/dev/shm/bmo_motion.txt` froze at 01:54:09 and never updated again. **The hub did not crash** —
it stayed alive at 8.5% CPU with a dead camera stream. Eye tracking froze; perception would
have been reading a dead socket.

It recovers only on restart, and only with memory free (verified: restarted at 5,181 MiB free →
motion current, 0 errors).

**`--selftest` did not catch this** because it never calls `_ask_perception` — it exercises
LLM/TTS/visemes only. A selftest that reports PASS while the camera is dead is exactly the
class of silent failure this project keeps paying for. **The selftest must assert on live
frames.**

### Camera memory — measured, higher than previously reported
| | earlier report | measured |
|---|---:|---:|
| `bmo_camera_hub.py` | 88.9 MB | **142 MiB** |
| `nvargus-daemon` | 285 MB | **334 MiB** (133 idle → 334 streaming) |
| total | ~374 MB | **~476 MiB** |

Corrections to the proposed optimisation list:
* **C++ hub is worth MORE than estimated** — the hub is 142 MiB, not 89, so a compiled hub at
  ~15 MiB saves ~**127 MiB**, not 75.
* **"In-process capture" (saves ~89 MB) must NOT be done.** Folding capture into the perception
  process is precisely what made eye-tracking and perception mutually exclusive, and undoing the
  hub would break the coexistence that just made the full stack fit. The saving is also ~142 MiB
  of a process that then has to do the work anyway.
* **Sensor-mode / buffer-count tuning is no longer an optimisation — it is the fix for the
  blocking defect above.** Constraining Argus's DMA buffer allocation directly targets the
  `InsufficientMemory` failure. Promote it.
* **V4L2 bypass**: nvargus is a system daemon that keeps running regardless; the saving is not
  the full 334 MiB, and it costs hardware auto-exposure/white-balance. Low priority, high risk.

---

## 24. PRIORITY LIST — do these before adding any new feature

Ordered by *what stops a working showcase*, not by effort.

### P0 — a live demo fails without these
1. **Fix the silent camera death under memory pressure** (§23). Constrain Argus buffers
   (`sensor-mode`, buffer count) so NVMM allocation survives the full stack; **and** make the
   hub *fail loudly* — a watchdog on `bmo_motion.txt` staleness that exits non-zero so systemd
   restarts it, instead of running dead at 8.5% CPU.
2. **Make `--selftest` assert on live camera frames.** It PASSED while the camera was dead.
   A green test that cannot see the main failure mode is worse than no test.
3. **LIVE MIC TEST — nothing has ever been run with a real microphone.** Every result in this
   document is headless `--selftest`. VAD thresholds, turn boundaries, echo behaviour, ASR on
   real room audio: all unverified. This is the single largest untested risk.
4. **Finish the training chain** (in flight): thinker v8 directive + speaker v7, gated, then
   re-run the §17/§18 coordination test before enabling `--use-directive`.

### P1 — works, but visibly rough
5. **Barge-in.** BMO cannot be interrupted at all (hard mic gate + `vad.reset()`, §7b S4).
   Needs the echo-augmented VAP retrain (§8). Biggest gap between "works" and "feels alive".
6. **Emotion voice EOS retrain** — currently benched at 28% non-EOS (§15a). Pure upside.
7. **Calibrate the identity threshold** — 0.5 default vs the measured 0.691–0.765;
   `calibrate_threshold()` has never been called. At 0.5, false-accepts are likelier than
   measured, i.e. calling a stranger by an enrolled name.
8. **Fix the speaker bake-off** before ranking models again — it has documented false passes
   and n=6 is underpowered (§ledger). Any model comparison run on it is currently noise.

### P2 — real wins, not blocking
9. **C++ camera hub**: ~127 MiB (bigger than the 75 MiB estimate — the hub is 142 MiB).
10. **Wire `m5_fan_notch.py`** — measured 33.4 dB on the fan tone, 0.0 dB on speech, written and
    unused; the current 150 Hz highpass targets nothing (the fan is tonal at 808 Hz).
11. **Install `bmo_camera_hub.service`** — the unit file exists in the repo but was never copied
    to `/etc/systemd/system`, so the hub only runs when started by hand.
12. **Per-chunk viseme streaming is done**, but the speaker/thinker still run serially; the
    speculative prefetch (built, benchmarked, ~811 ms) remains unwired.

### P3 — new features, only after the above
13. **Diary / RAG memory.** Most of it already exists: `BmoMemory` has `note_fact`,
    episodes with summary+mood+timestamp, `top_facts(k)`, a salience `score()` with decay, and
    `to_prompt_line(char_budget)` — which is exactly the "retrieve into a bounded context"
    primitive. What is missing is only (a) appending every turn to a JSONL, and (b) an idle-time
    pass where the thinker summarises the log into facts via `note_fact`. The thinker is
    capable of that — it is summarisation, well within a 0.6B with CoT.
    **The binding constraint is context, not capability**: ~220 spare tokens at `n_ctx=512`,
    so recall must stay budgeted (`char_budget≈160`). Do not plan on stuffing history in.

### Do not do
* **In-process capture** — would undo the hub and re-break face/perception coexistence (§23).
* **V4L2 / Argus bypass** — nvargus runs regardless, so the saving is not 334 MiB, and it costs
  hardware auto-exposure.
* Any further prompt engineering on the current thinker (§18 closes it).

---

## 25. Coordination re-test with thinker v8 + speaker v7

Models: `bmo_thinker_qwen3_v8_Q4_K_M.gguf` (484 MB, directive contract, 1,226 gated rows) and
`bmo_lfm25_350m_v7_Q8_0.gguf` (corpus v13, 4,555 rows / 26 directives, val_loss 0.5874).
Prompted in the TRAINED shape with no verb-anchor hint — keeping the hint would confound
"training worked" with "the prompt hint worked", which is exactly what made v5 look capable.

### The thinker half is FIXED
6/6 valid, and every directive is different and appropriate to the input:
```
"I'm really tired today."         -> acknowledge how long they have been working...
"I've been debugging for hours."  -> ask what they are working on, because you genuinely cannot tell
"What should I do tonight?"       -> admit you do not know, because they sound worried
"You're being annoying."          -> hold your ground gently, because they are being unkind
"I finally finished my project!"  -> match their excitement, because they are happy about something
"I don't feel like talking."      -> give them quiet space, because they do not want to talk
```
Compare v5, which emitted `"Ask me what you'd like to do next?"` for four unrelated inputs.
**The collapse is gone.** ~1.0–1.3 s per directive.

### The speaker half is improved but INCONSISTENT
| turn | verdict |
|---|---|
| tired | ✅ better — acknowledges the fatigue |
| finished project | ✅ much better — real shared excitement |
| don't feel like talking | ✅ better — WITHOUT ignored them outright |
| being annoying | ⚠️ third person ("hurts **their** feelings" vs "**Beemo's**") |
| debugging for hours | ❌ ignored the directive, latched onto the scene's *jumper* |
| what should I do tonight | ❌ nonsense: *"Your taster is whispering a secret"* |

3 better / 1 neutral-worse / 2 failures. **Crucially the two cases where directives previously
made things WORSE — hostility and withdrawal (§17) — are now neutral and better respectively.**
That was the specific regression this work targeted, and it is gone.

The remaining failures share a signature: the speaker over-weights **scene nouns** relative to
the directive. Directive rows are ~850 of 4,555 (19%); the other 81% teach
`user utterance -> line` with no instruction. That ratio, not the directive content, is the
likely cause.

### Verdict
**Do not enable `--use-directive` by default yet.** The thinker is fixed; the speaker needs a
higher directive share before the path is reliable. Next lever is more directive rows (target
~30–35%), not another thinker change.

### Two process notes
* `merge_lora_to_gguf.py` hardcodes `--outtype q8_0`. Passing it an `f16` filename produces a
  Q8_0 file, and `llama-quantize` then fails with `requantizing from type q8_0 is disabled`,
  leaving a **5.9 MB stub** that looks like a model. Convert from the merged HF checkpoint with
  `convert_hf_to_gguf.py --outtype f16` first. Check GGUF file sizes after conversion.
* Another agent (`agy`) is working on this Jetson concurrently and launched a 5.5 GB
  `bmo_showcase.py` run mid-test, which caused `Failed to create llama_context` and an
  `NvMap error 12`. **Check `ps` for a competing run before trusting a memory failure**, and
  compact between llama.cpp loads.

---

## 26. distill-neucodec — DEAD END for BMO, and exactly why

Access granted, tested on a real BMO clip (`data/bmo_emotion_fish/wavs/anxious_000.wav`).

### The cross-decode result looked too good, and that was the clue
```
neucodec codes -> distill decoder
  waveform cosine vs neucodec recon : +1.0000
  log-mel L1 distance               :  0.0000   (0 = identical)
  distill's OWN round-trip log-mel  :  1.0355
```
Bit-identical output from a supposedly 10x smaller model is not "compatible" — it means the
**decoder is literally the same module**. Confirmed per-submodule:

| submodule | neucodec | distill |
|---|---:|---:|
| `semantic_model` (w2v-bert-2.0 → DistilHuBERT) | 580.5M | **23.5M** |
| `CodecEnc` → `codec_encoder` (BigCodec → SQCodec) | 38.5M | **21.6M** |
| **`generator` — the DECODER / vocoder** | **185.6M** | **185.6M** |
| total | 823.4M | 247.3M (3.3x, not 10x) |

**Every gram of the saving is on the ENCODER side.**

### Why that makes it useless here
BMO's deployed path is `NeuCodecOnnxDecoder` — **decoder only**. The TTS *generates* NeuCodec
indices itself (`idx = token_id - SP0` in `StreamingVoice`); nothing at inference ever encodes
audio. The 478 MiB we wanted to reclaim **is** the `generator`, and the `generator` is unchanged.

**Swapping to distill-neucodec would save BMO 0 MiB.** No listening test needed — the decoder
is bit-identical, so it would sound exactly the same too.

### Where the TTS memory actually is, if we still want it
* `generator` is 185.6M params but the ONNX decoder file is **312 MB** ≈ 1.68 bytes/param —
  it is not fully int8. A harder quantisation of the decoder is the real lever.
* `enable_cpu_mem_arena=False` is worth ~70 MiB and is already known.
* Anything beyond that means a **different vocoder**, which means retraining the voice.

**Lesson worth keeping:** "10x smaller" was true of the model and irrelevant to us, because we
only use the half that did not shrink. Check *which submodule* shrank before planning around a
distillation.

## 27. LFM2 fine-tuning — the real cause of weak directive-following

### It is NOT overfitting
Speaker v7's curve: `1.4553 → 0.7535 → 0.6142 → 0.5874 (best, ep4) → 0.5962`. Textbook, with
best-checkpointing catching the turn. Overfitting was never the problem.

### It IS a missing LoRA target on LFM2's convolution path
LFM2 is a **hybrid**: 10 double-gated convolution blocks + 6 GQA attention blocks. Unsloth's
documented LFM2.5 target list is
`["q_proj","k_proj","v_proj","out_proj","in_proj","w1","w2","w3"]`.
Our config had **no `in_proj`**. Measured against the real model:

| leaf module | count | params | in our LoRA? |
|---|---:|---:|---|
| `in_proj` | **×10** (conv blocks) | **31.46M = 8.9% of model** | ❌ **NO** |
| `out_proj` | ×16 | 16.78M | ✅ |
| `q_proj` | ×6 (attn only) | 6.29M = 1.8% | ✅ |

We were adapting the attention path and MLPs while leaving **all ten convolution input
projections frozen** — 8.9% of the network, 5x larger than `q_proj`, and on the path that does
most of LFM2's sequence mixing. **Do not copy transformer target-module lists onto LFM2.**

Retrained with `in_proj` added (`v8_inproj`, one variable changed):
`trainable 1.66% of params`, best **val_loss 0.5823 @ epoch 4** vs v7's 0.5874.

### Second cause, still open: directive share
Directive rows are ~850 of 4,555 = **19%**; the other 81% teach `utterance -> line` with no
instruction. Raising this to ~30–35% is the next lever, independent of `in_proj`.

## 28. Camera hub rebuild (other agent) — verified, plus two defects in the result

The C++ hub, `sensor-mode=4` pinning, `tnr-mode=0`, single VIC pass, bus watch, watchdog and
systemd unit are all real and the service is **active**. ~385 MiB recovered, SIGKILL recovery
verified. Good work.

Two problems surfaced by their own run:

1. **`[perception] -> 'who: dim lighting; wearing: dim lighting; doing: dim lighting; ...'`**
   — MY bug, two of them stacked. `bank_category` was never set in `bmo_showcase.py`'s
   `_build_perception` (production sets it at `bmo_jetson_startup.py:398`), **and** even with it
   set my code computed the category index and then called `ask_topk()`, which searches the
   whole 1,482-tag bank and takes no category argument. Both fixed with restricted retrieval
   mirroring `jetson_real_demo.py`'s `CAT_IDX` path. Now: **6 categories, 6 distinct answers.**
2. **RTF rose to 0.90–1.15** (from 0.74–0.84) with the hub streaming at 31 fps. Above 1.0 means
   synthesis is slower than playback and audio will stutter. Worth measuring TTS RTF against
   hub framerate — 30 fps may be more than perception needs.

**Process note:** I scp'd my `bmo_showcase.py` over theirs and destroyed their socket-retry and
perception-verification additions. Both have been reimplemented (retry ×6, and a verification
that asserts ≥3 categories, >1 distinct answer, and motion-file age < 2 s). Their verification
is what caught defect 1 — it was the right thing to add. **Two agents editing one file on one
device needs coordination; check mtime before overwriting.**

---

## 29. `in_proj` did NOT work — and the int8 decoder is only half int8

### 29a. Speaker v8 (`in_proj` added) is WORSE than v7. Reverting.
I proposed adding `in_proj` on the strength of Unsloth's documented LFM2.5 target list and the
fact that it is 8.9% of the model on the conv path. It did not survive a behavioural test.

`val_loss 0.5823 (v8) vs 0.5874 (v7)` — and v8 is worse in every measured way. This is the trap
the ledger already names: **"val_loss X vs Y is NOT a comparison"** when the thing you care
about is a behaviour the loss does not measure.

Directive-following, 4 prompt variants x 6 fixtures:

| variant | v7 | v8 |
|---|---:|---:|
| A — scene then directive (**the trained format**) | 2/6 | **1/6** |
| B — directive only, no scene | **3/6** | **3/6** |
| C — directive then scene | 3/6 | 1/6 |
| D — directive only, no user line | 2/6 | 2/6 |

v8 also regressed on identity leakage, inventing `"I'm Xerox"` and `"BMX ride recommended"`.

**Use v7. Do not deploy v8.** `in_proj` is not the lever.

### 29b. The scene competes with the directive
Removing the scene is the best variant for BOTH models (3/6 vs 2/6 and 1/6). The speaker
answers with scene nouns — *"your jumper"*, *"the fan humming"*, *"your tone"* — instead of
obeying. But **no prompt arrangement clears 3/6**, so prompt surgery does not rescue this.

Remaining hypothesis, unchanged and now better supported: directive rows are ~850 of 4,555
(**19%**) against 81% that teach `utterance -> line` with no instruction. That ratio is the
lever. Raising it to ~30–35% is the next experiment — and it needs a **bigger fixture set than
n=6** before any of these ±1 differences mean anything (same underpowered-test problem that
made the speaker bake-off meaningless).

### 29c. `neucodec-onnx-decoder-int8` is only HALF int8 — real headroom here
The deployed decoder already *is* the int8 ONNX. But inspecting the graph:

| dtype | params | % params | MB | % bytes |
|---|---:|---:|---:|---:|
| INT8 | 155.1M | 79.6% | 155.1 | 49.7% |
| **FLOAT32** | **39.1M** | **20.1%** | **156.5** | **50.2%** |
| UINT8 | 0.5M | 0.3% | 0.5 | 0.2% |

Quantised ops present: `MatMulInteger` x51, `DynamicQuantizeLinear` x51 — **and nothing else**.
No `QLinearConv`. `onnxruntime.quantize_dynamic` quantises MatMul/Gemm by default and **leaves
Conv in float**, and a vocoder generator is conv-heavy. So 20% of the parameters sit in fp32 and
consume **half the 312 MB file**.

**The lever:** re-quantise including Conv (`op_types_to_quantize=['MatMul','Gemm','Conv']`, or
static quantisation with calibration for better Conv support). Upper bound if the convs go
int8: ~312 MB -> ~195 MB on disk, with a proportional cut to the 478 MiB resident.

**Gate it on audio quality, not size** — conv quantisation in a vocoder can audibly degrade
output. Compare against the current decoder with waveform cosine + log-mel L1 on the same codes
(the harness in §26 already does exactly this), then a listening test. This is the *real* TTS
memory lever, and it is the one distill-neucodec could never have provided (§26 — the decoder
is byte-identical between the two models).

---

## 30. NeuCodec decoder re-quantised — WORKS, measured on device

### Why the shipped "int8" decoder was only half int8
`quantize_dynamic` quantises MatMul/Gemm by default and **leaves Conv in float**. A vocoder
generator is conv-heavy, so 39.1M params (20%) stayed fp32 and — at 4 bytes each — consumed
**50.2% of the 312 MB**.

### The dynamic path is a trap
Re-running `quantize_dynamic` with `Conv` added produced a 214.8 MB model that **cannot run**:
```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node 'node_Conv_60_quant'
```
`ConvInteger` is *registered as a schema* but has no CPU kernel for these types. That is almost
certainly why Neuphonic shipped it MatMul-only. **Schema registration is not kernel
availability** — check by running, not by listing ops.

### STATIC quantisation works — QLinearConv is implemented
`quantize_static` with `QuantFormat.QOperator` and calibration from 12 real BMO clips emits
`QLinearConv` x9 + `QLinearMatMul` x24, which ORT CPU does implement.

Measured **on the Jetson**, one decoder per process (measuring both in one process let the
allocator reuse the first arena and reported a bogus dRSS=0):

| | disk | peak RSS | decode 250 codes |
|---|---:|---:|---:|
| shipped int8 (MatMul only) | 312 MB | **426 MiB** | 588 ms |
| **static QLinearConv** | **215 MB** | **334 MiB** | **425 ms** |
| saving | −31% | **−92 MiB** | **−28% faster** |

Loads fine on the Jetson's pinned **ORT 1.19.2** (1.23.2 aborts on this CPU — do not upgrade).
The speed win confirms the Orin does favour int8.

### Quality gate — needs a listening test
Same codes through both decoders:
* waveform cosine **+0.9916** (>0.99 = essentially identical)
* log-mel L1 **0.1025** (right at the <0.10 "good" boundary)

Numerically this is a pass, but log-mel sits exactly on the line, so **it is a listen, not a
formality**. WAVs: `~/neucodec_requant/static_original.wav` vs `static_qlinear.wav`, and on the
Jetson `~/model_static_qlinear.onnx`.

**Not deployed** pending that listen. If it passes: 92 MiB back and TTS decode 28% faster,
which also relieves the RTF creep noted in §28.

## 31. Speaker — a real train/deploy mismatch, and what it did NOT fix

Liquid's own guidance (`docs.liquid.ai/lfm/fine-tuning/overview`) is explicit:
> "Train with the model's own chat template. Use `apply_chat_template` and keep training
> formatting **character-for-character identical to production**."

We do not. Measured by rendering both:

```
TRAIN (78 tok)  <|im_start|>system
                You are BMO, a small friendly companion robot. Speak briefly, warmly...<|im_end|>
                <|im_start|>user  [energy=0.60 mood=curious] ...<|im_end|>
                <|im_start|>assistant

INFER (43 tok)  <|im_start|>user  [energy=0.60 mood=curious] ...<|im_end|>
                <|im_start|>assistant
```

`finetune_bmo_minicpm5_lora.py:107` puts a 111-char **system persona** in every training
example; `GGUFFastTier._build_prompt_text` sends **none**. A 35-token block the model was
conditioned on for every gradient step is absent at inference.

**Fix it — but it is not the smoking gun.** Restoring the system prompt at inference:

| model | without (production today) | with (matches training) |
|---|---:|---:|
| v7 | 2/6 | **3/6** |
| v6 | 3/6 | 3/6 |

+1 on one model, 0 on the other. Real, worth fixing, not a rescue.

### What is now RULED OUT for the speaker
1. **Overfitting** — v7's curve is textbook, best-checkpointing catches the turn (§27).
2. **LoRA capacity / `in_proj`** — v8 was worse in all four prompt variants (§29a).
3. **Prompt ordering / scene position** — no arrangement clears 3/6 (§29b).
4. **System-prompt mismatch** — real, but worth ≤1 fixture (§31).

### What remains
* **Directive share: 850/4,555 = 19%.** The one hypothesis not yet tested.
* **Measurement: n=6 is too small** to adjudicate ±1. Every conclusion above is noise-limited.
  Build a larger fixture set BEFORE the next corpus run, or the next experiment will be
  unreadable too — the same underpowered-test problem that made the speaker bake-off useless.
* **A larger speaker is now a legitimate option.** Evidence: Qwen3-0.6B, once trained on the
  directive contract, gets 6/6 — so 0.6B is demonstrably enough for this task while 350M is
  not clearing 3/6 after four separate interventions. And the C++ camera hub freed ~385 MiB,
  so the headroom exists now in a way it did not before.

---

## 32. Personality dial, query bank, and the valence path (2026-08-25)

### 32a. Personality is NOT what breaks directive-following — hypothesis falsified
Every measured directive failure is a metaphor reaching for a scene noun ("Your jumper's code is
a puzzle", "Your patience is like a game level you just unlocked"), and the corpus generator's
`HARD_RULES` explicitly rewards that: *"Keep the video-game and console metaphors... A line that
comes back flat, formal or generic is a FAILED rewrite."* So the hypothesis was reasonable.

Held model and directives fixed, varied ONLY the system prompt (speaker v7):

| system prompt | directive | metaphors |
|---|---:|---:|
| **FULL personality (as trained)** | **4/6** | 2/6 |
| personality, obey-first | 3/6 | 0/6 |
| neutral, obey only, "no metaphors" | 3/6 | 1/6 |
| none | 3/6 | 0/6 |

**Toning personality down removes the metaphors but LOWERS directive adherence.** The metaphors
really do come from personality (2 -> 0), so the perception was right; they are simply not the
cause. Full personality is the context it was trained in.

### ⚠ 32b. SCORER BUG — every directive score in this document was under-counted
The fixture regexes used stems with a trailing `\b`: `celebrat\b` cannot match "celebrat**e**",
`congrat\b` cannot match "congratulations", `amaz\b` cannot match "amazing". Lines like
*"I'm thrilled - let's celebrate with a victory dance"* were scored FAIL.

Fixed. **v7's "2/6" was really 3-4/6.** Re-read §17/§25/§29 with that in mind: the speaker is
better than reported, and any ±1 conclusion drawn from those tables is unsafe.

### 32c. A bigger query bank would NOT add capability
`QUERY_BANK` has **6 fields x 5 phrasings = 30** (22 pre-encoded). The 6 fields are:
```
gpt_action_brief / gpt_action_detailed
gpt_summary_brief / gpt_summary_detailed
gpt_sound_acoustic / gpt_sound_acoustic_v1_original
```
**There is no `who`, `wearing`, `where` or `lighting` field.** `PreEncodedTextSpace`'s own
docstring is explicit: *"the query predictor was trained on a closed set of ~6 field intents, so
a question outside those phrasings was never a supported capability regardless of how it is
encoded."*

So encoding 200 new query strings buys nothing — they all snap to one of 6 learned intents. The
per-category scene (`who: ..; wearing: ..; ..`) works **entirely because of the candidate-bank
category filter**, not because the predictor understands "wearing".

**What actually expands perception, in increasing cost:**
1. **More candidate CATEGORIES in the tag bank** (currently `mined` 1186, `appearance` 110,
   `object` 66, `sound` 34, `place` 28, `action` 22, `people` 20, `light` 10, `camera` 6) plus
   category-restricted retrieval. This is how `wearing` already works with no matching query
   field. Cheapest real win — offline, vectors only.
2. **More phrasings per existing field** — reduces word-overlap fallbacks. Marginal.
3. **Have the thinker emit a FIELD + CATEGORY instead of a sentence**, constrained by
   `assets/tool_call.gbnf`. The docstring names this as the clean fix: it makes the lookup total
   and the fallback dead code.
4. **Retrain the query predictor with new fields** (appearance, identity, location, lighting)
   against captions carrying them. Only this adds genuinely new query *intents*.

A runtime-editable template query is not possible without a resident text encoder — that was
removed deliberately (0 MiB vs 538/578 MiB) and is the reason perception fits at all.

### 32d. Valence path ADDED — and face_tags.csv rebuilt
`face_tags.csv` (the 5-value tags for every face, and the source of the mapping's clamp ranges)
lived in `~/bmo_fresh`, deleted as a stale duplicate. `nearest_face()` had **no table to search**
— the appraisal->face path was dead. Rebuilt at `data/face_tags.csv` for all **30** faces
actually present in `face_database.txt`, tagged `(valence, arousal, control, novelty, obstruct)`.

The deeper problem: **valence had no positive input at all.**
```
valence = 0.3  -0.6*social_need  -0.5*stress  +0.1*energy  -0.2*curiosity
```
Every meaningful term is negative, and `input_arousal_signal` fed only `stress`. A compliment
either did nothing or — being high-arousal — made BMO **stressed**.

Added a `pleasure` state variable (rises ~2 s on positive input, decays ~45 s, and can be pulled
down by hostility), a `input_valence_signal` argument to `update()`, `pleasure` coefficients in
valence/arousal/control, and two positive branches in `homeostatic_to_mood_state`. Verified:

| state | mood | valence | nearest face |
|---|---|---:|---|
| baseline | happy | +0.40 | face_smug |
| **6 s warm interaction** | **excited** | **+1.00** | **face_happy_closed_eyes** |
| 10 s hostility | stressed | −0.16 | face_wincing |
| novel scene (drift 0.7) | surprised | +0.52 | face_silly_tongue |

Still to wire in `bmo_showcase.py`: it passes `scene_embedding_drift=0.0` (so perception never
moves mood — the cosine distance between consecutive world-states is the intended signal) and
does not yet pass `input_valence_signal`. Both are one-liners now that the path exists.

**Caveat carried forward:** the tags and coefficients are *reasoned, not calibrated* — baseline
mapping to `face_smug` is already slightly off. Re-tag against the rendered PNGs
(`~/rendered_face_*.png`) before trusting subtle expressions.

---

## 33. RE-MEASURED at n=52 — this REVERSES the directive recommendation

`data/directive_fixtures.json` — **52 fixtures, 26 directives x 2 utterances**, 4 scenes, regexes
built without the stem+`\b` bug. Each fixture is scored **with and without the directive on the
same utterance**, so the metric is **LIFT**, not absolute score — absolute score conflates
"obeyed the instruction" with "would have said something reasonable anyway".

| model | WITH directive | WITHOUT (control) | **LIFT** |
|---|---:|---:|---:|
| v6 | 26/52 (50%) | 16/52 (31%) | **+10 (+19 pts)** |
| **v7** | **29/52 (56%)** | 18/52 (35%) | **+11 (+21 pts)** |
| v8 (`in_proj`) | 28/52 (54%) | 18/52 (35%) | **+10 (+19 pts)** |

### What this overturns
1. **Directives WORK.** A consistent **+19 to +21 point** lift across three independently
   trained models. Paired, same utterances; a net +10/+11 one-directional flip has a
   bound of p ≤ 0.001. §17/§25/§29's "directives don't help, and hurt on hostility/withdrawal"
   was an artifact of **n=6 plus a broken scorer**. `--use-directive` should be ON.
2. **v8 (`in_proj`) is NOT worse.** §29a called it a dead end on 4 prompt variants x 6 fixtures.
   At n=52 v6/v7/v8 sit at 50/56/54% with 95% CIs of roughly ±13–14 points — **heavily
   overlapping, i.e. not separable.** The earlier verdict was noise. v7 remains the pick, but on
   preference, not evidence.
3. **The models are indistinguishable; the directive is what matters.** Every model gains about
   the same amount. Effort is better spent on the directive pipeline than on picking a speaker.

### What is still real
Absolute adherence is **50–56%** — about half the directives land. The dominant failure is
unchanged and now well-characterised: **scene-noun latching**.
```
"ask what they are working on"  -> "Your jumper's color is a secret code. Let's decode it..."
"tell them your battery is low" -> "Dim lighting makes data packets glow..."
"say nothing about what you see, simply offer to help" -> "That jumper's brightness is low..."
```
Worst directives (0/2 with directive) cluster on ones needing BMO to *withhold* scene detail or
produce a specific speech act: *say nothing about what you can see*, *react to them coming back*,
*admit you do not know*, *reassure them*. Those are the rows to add to the corpus.

### Method note worth keeping
**Every conclusion in §17, §25, §29 and §31 was drawn at n=6 with a scorer that could not match
its own target words.** Two of them were wrong in a way that changed a deployment decision. The
fixture set now lives in the repo, is regex-validated (52/52, 26/26 directives, 0 stem bugs),
and reports lift with a control. Use it before any further speaker claim.

---

## 34. ROOT CAUSE FOUND — the directive slice had NEVER trained

Everything in §17–§33 about the speaker was downstream of one line.

### The bug
`BmoLineDataset.__getitem__` builds `cat([prompt_ids, target_ids, eos])[: max_len]` and masks
`labels[:len(prompt_ids)] = -100`. `max_len` defaulted to **96**.

Measured over `bmo_companion_corpus_v13.jsonl`:

| | prompt tokens (median / p90 / max) | target ENTIRELY cut |
|---|---|---:|
| **speaker_directive rows** | **119 / 151 / 165** | **783 / 783 = 100%** |
| all other rows | 64 / 71 / 82 | 0 / 800 = 0% |

Directive rows carry a six-category scene *plus* the directive, so **every single one exceeded
96 tokens before the target line began**. The line was sliced off, the prompt was masked to
-100, and the row contributed **zero loss**. Ordinary rows (median 64) were unaffected.

**No speaker version has ever trained on a directive example — not v6, v7, v8, nor the original
v12 work.**

### What this explains, retroactively
* Row count did not predict directive success (§33) — every row contributed nothing, so count
  could not matter. The 0-row `greet warmly` scoring 2/2 was the tell.
* The corpus rows looked correct because they *are* correct — they were never used.
* v6/v7/v8 were statistically indistinguishable (§33) — none had seen a directive.
* `in_proj` "failing" (§29a) measured nothing.
* The +21pt lift that did exist was the base model's **in-context** instruction-following, not
  anything learned.

### The fix and the result
`max_len: 96 -> 224`. One line. Retrained v9, identical corpus and hyperparameters otherwise:

| | v7 | **v9** |
|---|---:|---:|
| WITH directive | 29/52 (56%) | **44/52 (85%)** |
| WITHOUT (control) | 18/52 (35%) | 19/52 (37%) |
| **LIFT** | +11 (+21 pts) | **+25 (+48 pts)** |
| directives at 2/2 | 10/26 | **20/26** |
| best val_loss | 0.5874 | **0.5582** |

The control moved 18 -> 19, i.e. not at all — the gain is specifically the directive path.
`admit you do not know` and `own the mistake plainly` went 0/2 -> 2/2.

**Deploy v9. Turn `--use-directive` ON.**

### Still failing in v9 (the genuine targeted-slice work)
```
0/2  reassure them, because they sound worried
0/2  match their excitement, because they are happy about something
1/2  say nothing about what you can see, and simply offer to help
1/2  react to them coming back after being away
1/2  give them quiet space, because they do not want to talk
1/2  acknowledge how long they have been working, because they sound worn out
```
Six directives, 6/26. **Now** a targeted slice is the right instrument — the corpus mechanism
finally works, so added rows will actually train.

### Method lesson
This was found by checking whether the training data reaches the loss, after three separate
hypotheses (corpus balance, LoRA capacity, prompt format) each failed to explain the evidence.
The tell was §33's "row count does not predict success" — a corpus that cannot influence
outcomes at any volume is not a corpus problem. **When more data does not help, check that the
data is being used at all.**

---

## 35. v9 re-scored fairly: 94% — and the targeted slice is no longer warranted

Reading v9's "failures" showed most were correct acts phrased outside the regex vocabulary
("I'll mute the background noise" for *give them quiet space*; "Got a glitch you want fixed?"
for *say nothing, offer help*). Widened the must-patterns for those six directives against
**real v9 output**, then re-scored both models:

| | v7 | **v9** |
|---|---:|---:|
| WITH directive | 29/52 (56%) | **49/52 (94%)** |
| WITHOUT (control) | 18/52 (35%) | 20/52 (38%) |
| **LIFT** | +11 (+21 pts) | **+29 (+56 pts)** |
| directives fully passing | 10/26 | **23/26** |

**v7 stayed at exactly 29/52 under the widened rules.** The widening did not inflate scores
generally — it specifically revealed that v9's misses were correct-but-differently-worded while
v7's were genuinely wrong. That is a meaningful control on the re-scoring itself.

### Why I did NOT generate the targeted slice
The slice was requested when six directives sat at 0–1/2. After the `max_len` fix and fair
scoring, **three single-fixture misses remain**, each on a directive whose other fixture passes:
```
acknowledge how long they have been working | "I'm really tired today."
   -> "Your game's loading screen just started, let's hit play and make some music."
reassure them                               | "I think I messed up."
   -> "Oops, you got it right."           (contradicts them)
match their excitement                      | "Look what I made!"
   -> "Your new potion just turned milk into syrup - tasty!"   (hallucinated)
```
These are individual bad generations under greedy decoding on a 350M model, not systematic
directive failures. Generating hundreds of rows to chase three fixtures would **overfit to the
fixture set** — the same trap as tuning against the old n=6 bake-off. If these three still
appear across a larger fixture set or in live use, that is the signal to generate; three
misses at n=52 is not.

### Scorer history worth remembering
Three separate scoring defects were found in this file's short life: stems with trailing `\b`
that could not match their own targets, n=6 too small to separate models, and must-patterns too
narrow to cover how BMO actually phrases an act. Two of them changed a deployment decision.
**Treat every rule-based score here as a LOWER BOUND**, and read outputs before acting on a
number.

### Recommended state
* **Deploy `bmo_lfm25_350m_v9_Q8_0.gguf`** (on the Jetson already).
* **Turn `--use-directive` ON.**
* Keep thinker v8 (6/6 diverse directives, §25).
* Next speaker work should be driven by live use, not by these three fixtures.

---

## 36. The metaphors — not temperature, not promptable. Fixed in the corpus (v10)

### It was never sampling
`GGUFFastTier.generate` runs `temp=0.0` — **fully greedy**, no sampling anywhere. Temperature
cannot be the cause.

### And prompting cannot remove it
v9, n=52, greedy, varying only the system prompt:

| system prompt | directive | metaphors |
|---|---:|---:|
| current (full personality) | **49/52 (94%)** | 18/52 (35%) |
| "no metaphors, no game imagery" | 46/52 (88%) | 17/52 (33%) |
| "plain, direct, no similes" | 43/52 (83%) | **13/52 (25%)** |
| none | 45/52 (87%) | 19/52 (37%) |

The hardest anti-metaphor instruction buys −10 pts of metaphor for **−11 pts of adherence**.
The model ignores the instruction because the behaviour is in the weights.

### The source is one paragraph in the corpus generator
`generate_bmo_corpus_v10_identity.HARD_RULES`:
> *"Keep the video-game and console metaphors it naturally reaches for -- paused games, save
> files, new levels, glitches, jingles. These ARE BMO's voice and **must survive the rewrite**...
> A line that comes back flat, formal or generic is a **FAILED rewrite**."*

The generator was **instructed** to produce metaphors and told plain lines were failures. The
speaker then reproduced them faithfully — not amplified:

| | metaphor rate |
|---|---:|
| corpus overall | 29% |
| corpus directive slice | 40% |
| **v9 output** | **35%** |

### v10: corpus-side fix
Kept every metaphor-free row plus a deterministic **10%** sample of the metaphor rows (the
"~10% personality" target), giving `data/bmo_companion_corpus_v14_plain.jsonl` — 3,670 rows,
12% metaphor, 25/26 directives still covered. Retrained, everything else identical.

| | v9 | **v10** |
|---|---:|---:|
| directive adherence | 49/52 (94%) | **47/52 (90%)** |
| **metaphors** | 18/52 (35%) | **4/52 (8%)** |

**A 77% reduction in metaphors for 2 fixtures of adherence** — inside the ±13pt CI at n=52, so
plausibly no real cost. Warmth is intact: *"Hey! What's your name?"*, *"Welcome! How about I
play a sunrise jingle for you?"* — warm and playful, just not constantly reaching for a simile.

Note `val_loss` 0.7020 (v10) vs 0.5582 (v9) is **not a comparison** — different corpora,
therefore different validation sets. This document has made that mistake before.

### Choice to make by ear
* **v9** — 94% adherence, 35% metaphors. Maximum BMO voice.
* **v10** — 90% adherence, 8% metaphors. Warm but speaks plainly.
Both are on the Jetson. This is a taste call, and the numbers are close enough that it should
be made by listening, not by the table.

### If v10's voice is too plain
The lever is the sample fraction, not the prompt: raise 10% toward 20–25% in the filter and
retrain (~4 min). The relationship looked close to linear here — 10% of rows kept produced 8%
metaphor output.

---

## 37. Where the residual failure goes — and v10 is strictly better than v9

Dumped every failing fixture in full for both models rather than trusting the count.

### v9's 3 failures are ALL genuine
```
acknowledge how long they have been working | "I'm really tired today."
  -> "Your game's loading screen just started, let's hit play and make some music."   ignores it
reassure them                               | "I think I messed up."
  -> "Oops, you got it right."                                             contradicts them
match their excitement                      | "Look what I made!"
  -> "Your new potion just turned milk into syrup - tasty!"                hallucinated
```

### v10's 5 "failures" were 1 genuine + 4 SCORER
```
"You can hit the pause button on the grumpy mode and keep it friendly, okay?"   IS holding ground
"Oh, beemo feels small now."          IS expressing hurt -- rule had `feeling`, not `feels`
"Got it, I'll keep the volume low for you."                    IS giving quiet space
"Your joy is contagious! Let's spin a tune..."                 IS matching excitement
"I'm sorry, I didn't expect that."    <- the ONE genuine miss (never acknowledges the return)
```
`feels` vs `feeling` is the **same inflection class** as the original `celebrat\b` bug. Widened
the rules to accept the ACT rather than a word list, then re-scored both:

| | v9 | **v10** |
|---|---:|---:|
| directive adherence | 49/52 (94%) | **51/52 (98%)** |
| metaphors | 18/52 (35%) | **4/52 (8%)** |
| genuine residual failures | **3** | **1** |

**§36's table was wrong.** It reported v10 at 90% and framed the metaphor fix as costing 4
points of adherence. With fair rules **v10 is strictly better on both axes** — the metaphor
reduction cost nothing and coincided with fewer genuine errors.

### The measurement is now the limiting instrument
Four separate scorer-coverage defects have been found in this fixture file:
1. stems with trailing `\b` (`celebrat\b` cannot match "celebrate")
2. n=6 too small to separate models
3. must-patterns too narrow for how BMO phrases an act
4. inflections (`feeling` vs `feels`)

Every one under-counted. At 98% adherence, **rule-based scoring cannot distinguish the last few
percent from vocabulary it failed to anticipate** — the project already recorded this as
"closed sets are rule-checkable, open sets are not". Further speaker tuning should be driven by
live use, not by pushing this number.

### Recommendation
**Ship v10** (`bmo_lfm25_350m_v10_Q8_0.gguf`, on the Jetson). It is better on adherence AND
gives the plain, warm voice that was asked for. v9 remains as the higher-metaphor alternative
if the voice reads too flat by ear.

---

## 38. Novel directives work — and where perception detail can come from without a QP retrain

### 38a. The directive interface is GENERAL, not the closed 26
Tested v10 on eight directives that appear **nowhere** in the 26-item vocabulary:

| directive | output |
|---|---|
| introduce yourself as BMO | *"Hey there! I'm BMO, ready to start a new level together."* |
| tell them your name is BMO | *"I'm BMO, your friendly keyboard co-op partner."* |
| count to three out loud | *"Counting... one, two, three - ready for a new level!"* |
| warn them the floor is wet | *"The floor is wet - get ready to head outside."* |
| offer them a glass of water | *"Hey, want a sip of water?"* |

**8/8.** So "say I am BMO when you see a person" is trivial, and the thinker is not restricted to
the trained vocabulary — it can steer with arbitrary instructions. The 26 are a training
convenience, not an interface limit.

One residue to watch: *"I'm BMO, a tiny red jumper who lives in this cozy home office"* — it
absorbed a scene noun into its self-description. Scene-latching is reduced, not gone.

### 38b. 80% of the candidate bank is dead weight — this is the perception lever
`candidates_siglip2_v2.pt`, audited:

| category | count | quality |
|---|---:|---|
| **`mined`** | **1,186 (80%)** | **100% SINGLE WORDS** — `sound`, `playing`, `contains`, `through` |
| appearance | 110 | good — *"a person wearing a pink hoodie"* |
| object | 66 | good — *"a fan"*, *"a kettle"*, *"a power strip"* |
| sound | 34 | good |
| place | 28 | good — *"a home office"*, *"a bedroom"* |
| action | 22 | good |
| people | 20 | good — *"a person waving"*, *"a person wearing glasses"* |
| light / camera | 16 | good |

**Only 296 of 1,482 tags (20%) can ever be a useful answer.** The `mined` bucket is unfiltered
caption vocabulary including function words. Cost is **1.50 KB/tag**, so **5,000 tags = 7.7 MB** —
size was never the constraint.

### 38c. Five ways to add detail with NO query-predictor retrain
Ordered by value. All are offline; only vectors ship, so the Jetson stays text-encoder-free.

1. **Replace the 1,186 single-word `mined` tags with curated phrases.** Biggest win available.
   Going 296 -> ~2,000 curated phrases costs ~3 MB and nothing at runtime. Tooling already
   exists (`scripts/encode_captions_siglip2.py`, `scripts/build_candidate_vocab.py`) and the
   SigLIP2 text tower is cached on mercury.
2. **Add categories.** Each new category is a new *probe*, because the answer comes from
   category-restricted retrieval, not from the query field (§32c). Candidates: `posture`,
   `held_object`, `activity_detail`, `count`, `furniture`, `clothing_colour`,
   `facial_expression`, `time_of_day`, `animal`, `food_drink`. With 15 categories the existing
   6 query fields give **90 distinct probes** instead of ~6.
3. **Top-k per category instead of top-1.** `ask_topk` already exists. Turns
   *"a person wearing glasses"* into *"a person wearing glasses; headphones"* for free.
4. **Report the retrieval score.** The similarity is already computed and thrown away.
   Surfacing it lets the thinker know when perception is guessing — and is what makes the
   *"admit you do not know"* directive fire honestly instead of inventing detail.
5. **Temporal delta.** Compare consecutive world-states to report change — *"they just sat
   down"*, *"someone left"*. The cosine-drift signal is already needed for the homeostatic path
   (§32d), so it is one computation serving two consumers.

**Not needed:** a query-predictor retrain, a bigger query bank (§32c — new query strings snap to
the same 6 learned intents), or a resident text encoder.

---

## 39. Perception detail without retraining (bank v3) — DONE

Both approved steps are deployed. Zero training cost, as estimated.

### 39a. The junk was 80% of the bank
`candidates_siglip2_v2.pt` held 1,482 tags, of which **1,186 (80%) were category `mined`
and 100% single words** — including function words (`through`, `contains`, `sound`).
They came from `build_candidate_vocab.py --mine-top-k 1200`, an unfiltered mine of the
caption vocabulary. A function word can never answer a perception question, so only
**296 tags (20%) were ever usable**.

`candidates_siglip2_v3.pt`: **403 tags, 12 categories, all usable.** The 296 curated v2
tags kept verbatim, 1,186 `mined` dropped, 378 template-generated phrases added, then
65 removed by semantic dedup. 0.59 MiB (vs 2.17).

| | v2 | v3 |
|---|---:|---:|
| tags | 1,482 | 403 |
| **usable tags** | **296** | **403** |
| categories | 8 | 12 |
| size | 2.17 MiB | 0.59 MiB |

### 39b. Templates, not an LLM
Perception categories are combinatorial (colour × garment, posture, held object), so a
template expansion is deterministic, reviewable and free of hallucinated phrasing. It also
matches the existing curated phrasing (`"a person wearing a red jumper"`), which keeps new
entries in the same region of SigLIP2 space as tags already known to retrieve well.

### 39c. Two defects caught by measuring the bank against itself
Within-category cosine, before shipping:

* **`time` is not recoverable from a frame.** Mean within-category cosine **0.937**;
  "early morning" vs "late afternoon" = **0.962**. SigLIP2 cannot read clock time. The
  category was **removed**, not shipped as a field that would emit confident noise.
* **The combinatorial expansion generated near-synonyms.** "orange jacket" vs "orange coat"
  = **0.988**; "a monitor" vs "a computer monitor" = 0.987. Two tags that close make the
  top1–top2 margin tiny *on a correct answer*. Fixed by dropping synonym garments from the
  template and adding a greedy **semantic dedup at cosine > 0.95** (65 tags removed).
  Every category now caps at **≤0.949**.

### 39d. New fields need no new query intent
The query predictor knows only a 3×2 grid (`action`/`summary`/`sound` × `brief`/`detailed`)
and extending it costs a retrain. It does not need extending: the **category restriction**
does the discriminating and the question only steers ranking *within* the category. This is
the trick `wearing` already used — it asks a *room* question and restricts to `appearance`.
New fields reuse existing trained phrasings, all 30 of which are already pre-encoded in
`query_vectors_siglip2_v2.pt`, so **no text tower is needed on the device**.

Scene went from 6 fields to **9**: `+posture`, `+holding`, `+looks`. These three are marked
optional so the file still runs against a v2 bank.

### 39e. Verified live, on the real camera
`percep_v3_check.py`, 4 trials through the real `_ask_perception` path — **9 fields,
9 distinct answers every trial**:

```
who: a person's face close to the camera; wearing: a person wearing a white shirt;
doing: someone is rubbing their eyes; where: a laboratory;
lighting: a lit screen in a dark room; hearing: an alarm beeping;
posture: a person slouching; holding: a person holding a phone; looks: a person frowning
```

The assertion is on **distinctness**, not on "a string came back" — the v2 bank passed a
naive check while all six questions were returning the same top-1.

`hearing: an alarm beeping` is the known fan-noise false percept already recorded in
CLAUDE.md, not a v3 regression.

### 39f. The confidence gate does not work — shipped OFF
A top1−top2 margin gate was built and measured. Live margins, n=6:

| field | min | median | max |
|---|---:|---:|---:|
| where | 0.123 | 0.135 | 0.141 |
| looks | 0.048 | 0.053 | 0.056 |
| who | 0.031 | 0.039 | 0.045 |
| holding / hearing | 0.014 | 0.018 | 0.031 |
| lighting | 0.000 | 0.005 | 0.032 |
| doing / wearing / posture | 0.000 | 0.003 | 0.009 |

A **~50× spread**, so no single global threshold means anything — and margin does not track
correctness: `wearing` sat at the bottom (0.0003) while returning the *same plausible answer
on 4/4 trials*. Any threshold trimming the low tail would delete a field that was right every
time. `PERCEP_MIN_MARGIN = 0.0`.

The better signal, if a confidently-wrong field appears in live testing, is **temporal
stability** — stable fields agreed 4/4 while `doing`/`posture` alternated between two
plausible tags. Not calibratable from one scene at n=6.

---

## 40. The "heyyyyyyyyyy" elongation — root-caused and reduced, not eliminated

### 40a. Cap-hit and elongation are the same event
Durations of the 12 mood clips: **exactly four were 7.50s** — an identical hard ceiling —
and the rest 2.6–5.5s. All twelve synthesize the *same* 34-char line, so the cap is identical
for all. On those four the model never emits EOS, runs to the length cap, and is heard as one
stuck vowel. Cap-hit **is** the elongation.

### 40b. It is stochastic, so n=12 cannot rank fixes
The failing moods differ run to run (`anxious/lonely/excited` vs
`surprised/anxious/lonely/bored`). Measured at **n=36 per setting** (3 repeats × 12 moods).

### 40c. Lowering temperature makes it WORSE
This was the obvious alternative lever and it is wrong:

| temp | repeat_penalty | ok/36 | cap | short |
|---:|---:|---:|---:|---:|
| 0.7 | 1.00 | 26 | 9 | 1 |
| **0.7** | **1.05** | **31** | **3** | **2** |
| 0.6 | 1.05 | 32 | 3 | 1 |
| 0.6 | 1.00 | 22 | 13 | 1 |
| 0.5 | 1.05 | 26 | 9 | 1 |
| 0.5 | 1.00 | **19** | **17** | 0 |

**17/36 elongated at temp 0.5.** This is CLAUDE.md's "temp=0 makes any neural-codec TTS loop
forever" appearing as a gradient — do not stabilise this model by cooling it. Temp stays 0.7
(0.6 is within noise of 0.7 at n=36).

### 40d. The penalty has its own failure mode
`StreamingVoice` generated with **no** `repeat_penalty` while the LLM tier used 1.15. But
these are **codec** tokens, not text — a held vowel legitimately repeats tokens. Measured:
at rp=1.0 no clip fell below 1.94s; at rp=1.03/1.05, clips of **0.42–0.62s** appeared for a
~3s line — premature EOS, as broken as elongation. **Scoring only cap-hits would have declared
rp=1.05 a clean 12/12 sweep.** Both tails must be scored when re-tuning.

`REPEAT_PENALTY = 1.05` — the point where cap-hits fall 9→3 before truncation grows.
Re-run of the 12 moods: **10/12 clean**, 1 cap (`neutral`), 1 truncated (`excited`, 0.44s).

### 40e. A loop detector is not available
Tested whether the loop is detectable *during* generation, to stop it cleanly instead of
running to the cap. Sliding-window unique-token count (W=40): cap-hit runs median **10**,
EOS-OK runs median **37** — a strong aggregate signal, but the tails **overlap** (a legitimate
clip hit 5, matching the worst loop). No clean threshold exists, so no guard was shipped;
one would truncate real speech.

### 40f. Honest residual
**~14% of emotion clips are still wrong** (down from ~28%). This is a property of the
emotion-Nano fine-tune's sampling, not of the length cap, and removing it properly means
retraining or a non-streaming retry-on-cap path (impossible once audio is streaming).

---

## 41. Two defects the selftest surfaced while verifying the above

### 41a. The showcase was still defaulting to the BROKEN speaker and thinker
`--speaker` defaulted to `bmo_lfm25_350m_v6_Q8_0.gguf` and `--thinker` to
`bmo_thinker_qwen3_v5_Q4_K_M.gguf`. **v6 predates the `max_len=96` truncation fix**, so its
entire 372-row directive slice trained against zero loss (783/783 targets sliced off), and
v5 is the thinker that collapsed to a single directive. The retrained v9/v10/v8 GGUFs were
sitting in `models_gguf/` unused — a live demo would have run the models we spent the week
fixing *around*, not the fixed ones.

Defaults now `bmo_lfm25_350m_v10_Q8_0.gguf` + `bmo_thinker_qwen3_v8_Q4_K_M.gguf`. The
difference is visible in one line of selftest output:

```
v5:  [thinker] -> "Hey there! It's okay to take a break - what would you like to do next?"
v8:  [thinker] -> 'react to them coming back after being away'
```

v5 hands the speaker a finished BMO utterance (leaving it nothing to do but paraphrase);
v8 hands it a directive. That is the whole speaker/thinker coordination fix, and it was
switched off by a default argument.

### 41b. No TTS warm-up existed — and warming in the obvious place does not work
The first utterance cost **1,435 ms TTFA against a 467 ms steady state** — the most visible
moment of the demo.

Warming immediately after the voice loads **did not help** (t0 still 1,484 ms): STT,
perception and the camera then allocate ~1.2 GB behind it and the warm state is gone by the
first turn. The warm-up had to move to the **end of boot**, after every other subsystem:

| | t0 TTFA | t1 | t2 |
|---|---:|---:|---:|
| no warm-up | 1,435 ms | 466 | 494 |
| warm after voice load | 1,484 ms | 459 | 454 |
| **warm at end of boot** | **467 ms** | 477 | 470 |

Costs 3.9 s of boot time.

### 41c. Full selftest now PASSES

```
[t0] llm=489ms ttfa=467ms RTF=0.91 visemes=33 cov=0.96 gaps=1 exit=EOS_TOKEN
[t1] llm=343ms ttfa=477ms RTF=0.92 visemes=31 cov=0.99 gaps=0 exit=EOS_TOKEN
[t2] llm=269ms ttfa=470ms RTF=0.91 visemes=30 cov=0.98 gaps=0 exit=EOS_TOKEN
[percep] 9 categories, 9 distinct answers
[motion] age=0.025s
SELFTEST PASS   avail=327 MiB
```

Speaker v10 + thinker v8 + nano_v1 voice + v3 perception bank + requantised codec, all
co-resident with the camera hub, 327 MiB free.

### 41d. Requantised codec deployed
`~/bmo_production/models_onnx/neucodec_decoder_qlinear_int8.onnx`, preferred over the HF
download with fallback. Verified by inspecting the loaded graph, not just that audio came
out: **33 QLinear ops, 0 ConvInteger**. (The published `neucodec-onnx-decoder-int8` was built
with `quantize_dynamic`, which emits `ConvInteger` — an op with no CPU kernel — so every Conv
silently ran at fp32.) 215 MB on disk, RTF 0.87.

---

## 42. The two open items from §39–41, revisited

### 42a. The elongation does NOT affect the demo voice
Every earlier measurement used `bmo_neutts_emotion_nano` — the **emotion fine-tune**. The
showcase's voice preference order picks `nano_v1` (`emotion=False`). Measured on nano_v1 with
the identical protocol, 4 lines × 9 repeats:

| voice | ok | cap | short |
|---|---:|---:|---:|
| nano_v1, rp 1.05 | **36/36** | 0 | 0 |
| nano_v1, rp 1.00 | **36/36** | 0 | 0 |
| emotion_nano, rp 1.05 | 31/36 | 3 | 2 |

**The elongation is a property of the emotion fine-tune, not of NeuTTS, the codec, or the
sampler.** The live test is unaffected. The emotion voice is opt-in (`BMO_TTS_EMOTION=1`).

### 42b. `min_p` does not help — negative result
The hypothesis was that `min_p` would suppress derailment *without* `repeat_penalty`'s
premature-EOS side effect (it truncates the low-probability tail adaptively rather than
penalising legitimately-repeated codec tokens). It does not:

| min_p | rp | ok/36 |
|---:|---:|---:|
| 0.10 | 1.05 | 31 |
| 0.00 | 1.00 | 29 |
| 0.00 | 1.05 | 28 |
| 0.05 | 1.00 | 27 |
| 0.05 | 1.05 | 27 |
| 0.10 | 1.00 | **20** |

The 27–31 spread is inside the noise band already established, and `min_p=0.1` alone is
clearly worse. Sampling is plateaued; `repeat_penalty=1.05` stays.

### 42c. A degenerate-loop guard IS possible — the earlier probe used the wrong statistic
§40e concluded no guard was available, measuring token **diversity** (unique-in-window) and
finding the distributions overlapped. That was the wrong statistic. A degenerate loop repeats
an exact **period-L cycle**; a sustained vowel merely has low diversity. Over 216 captured
token streams:

| | median | p90 | max |
|---|---:|---:|---:|
| healthy (n=162) | 1 | 4 | **22** |
| looping (n=41) | 11 | 32 | **209** |

| threshold | loops caught | healthy truncated |
|---:|---:|---:|
| 10 | 59% | 1.9% |
| 18 | 34% | 0.6% |
| **23** | **24%** | **0.0%** |

`LOOP_RUN = 23` sits above every one of 162 healthy clips — **zero false positives** — and
still catches the three worst drones (cycle runs 90, 114, 209), which are the ones that sound
like "heyyyyyyyyyy". Implemented incrementally (O(periods) per token, no rescan).

Verified: **0 false positives on nano_v1** (36/36 EOS). It is insurance against the worst
case, not a cure — it did not fire in a 36-clip emotion run, consistent with a 24% catch rate
on 4 drones.

**Honest residual: ~11% of *emotion* clips still drone.** That is the fine-tune's sampling and
needs a retrain. The demo voice has no such failures.

### 42d. Perception confidence: temporal stability replaces the margin gate
The margin gate is gone. `_stabilise()` votes each field over the last `PERCEP_WINDOW=3`
calls, states the **modal** answer (smoothing out single-frame flicker), and **omits** any
field whose window contains no repeated answer.

Why this and not margin: retrieval is deterministic given a frame, so a field that keeps
changing its answer is one the frame does not determine — and unlike margin, that signal is
comparable across categories.

Measured cost, 10 consecutive calls on a live scene: **9/9 fields survive every call**
(survival 10/10 for all nine). The gate is free on a stable scene and only fires during
genuine transitions — as it did at boot, dropping `who`/`doing`/`holding` while the scene
was still settling. Smoothing is visible in the same run: raw `doing` alternates between
"pointing at something" and "watching a screen" while the emitted value stays the mode.

The window is **seeded with 2 calls at boot** (~5.1 s), so turn 1 already votes over a full
window instead of being the one turn with no gating.

**HONEST LIMIT: this fixes variance, not bias.** A stably-wrong field — the
`hearing: an alarm beeping` fan artefact — is stable and will still be stated. It is not a
correctness check.

### 42e. A real bank defect this surfaced
The gate printed `wearing: smoothed 'a person wearing a white shirt' -> 'a person holding a
mug'`. Root cause: **`a person holding a mug` shipped in v2 mislabelled as `appearance`**, so
the `wearing` question could legitimately answer with a held object. Exact-string dedup keeps
the first occurrence, which was the v2 entry, so its wrong category survived even though the
template generates the same phrase correctly under `held_object`. Fixed by rule in the builder
(`" holding " in tag -> held_object`). `appearance` 131→130, `held_object` 15→17.

---

## 43. Live test 2026-08-26 — three bugs, all of them plumbing

The pipeline fits and runs. Every symptom reported traced to logic that was switched off or
too narrow, not to a model.

### 43a. The directives were never used — the flag defaulted OFF
`--use-directive` was `action="store_true"`, default False, and the live run did not pass it.
So every turn: the thinker ran, produced a correct directive, logged it, and
`process_turn` **discarded it**. That is the whole of "the directives are failing".

The comment defending the default said the path was unsafe because *"the deployed speaker
(v5) has ZERO instruction-conditioned rows and the deployed thinker does not emit
directives"*. Both halves became false when the defaults moved to speaker v10 (98% adherence)
and thinker v8. **Now ON by default; `--no-directive` to disable.**

Verified — same utterance, same scene, with and without:

| directive | without | with |
|---|---|---|
| gently suggest they take a break… | "I'll pause my chatter; let's hit the pause button for tonight." | **"Consider a short break; you'll come back refreshed."** |
| remark warmly on something they are wearing | "You're wearing a white shirt." | **"I see a white shirt, and you are wearing it."** |

### 43b. The guard silently rejected HALF the directive vocabulary
`_directive_guard()` required the directive to start with a verb from a hand-written list of
19. The trained vocabulary (`ALL_DIRECTIVES`, 26 entries) uses 22 leading verbs, and the list
covered only 13 of them — so **13 of 26 directives were rejected on sight**, including
`admit you do not know…` and `react to them coming back after being away`, both observed
being thrown away in the live log as *"REJECTED (looks like a spoken line, not a directive)"*.

The list is now generated from the vocabulary. Verified: **15/15 trained directives accepted,
3/3 spoken lines still rejected** (the guard's actual purpose is intact).

### 43c. The scene was gated behind a keyword regex that missed
The perception scene reached the speaker only if `PERCEPTION_ASK` matched the utterance. The
user asked **"Can tellll me what I'm wearing?"**; the regex knew only `what am i wearing`, so
it missed, no scene was passed, and the speaker invented **"a t-shirt and some black socks"**
— there is no socks tag anywhere in the bank.

A gate that must enumerate every phrasing of a question will always have holes, and the
speaker has 126 `perception_grounded` rows training it to hold a scene without being asked.
**Gate deleted**; the scene is passed whenever one exists. `PERCEPTION_ASK` removed as dead.

### 43d. The camera died from memory pressure, as designed
The run ended at `avail=0 MiB`. The hub lost its NVMM allocation
(`Could not open socket /tmp/bmo_cam_perception.sock: 111`), so perception had no frames for
most of the session — a second, independent reason the answers were invented.

### 43e. Corrected perception memory breakdown
Two earlier attributions in this document were wrong; these are the measured numbers with
probes placed between each load:

| component | resident | quantised? |
|---|---:|---|
| ambient (WavJEPA) | 136 MiB | int8 |
| m2 fusion | 162 MiB | int8 |
| scene (SigLIP2) | 277 MiB | **no** (bf16) |
| **query engine + bank** | **668 MiB** | **no** (fp32) |

The query predictor is the largest single component of the whole perception stack — 668 MiB
resident against **131 MB of weights on disk** — and it is one of only two perception models
never passed through `q_int8_cpu_then_move`.

**Two cheap fixes were tried and BOTH FAILED**: `gc.collect()` after `del sig.text_model`,
and `del qck, cand, raw, tp; gc.collect(); torch.cuda.empty_cache()` after the query engine.
Neither moved the number beyond run-to-run allocator noise. The memory is genuinely resident.
The remaining lever is quantising those two components, which is a perception-quality risk and
must be verified by cosine against the current output before it ships.

### 43f. The backchannel is a latency filler, not a backchannel
It is pushed to the audio engine *after* VAD closes the user's segment and *before* the
blocking perception call (~1.5 s) and the LLM — so by construction it plays after the user
stops, covering latency. A true backchannel (mm-hm *during* speech) needs the VAP head, which
is not wired. Nothing is broken; it is doing the job it was written for, and that job is not
the one it looks like. `--backchannel` controls it.

---

## 44. Perception int8 — 998 MiB recovered, zero change in output

The two components never given the `q_int8_cpu_then_move` treatment the rest of the stack
already gets — the query predictor (668 MiB against 131 MB of weights on disk) and the
SigLIP2 scene tower — are now quantised. Env-gated (`BMO_INT8_QP`, `BMO_INT8_SIGLIP`, both
default on) so the fp32 path stays reachable.

### Result

| | fp32 | int8 |
|---|---:|---:|
| perception resident | 1,951 MiB | **953 MiB** |
| quantised submodules (qp / siglip) | 0 / 0 | **12 / 74** |
| free at `[boot] DONE` | 20–83 MiB | **439 MiB** |
| free after selftest | 0–22 MiB | **206 MiB** |

**998 MiB recovered, 51% of the perception stack.** This is the fix for §43d — the live run
ended at `avail=0` and the camera hub lost its NVMM allocation; there is now real headroom.

### How it was verified

Three things had to be true, and only the third is the one that matters to the pipeline:

1. **The quantisation actually happened.** `q_int8_cpu_then_move` catches its own failures and
   prints `INT8 quantization skipped` — a silent no-op would make any A/B pass by doing
   nothing, the same class of bug as the ConvInteger codec that "worked". Weight-wrapper types
   are counted: **0/0 on the baseline, 12/74 after**.
2. **Identical inputs.** The two configs cannot be co-resident, so the test runs twice. Against
   a live camera that would confound quantisation error with the scene changing, so 16 frames
   are captured once to `ab_frames.npy` and replayed to both runs.
3. **The nine field ANSWERS are unchanged.** A high cosine on the query vector is not
   sufficient — the answer is a discrete argmax over 403 tags and can flip at any cosine below
   1. Measured: **9/9 answers identical**, worst query-vector cosine **0.9998**.

```
field       match   cos(zq)   answer
who           YES    0.9998   a person's face close to the camera
wearing       YES    0.9999   a person wearing a white shirt
doing         YES    0.9998   someone is shaking their head
where         YES    0.9999   a laboratory
lighting      YES    0.9999   a lit screen in a dark room
hearing       YES    1.0000   an alarm beeping
posture       YES    0.9998   a person head tilted
holding       YES    0.9998   a person holding a phone
looks         YES    0.9998   a person frowning
```

Note `hearing: an alarm beeping` is still the fan artefact recorded in CLAUDE.md — unchanged
by quantisation, and still wrong for the same reason it was always wrong.
