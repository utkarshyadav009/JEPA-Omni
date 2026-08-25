---
name: bmo-jetson-deploy
description: Operational knowledge for the real BMO deployment on the Jetson Orin (SSH bmo@bmo-desktop): ~/bmo_production/ layout, mandatory model load order, measured end-to-end latency, streaming and emotion voice (NeuTTS/NeuCodec), NvMap ENOMEM and memory-compaction forensics, power and fan acoustics, deploy history. Use when launching, debugging, or measuring anything on the Jetson, when touching bmo_jetson_startup.py, bmo_launch.sh, m5_streaming_voice.py or m5_tools.py, or when hitting GGUF load failures, TTS crashes, CUDA allocator assertions, or thermal/battery issues on device.
---

# BMO Jetson production deployment

Hard-won operational knowledge for the running Jetson Orin deployment. Everything here was
measured on real hardware; several sections record hypotheses that were tested and
FALSIFIED — do not re-investigate those leads.

## BMO Jetson production deployment

Real, running deployment on the Jetson Orin (7.6GB shared memory, ARM64), separate from
the M1/M2 training work above. SSH: `bmo@bmo-desktop` (Tailscale, key auth already set up).

**File layout** — `~/bmo_production/` on the Jetson (consolidated from a previously
scattered layout; do not recreate the old `~/jepa_omni_transfer`, `~/gguf_models`,
`~/bmo_stt_test`, `~/BMO-Project` paths, they no longer exist):
- `pipeline/` — this repo's `models/`, `checkpoints/`, `data/`, `scripts/`, `assets/`
- `face_engine/` — the real BMO Face Engine (C++, compiled `BMO_Engine` binary, real git
  repo). `~/bmo_fresh` was a confirmed-stale duplicate, deleted 2026-08-07.
- `models_gguf/` — deployed GGUF models (LLM tiers + TTS)
- `tokenizers/` — LFM2/MiniCPM5 tokenizer dirs
- `scripts/` — `bmo_jetson_startup.py` (real `build_bmo_stack()`), `bmo_launch.sh` (**the
  real entry point** — does a privileged memory-compaction step via a narrowly-scoped
  NOPASSWD sudoers rule, then execs the startup script; don't run
  `bmo_jetson_startup.py` directly in production, use the wrapper)

**Required load order** (documented in `scripts/bmo_jetson_startup.py`'s module
docstring, measured/reproducible): llama.cpp models (LFM2 fast tier, MiniCPM5 reasoning
tier, TTS) MUST load before the torch/int8 perception stack (vision, WavJEPA x2, M2
predictor, M3 connector, STT encoder, decision head). Reversing this fragments Jetson's
unified memory badly enough that the LLM/TTS GGUFs fail to load.

**Real, measured latency breakdown** (2026-08-07, `scripts/jetson_e2e_full_test.py`,
steady-state medians): STT 38ms, perception (ViT-L+WavJEPA) 985ms, fast-tier LLM 337ms,
TTS backbone 829ms, NeuCodec decode ~53ms. **Total ~2.19s.** Perception and TTS backbone
are roughly TIED as the two dominant costs (~45% and ~38% of total respectively) — TTS
alone is NOT "the majority" of latency, despite intuition; neither is perception alone.

**STT**: `models/m4_speech.py::MoonshineSpeechEncoder` (UsefulSensors/moonshine-base,
native `transformers` support) replaced Whisper-medium for the deployed 3-class
turn-taking decision head — 10-13x faster (277ms->37ms), real accuracy cost (95%->90.67%,
mostly backchannel recall). Checkpoint:
`checkpoints/m4_decision_head_3class_speechonly_moonshine/best.pt`. The DEEPER AV-grounded
generation path (`models/m4_duplex_loop.py::DuplexLoop`, M4b/Ultravox-style speech
projector) still uses Whisper and is **not wired into production** (`build_bmo_stack()`
never touches it) — only retrain that path's projector if it actually gets deployed.

**Vision frame count**: `models/m5_streaming_loop.py::StreamingConfig.n_vision_frames`
defaults to **16** (not 64) — real 2.3x win on the perception leg (2309ms->1014ms),
same temporal span (10s), fewer frames within it (matches V-JEPA2's own published
16-frame eval point). `RollingVideoBuffer.get_window()` already uniformly subsamples
across the full buffered window, so this isn't a truncation.

**TTS reliability**: the TTS GGUF load intermittently fails with NvMap ENOMEM (~50% raw
rate, root cause is general Jetson kernel memory fragmentation — a CMA-region hypothesis
was tested and directly RULED OUT via live `/sys/kernel/debug/nvmap` monitoring, don't
re-investigate that lead). Real fix in production: `build_bmo_stack()` retries the TTS
load up to 5x with `_compact_memory()` between attempts (100% success over a real
10-trial batch, vs 50% with compaction-once). Root cause itself remains unsolved.

**Streaming voice — WIRED INTO PRODUCTION (2026-08-07).** The token→audio path that was
previously unbuilt is now the deployed voice: `models/m5_streaming_voice.py::StreamingVoice`
loads the NeuTTS-Air BMO GGUF backbone (5x retry+compaction wrapper) and streams speech
tokens → seam-free chunked decode → audio. `bmo_jetson_startup.py` constructs it as the
stack's `tts` and the smoke test calls `stack["tts"].speak(...)`. Hard-won specifics, all
required:
  - **MUST sample** (temp=0.7, top_k=50). Greedy (temp=0) makes neural-codec TTS loop
    forever and never emit `<|SPEECH_GENERATION_END|>`. This was the real cause of the old
    "Nano can't emit EOS" verdict — a greedy-decode testing bug, NOT a model defect.
  - Speech tokens come from the **low-level `Llama.generate()` iterator** (`idx = tid - SP0`,
    SP0 = id of `<|speech_0|>`); the high-level completion API silently suppresses the
    `<|speech_N|>` special tokens.
  - Decode uses the **NeuCodec INT8 ONNX decoder** (`neucodec.NeuCodecOnnxDecoder`,
    `neuphonic/neucodec-onnx-decoder-int8`), CPU/onnxruntime, ~54ms per 25-token chunk,
    full-utterance RTF ~0.08. This is the decisive fix over the torch NeuCodec: the torch
    path's `.to('cuda')` hits a `CUDACachingAllocator` NVML assertion in the FULL stack
    (the 3 resident llama.cpp models already hold the CUDA context), and its fp32-CPU
    fallback is RTF ~2.0 (unusable). ONNX-CPU is both faster AND conflict-free.
  - Seam-free audio via encodec-style `_linear_overlap_add` triangular-window cross-fade
    of overlapping decoded windows (CHUNK=25, LOOKFWD=5, LOOKBACK=50, OVERLAP=1, HOP=480);
    naive per-chunk decode clicks at boundaries. Plus a leading breath/silence trim + 8ms
    fades on the start.
  - **Real measured (full stack smoke test 4):** TTFA 625ms, full-utterance synth ~RTF 1.0
    (token generation, not decode, is the bottleneck), down from 2813ms TTFA with the fp32
    CPU codec. The `thinking_filler` backchannel masks the 625ms. Benign shutdown noise:
    3× `Exception ignored in: <function Llama.__del__>` / `free_model` TypeError at
    interpreter exit (one per GGUF) — harmless, Python already swallowed them.

**CORRECTION — NeuTTS Nano IS viable (greedy-decode bug, not a model defect).** The earlier
"Nano can't reliably emit EOS" verdict was wrong: it was tested with greedy decoding, which
makes ANY neural-codec TTS (Nano AND Air) loop forever. With sampling (temp=0.7, top_k=50)
the EOS emits fine. Air (`bmo_neutts_v5`) stays deployed as the known-good voice; Nano is a
live option again if a smaller/faster backbone is wanted. Separately still true: the
`neutts.NeuTTS.infer()` reference-audio-conditioned API is the wrong tool for these
checkpoints — `finetune_bmo_neutts.py` trains `text -> that utterance's own codes` with NO
reference concatenation (fixed-voice fine-tune), so `StreamingVoice` uses the raw
training-format prompt directly, not that API.

**Emotion-capable voice (in progress, 2026-08-07).** One homeostatic state drives words
(LLM, already wired via `_state_prefix`), voice, and face. Voice path: fine-tune the
deployed BMO voice with 12 emotion control tokens (`<|NEUTRAL|>` + one per mood, matching
the strings `homeostatic_to_mood_state()` emits, so emotion=mood with zero translation).
`StreamingVoice._prompt(text, emotion)` already prepends `<|EMOTION|>` before
`<|SPEECH_GENERATION_START|>`. Data: 1421 Fish-rendered clips (community BMO voice,
s2.1-pro-free, verified tags in `EMOTION_MAPPING.md`) across 11 moods (min 100/mood) at
`data/bmo_emotion_fish/`. Scripts: `scripts/prep_bmo_emotion_neutts_dataset.py` (encode →
codes+mood) and `scripts/finetune_bmo_emotion_neutts.py` (restore from
`bmo_neutts_finetune_v5/best`, add tokens, mix Fish-emotion + real-recording-neutral anchor
to hold the production neutral voice identical). Mood→Fish-tag→face map in `EMOTION_MAPPING.md`.

**Emotion voice — TRAINED + DEPLOYED, opt-in (2026-08-07).** Fine-tune done: eval_loss
plateaued 0.516 (no overfit), `checkpoints/bmo_neutts_emotion/best`, converted to
`bmo_neutts_emotion_Q8_0.gguf` (568MB — SMALLER than v5's 765MB). Full mood→voice path is
wired end-to-end: `FastTierResult.mood` (set from `state["mood"]`) → `m5_streaming_loop.py`
passes it to `StreamingVoice.speak(emotion=mood)` → `<|MOOD|>` token. `StreamingVoice`
self-detects emotion capability (`has_emotion` = does `<|NEUTRAL|>` tokenize to 1 token) so
passing a mood is a NO-OP on v5 — the loop change is safe on either model. `speak()` now
returns a `SpeakResult` dataclass (`.duration_sec`/`.synth_latency_sec` for the loop, still
tuple-unpackable for the smoke test — a real integration gap that was fixed).
  - **Standalone on Jetson: works well.** Loads 5.2s, TTFA 450-457ms warm (622ms first),
    all 12 moods objectively distinct — f0 maps sensibly (tired/stressed/concerned ~145Hz,
    anxious/content ~360Hz). Spot-check: `scripts/emotion_voice_spotcheck.py`.
  - **KNOWN ISSUE — thin moods.** After the text filter, `lonely` (19 clips) and `happy`
    (39) are under-represented; `<|LONELY|>` is unstable (ran to the MAX_TOK length cap on
    Jetson). Fix = targeted data augmentation for those moods before trusting them.
  - **Deployment gating: emotion is OPT-IN via `BMO_TTS_EMOTION=1`; v5 is the default.**
    Not auto-flipped because it needs human listen-approval AND — the real blocker —
    **the emotion GGUF is not yet stable in the FULL stack.** Reproducible split (small-n
    but clean): v5 full-stack passes 2/2 (smoke 4 + 7); the emotion GGUF full-stack fails
    2/2 (smoke 5 + 8) — crashing at `M3Connector(...).to(device)` during PERCEPTION load
    with the flaky `NVML_SUCCESS == r` CUDACachingAllocator assertion (and once, later, at
    the first `llama_decode` with `NvMapMemAllocInternalTagged error 12`). It is NOT an OOM
    of the model (emotion GGUF is 568MB < v5's 765MB) and compaction alone does NOT fix it
    (smoke 8 had compaction and still crashed). **REAL CAUSE (per the user + jetson_preflight.sh's
    own service-note):** the M3 load runs within ~200-400 MiB of the memory edge, and
    BACKGROUND SERVICES (bmo_app, bmo_tunnel, burningtruth_app + its cloudflared tunnel,
    jtop, snapd, packagekit) — plus the coding-agent's own ssh/python RSS — each
    independently eat that margin and tip a passing boot into NvMap error-12 OOM at layers
    25/26/28. The v5-pass/emotion-fail 2/2 split was COINCIDENTAL service-memory timing,
    not the GGUF. **Fix = run `sudo bash jetson_preflight.sh` FIRST** (it stops those
    services + drop_caches + compact_memory, gating on /proc/buddyinfo order≥10 blocks);
    a static preflight PASS still doesn't protect the multi-minute load, so the services
    must actually be stopped. After the run, restart them:
    `sudo systemctl start bmo_app.service bmo_tunnel.service burningtruth_app.service
    burningtruth_tunnel.service jtop.service`.
    **RESOLVED (task #183, 2026-08-08): the full stack now loads reliably WITHOUT preflight.**
    Added `_load_gguf_retry` (LLM tiers) + `_to_device_retry` (perception `.to(device)`) to
    `build_bmo_stack()`, mirroring the TTS 5x-retry+compaction. Verified live (smoke10, emotion
    voice, no preflight): the MiniCPM5 GGUF load failed once → retry compacted → succeeded →
    whole stack loaded + spoke (VOICE TTFA 723ms in the full stack, SMOKE_TEST_DONE). Preflight
    is still nice-to-have under heavy service load but no longer required for a clean boot.

**Compaction fix (2026-08-07): `_compact_memory()` now works when not root.** The Jetson's
NvMap ENOMEM is fragmentation (plenty of free MiB but too few order≥10 / 4MiB+ contiguous
blocks). `_compact_memory()` used to warn "not root, skipped" and no-op, leaving the 5x
load-retry with no safety net → the full stack OOMs at the ragged edge. Fixed to fall back
to the narrowly-scoped NOPASSWD sudoers rule (verified live): `(root) NOPASSWD: /usr/bin/tee
/proc/sys/vm/compact_memory`, i.e. `echo 1 | sudo -n tee /proc/sys/vm/compact_memory`
(NOT `sudo sh -c 'echo...'`, which the rule does not cover and which prompts for a
password). Manual defrag anytime: `echo 1 | sudo tee /proc/sys/vm/compact_memory`.

**NeuTTS Nano: verdict REVERSED — see the "CORRECTION — NeuTTS Nano IS viable" note in the
production section above.** The old "not viable / can't emit EOS" conclusion was a
greedy-decode testing bug (temp=0 makes any neural-codec TTS loop forever); with sampling it
emits EOS fine. Still true and separate: `neutts.NeuTTS.infer()`'s reference-audio-conditioned
API is the wrong tool for these fixed-voice checkpoints (`finetune_bmo_neutts.py` trains
`text -> own codes`, no reference concatenation), which is why `StreamingVoice` uses the raw
training-format prompt via the low-level generate iterator instead. Air (`bmo_neutts_v5`)
stays deployed as the known-good voice.

**ToMe (token merging) on ViT-L: real structural incompatibility, not implemented.**
`models/world_state_builder.py` asserts vision tokens are a multiple of 256 (the M2
predictor's temporal-bin structure depends on this regular spatial/temporal grid).
Free-form similarity-based token merging breaks that grid. If revisited, use
grid-preserving regular spatial pooling (e.g. 2x2 avg-pool per temporal bin) instead of
real ToMe's arbitrary bipartite matching.

**Known gotcha**: `pip install neucodec` on the Jetson silently upgrades numpy
1.26.4->2.2.6, which breaks torch's compiled extensions (real incident, caught and
fixed 2026-08-07). Always check/pin `numpy==1.26.4` after installing any new package
there. Also: loading `NeuCodec` fp32 directly to CUDA reproducibly crashes
(`CUDACachingAllocator` NVML assertion) — convert to bf16 on CPU first, matching the
`q_int8_cpu_then_move` pattern already used for the perception stack.

**Backchannel/non-verbal cues**: `models/m5_tts.py::PrebuiltVoiceBank` — real BMO-voiced
clips (continuer/reactive/thinking_filler categories) rendered offline via the Fish Audio
API (public "BMO from Adventure Time" community voice, NOT our own NeuTTS fine-tune,
which loops on short 1-3 word phrases). Wired into `models/m5_streaming_loop.py`'s
decision path (`label=="backchannel"`) and a new latency-masking hook that fires
`thinking_filler` at the start of any SPEAK turn.

## Power, fan acoustics and GLR (2026-08-16, later)

**BMO can act on its own body.** `models/m5_tools.py` registers `power`/`battery`/`temperature`,
`set_power_mode`, `fan`, plus `power_status()`, `battery_to_energy()` (battery → homeostatic
`energy`) and `power_guard()` (spec §7 rules as data; reports, does not act). Backed by the
user's `bmo-power` (INA219 I2C bus 7 @ 0x41, Waveshare UPS C 3S, PWM fan hwmon0/pwm1 + tach,
nvpmodel). **Always `import bmo_power`, never shell out** — the CLI raises `PermissionError`
non-interactively as the `bmo` user while the module works. Note also that
`get_power_status()` reports `mode:'manual'`/`nvfancontrol_active:False` even when
`systemctl is-active nvfancontrol` is active and PWM tracks temperature — those two fields are
unreliable; trust systemd and the PWM value.

**CORRECTION — the fan is TONAL, not broadband.** The earlier note above ("lo/hi 0.12–0.19, so
broadband, high-pass useless") was measured on the *total* room floor and could not separate fan
from room. Differencing two commanded fan speeds isolates it: **38.6% of all fan energy sits in
one peak at 808–812 Hz**, and 9 blades predicts BPF = rpm/60×9 = 810.4 Hz against 808.6 Hz
measured — **1.9 Hz error**. Reproduce with `scripts/measure_fan_signature.py`.

`models/m5_fan_notch.py` is the fix: tach-driven zero-phase IIR notch at BPF + 2 harmonics,
Q=30, 1% RPM hysteresis. **33.4 dB on the tone, 0.0 dB on speech, 1.6 ms** per 10 s buffer —
but ONLY via `scipy.signal.sosfiltfilt`; the pure-numpy fallback is **374 ms** (57% of the
perception leg). Zero-phase is required because WavJEPA is temporal and the world-state builder
aligns audio against video. **Its effect on the `hearing` percept is still unproven** — the test
room was at the noise floor (rms 0.0018 vs 0.0016), and removing a tone does not make silence
read as silence. Re-test with real speech.

Any script that pins the fan MUST restore `auto` in a `finally:` — the Jetson idles at ~68 °C
and a latched `quiet` after an exception is a thermal throttle waiting to happen.

**GLR runs under llama.cpp — both halves proven.** Embeddings in: `llama_batch.embd`
(byte-identical, already banked). Hidden states out: `llama_get_embeddings_ith` read as a **raw
ctypes pointer**. The wrapper's `_ctx.get_embeddings()` raises `ValueError: '&<f' is not a valid
PEP 3118 buffer format string` — the SAME broken buffer binding documented for `get_logits_ith`
at `m4_cognitive_core.py:227`, not an absence of the feature. `scripts/probe_llamacpp_hidden_states.py`
gates this; it PASSES (per-token `(T,1024)`, cos 0.908 on a shared-prefix pair confirms
last-token states, per-step updates work with KV cache).

`checkpoints/glr_thinker_v1/best.pt`: 1,049,600-param linear head on a frozen 752M Qwen3-0.6B,
val_loss 3.5002 / ce 1.9951 / delta 1505 at epoch 4. **λ=1e-3, measured not guessed** — the paper
omits λ and λ=1.0 makes `L_Δ` (squared L2 summed over 1024 dims, in the thousands) swamp `L_CE`
(1–3), training the head as a pure regressor with no pressure to keep the answer correct.

**Speaker corpus: assert no `{name}` survives.** 54 rows in every v10 variant shipped an
unsubstituted placeholder, and **not one of them had a name anywhere in the conversation** —
substituting a real name would teach BMO to address strangers by invented names, which is the
same defect as the "why does it think I'm Alice" incident. `scripts/fix_name_placeholders.py`
removes the address instead and asserts the output is clean → `bmo_companion_corpus_v11.jsonl`.
The speaker has **zero** instruction-conditioned rows in any version; that gap, not a v1→v5
regression, is why it ignores the thinker (v1 scores 1/6 on intent adherence, v5 4–5/6).

**Never `pkill -f <script>` over SSH.** Fourth occurrence of this class: the tailscale-ssh child
and the `bash -c` wrapper both carry the pattern in their own argv, so pkill kills its own shell
(exit 255). Wait on and signal **PIDs** (`kill -0 <pid>`).

## PRODUCTION DEPLOY 2026-08-16 — v5 tiers live, OOM root-caused and fixed

**Production had been running speaker v1 + thinker v2 since 2026-08-08.** Every speaker v2-v6 and
thinker v3-v7 existed only in test harnesses. v1 is the version measured at **1/6** on intent
adherence — the worst of the five. Check what `build_bmo_stack` actually loads before optimising
anything downstream of it.

Now deployed: **`bmo_lfm25_350m_v5`** (speaker) + **`bmo_thinker_qwen3_v5`** (thinker), with the
`enable_thinking=True` fix live — verified on-device, 4.7 s, `.reasoning` populated.

### THE OOM WAS TWO DEAD LOADS, both already documented as retired

`build_bmo_stack` was still loading, every boot:
  * **WavJEPA-nat** — `+701 MiB`, `+326 ms`, **zero measured gain**, listed under "Already
    settled, do not reopen". `build_world_state_features` treats `nat_encoder=None` as
    `audio_mode="base"` (0.608 audio-following, same as base+nat). Now `None`.
  * **M3 connector** — the thinker↔perception hookup moved to the prediction-style integration
    and M3 was DROPPED; only three test scripts still read it. Now `None`.

Dict keys retained as `None` so older test scripts get `None` (which they pass straight through)
rather than `KeyError`.

**Measured, full stack including TTS, camera opened:**

```
before boot                                    4898 MiB available
FULL STACK RESIDENT (incl TTS)   3923 MiB used  975 MiB available
CAMERA OPEN=True                  234 MiB NVMM  741 MiB available
```

Previously the same stack left **280 MiB** and the camera's NVMM allocation failed. Boot time
46 s without TTS; VOICE TTFA 997 ms in the full stack.

### THE TTS CRASH WAS A POWER-MODE BUG, NOT A TTS BUG

The boot aborted with a C++ `Assertion '__n < this->size()' failed` inside onnxruntime — a
**SIGABRT, so the 5x retry+compaction wrapper cannot catch it**. Reproduced in isolation with
6.6 GB free, so not memory. Two independent causes, both needed:

1. **`import onnxruntime` aborted outright.** ORT **1.23.2**, pulled into `~/.local` by
   **`piper-tts`**, dies parsing this CPU (`Unknown CPU vendor, cpuinfo_vendor value: 0`).
   Fixed: `pip install --no-deps onnxruntime==1.19.2` (`--no-deps` so numpy stays 1.26.4 — the
   documented hazard on this device). Reverse with `pip install onnxruntime==1.23.2`.
   *Not verified whether 1.23.2 would work once cause 2 was fixed; the downgrade stands.*
2. **The Jetson was in 7W mode with only 4 of 6 cores online** (`/sys/devices/system/cpu/online`
   = `0-3`). ORT enumerates all 6 and tries to pin a thread to core 4:
   `pthread_setaffinity_np failed ... mask: {4,} ... Specify the number of threads explicitly`,
   then indexes an empty vector and aborts. `bmo_power.set_power_mode("MAXN_SUPER")` brought
   cores 0-5 online and TTS loaded and spoke (2286 ms).

**So a power mode can break TTS.** Anything using onnxruntime needs all cores online, or an
explicit `intra_op_num_threads`. Worth pinning threads explicitly so 7W/eco mode is survivable —
`power_guard()` drops to 7W at ≤20% battery, which would break the voice exactly when it matters.
