# Pipeline latency optimization plan (2026-08-08)

Diagnosis (user's live test + research): latency is **ASR-dominated** — the energy-VAD
waits 1000ms of silence + Python `asr.generate()` adds ~100ms GIL/CUDA-launch. TTS is
next (llama.cpp talker + CPU-ONNX NeuCodec decode).

Jetson env: **JetPack 6.2.1, L4T R36.4.7, Python 3.10, CUDA 12.x, TensorRT 10.3, aarch64.**
Key gate found: installed `onnxruntime` (1.23.2) is **CPU-ONLY** (providers = Azure/CPU, no
CUDA/TRT EP). TensorRT 10.3 IS present. No `neutts` pkg (talker = GGUF via llama.cpp,
self-contained StreamingVoice). No `sherpa-onnx`.

## Ordered plan (each step ~unblocks the next)
0. ✅ **VAD wait 1000 → 500ms** (done; immediate ~500ms/turn cut). Cheap, low risk.
1. **Streaming STT (biggest ASR win, ADDITIVE / low risk).** Install `sherpa-onnx` + a
   streaming Zipformer RNN-T; process 40ms chunks continuously, emit finals *before* the
   user pauses → removes the silence-wait entirely. + persistent warm audio ring buffer
   (no per-turn InputStream open/close). Fallback: keep Moonshine.
2. **onnxruntime-gpu for Jetson (RISKIER — swaps a working dep).** Install the JP6.2 /
   py3.10 / cu12 / TRT10 aarch64 wheel (NVIDIA Jetson index). Enables GPU ONNX. MUST verify
   the existing CPU-ONNX NeuCodec path still works as fallback; ideally test in a venv first.
3. **NeuCodec decode → GPU** (already ONNX; just switch to CUDA/TRT EP once #2 lands). Quick.
4. **NeuTTS talker → ONNX (GPU).** Export the emotion-v4 HF checkpoint via neuphonic's ONNX
   export script (from the neutts-air repo, not installed) → run on ORT-GPU / TensorRT,
   removing llama.cpp + GIL. Target sub-100ms first chunk (aspirational; real floor depends
   on the autoregressive talker — TRT + no-GIL should help a lot).
5. (bonus) **Ultravox-style STT** (Moonshine projector, no text) retrained for the new LLM,
   on top of the fast engine.

## Risk notes
- #2 (onnxruntime-gpu) is the one that can break the working voice (NeuCodecOnnxDecoder uses
  the current CPU onnxruntime). Do it carefully / with a fallback / venv.
- #1 (sherpa-onnx) is additive and the biggest single latency win — good first real step.
- Backlog of FEATURE tasks (198/199/200/184/182 + perception) preserved in BACKLOG.md;
  resume after this push.
