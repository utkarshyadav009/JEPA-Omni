# BMO backlog — deferred feature tasks (SAVED 2026-08-08, do NOT lose)

Per user: we're pausing these to **optimize the pipeline hard first** (latency:
ASR-dominated + TTS via ONNX), then coming back to everything below. This file
is the durable copy of the task list (mirrors the task tracker).

## Come back to these AFTER the optimization push
- **#198 — LLM generalization retrain.** Live test: replies are generic and don't
  match the mood (voice gets stressed, words stay off). Retrain the fast tier (+ thinker)
  on a broader, emotion-varied, **confrontation/hostility-aware** dialogue corpus so the
  WORDS match the emotional state. Biggest lever for "generic replies."
- **#199 — Tool-calling end-to-end live test** with the trained Qwen3 v2 thinker + the
  ToolDispatcher (m5_tools.py). weather/time/timer/reminder, parse→execute→inject→speak.
- **#200 — Backchannels + duplex** in the live loop: MicGate (don't hear self), PrebuiltVoiceBank
  thinking_filler/backchannels while generating, barge-in (interrupt).
- **#184 — Companion capability block**: expanded corpus (6 tools + companion behaviors)
  DRAFT is generated (data/bmo_companion_tools_v8_DRAFT.jsonl); LoRA-retrain the LLMs on it.
- **#182 — Wire Tier 1 speculative prefetch into the live loop** (m5_speculative.py; built +
  benchmarked ~893ms hidden/hit; now that live mic+STT exist, run partial-transcript ASR
  during the turn → speculate → commit on match).

## Also still pending (older tracks)
- #181 backchannel voice consistency + emotion-tagging; #172 emotion-TTS alt eval;
  #173 C1 frame info-cost; #174 C2 perception research memo; #175 D opportunistic vision
  scheduling; #176 E fast-tier token/ctx tuning; #177 NeuTTS Nano sampled revisit.

## North-star (memory: bmo-pipeline-vision)
JEPA-space multimodal memory (recognize people by abstract visual+audio embeddings),
query-style predictor (thinker asks perception for scene detail), VJEPA 2.1, single wavJEPA.
AFTER the pipeline is production-level.

**ACTIVE ASYNCHRONOUSLY (2026-08-10): see `JEPA_MEMORY_PLAN.md`** — plan, verified M2/M3/
embed-predictor baselines, the V-JEPA 2.1 latency evidence, and the run log for this track.
Runs on mercury alongside (not blocking) the optimization push.

---
## ACTIVE NOW: latency optimization push (see OPTIMIZATION section in CLAUDE.md / tasks #201+)
User research: latency is ASR-dominated (1000ms VAD silence wait + ~100ms Python GIL/CUDA
launch on asr.generate()). Plan:
1. **STT streaming**: chunked RNN-T (sherpa-onnx Zipformer) — emits words as you speak, no
   silence wait; OR Moonshine in TensorRT/native C++. + persistent warm audio ring buffer.
   + reduce/replace the energy-VAD silence wait.
2. **TTS ONNX**: export NeuTTS talker + NeuCodec to ONNX (neuphonic provides an export
   script), run via ONNX Runtime GPU / C++ — removes Python GIL + llama.cpp wrapper delay.
   Feasible now that fast+thinker are sub-1B (GPU headroom). Target sub-100ms first audio.
3. Then (bonus) Ultravox-style STT (Moonshine projector, no text) on top of the fast engine.
