# jetson_artifacts/ — benchmark numbers and logs pulled off the Jetson

Transferred from `bmo@bmo-desktop` (Tailscale) on 2026-08-14 so every measurement for this
project lives in one place on mercury rather than being split across two machines.

## What is here
- `benchmarks/home/` — the Jetson's `~` benchmark JSONs and run logs (latency probes,
  full-stack memory runs, echo tests, production smoke tests, live-loop logs).
- `benchmarks/pipeline_checkpoints/` — result JSONs written under
  `~/bmo_production/pipeline/checkpoints/`.

## Deduplication
11 files were byte-identical (md5) to results already committed on mercury and were
**deleted rather than kept twice**. They are listed here so the provenance chain is not
lost — the mercury copy is authoritative for each:

| Jetson file | already on mercury as |
|---|---|
| `jetson_phase4_memory_v2_withqwen_results_full60.json` | `checkpoints/vjepa21_shelved/JETSON_PHASE4_MEMORY_LIKEFORLIKE_WITHQWEN.json` |
| `jetson_minicpm5_latency_results_v4.json` | `checkpoints/m5_jetson/JETSON_MINICPM5_LATENCY_RESULTS.json` |
| `jetson_full_stack_v3_results.json` | `checkpoints/JETSON_FULL_STACK_V3_BAD_ORDER_RESULTS.json` |
| `vjepa21_vitl_jetson_results.json` | `checkpoints/vjepa21_shelved/vjepa21_vitl_jetson_results.json` |
| `jetson_phase4_memory_results.json` | `checkpoints/vjepa21_shelved/JETSON_PHASE4_MEMORY_RESULTS.json` |
| `jetson_phase4_memory_LOCKED_results.json` | `checkpoints/m5_jetson/PHASE_A1_LOCKED_CHECKPOINTS_MEMORY_RESULTS.json` |
| `jetson_phase4_4_results.json` | `checkpoints/vjepa21_shelved/JETSON_PHASE4_4_RESULTS.json` |
| `jetson_phase4_2_3_results.json` | `checkpoints/vjepa21_shelved/JETSON_PHASE4_2_3_RESULTS.json` |
| `jetson_tts_latency_results.json` | `checkpoints/vjepa21_shelved/JETSON_TTS_LATENCY_RESULTS.json` |
| `jetson_vad_cpu_results.json` | `checkpoints/vjepa21_shelved/JETSON_VAD_CPU_RESULTS.json` |
| `m4_echo_test_real_hardware_results_v3.json` | `checkpoints/m5_jetson/PHASE_B4_REAL_ECHO_TEST_RESULTS.json` |

## Deliberately NOT transferred
A first rsync pass pulled 360MB because it recursed into everything. That was discarded and
redone surgically. Excluded on purpose: tokenizers (~30MB of `tokenizer.json`), the
`face_engine` binary, `site-packages`, `sherpa-onnx-src`, and the unrelated `TreasureHunt`
project — none are benchmark artifacts, and the GGUFs/checkpoints already exist on mercury
as the source of truth.
