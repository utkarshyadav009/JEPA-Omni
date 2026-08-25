# CURRENT ARCHITECTURE — Ground Truth

**Source: `EVIDENCE_LEDGER_V2.md` Part B (forensics) + Part F (live Jetson sync, 2026-08-16)**
**Paste this into every writing chat. It supersedes anything said earlier in this project.**

---

## THE LIVE PRODUCTION PATH

`bmo_launch.sh` → `bmo_jetson_startup.py::build_bmo_stack()`

Everything below is what the **live Jetson copy** loads. The dev-machine checkout was stale
and has now been synced.

### Perception — all frozen, all int8

| Component | Model | Notes |
|---|---|---|
| Vision | `facebook/vjepa2-vitl-fpc64-256` | **16 frames** in streaming (`m5_streaming_loop.py:104`), not 64 |
| Ambient audio | `labhamlet/wavjepa-base` | base only |
| Ambient (nat) | **None** | removed — measured +701 MiB / +326 ms for zero gain |
| Scene | SigLIP2 `base-patch16-224` | large variant benchmarked and rejected |
| Speech | `UsefulSensors/moonshine-base` encoder | feeds decision head only |

### Trained components in the live path

| Component | Checkpoint | Status |
|---|---|---|
| M2 fusion predictor | `m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` | **LOADED** |
| Query predictor | `qp_runD.pt` (= `sig_runD_proj768/best.pt`, byte-identical) | **LOADED** |
| Identity head | `identity_head_joint.pt` (= `jepa_identity_head_av_full/head_joint.pt`) | **LOADED** — the best head in the track |
| Decision head | `m4_decision_head_3class_speechonly_moonshine/best.pt` | **LOADED** — 416-d Moonshine features, acc 90.67% |
| M3 soft-prompt connector | `m3_multigran_richcaption_v2` | **`None`** — dropped 2026-08-16 |

### Cognition — two tiers

| Tier | Checkpoint | Base | Config |
|---|---|---|---|
| Fast / **speaker** | `bmo_lfm25_350m_v5_Q8_0.gguf` | ~350M [VERIFY base family] | `enable_thinking=False` |
| Reasoning / **thinker** | `bmo_thinker_qwen3_v5_Q8_0.gguf` | Qwen3-0.6B | `enable_thinking=True`, `max_new_tokens=320` |

Routed by `CognitiveCoreRouter.route()` — fast tier always runs first, escalates to thinker.
**Qwen2.5-1.5B is no longer in the architecture.**

### Output

`StreamingVoice` (`m5_streaming_voice.py`): NeuTTS-Air GGUF backbone (temp 0.7, top_k 50) →
NeuCodec INT8 ONNX chunked decode → overlap-add → float32 24 kHz.

---

## HOW PERCEPTION REACHES THE LLM — read this carefully

**Not via embeddings.** The soft-prompt connector is gone. The path is:

```
frozen encoders → M2 fusion → World-State (1024-d)
                                    ↓
                    query predictor (sig_runD_proj768)
                                    ↓
              retrieval against candidates_siglip2_v2.pt (1,482 tags)
                                    ↓
                        TEXT TAGS in the chat-templated prompt
                                    ↓
                          fast tier (GGUF, text in / text out)
```

The World-State is the shared latent that the **perception-side specialists** read — query
predictor, identity head, decision head. The interface to the language models is **text**.

This matters for how you state the Aim. The honest version:

> Perception is non-autoregressive and organised around a shared latent World-State read by
> multiple independent specialists. The interface to the autoregressive generator is a
> retrieved text description rather than a learned embedding projection — a change made after
> the embedding-projection approach (M3) was found too slow for the edge target.

That is a **deployment finding**, not a weakness. You tried the embedding interface, measured
it, and it lost. Say so.

---

## BUILT BUT NOT WIRED (must not be described as deployed)

| Component | Status |
|---|---|
| `DuplexLoop` (`m4_duplex_loop.py`) | Tick orchestrator — `build_bmo_stack` never constructs one. **No confirmed live caller.** |
| `BmoDuplexTick` | The real integration of homeostatic state + router + async thinker — **no caller anywhere** |
| GLR transition head v2 | Passed rollout gate (eos_rate 0.99–1.0 through K=15–20) — **not deployed** |
| `bmo_lfm25_350m_v6` | Trained, bake-off 8/12 vs v5's 9/12 — **not deployed** |
| `bmo_thinker_qwen3_v6/_v7` LoRA | Never merged/quantised |
| `bmo_neutts_emotion_v2/v3/v4` | Production hardcodes the **unversioned (v1) filename** |
| Moonshine STT projector | WER plateaus at 0.94 — research track |
| `perception_prefix.py` | Generative alternative to soft prompt — unwired |

---

## KNOWN DIVERGENCES STILL OPEN (these belong in Limitations)

1. **Identity threshold hardcoded at 0.5**, not the calibrated 0.691–0.765 operating point.
   `calibrate_threshold` is never called.
2. **Homeostatic state never computed live.** Both runtime paths hardcode
   `{"energy": 0.5, "mood": "curious"}`. `HomeostaticState.update()` exists and
   `BmoDuplexTick` wires it correctly, but has no caller. The parameters are also
   uncalibrated guesses, flagged in their own docstring.
3. **Thinker reasoning discarded.** `.reasoning` is populated correctly, but
   `bmo_duplex_tick.py:70` speaks `fast_result.text` and drops it. Conditioning the speaker
   on `.reasoning` exists only in `jetson_real_demo.py`.
4. **Timer and reminder tools format strings and take no action.** Weather, web search, and
   encyclopedic lookup are real HTTP calls.
5. **STT conflict unresolved.** `ARCHITECTURE.md` says SenseVoice is live; the live
   `build_bmo_stack()` uses Moonshine with no SenseVoice reference. Five `live_bmo_*.py`
   dev harnesses exist outside the production tree. Report as unresolved.

---

## PROMPT PATCHES — for subchapters already drafted

### §3.3 Perception Layer — REDO the language-model paragraph

The drafted version names Qwen2.5-1.5B. Replace with:

```
Replace the final paragraph on the language model with the following, drawn from
EVIDENCE_LEDGER_V2.md Table 2 and Part B1:

Generation is handled by TWO models in a fast/slow arrangement, not one. A ~350M fast tier
(bmo_lfm25_350m_v5, enable_thinking=False) produces immediate conversational output; a
Qwen3-0.6B reasoning tier (bmo_thinker_qwen3_v5, enable_thinking=True, max_new_tokens=320)
handles queries the fast tier escalates. Both run as llama.cpp GGUF at Q8_0.
CognitiveCoreRouter.route() always calls the fast tier first. Note that Qwen2.5-1.5B was used
in earlier milestones and is no longer part of the architecture.
Mark [YOUR REASONING] for why a two-tier generator was chosen over a single mid-size model.
```

### §3.2 System Architecture — ADD the fast/slow split and fix the interface

```
Add to §3.2 two items, from EVIDENCE_LEDGER_V2.md Part B1, B4, and Table 2:

(a) The generation side is split into a fast speaker and a slower thinker, mirroring the
    perception/generation split: a path that must meet conversational timing and one that
    need not. Describe the router. Mention the GLR latent-reasoning transition head as a
    validated component (v1 diverged at K=10, mean latent norm 423, eos_rate 0.375; v2 with
    zero-init and normalised loss held eos_rate 0.99-1.0 through K=15-20) that is NOT
    deployed — per Part B4 it has no production caller. Do not describe it as in use.
    Mark [YOUR REASONING] for why reasoning was separated from speaking.

(b) CORRECT the perception-to-LLM interface. The M3 soft-prompt connector was dropped
    (live build_bmo_stack sets m3_connector=None). Perception reaches the language models as
    RETRIEVED TEXT TAGS via the query predictor against a 1,482-tag candidate set, not as a
    learned embedding projection. State this as a deployment-driven design change and
    forward-reference the results that motivated it. The World-State remains the shared
    latent read by the perception-side specialists.
```

### §3.5 Downstream Specialists — ADD a fifth

```
Add a fifth specialist, from EVIDENCE_LEDGER_V2.md Table 2 and Part B4:

5. Latent reasoning transition head (GLR). 1,049,600 trainable parameters over a frozen
   Qwen3-0.6B backbone. v1 (non-zero init, unnormalised loss) diverged on multi-step
   rollout; v2 (zero init, normalised loss) passed, with a measured 1.47x token reduction
   at K<=10 — considerably below the 5-7x reported in the source literature. State per
   Part B4 that it has no production caller and is sequenced behind speaker
   directive-conditioning. Do not describe it as deployed.
```

---

## ONE THING TO CHECK BEFORE WRITING RESULTS

The streaming path uses **16 frames**, but M2 was trained on 64. The frame-reduction study
measured R@1 34.4 / 33.0 / 32.0 at 64 / 32 / 16 — so the cost is known and the choice is
documented, not silent. Report it that way: a measured accuracy-latency trade, with the
number attached.
