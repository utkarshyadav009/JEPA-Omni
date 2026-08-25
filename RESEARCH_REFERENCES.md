# BMO — research references & model provenance

Papers, models, and datasets underpinning the BMO pipeline, grouped by role.
"Used" = currently in the deployed/attempted pipeline; "Candidate/researched" =
evaluated or on the roadmap. **arXiv IDs are best-effort — verify before formal
citation** (model *releases* often have no paper; those link to repo/HF).

---

## 1. Full-duplex dialogue / turn-taking / backchannels
*(the target UX: continuous listen+speak, backchannels, barge-in)*
| Work | Org / Authors | Yr | Role in BMO | Ref |
|---|---|---|---|---|
| **Moshi** — speech-text foundation model for real-time full-duplex dialogue | Kyutai | 2024 | north-star for full-duplex + user-prediction ("predict what the user says next") | arXiv:2410.00037 |
| **dGSLM** — Generative Spoken Dialogue Language Modeling | Meta (Nguyen et al.) | 2022 | backchannels emerge as a *separate stream* from turn speech (design basis of `PrebuiltVoiceBank`) | arXiv:2203.16502 |
| **VAP** — Voice Activity Projection (turn-taking prediction) | Ekstedt & Skantze | 2022 | short-horizon prosody prediction decides *when* to backchannel/take turn | arXiv:2205.09812 *(verify)* |
| Filler/thinking audio at turn-start (voice-agent practice) | industry (Lex/Twilio/Sierra) | — | `thinking_filler` masks LLM/TTS latency at SPEAK start | (engineering practice) |

## 2. Streaming STT + VAD  *(the current live path)*
| Work | Org | Yr | Role | Ref |
|---|---|---|---|---|
| **SenseVoice-Small** (FunAudioLLM) | Alibaba | 2024 | **chosen live STT** — non-autoregressive, text+emotion, RTF ~0.075 on Jetson | arXiv:2407.04051 *(FunAudioLLM report)* / HF `FunAudioLLM/SenseVoiceSmall` |
| **silero-VAD** | snakers4 | — | endpointing inside sherpa-onnx (C++), replaced the failing Python Silero | github.com/snakers4/silero-vad |
| **Moonshine** — ASR for live transcription | Useful Sensors | 2024 | prior live STT (~97 ms); Ultravox projector encoder | arXiv:2410.15608 *(verify)* |
| **Whisper** — robust ASR via weak supervision | OpenAI | 2022 | earlier STT + projector-encoder baseline | arXiv:2212.04356 |
| **Zipformer** — faster/better ASR encoder | k2-fsa | 2023 | streaming transducer candidate (true mid-utterance partials → unblocks Tier-1) | arXiv:2310.11230 |
| **sherpa-onnx / next-gen Kaldi** | k2-fsa | — | the C++/onnxruntime STT+VAD runtime BMO now uses | github.com/k2-fsa/sherpa-onnx |

## 3. Speech↔LLM fusion  *(Ultravox track — now closed as research)*
| Work | Org / Authors | Yr | Role | Ref |
|---|---|---|---|---|
| **Ultravox** — speech→LLM, no ASR text stage | Fixie.ai | 2024 | the no-text projector approach we implemented (Moonshine→proj→LFM2.5) | github.com/fixie-ai/ultravox · HF `fixie-ai/ultravox` |
| **Flamingo** (Perceiver Resampler) | DeepMind (Alayrac et al.) | 2022 | the resampler projector variant tested (Exp C) | arXiv:2204.14198 |
| **Perceiver / Perceiver IO** | DeepMind (Jaegle et al.) | 2021 | fixed-latent cross-attention resampling | arXiv:2103.03206 · 2107.14795 |
| SALMONN / Qwen-Audio | Tsinghua / Alibaba | 2023 | related audio-LLM fusion (context) | arXiv:2310.13289 · 2311.07919 |
| KL-distillation from the input side (speech-LLM as prosody-aware text-LLM) | (arXiv search) | 2025 | basis for our CE + KL-teacher projector objective | *title-cite; verify ID* |

## 4. LLM backbones  *(BMO's brains)*
| Model | Org | Role | Ref |
|---|---|---|---|
| **LFM2 / LFM2.5-350M** | Liquid AI | **fast tier** (v2, val 0.64) | HF `LiquidAI/LFM2.5-350M` (release) |
| **Qwen3-0.6B** | Alibaba | **thinker tier** (v3, native `<think>` CoT) | arXiv:2505.09388 *(Qwen3 report, verify)* / HF `Qwen/Qwen3-0.6B` |
| **GPT-OSS-120B** | OpenAI | local teacher for corpus distillation (v9 + thinker) | HF `openai/gpt-oss-120b` (release) |

## 5. TTS / neural codec  *(BMO's voice)*
| Model | Org | Role | Ref |
|---|---|---|---|
| **NeuTTS-Air + NeuCodec** | Neuphonic | deployed streaming emotion voice (GGUF + INT8 ONNX decoder) | HF `neuphonic/neutts-air`, `neuphonic/neucodec` (release) |
| Chatterbox / CosyVoice2-0.5B / IndexTTS-2 | Resemble / Alibaba / — | emotion-TTS candidates (researched, not deployed) | CosyVoice2 arXiv:2412.10117 *(verify)* |

## 6. Companion / emotional-support dialogue  *(v9 corpus design)*
| Dataset / Work | Authors | Yr | Role | Ref |
|---|---|---|---|---|
| **PERSONA-CHAT** | Zhang et al. | 2018 | persona-grounded dialogue (companion identity) | arXiv:1801.07243 |
| **ESConv** — Towards Emotional Support Dialog Systems | Liu et al. | 2021 | 8 support strategies → `emotional_support` slice | arXiv:2106.01144 |
| **EmpatheticDialogues** | Rashkin et al. | 2019 | empathetic response modeling | arXiv:1811.00207 |
| **PAL** — Persona-Augmented Emotional Support | — | 2022 | persona + support generation | arXiv:2212.09235 |
| Persona-grounded safety of AI companions | — | 2026 | *warning:* narrow/mirroring companions = failure mode → drove non-sycophantic + hostility slices | ACL 2026 |
| CPED — Chinese personalized & emotional dialogue | scutcyr | — | personality+emotion annotation reference | github.com/scutcyr/CPED |

## 7. Perception / JEPA  *(north-star; JEPA-memory agent's track)*
| Work | Org | Role | Ref |
|---|---|---|---|
| **V-JEPA 2** — video JEPA | Meta | vision encoder (ViT-L, 16-frame) | arXiv:2506.09985 *(verify)* / HF `facebook/vjepa2-vitl-fpc64-256` |
| **I-JEPA** — image JEPA (method origin) | Meta (Assran et al.) | predictive-embedding paradigm | arXiv:2301.08243 |
| **WavJEPA** | — | audio JEPA (ambient World-State) | (see repo `audio_encoder.py`) |
| **EmbeddingGemma** | Google | text-target space for M1 alignment | HF `google/embeddinggemma` (release) |

## 8. Training / inference methods
| Method | Authors | Role | Ref |
|---|---|---|---|
| **LoRA** — low-rank adaptation | Hu et al. | all BMO fine-tunes; projector Stage-2 LLM unfreeze | arXiv:2106.09685 |
| **SWA** — stochastic weight averaging | Izmailov et al. | projector checkpoint averaging (Exp) | arXiv:1803.05407 |
| **Knowledge distillation** | Hinton et al. | GPT-OSS→BMO corpus + projector KL teacher | arXiv:1503.02531 |
| **Speculative decoding** | Leviathan / Chen et al. | Tier-1 "predict the user, run ahead" concept | arXiv:2211.17192 · 2302.01318 |

---

### How these map to the deployed pipeline
`ReSpeaker mic → sherpa-onnx VAD (silero) → SenseVoice STT → CognitiveCoreRouter
(LFM2.5-350M fast ⇄ Qwen3-0.6B thinker) → ToolDispatcher → NeuTTS-Air emotion voice`,
with `PrebuiltVoiceBank` (dGSLM/Moshi-inspired backchannels) masking latency, and a
`HomeostaticState` driving mood. Perception (V-JEPA 2 + WavJEPA) and JEPA-space
identity memory are the north-star layer merged next.

*Companion note: the user plans to swap to **smaller STT/TTS** later to free memory
for the perception stack (SenseVoice ~600 MB + NeuTTS voice ~700 MB are the swap
candidates); tracked, not for now.*

## Latent-space reasoning (surveyed 2026-08-16, for the BMO thinker)

Motivation is measured, not speculative: enabling Qwen3's native CoT took the thinker from
650 ms to **1,749–3,509 ms**, making it the dominant leg of the whole pipeline.

* **[GLR — Geometric Latent Reasoning Induces Shorter Generations in LLMs](https://arxiv.org/abs/2606.02248)**
  — *the applicable one.* A lightweight **transition head** predicts iterative direction updates
  within the model's pretrained token-embedding space, formulating latent reasoning as geometric
  path approximation. Textual CoT traces act as anchors while permitting continuous deviation
  from exact token embeddings. At inference it replaces an **initial segment** of explicit
  reasoning with a fixed number of continuous latent steps, then resumes normal token decoding.
  **Evaluated on Qwen3 0.6B and 1.7B — our exact thinker.** Fewer generated tokens without any
  explicit length objective. Published ~June 2026; **no public weights**, so the head is ours to
  train (tasks #13/#14).
* **[JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation](https://arxiv.org/abs/2512.19171)**
  — JEPA reasoning engine + a separate **"Talker"** module for linguistic reconstruction. Claims
  error containment (token-level failures cannot propagate into the latent chain), continuous
  guidance from the whole lossless trajectory, and uncertainty via mixed latent vectors. This is
  **architecturally BMO's thinker/speaker split with a latent interface** — the north star for
  the split we already have. No released model.
* **[LaSER — Internalizing Explicit Reasoning into Latent Space for Dense Retrieval](https://arxiv.org/abs/2603.01425)**
  (SIGIR 2026) — **dense retrieval, not dialogue.** Dual explicit/latent view self-distillation
  with trajectory alignment; the >99% latency reduction is measured against rewrite-then-retrieve
  pipelines. Frequently mis-cited as a general "0.6B latent-reasoning drop-in" — it is not.
  Potentially relevant to the *retrieval* side of memory, not to the thinker.
* **[COCONUT — Training LLMs to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)**
  — foundational: feed the last hidden state back as the next input embedding. Reproduced on
  GPT-2 / Llama3-1B / Qwen3-4B; CODI reports matching explicit CoT on GSM8k with 3.1× fewer tokens.
* **[ThinkJEPA](https://arxiv.org/abs/2603.22281)** — a large VLM as high-level "thinker" for a
  V-JEPA2-style latent predictor. Adjacent to the perception-prefix track, not the thinker track.
* **"Non-Causal Latent Alignment (Full-Duplex Specialized)"** — searched, **not found**. Treat as
  unverified until a citation surfaces.

**Deployment hook is already banked:** `llama_batch.embd` accepts a custom embedding prefix and
produces **byte-identical** output to the token path (`scripts/prototype_llama_embd_input.py`,
llama-cpp-python 0.3.34). No C++ fork. The same mechanism serves the JEPA-perception-prefix track.

**Do not justify this work on KV cache.** At `n_ctx=512` on a 0.6B model the KV is negligible —
KV quantization was already measured worthless at this context size. The win is wall-clock tokens.
