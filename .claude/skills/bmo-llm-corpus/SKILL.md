---
name: bmo-llm-corpus
description: How BMO's cognitive core and its training corpora are built: Tier-1 speculative turn-taking and tool-calling, the 350M fast tier + Qwen3-0.6B thinker rework, the v9 companion corpus and v2/v3 LLM retrain with Moonshine/Ultravox, and the corpus-generation rules earned from three failed attempts. Use when generating or filtering companion-dialogue corpora, retraining or swapping the fast/reasoning LLM tiers, converting or deploying new GGUFs, or working on speculative prefetch, tool-calling, or the speech projector.
---

# BMO cognitive core, LLM tiers and corpus generation

Design decisions and measured results for BMO's LLM tiers and the corpora that train them.
The corpus-generation rules at the end were paid for with three failed attempts — read them
before generating any new corpus.

## Tier 1 speculative turn-taking + tool-calling (2026-08-08)

**Tier 1 speculative prefetch — BUILT + BENCHMARKED.** `models/m5_speculative.py::
SpeculativePrefetcher` is the Jetson-feasible "predict the user, run ahead" (Moshi-flavored)
mechanism: on partial transcripts during the user's turn it speculatively runs the fast
tier (+ pre-decodes the first audio chunk) in the background; at end-of-turn it COMMITS the
result if the final transcript matches (token-overlap ≥ 0.80), else silently falls back
(a wrong guess is NEVER spoken). `scripts/bench_tier1_speculative.py` on the REAL Jetson
models (fast tier + StreamingVoice, no perception so no M3/preflight issue) measured:
**baseline response path = fast-tier 430ms + TTS-first-audio 463ms = 893ms; on a HIT
perceived latency ~0ms (~811ms avg removed).** Full design + results in
`SPECULATIVE_TURNTAKING.md`. NOT yet wired into `m5_streaming_loop` (needs mid-turn
Moonshine decodes + mic hardware to verify) — the component + benchmark stand alone.

**Tool-calling — the missing execution layer is BUILT.** The LLM was fine-tuned to EMIT
`<tool_call name=weather day=today/>` but nothing parsed/executed it (tool use was dead
end-to-end). `models/m5_tools.py::ToolDispatcher` closes it: `parse_tool_calls` →
`ToolRegistry.execute` → fold the result back into BMO's spoken line (template mode, or an
optional LLM second pass once trained to consume `[tool_result ...]`). Handlers: time/date
are REAL; weather/timer/reminder/search are clean stubs with obvious real-API injection
points. Tested end-to-end (e.g. `<tool_call name=time/>` → "It's 9:38 AM right now!").
`assets/tool_call.gbnf` constrains llama.cpp decoding to valid tool-call syntax (fine-tune
teaches WHEN/WHAT, grammar guarantees FORM). Corpus expansion (`generate_bmo_text_corpus_
gptoss.py` extended to 6 tools + a `companion` slice) generates a DRAFT for review
(`data/bmo_companion_tools_v8_DRAFT.jsonl`); LLM LoRA retrain on it is left for human review
(the corpus is an unverified draft by design), then it's a straightforward retrain+deploy.

**Emotion voice v3 (2026-08-08).** Fixes the two v2 defects the user heard: (1) BMO
mispronounced — root cause was the Fish TRAINING audio spelling "B-M-O" as letters
(confirmed by A/B), fixed by regenerating the 338 BMO-line audios with "Beemo" spelling
(`scripts/regen_bmo_lines_beemo.py`) so audio matches the Beemo-normalized text; (2) 7s
garble on happy/content/anxious — EOS over-run because those moods had the longest training
clips (happy max 431 codes), fixed by a >320-code length filter in prep + finetune, plus a
length-aware inference safety cap in `StreamingVoice.stream`. Pipeline: `_run_voice_v3b.sh`
(prep→retrain→GGUF) → `bmo_neutts_emotion_v3`.

**Jetson training-box disk is chronically near-full (879G, was 100%).** Full NeuTTS
fine-tunes save 3.1G/checkpoint (model+optimizer); `save_total_limit=3` × several runs fills
it fast and a save fails mid-train with `No space left on device`. Safe cleanup that unblocks:
`rm -rf checkpoints/*/checkpoint-*` (intermediate checkpoints; every `/best`, GGUF, and the
deployed v5 are preserved). Freed ~70G this way. Watch free space during any NeuTTS retrain.

## Cognitive core rework (2026-08-08): 350M fast + Qwen3-0.6B thinker

Replaced BOTH tiers after on-Jetson benchmarking (all cleaned corpus, typographic Unicode
scrubbed DB-wide -- see the ascii_normalize note):
- **Fast tier: LFM2-700M v9 -> LFM2.5-350M** (`bmo_lfm25_350m_v1_Q8_0.gguf` +
  `tokenizers/lfm25_350m_tok`). Retrained (LoRA, `finetune_bmo_minicpm5_lora.py` is
  model-agnostic; LFM2 target modules are `q_proj,k_proj,v_proj,out_proj,w1,w2,w3`, NOT the
  MiniCPM names). ~2x faster + lighter: 226 vs 405ms/24tok, 9.4 vs 16.9ms/tok, ~half memory.
  val_loss 0.457, personality + state-conditioning intact. Watch: occasionally emits "BMW"
  for "BMO" (folded to Beemo in `normalize_bmo_text`).
- **Reasoning tier: MiniCPM5-1B -> Qwen3-0.6B thinker** (`bmo_thinker_qwen3_v2_Q8_0.gguf` +
  `tokenizers/qwen3_thinker_tok`). The old MiniCPM5 had reasoning OFF (just a bigger
  conversational model = redundant with the fast tier). Qwen3-0.6B is a REAL System-2
  reasoner (native `<think>` CoT), fine-tuned (`finetune_thinker_qwen3.py`, LoRA) on a
  DEDICATED reasoning corpus (`generate_thinker_corpus_gptoss.py`: 448 examples =
  reasoning + tool-orchestration + state-grounded, prose CoT distilled from GPT-OSS-120B).
  Benchmarked best of the candidates: 18.2ms/tok, reasoning-capable, smallest (DeepSeek-R1-
  Distill-1.5B was heavier/slower, 1.3GB Q8). Tested: reasons in-character AND emits tool
  calls. `GGUFReasoningTier.generate` now STRIPS the `<think>...</think>` block (only the
  answer is spoken) and uses max_new_tokens=320 (60 would cut off mid-CoT). Escalation is
  now genuinely slower (~5s full CoT) but that's the exception path + real deliberation.
  Rationale + benchmark table in `COGNITIVE_CORE_ANALYSIS.md`. Thinker↔perception hookup
  follows the newer prediction-style integration (M3 connector DROPPED) -- see the
  `bmo-pipeline-vision` memory.

## v9 companion corpus + v2/v3 LLM retrain + Moonshine Ultravox (2026-08-09, overnight on mecury)

**Root-caused the "generic reply / didn't get mad at insults" bug**: the old v7 corpus was
1595 rows but only **12% carried a real user `prompt`** -- the other 88% were bare
mood-expression lines, which `finetune_bmo_minicpm5_lora.py` trains with the fixed user turn
"Say something." So the model learned to emit a mood line regardless of input, and had ZERO
confrontation data. **Fix = a corpus of (user-utterance -> state-conditioned response) PAIRS.**

- **v9 corpus** (`scripts/generate_bmo_companion_corpus_gptoss.py`, GPT-OSS-120B local ->
  `data/bmo_companion_corpus_v9.jsonl`): **3,336 rows, 92.7% prompted, balanced** (general 19%,
  support 15%, playful 12%, memory 11%, hostility **10.5%**, tools 10%, warmth 9%, identity 5%,
  mood-lines 7%). Hostility labeled high-stress `stressed` (matches `homeostatic_to_mood_state`
  + `live_bmo.py::update_emotion`). Support slice uses ESConv strategies. Generator prints a
  category histogram + hostility fraction at the end.
- **Fast tier v2** (`checkpoints/bmo_lfm25_350m_v2_Q8_0.gguf`, 379MB, val_loss 0.64): retrained on
  real 971 lines + v9 (synth-upsample **2x**, not 8x -- corpus is big now). `finetune_bmo_minicpm5_
  lora.py` gained `--synth-upsample` + hostility/support/general sample checks. Held-out verified:
  insults -> hurt/firm in-character; support -> ESConv empathy; tools fire; NOT chirpy.
- **Thinker v3** (`checkpoints/bmo_thinker_qwen3_v3_Q8_0.gguf`, 805MB, best val_loss 1.95 @ epoch 2;
  later epochs overfit the 336-ex corpus, best-checkpointing handles it): `generate_thinker_corpus_
  gptoss.py` gained a `companion` scenario category (hostility/support/boundary/repair reasoning).
- **Deploy**: `scripts/deploy_v2_to_jetson.sh` (STAGED, reversible; backs up current GGUFs, scp
  v2/v3, repoints `~/live_bmo.py`). NOT auto-deployed; needs a listen first. Tokenizers unchanged.

**Moonshine Ultravox projector (per user: Moonshine, NOT Whisper).** `models/m4d_stt_projector_
moonshine.py` (frozen `moonshine-base` 416-d + frame-stack projector -> LFM2.5-350M 1024-d),
trained `scripts/train_stt_projector_moonshine.py` on LibriSpeech clean-100 with the real Ultravox
objective (CE + **KL-distillation from a text teacher**). Fixes per user guidance: EOS-terminated
targets (was rambling into WER>1 insertions), **position-based loss masking** (pad==eos in LFM2.5,
so id-masking drops EOS supervision), attention masks, per-2k checkpoints, length-capped +
rep-penalty-1.2 + EOS-stop eval, SWA. **WER plateaus ~0.94 -- model-scale limit** (350M *frozen*
LLM is a weak ASR decoder on 100h). **Verdict: keep Moonshine text decode as live STT; projector =
research track** (needs unfrozen-LLM Stage-2 / 10x data).

**DEPLOYMENT UNLOCK:** `llama-cpp-python 0.3.34` accepts a **custom embedding prefix** via the
`llama_batch.embd` buffer (`llama_batch_init(n_tokens, embd_dim, n_seq_max)` + `llama_decode`) --
NO C++ fork. Proven in `scripts/prototype_llama_embd_input.py`: HF embeddings fed to the GGUF
produced **byte-identical output to the token path**. So a projector (or ANY embedding prefix,
incl. future perception/JEPA embeddings) can drive the deployed GGUF fast tier directly -- the
mechanism the `bmo-pipeline-vision` north-star needs.

## Corpus-generation rules earned 2026-08-16 (three failed attempts)

**Count is not quality.** A directive corpus passed a 150-row count gate with 395 rows and was
still actively harmful: 197 of them paired a RANDOM chain-of-thought with a line written for a
different directive, which trains the speaker to IGNORE instructions — worse than the defect
being fixed. Found by reading rows, not by any count. **When a generator produces a (context,
target) pair, the context must be generated WITH its target, never substituted afterwards**, and
rows should carry a provenance flag a gate can verify.

**Salvage malformed JSON per-item, not per-array.** 48% of generations (27/56) died on
`json.loads` because BMO's lines are full of apostrophes and inner quotation; one bad entry
discarded the other three. Extracting each `"key": "..."` independently took yield 82 → 395.
Then check for TRUNCATION — salvage can stop at an unescaped inner quote and yield half a
sentence, which is worse than dropping the line.

**A directive must be satisfiable from the scene.** *"greet them by name"* with no name in the
scene makes the generator refuse the instruction (correctly, per HARD_RULES) and emit a line that
does something else — 23 rows teaching "instruction X → do Y".

**Normalize every text field, not just the target.** `ascii_normalize` was applied to `text` and
not `prompt`, leaving 100 prompts with typographic Unicode in a DB scrubbed of it everywhere else.

**`--real-metadata` is a Jetson path.** It defaults to `~/../home/bmo/BMO-LabelData/...` =
`/home/bmo/...`, which does not exist on mercury; training dies there. `load_real_lines` reads
TEXT only, so the CSV now lives at `data/real_speech/metadata.csv` (916 lines, matches v5). Pass
`--real-metadata data/real_speech/metadata.csv` when training on mercury.

**Never wait on or signal a process NAME — 5th and 6th occurrences this session.** `pkill -f
jetson_real_demo` over SSH matched the tailscale-ssh child and the `bash -c` wrapper (both carry
the pattern in their own argv) and killed its own shell with exit 255. Separately, `ps`/`pgrep`
PID capture twice raced the model load and returned the setsid wrapper, so `kill -0` reported
"not running" for a healthy job and a chain fired early against a stale file. **Wait on a LOG
MARKER printed after the work completes** — it cannot be true early and cannot match an argv.
