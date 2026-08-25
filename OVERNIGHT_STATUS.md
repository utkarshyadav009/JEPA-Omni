# BMO overnight run — 2026-08-08 → 09 (mecury, 4× RTX PRO 6000)

Autonomous run while you slept. Two goals: (1) train the **Moonshine** Ultravox
projector, (2) generate a **good companion corpus** (with real hostility →
emotion-appropriate reactions) and retrain **both** LLMs on it. Everything below
runs on mecury; **nothing was deployed to the Jetson** — deploy is your morning
call (staged commands at the bottom).

## TL;DR status — ALL COMPLETE (03:20, 2026-08-09)
| Piece | State |
|---|---|
| v9 companion corpus (#204/#205) | ✅ **3,336 rows, 92.7% prompted, hostility 10.5%** |
| v3 thinker corpus (#206) | ✅ 336 examples incl. companion/hostility reasoning |
| Fast-tier **LFM2.5-350M v2** (#209) | ✅ retrained, val_loss **0.64**, `checkpoints/bmo_lfm25_350m_v2_Q8_0.gguf` (379MB) |
| **Qwen3-0.6B thinker v3** (#210) | ✅ retrained, best val_loss **1.95**, `checkpoints/bmo_thinker_qwen3_v3_Q8_0.gguf` (805MB) |
| Moonshine Ultravox projector (#207/#208/#212/#213) | ✅ trained; WER plateau ~0.94 → research track (see §1) |
| llama.cpp embd-input deploy path (#214) | ✅ **PROVEN** — byte-identical to token path, no C++ fork |

### The fix, in the model's own words (held-out, `scratchpad/retrain_fast.log`):
```
USER "You are so stupid and useless."      BMO "Did Beemo do something wrong? Why are you being mean?"
USER "Shut up, nobody likes you."          BMO "That makes BMO feel small and lonely."
USER "I hate you, you're the worst."       BMO "That hurts, Beemo feels sad."
USER "I had the worst day of my life."     BMO "...that sounds exhausting. Want to share one tiny thing that made you smile, no more?"
USER "What's the weather like tomorrow?"   BMO "Let me peek at the sky! <tool_call name=weather day=tomorrow/>"
USER "I love you BMO."                      BMO "Beemo loves you too!"
```
Contextual + emotion-appropriate + tool-calling — the generic-reply bug is fixed.

### Deploy in the morning (after you listen — NOT auto-deployed):
```
bash scripts/deploy_v2_to_jetson.sh      # reversible; backs up current GGUFs, scp v2/v3, repoint live_bmo.py
# then on Jetson:  python3 ~/live_bmo.py
```

## §0 The bug we're fixing (measured, not guessed)
The old v7 corpus was 1595 rows but **only 12% had a real user `prompt`** — 88%
were bare mood lines that `finetune_bmo_minicpm5_lora.py` trains with the fixed
user turn `"Say something."`. So the model learned to emit a mood line **regardless
of what you said**, and had **zero** confrontation examples → it stayed chirpy when
insulted ("I said 'fuck you' and it wasn't mad"). The fix is a corpus dominated by
real **(user utterance → state-conditioned response)** pairs, with a big hostility
slice.

Sample of the new hostility data (GPT-OSS-120B, state = high-stress `stressed`):
```
USER "You're annoying."   → BMO "That... that really hurts BMO's feelings."
USER "Shut up, BMO."      → BMO "Beemo doesn't like being spoken to like that. Please stop."
USER "You're useless."    → BMO "Oh... okay. Beemo will just be quiet then."
USER "I hate you."        → BMO "BMO feels sad when you say that, it hurts."
USER "Nobody likes you."  → BMO "Beemo feels hurt and a little angry, but won't say anything mean back."
```
Genuine range — hurt / firm-boundary / sad / confused — never breaks character,
never chirpy. Maps to `homeostatic_to_mood_state`: hostility bumps stress → mood
`stressed`, so training labels match what `live_bmo.py::update_emotion` produces.

## §1 Moonshine Ultravox projector (per your correction: Moonshine, NOT Whisper)
- New additive module `models/m4d_stt_projector_moonshine.py` — frozen Moonshine
  encoder (416-d, the SAME `moonshine-base` the live loop already loads) + frame-stack
  projector → LFM2.5-350M embedding space (1024-d). One encoder in memory, no
  autoregressive text-decode on this path.
- Trained `scripts/train_stt_projector_moonshine.py` on LibriSpeech clean-100 with
  the **real Ultravox objective**: CE on the transcript **+ KL-distillation from a
  text-only teacher** (the same LLM reading ground-truth text). Result:
  ckpt `checkpoints/bmo_stt_projector_moonshine/projector.pt` (3568 steps ≈ 1 epoch,
  ce≈1.6, kl≈1.5).
- Data note: did **not** download GigaSpeech (disk is 95% full / ~45G free);
  LibriSpeech clean-100 (already present) is the right data for Stage-1 projector
  training anyway.
- Improvements from your guidance (all implemented): **explicit EOS-terminated
  targets** (model learns to stop; was rambling into insertions), **position-based
  loss masking** (pad==eos in LFM2.5, so id-masking would drop EOS supervision),
  **attention masks** on teacher/student forwards, **per-2k checkpoints**, and an
  eval with **rep-penalty 1.2 + audio-length token cap + EOS stop**.
- WER-vs-step sweep (held-out test.clean, 16 utts): 6k=0.98, 8k=0.98, 10k=0.94,
  **SWA(8k,10k,12k)=0.95**. Best individual utterances transcribe near-perfectly,
  but mean WER plateaus ~0.94. **Honest verdict: this is a model-scale limit** — a
  350M *frozen* LLM as the ASR decoder on 100h in ~3 epochs is a weak decoder. The
  projector proves the concept but is NOT production ASR yet. To make it real:
  unfreeze the LLM (Ultravox Stage-2 LoRA), 10× more data, or a bigger decoder.
  **Recommendation: keep Moonshine's text decode as the live STT; projector = research track.**
- ✅ **DEPLOYMENT UNBLOCKED (your ask #3) — no llama.cpp fork needed.** Proved that
  `llama-cpp-python 0.3.34` accepts a **custom embedding prefix** via the
  `llama_batch.embd` buffer: fed HF-computed embeddings into the GGUF and got byte-identical
  output to the token path (`scripts/prototype_llama_embd_input.py`):
  `EMBD-path: ' Paris. The French language...'` == `TOKEN-path: ' Paris. The French language...'`,
  `llama_decode rc=0`. So a projector (or ANY embedding prefix — incl. future perception/JEPA
  embeddings) can drive the deployed GGUF fast tier directly. This is the real architectural
  unlock; it just needs a projector good enough to be worth wiring in.

## §2 Corpus + retrain (auto-chaining)
- `scripts/generate_bmo_companion_corpus_gptoss.py` (v9) slices: hostility (~200),
  emotional_support (~150, ESConv strategies), general_conversation (~150, non-puppet
  range), warmth, companion_memory, tool_use (6 tools), identity, + a MINORITY of
  bare mood lines. All conversational slices carry a real `prompt`.
- Thinker corpus extended with a `companion` category (hostility/emotional-support/
  boundary/repair reasoning) → `data/bmo_thinker_corpus_v3_DRAFT.jsonl`.
- Retrain (`scratchpad/launch_retrain.sh`, fires on corpus-done): fast tier LoRA on
  LFM2.5-350M (LFM2 target modules, synth-upsample 2×, 5 epochs) + thinker LoRA on
  Qwen3-0.6B, then merge→Q8_0 GGUF for both. Held-out **hostility/support/general
  sample generations** are printed at the end of the fast-tier log to verify the fix.

## §3 Morning deploy plan (NOT done automatically — your decision)
Artifacts will be at:
- `checkpoints/bmo_lfm25_350m_v2_Q8_0.gguf`  (new fast tier)
- `checkpoints/bmo_thinker_qwen3_v3_Q8_0.gguf` (new thinker)
- `checkpoints/bmo_stt_projector_moonshine/projector.pt` (Ultravox projector)

Review the sample generations in `scratchpad/retrain_fast.log` first. If the
hostility/companion behavior reads right, deploy is: scp the two GGUFs to the Jetson
`~/bmo_production/models_gguf/`, back up the current ones, point `live_bmo.py` at the
v2/v3 filenames, and re-run. (I'll give exact scp commands in the morning once the
files exist + you've listened.)

---
*(This file is updated live as jobs finish.)*
