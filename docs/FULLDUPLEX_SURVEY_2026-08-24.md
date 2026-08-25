# Awesome-Full-Duplex-SDM — what is actually usable for BMO

Source: `github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM` (54 entries). Read in full, filtered
against BMO's real constraints: **7.6 GB unified memory with ~350–500 MiB free after the
current stack**, an existing fine-tuned speaker/thinker/voice, and a cascaded architecture.

The repo's own taxonomy does most of the filtering: `Component` and `Cascaded` entries bolt
onto BMO; `End-to-end` entries replace it.

---

## 1. THE ONE TO STEAL: KAME's oracle-stream injection

**KAME — Tandem Architecture for Enhancing Knowledge in Real-Time S2S** (SakanaAI,
[arXiv 2510.02327](https://arxiv.org/abs/2510.02327), [code](https://github.com/SakanaAI/kame))

This is BMO's thinker/speaker split, already built and measured, with the one piece our design
is missing.

**Mechanism.** User speech goes to a fast S2S model *and simultaneously* to a backend LLM. The
LLM's text answer is injected into the running S2S generation as a **fourth "oracle stream"**
alongside audio / inner-monologue / input, at the model's own ~80 ms token cycle. Each
injection is prefixed with **a dedicated boundary special token**. When the LLM is late, KAME
**prioritises the most recent response** (it came from a longer, more complete transcript).
Critically, they add **random jitter to oracle arrival timing during training**, so the model
learns to tolerate a slow, variable backend.

| | MT-Bench (reasoning/STEM/humanities avg) | median latency |
|---|---:|---:|
| Moshi (S2S baseline) | 2.05 | 0.0 s |
| **KAME** (GPT-4.1 backend) | **6.43** | **0.0 s** |
| Unmute (cascaded) | 7.70 | 2.1 s |

**Why this matters to us.** In our §18 design the thinker's directive lands *between turns*.
KAME injects it **while the speaker is already talking** — which is exactly the "parallel
decoding" you were reaching for. And the jitter trick is the direct answer to "the thinker
takes 1.0–1.3 s and varies".

**The BMO translation — no new model required.** BMO's TTS is already sentence-chunked at
RTF 0.62, so we do not need 80 ms token-level injection. The analogue is:
1. Speaker emits sentence 1 immediately (measured ~176 ms + 290 ms TTFA).
2. Thinker runs concurrently; its directive arrives ~1.0–1.3 s later — i.e. **during sentence 1's
   playback** (utterances are 2.5–5 s).
3. Sentence 2 is generated conditioned on the directive, with a boundary marker.
4. If the directive is late, use the most recent one or none — never block.

This needs corpus rows teaching *"continue coherently given a mid-turn directive"*, plus
arrival jitter in training. It is a scheduling change on the pipeline we already have.

**Honest caveat from the paper:** KAME "can correct its speech content after getting
contradictory oracles", producing redundant expressions that hurt its own scores. Mid-utterance
steering has a real failure mode — the second sentence can contradict the first.

---

## 2. THE ANSWER TO "I don't mind retraining": Freeze-Omni

**Freeze-Omni** ([arXiv 2411.00774](https://arxiv.org/abs/2411.00774), **code + weights**)

Speech in *and* speech out attached to a **frozen LLM** — adapters trained, backbone untouched.
Trained on **60k multi-round text Q&A on 8 GPUs**, reusing ordinary ASR/TTS pairs rather than
scarce duplex speech corpora. Duplex turn-taking comes from multi-task training. The authors
verify "intelligence in the speech modality is at the same level as the text modality of its
backbone".

**Why this is the most BMO-compatible end-to-end option:** every other end-to-end model asks you
to throw away the fine-tuned speaker, the 4,555-row companion corpus, the 26-directive
vocabulary, and BMO's voice. Freeze-Omni asks you to keep the LLM **exactly as it is** and train
adapters around it. The personality survives by construction, not by luck.

This is the one to prototype if you want a genuinely duplex BMO without restarting the corpus
work.

---

## 3. THE BLUEPRINT for "speaker ⇄ thinker directives + tool calls": DuplexSLA

**DuplexSLA — Synchronized Speech, Language and Action**
([repo](https://github.com/hyzhang24/DuplexSLA))

You asked for a speaker that takes directives from the thinker *and issues directives back for
tool calls*. DuplexSLA is that, as a published architecture:

* **user audio channel** — continuous input
* **assistant audio channel** — TA4 layout: **text anchor + 4 audio tokens per 160 ms chunk**
* **action channel** — a *rate-limited* text stream (≤10 tokens/chunk) carrying delayed
  transcripts, **planning text, interaction-control labels, and structured tool calls**

All three decode jointly from one backbone, so "tool calls are anchored to their own chunks and
run in semantic order" — tool use *without interrupting speech generation*.

**But it is not usable:** base is Step-Audio-2-mini (**~7B**), and only the tech report is out —
"inference code, model checkpoints and DuplexSLA-Bench coming soon". 7B does not fit 7.6 GB
alongside perception.

**Take the idea, not the model:** a separate, rate-limited **action channel** distinct from the
spoken channel is a clean design for BMO. Our `m5_tools.py` ToolDispatcher already parses
`<tool_call .../>` out of the spoken text — which conflates the two channels. Separating them
is the right direction and costs nothing architecturally.

---

## 4. DROP-IN COMPONENTS worth evaluating

| entry | type | open | verdict for BMO |
|---|---|---|---|
| **SoulX-Duplug** ([2603.14877](https://arxiv.org/abs/2603.14877)) | plug-and-play streaming state prediction | **code + weights** | **Evaluate first.** Explicitly designed to bolt onto an existing system — the same slot as our VAP head, but pretrained. Could shortcut the echo-augmented retrain. |
| **X2-Turn** ([2608.10878](https://arxiv.org/abs/2608.10878)) | joint streaming ASR + turn state, dual head | **code + weights** | Right idea — one model replacing SenseVoice **and** the turn head, distinguishing interruption / backchannel / utterance-completion. **But the checkpoint is 4B** (`X2-Turn-4B-0812`), built on Voxtral Realtime. Does not fit. |
| **TurnSense / Turnsense** | lightweight EOU detection | **code + weights** | Small by design — worth a look purely on size. |
| **KAME / MoshiRAG** | retrieval during duplex speech | **code + weights** | Directly relevant to your diary/RAG question — asynchronous knowledge retrieval *while speaking*. |
| **VoXtream / VoXtream2** ([2603.13518](https://arxiv.org/abs/2603.13518)) | full-stream TTS, "extremely low latency", dynamic speaking-rate control | code | Only interesting if we ever leave NeuTTS. We won't — that is BMO's voice. |
| **FlexDuo** | pluggable duplex control (Speak/Listen/**Idle**) | — | Already analysed (§8); the Idle-state labelling scheme still stands. |

---

## 5. VALIDATION: our architecture is the one the field is re-discovering

**X-Talk — "On the Underestimated Potential of Modular Speech-to-Speech Dialogue Systems"**
([2512.18706](https://arxiv.org/abs/2512.18706), code)

A decoupled cascade — VAD, speech enhancement, ASR, emotion, environmental sound, LLM with RAG
and tool use — achieving **sub-second latency without sacrificing modular flexibility**. Their
thesis: specialised end-to-end omni-models "struggle balancing competing objectives", while
modular systems optimise each part independently.

That is BMO, feature for feature, including the environmental-sound stream almost nobody else
has. **This is not a reason to be complacent, but it is a reason to stop treating the cascade as
a compromise.** The two independent failures we measured — the camera dying silently, the
speaker ignoring directives — are integration and data problems, not architectural ones.

---

## 6. DATASETS — relevant to the VAP retrain

The blocking item in §8 is that our VAP head never saw its own echo. Several 2026 releases are
directly on point:

* **TURNS-2K** (data) — turn-detection labelled.
* **DuplexGen** ([2607.26178](https://arxiv.org/abs/2607.26178), data + code) — *adaptive
  synthesis* of human-AI turn-taking dialogues. Synthesis is what we need, because we must
  inject echo ourselves.
* **SOMMELIER** ([2603.25750](https://arxiv.org/abs/2603.25750), Naver, pipeline code) —
  a **data pipeline** for full-duplex SLMs. Pipeline > dataset for us, since our corpus must
  contain BMO's own voice as the reference channel.
* **ConversationalVoice**, **DuplexChat**, **SmoothConv/DuplexConv** — real and synthetic
  full-duplex corpora.

---

## 7. WHAT I WOULD ACTUALLY DO

**Now, no new model:**
1. **KAME-style mid-utterance directive injection**, at sentence granularity, with a boundary
   marker, most-recent-wins, and never blocking. This is the parallel-decoding win and it fits
   the existing pipeline. Add **arrival jitter** to the training rows so the speaker tolerates a
   variable thinker.
2. **Separate the action channel from the spoken channel** (DuplexSLA's idea). Tool calls should
   not be parsed out of speech text.
3. **Evaluate SoulX-Duplug** before investing further in the VAP retrain — it may already be
   what we were about to train.

**If you want a genuinely duplex BMO and accept retraining:**
4. **Freeze-Omni**, because it is the only end-to-end route that keeps the fine-tuned LLM frozen
   and therefore keeps the personality and the corpus investment. Everything else discards them.

**Do not pursue:** DuplexSLA (7B, unreleased), X2-Turn (4B), Moshi / Qwen3.5-Omni / Seeduplex
(too large or closed), VoXtream (we are keeping NeuTTS).

**The memory reality check that filters most of this list:** the current stack leaves
~350–500 MiB. Any model above ~1B is not a candidate regardless of merit, and that rules out
most of the end-to-end column.
