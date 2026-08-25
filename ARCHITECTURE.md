# BMO — what actually exists, what runs, and what it's for

Written 2026-08-14 because a lot has been trained and it is genuinely hard to track. This is
the map. **Every number here is measured and sourced**; anything unproven is labelled.

---

## 0. TL;DR — the one-paragraph version

BMO has **two brains** (a fast tier for instant replies, a thinker for reasoning), **one
voice**, **one face**, and **a perception stack**. The brains, voice and face work. Perception
works for *what is happening* (actions, sounds) but **not for *what things are*** (rooms,
objects) — because its encoder was trained on action-and-sound video, not scenes. That single
gap is why "describe my room" fails, and it is the only genuinely broken thing.

---

## 1. What is DEPLOYED on the Jetson right now

| component | what it is | status / measured |
|---|---|---|
| **Fast tier** | LFM2.5-350M v2, Q8 GGUF, LoRA'd on the v9 companion corpus | live · **9.4 ms/token** |
| **Thinker** | Qwen3-0.6B v3, Q8 GGUF, real `<think>` CoT (stripped before speaking) | live · **18.2 ms/token** · 1103 MiB |
| **Voice** | NeuTTS-Air BMO fine-tune + 12 emotion tokens → NeuCodec ONNX decode | **BROKEN** — `import onnxruntime` aborts on this device (not memory) |
| **STT** | SenseVoice-Small via sherpa-onnx (+ Moonshine for the turn-taking head) | live · RTF 0.075 |
| **Face** | `BMO_Engine` C++/raylib on its own X server, CSI camera via nvarguscamerasrc | live · 350 MiB + 86 MiB Xorg |
| **Perception** | V-JEPA2 ViT-L + WavJEPA base + WavJEPA nat → **M2** predictor | live · **1.1–1.7 s** per refresh (MAXN_SUPER) |

**Power mode matters more than anything else here:** the device was found in **7W mode**
(4/6 cores, GPU 408 MHz). Restoring **MAXN_SUPER** (`nvpmodel -m 2` + reboot + `jetson_clocks`)
made perception **~7x faster**. Check `nvpmodel -q` before trusting any latency number.

---

## 1b. THE ONE QUESTION THAT KEEPS COMING UP — what actually feeds the LLM?

**Nothing feeds the LLM an embedding. The LLM receives TEXT.** That is the whole answer, and
every other confusion downstream of it dissolves once it is stated plainly.

```
  camera ──┬─► V-JEPA2 ViT-L ──┐
           │                   ├─► M2 ──┐
  mic ─────┴─► WavJEPA base ───┘        │      ┌── these four are STREAMS ──┐
           │                            ├──────┤ m2 · vision · ambient ·    │
           └─► SigLIP2 vision ──────────┘      │ scene                      │
                                               └──────────┬─────────────────┘
                                                          ▼
                                             QUERY PREDICTOR  (asks: "what is X?")
                                                          │
                                                   one 768-d vector
                                                          │
                                            nearest-neighbour vs 1,372 TAGS
                                                          │
                                                    ▼ PLAIN TEXT ▼
                                          "a person sitting", "a desk"
                                                          │
                                                       THINKER  ──►  SPEAKER
```

**SigLIP2 did NOT replace M2.** They are different jobs and both are in the deployed
checkpoint (`sig_runD_proj768`, `token_sources = m2,vision,ambient,scene`):

| stream | encoder | answers | why it stays |
|---|---|---|---|
| `m2` | AVJepaPredictor over V-JEPA2 + WavJEPA | audio-visual congruence — do the sound and picture agree? | dropping audio makes the model answer sound questions from the picture 93% of the time |
| `vision` | V-JEPA2 ViT-L | motion, actions, temporal order | SigLIP2 is per-frame and cannot see "door opening" vs "door closing" |
| `ambient` | WavJEPA base | sound events | audio-following collapses to ~0 without it |
| `scene` | SigLIP2 vision tower | what things ARE — rooms, objects, people | V-JEPA2 was blind to this; it retrieved "opening a microwave oven" for a bedroom |

**And SigLIP2 did NOT replace EmbeddingGemma either — pre-encoding did.** EmbeddingGemma was
only ever there to turn the *question* and the *candidate answers* into vectors. Both sets are
now encoded once, offline, on mercury, and only the vectors ship. Neither text tower is
resident on the Jetson. Measured: the question encoder costs **0 MiB**, the candidates
**2 MiB**, against 578 + 355 = **933 MiB** before.

### Why M2 → LLM is dead, and what replaced it
**M3 was the M2-embedding-into-the-LLM path, and it is DEAD.** It soft-prompted the world
state into a 1.5B LLM to *generate* a caption: F1 0.317, 1–6 s per generation, and
confidently wrong. A second attempt (perception-prefix, 16 soft tokens into the thinker)
scored F1 **0.269** — worse. Both were beaten by retrieval, which runs in **25 ms** on device.

So the ordering is settled by measurement, not preference:

| approach | result | status |
|---|---|---|
| M3 connector: world-state → soft prompt → LLM generates | F1 0.317, 1–6 s | **DEAD** |
| perception prefix: 16 soft tokens → thinker generates | F1 0.269 | **DEAD** |
| **query predictor → retrieve text → LLM reads text** | **R@1 0.737, 25 ms** | **DEPLOYED** |

The `llama_batch.embd` mechanism that would let embeddings drive the GGUF directly is proven
and banked (`scripts/prototype_llama_embd_input.py`, byte-identical output to the token path)
— but nothing uses it, because the thinker has never been trained to interpret JEPA-space
vectors and this project paid for that lesson twice.

---

## 2. What each trained thing IS (the part that gets confusing)

Think of it as **one frozen perception trunk** with **several heads** bolted on over time.
The trunk never changed; the heads are different attempts to *use* it.

```
                  ┌─────────── FROZEN TRUNK ───────────┐
 camera ─────────►│ V-JEPA2 ViT-L ──┐                  │
 mic (ambient) ──►│ WavJEPA base    ├──► M2 predictor ─┼──► world-state + pre-pool tokens
                  │ WavJEPA nat  ───┘                  │
                  └────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┬─────────────────┐
        ▼                           ▼                           ▼                 ▼
  [1] M3 connector           [2] embed predictor        [3] query predictor   [4] identity head
      DEAD                       superseded                  LIVE-ready           LIVE-ready
```

| # | name | what it does | trained on | result | verdict |
|---|---|---|---|---|---|
| 1 | **M3 connector** | soft-prompts the world-state into a 1.5B LLM to generate captions | VGGSound rich captions | F1 **0.317** | **DEAD** — 1–6 s to generate, hallucinated |
| 2 | **M2 embed predictor** | one vector per clip, matched to caption embeddings (VL-JEPA style) | VGGSound + Action100M | score **62.05** | superseded by #3 |
| 3 | **Query predictor** | *ask a question*, get a query-conditioned embedding → retrieve a caption | VGGSound ×6 + Action100M ×2 | within-clip **0.897** (chance 0.167), R@1 **0.715** | **works in-domain** |
| 4 | **Identity head** | joint audio+visual → 256-d identity embedding for the memory | VoxCeleb2, 4,420 speakers | TAR@FAR1% **0.765**, AUC **0.966** | **works** |
| 5 | **Perception prefix** | scene → 16 soft tokens → thinker generates free-form text | full corpus, 8k steps | F1 **0.269** | **underperforms** #1 and #3 |

### Also trained, and closed
* **Ultravox / Moonshine speech projector** — no-text speech→LLM. WER plateaued 0.903.
  **CLOSED**: 350M decoder is capacity-walled for ASR. The `llama_batch.embd` mechanism it
  proved is banked and reused.
* **V-JEPA 2.1** — better encoder on paper. **37.98 s/forward on Jetson** (15.6x slower).
  Dissertation-only, never on-device.

---

## 3. The memory (the north-star feature) — how it works

```
face crop ──► ViT-L ──┐
                      ├──► IDENTITY HEAD ──► 256-d embedding ──► JepaMemory
voice ─────► WavJEPA ─┘                                            │
                                                                   ▼
                                            cosine vs enrolled centroids + threshold
                                                                   │
                                    ┌──────────────────────────────┼───────────────┐
                                    ▼                              ▼               ▼
                              "that's Alice"              "I don't know you"   (ambiguous)
```

Measured on **884 unseen identities**: at a 1%-false-accept operating point, **51.5 %
recognised, 48.1 % "I don't know you", 0.4 % wrong name**. Enrolment sharpens it (1 clip
0.584 → 8 clips 0.760). Vision and voice are **complementary** — joint 0.765 beats voice-only
0.694 and vision-only 0.571.

**Not yet wired into the live loop**, and it needs a face detector (the deployed
`motion_tracker.cpp` has the camera pipeline but no face crops for this head).

---

## 4. THE PROBLEM, precisely

V-JEPA2 + M2 were trained on **VGGSound + Ego4D** — *actions and sound events in video*.
They encode "something is happening". They do **not** encode "this is a bedroom with a desk".

Measured on a real frame of your room, same 121k caption bank, only the encoder swapped:

| encoder | what it retrieved |
|---|---|
| V-JEPA2 → M2 | *"opening a microwave oven"* ❌ |
| **SigLIP2** | *"someone seated on a black leather couch inside a room adorned with wall art"* ✅ |

And on a probe of 4 room sentences vs 6 event sentences, SigLIP2 ranked **room 1-2-3-4**, with
*microwave* (−0.019), *car door* (−0.041) and *military parade* (−0.043) **negative**.

This is not a head problem. Retrieval AND generation both failed, because both read the same
blind representation. **The encoder is the bottleneck.**

---

## 5. "SigLIP2 is an image model — what about video?"

**Already answered by your own M1 gate.** `baseline_siglip2.py` uses the standard *meanP*
recipe: sample K frames, encode each independently with SigLIP2, mean-pool into one video
embedding. Zero-shot, that scored **R@1 32.5 on MSR-VTT video retrieval — versus the trained
V-JEPA2 spine's 22.5.** A frame-wise image model, pooled, already beat the video model at
video retrieval in this project.

What SigLIP2 genuinely cannot do is **motion and temporal order** — "the door is opening" vs
"the door is closing" look the same to it. That is exactly what V-JEPA2 + WavJEPA + M2 are
good at. So they are complements, not substitutes:

| | V-JEPA2 + WavJEPA + M2 | SigLIP2-base |
|---|---|---|
| what is HAPPENING (motion, actions, sound) | **strong** | weak |
| what things ARE (rooms, objects, people) | **blind** | **strong** |
| Jetson cost | 1.1–1.7 s / refresh | **19 ms / frame** |
| streaming-friendly | full window re-encode | **per-frame, incremental** |

**On streaming specifically — this is the part that helps you:** SigLIP2 is per-frame, so it
can run on *one* frame whenever you want (19 ms) instead of re-encoding a 10 s window. That
is a genuinely streaming-friendly perception signal, which V-JEPA2 is not.

---

## 6. Proposed target architecture

```
                 ┌── SigLIP2-base (1–4 frames, 19 ms each) ──► SCENE stream ──┐
 CSI camera ─────┤                                                            │
                 └── V-JEPA2 ViT-L (16 frames) ──┐                            │
                                                 ├── M2 ──► MOTION/AV stream ─┤
 mic (ambient) ───── WavJEPA base + nat ─────────┘                            │
                                                                              ▼
                                                              QUERY PREDICTOR (multi-source)
                                                                              │
                                              ┌───────────────────────────────┼──────────────┐
                                              ▼                               ▼              ▼
                                     retrieve a caption            identity head      (future) generate
                                              │                          │
                                              └────────► THINKER ◄───────┘
                                                          │
                                                   fast tier → voice → face
```

The query predictor **already accepts arbitrary named streams** (`source_dims`), so adding
SigLIP2 is a config entry plus a retrain — not new architecture. And your own ablation says
combining streams wins: `m2+vision` **0.478** > `vision` 0.447 > `m2` 0.385.

### The memory trade, honestly

> **CORRECTION (2026-08-14).** An earlier version of this section claimed SigLIP2 means "no
> bank to curate". **That was wrong.** SigLIP2 does not *produce* words, it *scores* them —
> so a candidate list is inherent to retrieval itself, not an artifact of EmbeddingGemma. The
> only bank-free alternative is generation, and both generation attempts measured worse
> (M3 F1 0.317, perception-prefix 0.269). What SigLIP2 actually removes is the **121k
> corpus-shaped** bank, which is a smaller claim but a more useful one. Corrected below.
>
> This mattered in practice: the Jetson fit test loaded SigLIP2 **and** EmbeddingGemma **and**
> the bank, and OOM'd at 659 MiB free.

The 355 MiB bank is 121,104 VGGSound/Action100M captions, mean **132 characters**. It is that
large because the query predictor emits vectors into EmbeddingGemma's space that only make
sense as **nearest neighbours of the training distribution** — it is a lookup table over the
corpus. That is precisely the "live scenarios won't match the corpus" failure mode.

SigLIP2 generalises zero-shot, so candidates need not come from the corpus. Measured proof:
in the room test the winning sentences were **10 hand-written probes**, while corpus-style
event captions scored *negative* (microwave −0.019, car door −0.041, parade −0.043).

**Measured tower split** (`google/siglip2-base-patch16-224`, fp16):

| | params | fp16 |
|---|---|---|
| vision tower | 92.9 M | **177 MiB** |
| **text tower** | 282.3 M | **538 MiB** |
| total | 375.2 M | 716 MiB |

The text tower is **3x the vision tower**, because it carries a 256k-token Gemma vocab
embedding (256000x768 ~ 197 M params alone) — the same table that broke under int8 on
EmbeddingGemma, and the same reason EmbeddingGemma is 578 MiB.

**Consequence: if the candidate set is fixed at build time, pre-encode it once on mercury and
ship only the vectors. No text tower on the Jetson at all.**

| | now (measured resident) | after |
|---|---|---|
| EmbeddingGemma | 578 MiB | **0** |
| SigLIP2 | 1,441 MiB | ~900 MiB (vision tower only, est.) |
| bank | 355 MiB | ~5 MiB (few thousand phrases, 768-d fp16) |
| **freed** | | **~1.4 GiB** |

### Retrieve TAGS, not sentences
Perception currently hands the LLM a pre-written 132-char narration — i.e. perception is doing
the LLM's job, in VGGSound's voice. The thinker is already resident at 18.2 ms/token. Give it
grounded facts (`indoor room`, `desk`, `office chair`, `posters on wall`, `one person seated`,
`dim lighting`) and let it compose the sentence. Two non-preferential reasons:

1. **SigLIP2 trains with a sigmoid loss** — independent yes/no per (image, text) pair, not
   CLIP's in-batch softmax. It is built for absolute-threshold multi-label scoring; a tag
   vocabulary is its native use case, whole-caption ranking is not.
2. **Its text tower maxes at 64 tokens** and was trained on short alt-text. The 132-char
   narrations sit at the edge of what that space represents well.

Cost: loses natural-narration specificity, and needs a different eval (per-tag precision /
recall with a calibrated threshold, not caption R@1).

### BUILT + FALSIFIED + FIXED (2026-08-15)

**The design in the section above was implemented, trained head-to-head, and one of its
central claims was falsified. All four runs below share the same 518k-clip pool, the same
1024 effective batch, the same 2048/6144 negative pool, the same LR schedule and
λ_within=0.3. Only the marked variable changes.**

| run | target space | streams | VGG within-clip | VGG R@1 | A100M R@1 |
|---|---|---|---|---|---|
| `query_predictor_ddp_lw0.3` *(reference)* | EmbeddingGemma + trained proj 1536 | 3 | **0.883** | 0.681 | 0.090 |
| `sig_runA_matched3stream` | SigLIP2 **frozen (Identity)** | 3 | 0.654 | 0.489 | 0.051 |
| `sig_runB_scene4stream` | SigLIP2 **frozen (Identity)** | 4 | 0.688 | 0.627 | 0.069 |
| `sig_runC_proj1536` | SigLIP2 + trained proj **1536** | 4 | 0.747 | **0.739** | 0.091 |
| **`sig_runD_proj768`** ← **deployment choice** | SigLIP2 + trained proj **768** | 4 | **0.811** | **0.737** | 0.085 |

#### What was falsified
The claim was: freeze the target in SigLIP2's pretrained joint space, because that buys
zero-shot tags and checkpoint-independent banks. **Run A tested it directly against the
EmbeddingGemma reference and lost — within-clip 0.654 vs 0.883, R@1 0.489 vs 0.681 — with
the within-clip curve FLAT from step 249 (0.651 → 0.645 → 0.642 → 0.655 → 0.658 → 0.654).**
A flat curve is a ceiling, not slow convergence.

#### Root cause (the first hypothesis was also wrong)
The initial guess was that SigLIP2's space collapses long captions. Measured on 400 held-out
clips' 6 caption fields, **it does not**:

| text space | within-clip cos | cross-clip cos |
|---|---|---|
| SigLIP2 raw | 0.7556 | 0.6648 |
| EmbeddingGemma raw | 0.7306 | 0.6075 |
| **EmbeddingGemma + trained proj** | **0.4340** | **0.1533** |

SigLIP2's raw space is barely different from EmbeddingGemma's raw space. **The trainable
projection is where the representation learning happens** — it takes a space crammed between
0.61–0.73 cosine and spreads it to 0.15–0.43. The frozen design deleted it.

#### The fix, and why it keeps the memory win
Two things had been conflated:
* **"No text encoder on-device"** comes from *pre-encoding*, and is independent of the
  projection — a projection is a matmul applied OFFLINE at bank-build time
  (`scripts/build_bank_siglip2.py --ckpt`). **Survives.**
* **"Frozen target space"** comes from Identity. **That is what cost the accuracy, and it
  was never required for the memory win.**

Restoring the Linear (B → D, changing nothing else) moved within-clip 0.688 → 0.811 and
R@1 0.627 → 0.737. **The 768 projection beats the 1536 one** — same R@1 (0.737 vs 0.739),
much better within-clip (0.811 vs 0.747) — *and* halves the bank. Cheaper and better; no
trade.

**Net versus the reference: R@1 0.737 vs 0.681 (+8.2% relative) while on-device text
machinery drops from 933 MiB to 177 MiB.** The one metric still behind is within-clip
(0.811 vs 0.883), i.e. query-sensitivity; the swapped-query control is 0.006 against a
0.167 chance level, so the model genuinely reads the question.

**Accepted cost:** banks are checkpoint-coupled again and must be rebuilt on retrain — a
build-time chore, not memory. Zero-shot tag scoring still works but must read
`encode_text_frozen_raw` (pre-proj), never the projected target.

### Candidate sets: captions vs tags (run D, 512 held-out clips)

| path | space | metric | result |
|---|---|---|---|
| captions via **predictor** | learned | caption R@1 | **0.705** |
| captions via zero-shot | raw | caption R@1 | 0.619 |
| tags via **predictor** | learned | tag p@5 | **0.418** (shuffled 0.021, gap **+0.396**) |
| tags via zero-shot | raw | tag p@5 | 0.387 (shuffled 0.048, gap +0.339) |

Two findings:
1. **The JEPA streams earn their place.** The trained predictor (V-JEPA2 + WavJEPA + M2 +
   query conditioning) beats raw zero-shot SigLIP2 image retrieval 0.705 vs 0.619. SigLIP2
   alone is not sufficient.
2. **Tags survive the predictor path** — the extrapolation worry was unfounded. Tags score
   *better* through the predictor (0.418) than zero-shot (0.387). Caveat: tag p@5 is a
   word-overlap proxy, NOT comparable to caption R@1; it is only meaningful by its gap to
   the shuffled control.

### The room frame — the actual deployment condition

Zero-shot on a real frame from BMO's camera:

| candidates | top results |
|---|---|
| **tags (2.04 MiB)** | *an office chair* (+0.118), *a cluttered room*, *desk*, *a tidy room*, *a desk*, *chair*, *a cluttered scene* |
| captions (177 MiB) | *"a musician practicing the clarinet within the confines of an office space"* (+0.150) |

**Tags return grounded, correct facts. Captions hallucinate a clarinet player.** This is the
strongest argument for handing the thinker tags and letting it compose the sentence, and it
is measured on the real deployment frame rather than a corpus metric.

### AV congruence in the new space (audio swapped, asked what it HEARS)

| run | follows EARS | matched control |
|---|---|---|
| **`sig_runD_proj768`** (base audio, **no nat**) | **0.608** | **0.967** |
| `sig_runB_scene4stream` (frozen) | 0.502 | 0.502 |
| `sig_runA_matched3stream` (frozen) | 0.467 | 0.541 |
| *old EmbeddingGemma arm B (base+nat)* | *0.609* | — |
| *old EmbeddingGemma arm C (base only)* | *0.562* | — |

**Run D reaches 0.608 audio-following with base audio ONLY — matching the old arm B (0.609)
that needed nat, and beating the old base-only arm C (0.562). Dropping nat now costs
nothing**, which retires the 469 ms nat forward on the Jetson at no measured quality cost.
This supersedes the earlier "dropping nat costs 4.7 points of audio-following" finding,
which was measured in EmbeddingGemma geometry.

Note also that both frozen runs sit at matched_control ≈ 0.50 — chance for a 2-way choice.
The frozen target could not reliably tell a clip's own sound caption from another clip's at
all; run D is at 0.967. Independent confirmation that the projection was load-bearing.

### Measured on-device artifact sizes

| artifact | size | replaces |
|---|---|---|
| `query_vectors_siglip2.pt` (30 phrasings, 6 fields) | **0.05 MiB** | EmbeddingGemma encoding the question |
| `candidates_siglip2.pt` (1,372 tags) | **2.04 MiB** | — |
| `bank_runD_fp16.pt` (121,104 captions, 768-d, proj applied) | 177 MiB | the 355 MiB 1536-d bank |
| SigLIP2 text tower | **not shipped** | — |
| EmbeddingGemma | **not shipped** | 578 MiB |

Measured tower split (`siglip2-base-patch16-224`, fp16): vision 92.9 M / **177 MiB**, text
282.3 M / **538 MiB** — the text side 3× larger because of a 256k-token Gemma vocab
embedding, the same table that broke under int8.

`models/text_target.py::PreEncodedTextSpace` is the device-side stand-in for a text encoder:
pure lookup, no weights. Misses route to the nearest known phrasing by word overlap and are
recorded in `last_fallbacks`, so guessing is visible (`strict=True` refuses instead).
Measured: *"what do you hear right now?"* → *"What do you hear?"* (0.429); gibberish → 0.000.

`load_perception_query_engine` reads `shared_dim`/`query_dim` **off the checkpoint** rather
than hardcoding 1536, and raises if the query encoder does not match — the silent-mismatch
bug class, made loud.

### Corpus fix that made the comparison valid
Action100M scene coverage was 80,000 of 399,934 clips, so `--restrict-to-scene` was gutting
the corpus 345,754 → 69,339 (20%). Extracted the missing 319,934 segments (64 shards ×
4 threads, ~249 clips/s, ~22 min, `bad=0`). Coverage is now 345,754 → **345,751**, and the
scene-restricted pool (517,181) matches the unrestricted pool (518,347) to **0.2%** — so
3-stream vs 4-stream is now a clean comparison too.


---

## 7. Honest status board

| capability | state |
|---|---|
| two-tier brains + personality | ✅ works, deployed |
| face engine + camera | ✅ works, deployed |
| STT | ✅ works (SenseVoice, RTF 0.075) |
| TTS | ❌ **broken** — onnxruntime aborts on-device (fixable, unrelated to perception) |
| perception: *what is happening* | ✅ works in-domain |
| perception: *what things are* | ❌ **broken** — wrong encoder; SigLIP2 fixes it (measured) |
| identity memory | ✅ works (0.765 TAR), ⚠️ not wired live, needs face crops |
| thinker ↔ perception query loop | ✅ built + wired, opt-in |
| generation instead of retrieval | ❌ tried, underperforms (F1 0.269 vs 0.317) — not the fix |

**Two things are actually broken: the TTS import, and the perception encoder.** Everything
else measured works. The encoder fix is identified, benchmarked on-device, and net-frees
memory.

---

## 8. Jetson fit test — 2026-08-15 (fresh reboot + `PREFLIGHT: PASS` before each run)

Conditions: MAXN_SUPER, face engine + Xorg left running, **no TTS/STT loaded**, CSI camera
live, `qp_runD.pt` (SigLIP2 + proj-768, 4 streams), 1,372 tags as candidates,
`--text-mode preencoded` (**no text encoder resident**).

### 8.1 IT FITS

| stage | no-nat Δ | no-nat avail | with-nat Δ | with-nat avail |
|---|---|---|---|---|
| speaker (LFM2.5-350M) | +253 | 4,687 | +832 | 5,215 |
| thinker (Qwen3-0.6B) | +780 | 3,907 | +1,108 | 4,107 |
| ViT-L int8 | +629 | 3,278 | +896 | 3,211 |
| WavJEPA base | +618 | 2,660 | +699 | 2,512 |
| **WavJEPA nat** | — | — | **+701** | 1,811 |
| M2 | −53 | 2,713 | −366 | 2,176 |
| SigLIP2 (**vision tower only**) | +1,170 | 1,543 | +1,048 | 1,129 |
| **pre-encoded queries** | **−1** | 1,544 | **+0** | 1,128 |
| query predictor + 1,372 tags | +134 | **1,410** | +193 | **935** |
| camera | +18 | 1,392 | −13 | 948 |
| after 3 rounds | | **429** | | **36** |

Against the configuration that OOM'd at 659 MiB free, **EmbeddingGemma's 578 MiB and the
355 MiB bank are replaced by 0 MiB and ~2 MiB.** Dropping SigLIP2's text tower cut it
1,527 → 1,170 MiB (verified numerically safe first: `vision_model(px).pooler_output` vs
`get_image_features(px)` gives **cosine 1.0**, which matters because the cached scene
features were built with `get_image_features`).

### 8.2 Latency (steady state, rounds 2–3; round 1 is warm-up)

| leg | no-nat | with-nat |
|---|---|---|
| capture | 905 | 929 |
| **perception** (V-JEPA2 + WavJEPA + M2) | **792** | **1,118** |
| SigLIP2 scene | 104 | 94 |
| **query predictor + retrieval** | **33** | 23 |
| thinker | 817 | 705 |
| speaker | 212 | 190 |
| **total** | **2,886** | **3,082** |

**The whole perception-query addition costs 33 ms.**

### 8.3 VERDICT ON nat — drop it, now confirmed on-device
nat costs **+701 MiB** and **+326 ms** of perception (792 → 1,118), and leaves only **36 MiB**
free after three rounds — effectively zero headroom. The congruence eval measured it buying
**nothing** in this target space (base-only 0.608 vs base+nat 0.609). It is a pure loss.
(Note: nat's cost was previously recorded as 469 ms; **measured here at 326 ms**.)

### 8.4 Does it describe the right thing?
* **Perception** — *"a person lying down"* → *"a person sitting"*. Grounded and plausible from
  the live CSI camera. **Works.**
* **Thinker** — *"I see a person lying down, and I'm curious about them. Maybe they're resting
  or just want to be quiet — I should ask if they need anything…"* Correctly consumes
  perception and reasons about it. **Works.**
* **Speaker** — *"Finn, Jake — just how you like it!"* / *"Sweet, sweet music? Let's make it
  together!"* **Generic BMO flavour that ignores the scene entirely.**

**So perception → thinker is sound; thinker → speaker is where grounding is lost.** That is a
fast-tier prompting/corpus issue, not a perception one, and it is the next thing to fix. It is
independent of everything else measured here.

### 8.5 Bugs found and fixed by this test
1. `load_perception_query_engine` tried to load a trained proj into `PreEncodedTextSpace`'s
   `nn.Identity`. Now takes the pre-projected path explicitly and **raises** if candidates
   were not projected offline, rather than silently retrieving in the wrong geometry.
2. **The no-nat path passed the 1-channel BASE encoder into the nat slot**, which crashes on
   the 2-channel tensor nat expects. `world_state_builder.build_world_state_features` now
   accepts `nat_encoder=None` meaning `audio_mode="base"` — matching how `qp_runD` was
   trained. Had this not crashed it would have produced silently wrong ambient features.
3. A transfer that hit a 2-minute timeout left a **truncated** `bank_runD_fp16.pt`
   (192 of 227 MiB) on the device. Deleted. Always verify byte counts after a push.

Raw results: `jetson_artifacts/benchmarks/fit_2026-08-15/fit_{nat,nonat}.json`.

---

## 9. Capture + speaker fixes — 2026-08-15

### 9.1 Capture was 905 ms of my own `time.sleep`, not the camera

`jetson_core_pipeline_test.grab()` read frames SYNCHRONOUSLY with `time.sleep(0.05)` between
them: **16 x 50 ms = 800 ms**, ~88% of the measured 905 ms. It was never a camera problem —
a correctly-formed Jetson `nvarguscamerasrc` path is **10-30 ms glass-to-glass**.

**Deleting the sleep alone would have been WRONG.** V-JEPA2 is handed
`true_window_dur_sec=10.0`, and 16 back-to-back frames at 30 fps span 0.53 s. Claiming that
is a 10 s window corrupts the temporal bins the model reads.

**Fix: producer/consumer ring buffer** — `CaptureThread` fills a deque on a background
thread; inference uniformly samples 16 frames across the buffered window. This is the same
model NVIDIA's libargus samples use (EGLStream producer/consumer), and mirrors
`m5_streaming_loop.py::RollingVideoBuffer`, which **production already used** — so the old
benchmark was measuring a synchronous stand-in, not the deployed architecture.

Pipeline also tightened per NVIDIA guidance: `queue` before the consumer (prevents Argus
"producer frame drop" from NVMM buffer exhaustion), `max-buffers=1 drop=1 sync=false`
(freshest frame, never a stale queued one — `max-buffers=2` lets one stale frame sit in
front of the live one), and BGRx->BGR done as a numpy channel-slice instead of a per-frame
CPU `videoconvert`.

| leg | before | after |
|---|---|---|
| **capture** | **905** | **1-9** |
| perception | 792 | 662-668 |
| SigLIP2 | 104 | 90-91 |
| query | 33 | 22 |
| thinker | 817 | 770-776 |
| speaker | 212 | 218-223 |
| **total** | **2,886** | **1,795** |

**−1,091 ms end-to-end (−38%)**, and the 10 s window is now *true* rather than asserted.
Buffer warms in 0.6 s, once, at startup. The capture thread is throttled to production's
`video_fps=6.4` (64 frames/10 s) rather than the sensor's 30 fps: 300 buffered frames cost
~59 MiB for no benefit. **That throttle sleep is on a background producer and blocks
nothing — it is not the bug that was removed, which was on the inference path.**

### 9.2 The speaker was NOT the problem — three hypotheses, two falsified

Probes on the real deployed GGUFs (`jetson_speaker_prompt_probe.py`, `jetson_chain_probe{2,3}.py`):

* **Prompt shape — FALSIFIED.** The current all-in-one-user-turn format is the ONLY shape
  that grounds this model. system-turn, assistant-turn, and "voice the plan" variants all
  produced generic output indistinguishable from the no-grounding control. **This model
  ignores system/assistant history turns.**
* **Truncation — FALSIFIED.** Full thought vs `[:200]` produced byte-identical output.
* **Thin perception (top-1 tag) — FALSIFIED, and backwards.** Richer scenes made the
  THINKER *less* grounded: top-k tags and a full caption both collapsed to "sweet, sweet
  music", while the single tag "a person lying down" was the only input that grounded it.

**ACTUAL CAUSE: the 350M speaker keys on the most recent salient ENTITY in its prompt.** Two
runs differing only in the thought's trailing two words:

| thought ends | speaker says | grounded |
|---|---|---|
| "...mood bright for **Finn and Jake**." | "Finn, Jake - just how you like it!" | ❌ |
| (character names stripped) | "Finn, you look a bit tired - maybe I can queue a gentle lullaby while you rest." | ✅ |

The thinker habitually signs off with Adventure Time character names learned from the v9
companion corpus. Those are **persona artifacts, not scene content** — pure noise to the
speaker, and they hijack the whole line. `sanitize_thought()` strips them before the handoff.

Live result after the fix:

| round | says |
|---|---|
| 1 | *"Finn, you look a bit tired - maybe I can queue a gentle lullaby while you rest."* |
| 2-3 | *"They might just be relaxing, or maybe they're having a quiet moment."* |

versus *"Finn, Jake - just how you like it!"* before. Rounds 2-3 are grounded **and carry no
character name at all**.

### 9.3 The identity head is NOT wired — this is why BMO calls the user "Finn"

Verified by grep: **zero** references to the identity head, `IdentityHead`, or `JepaMemory`
in `scripts/bmo_jetson_startup.py` or `models/m5_streaming_loop.py`. No name-handling logic
anywhere in the loop.

**"Finn" is not an identity guess. It is a hallucination from the v9 companion corpus.** The
identity head is trained and validated (TAR@FAR1% **0.765**, AUC 0.966, 884 unseen
identities) and unused. Sanitising the thinker's text fixes GROUNDING but cannot stop the
speaker addressing the user as Finn, because that comes from its own LoRA weights, not its
input — confirmed: round 1 still says "Finn" with a fully sanitised prompt.

What it would take, in order of real work:
1. **A face detector** feeding crops to the head. This is the actual blocker —
   `motion_tracker.cpp` has the camera pipeline but produces no face crops.
2. **Enrolment** — `JepaMemory.enroll()` exists with threshold calibration (1 clip 0.584 ->
   8 clips 0.760).
3. **Name into the prompt.** The head's honest output is three-way: at the 1%-FAR operating
   point on unseen identities, **51.5% recognised / 48.1% "I don't know you" / 0.4% wrong
   name.** That 0.4% is the point — a companion that asks is better than one that guesses.
   Today the wrong-name rate is effectively 100%, confidently, with no identity signal at all.

**Interim fix needing no face detector: drop the vocative when identity is unknown.** "You
look a bit tired" instead of "Finn, you look a bit tired" — a post-processing rule on a
one-sentence output, strictly better than inventing a name.

### 9.4 Bug introduced and fixed
`CaptureThread` set `self._stop = threading.Event()`, shadowing `threading.Thread`'s own
private `_stop()` that `join()` calls internally -> `TypeError: 'Event' object is not
callable` at shutdown. Renamed `_stop_evt`. **Never assign to `_stop`/`_started`/`_target`
on a Thread subclass.**

---

## 10. Identity head — MEASURED TO FIT (2026-08-15)

### 10.1 Cost
| | |
|---|---|
| identity head (8.15 M params, joint AV) | **+96 MiB** |
| identity query latency, steady state | **4–9 ms** |
| avail after full load, with identity | **1,306 MiB** |
| avail at end of 3 rounds | 266 MiB |

It reuses the ViT-L and WavJEPA token streams the perception tick already produced
(`stats_pool` → 2048-d vision, 1536-d audio), so there is **no extra encoder load** — the
96 MiB is the head itself.

### 10.2 The behaviour contract, running on-device
Cold start (empty memory) → enrolment → recognition, in one run:

| round | identity | BMO says |
|---|---|---|
| 1 | `unknown (empty_memory)` | *"Name: BMO! Thanks, I'm happy to help."* |
| 2–3 | `Alice (match)` | *"A resting person is like a paused game; maybe we can play a quick mini-game together."* |

**The unknown branch fixes the actual complaint: no invented name.** No Finn, no Jake — it
introduces itself as BMO. Prompt-side contract implemented in
`jetson_core_pipeline_test.py`: recognised → the speaker is told the name and asked to greet
by it; unrecognised → explicitly told *not* to guess, and to introduce itself and ask.

**The recognised branch does NOT yet work: it knew "Alice" and never said "Alice".** The
identity signal reaches the speaker and the speaker cannot consume it. That is the corpus
problem, and it is now evidenced rather than assumed.

### 10.3 Two honesty caveats
* **`conf=1.000` is not a recognition result.** Enrolling and querying near-identical frames
  of a static scene is a plumbing test. Real accuracy remains the offline **TAR@FAR1% 0.765 /
  AUC 0.966** on 884 unseen identities.
* Round 1's line ("Name: BMO! Thanks…") introduces but does not really *ask*. Also corpus.

### 10.4 What the corpus retrain must teach
Concretely, from the observed failures:
1. **Use a supplied name.** Given "their name is Alice", greet Alice by name.
2. **Never invent one.** No Finn/Jake/Adventure-Time vocatives — they also break grounding
   (§9.2), so this fixes two things at once.
3. **A real first-meeting turn.** Introduce, then actually ask the name, then hand off to
   enrolment.
4. **Drop the vocative when unknown** rather than substituting a placeholder.

This is also why the head's honest three-way output matters: at the 1%-FAR operating point on
unseen identities it is **51.5% recognised / 48.1% "I don't know you" / 0.4% wrong name**.
"I don't know you" is the *common* case, so branch 3 must be good behaviour, not a stumble.

### 10.5 Still blocking live identity
The head consumes **face crops**, and the deployed `motion_tracker.cpp` has the camera
pipeline but produces none. The fit test above feeds it whole-frame ViT-L features, which is
enough to prove memory and latency but is **not** the trained input distribution
(VoxCeleb2 face crops). A face detector remains the real prerequisite.

---

## 11. Face crops + amortized identity — 2026-08-15

### 11.1 The CSI sensor is EXCLUSIVE (this drove the whole design)
With `face_engine/motion_tracker` holding sensor-id=0, a second `nvarguscamerasrc` client
fails: **`Failed to create CaptureSession`** — and silently, `isOpened() == True` followed by
`read() == False`. So with two processes it was strictly either/or:

* motion_tracker running → the eyes track you, **perception is blind**
* perception running → perception works, **the eyes are dead**

**Fix: one camera client, three consumers.** `models/m5_motion_crop.py` folds motion
differencing into the capture thread that already owns the camera:

```
camera ──► capture thread ─┬─► perception window (V-JEPA2 / SigLIP2 / WavJEPA)
                           ├─► motion centroid ──► /dev/shm/bmo_motion.txt (face engine)
                           └─► crop around centroid ──► identity head
```

Output is byte-identical to `motion_tracker.cpp`, so the face engine's poller is unchanged.
Verified live: `1 0.3552 -0.0546` written while perception ran. Also retires
`motion_tracker`'s **133 MiB** RSS (measured — not the "tens of MB" its own comment claims).

`nvargus-daemon` is resident at **140 MB regardless** — it is the shared Argus daemon, not a
per-client cost. That answers the "is the daemon eating VRAM" question: it is already there.

### 11.2 Face crops work, and cost 218–239 ms
The identity head was trained on VoxCeleb2 **face crops**, so a whole 1280x720 frame is out
of distribution. The motion centroid localises the person for free. Measured centroids:
(0.516, −0.057) and (0.416, −0.062) — right of centre, slightly high, consistent with a
seated person.

Cost: **+210 ms**, because the crop needs a *second* V-JEPA2 forward. Tick went
1,891 → 2,295 ms.

**Honest limit:** a motion centroid is not a face box. It is the centre of mass of what
MOVED — chest height for a seated person, and **nonexistent for a still one** (round 2 of the
first run had no motion, so no crop, and fell back to whole-frame). `HEAD_BIAS=0.22` lifts the
crop toward the head; that is a heuristic, not a detector. A real detector remains the fix.

### 11.3 Amortized identity — the fix, borrowed from rendering
`models/m5_identity_schedule.py`. "Who am I talking to" changes on the scale of minutes; the
tick runs at ~2 s. Recomputing every tick is exactly the redundancy temporal reuse exists to
remove: gate expensive work behind a cheap always-on signal (the motion centroid, already
computed), reuse otherwise.

Runs on: **no answer yet · motion onset · staleness (`max_age_s`) · low-confidence retry.**
Reuses the cached answer otherwise, at zero cost.

MEASURED over 6 rounds:

| round | decision | identity_ms | total_ms |
|---|---|---|---|
| 1 | ran (`no_answer_yet`) | 19 | 3,455 |
| 2 | ran (post-enrolment, crop path) | **240** | 2,428 |
| 3–6 | **skipped** (`reuse_cached`) | **0** | 1,782–1,885 |

**`run_rate 0.333`. Amortized cost 43 ms/tick vs 218–240 ms unamortized (~5x), and the
steady-state tick returns to 1,782 ms — identical to before face cropping existed.**

Caveat: 0.333 is flattered by a static test scene; real use fires more motion onsets.
`max_age_s=30` bounds the worst case.

**This is the template for §C of `PIPELINE_REMAINING.md`** — the same argument applies to the
651 ms perception leg, which is likewise recomputed every tick regardless of whether anything
changed.

---

## 12. Persistent memory — 2026-08-15 (`models/bmo_memory.py`)

### 12.1 Why it is NOT RAG and NOT a knowledge graph
The 2026 agent-memory literature (Mem0, Zep, A-MEM, GraphRAG) targets cloud agents with big
contexts and an always-available text encoder. **BMO has neither, by decisions made and
measured here:**

1. **No text encoder is resident.** EmbeddingGemma (578 MiB) and SigLIP2's text tower
   (538 MiB) were both removed by pre-encoding offline. Vector RAG needs to embed *new* text
   at runtime — adopting it means putting ~538 MiB back and undoing §6/§8 entirely.
2. **`n_ctx=512`, real prompt 34 tokens.** A memory that returns chunks cannot be consumed.
3. **~0 ms retrieval budget** on a 1.8 s tick that is already full.

| idea | source | verdict |
|---|---|---|
| no LLM calls at retrieval | Zep | **adopted** — recall is a dict lookup |
| `[[wiki links]]` between people | Obsidian | **adopted** — relations without a graph DB |
| semantic vs episodic split | A-MEM | **adopted** |
| budget-curated eviction | MOBIMEM | **adopted** — capped, decays by reinforcement |
| vector search over history | RAG | **dropped** — needs the encoder we removed |
| LLM entity extraction on read | Mem0/GraphRAG | **dropped** — 700–1500 ms per call |
| graph database | Zep/Neo4j | **dropped** — tens of people, not millions |

### 12.2 The design
Semantic memory **keyed by identity, not by text embedding** — the head already answers "who
is this", so that label is the primary key:

```
identity embedding ──► JepaMemory ──► "Alice" ──► PersonProfile ──► one prompt line
```

`to_prompt_line(char_budget)` is budget-aware by construction and drops in value order
(episodic before semantic, weakest facts first). **It returns "" rather than a fragment** when
even the name will not fit — a mangled line still steals tokens from the scene description.

| budget | tokens | output |
|---|---|---|
| 200 | 28 | *"You know Alice: is writing a thesis on robotics; has a cat named Pixel; drinks tea, not coffee. Also knows Bob."* |
| 40 | 8 | *"You know Alice. Also knows Bob."* |
| 12 | **0** | **""** |

Measured on-device: **`memory_ms=0`**, tick unchanged at **1,711–1,728 ms**.

### 12.3 MEASURED FAILURE — perception tags are NOT facts about people
First wiring wrote the scene tag in as a durable fact (`note_fact(who, f"was {seen}")`). The
tag was "a closet", so memory stored **"You know Alice: was a closet."** and the speaker
greeted her as **"closet queen"**.

**Scene tags describe the ROOM.** Even person-ish tags ("a person sitting") are transient
states, not attributes. This is precisely the distinction the semantic/episodic split exists
to enforce, and it was violated by accident within minutes of building it — so the guard is
now in `note_fact`'s docstring. Observations go to the episode ring
(`note_encounter`), which decays and is dropped from the prompt first.

This is the failure mode that makes memory systems actively harmful rather than merely
unhelpful: confidently remembering something false, then acting on it.

### 12.4 VERIFIED across processes (the reboot test)
Run 1 cold → introduces itself, enrols, saves. Run 2 is a **fresh process** reading only the
files:

```
[info] persistent memory: {'people': 1, ...} from ~/bmo_memory.json
  [IDENTITY]   Alice (conf=1.000, match)        <- recognised from disk, NO enrolment
  [MEMORY]     You know Alice.
  [BMO SAYS]   Alice, thanks! I've got a fresh set of shelves for you.
  [MEMORY]     You know Alice. You last spoke just now.
```

**BOTH halves must persist.** The first attempt saved only the profiles, so run 2 reported
`empty_memory` and could never reach Alice's profile — every person a stranger after reboot,
the profile findable only by coincidence:

| half | stores | file | size |
|---|---|---|---|
| `JepaMemory` | **who** — identity centroids | `~/bmo_identity.pt` | ~1 KB/person |
| `BmoMemory` | **what** — name, facts, episodes | `~/bmo_memory.json` | ~0.4 KB/person |

Third bug found the same way: `JepaMemory.load` uses `map_location="cpu"` (correct — it must
load on GPU-less machines), so the centroids met a CUDA query embedding and died with
*"Expected all tensors to be on the same device"*. Added `JepaMemory.to(device)`.

### 12.5 Writes never call a model
`note_fact` is an append plus a cap. *Deciding* what is worth remembering is a language task
and belongs off the tick — the thinker proposes facts after a turn, not during one.

---

## 13. Speaker v3 deployed + companion tools — 2026-08-15

### 13.1 Speaker v3: the corpus fix worked on-device
Trained on corpus v10c (3,641 rows, **0 cartoon references**, BMO-voice retention 29% -> 33%),
`best_val_loss=0.7093` at epoch 3. GGUF 379 MB, deployed byte-exact, v2 backed up.

**All three failures the corpus was built to fix are gone:**

| | v2 | **v3** |
|---|---|---|
| unknown branch | *"Name: BMO! Thanks, I'm happy to help."* | *"I'm BMO, happy to join the jam session! Could you name me?"* |
| **recognised branch** | *"A resting person is like a paused game…"* (never said Alice) | ***"Alice!** You're the best friend, and you've got the sweetest song yet."* |
| invented names | "Finn", "Finn, Jake" | **none** |

**It uses a name it is handed** -- the behaviour v2 could not perform no matter how explicitly
the prompt asked, because zero examples of it existed. Not a prompting problem.

The personality survived de-cartooning: "jam session", "sweetest song", `*beep boop*`, and the
empathy-by-analogy move intact (*"Beemo feels wobbly sometimes too when things go \*blip\*"*)
with the referent moved off the treehouse.

### 13.2 `name_stranger` -- a bad exemplar, not scarcity
v3 still said *"Could you name me?"* -- asking to BE named. Reading all 39 rows rather than
assuming the class was too small found three separate problems:

1. **Contamination:** 9 of 39 have the user ALREADY giving their name (*"I'm Carlos."*).
   Those are `name_just_told` filed as `name_stranger`, so the class taught two behaviours.
2. **A malformed exemplar:** *"Thanks! I'm BMO. **Who should I call?**"* -- the exact shape of
   the bad output. It was being imitated.
3. **Scarcity:** 39 examples for the class that fires FIRST in every real interaction, vs 77
   for `name_just_told`.

Fixing 1 and 2 matters more than 3: adding examples on top of a confused class teaches the
confusion harder. `scripts/expand_name_stranger.py` reclassifies, drops malformed, and
generates ~90 more behind a **code-asserted** check that BMO asks for THEIR name and never to
be named.

### 13.3 APPEARANCE tags -- perception could not see clothing
Requested behaviour: *identity says new person + SigLIP2 sees a red jumper -> compliment it.*
**BMO could not have done this.** The vocabulary had bare mined colour words ("blue", "red")
and loose garment words, but SigLIP2 scores whole phrases -- "blue" matches anything blue in
the room, not what a person wears.

Added **110 composed appearance tags** (11 colours x 9 garments + glasses/cap/beard/backpack/
blanket/pyjamas). Vocabulary **1,372 -> 1,482 tags, 2.02 -> 2.17 MiB**. Appearance is also the
most socially useful thing to notice cheaply: unlike the room, it changes day to day.

### 13.4 Generalising companion behaviour instead of hard-coding it
Hard-coding "new person + garment -> compliment" yields a robot that compliments jumpers and
nothing else. The general form, now a `perception_social` scenario class in the thinker corpus:

> **(identity state, perception tags, memory) -> a social move**

where identity state is the head's real three-way output (recognised / don't-know-you /
uncertain). Twelve scenarios teach: notice something specific about a stranger; remark on what
is NEW rather than what is visible; follow up a remembered fact; **say nothing** when someone
is still and silent; acknowledge an absence without guilt; ask to confirm rather than risk a
wrong name; do not interrupt focused work; do not talk to an empty room.

"Remark on what is NEW" falls out of the memory design for free -- episodes already store
per-encounter observations, so appearance *change* is derivable without new machinery.

### 13.5 Tools: what BMO can actually retrieve
**Two different things are both called "RAG", and BMO's answer differs:**
* **Vector RAG over conversation history** needs a text encoder at runtime -> still off the
  table (~538 MiB to reinstate a tower we deliberately removed).
* **Tool/API retrieval** needs **no embedding model at all** -> already built and now real.

| tool | backend | key | latency | status |
|---|---|---|---|---|
| `time` / `date` | local | none | 0 | ✅ real (Jetson already `Europe/London`) |
| `weather` | OpenWeatherMap | yes, stored mode-600 at `~/.config/bmo/openweather_api_key` | ~0.3 s | ✅ real |
| `facts` / `lookup` | Wikipedia REST | **none** | 0.2–0.8 s | ✅ real |
| `search` | DuckDuckGo Instant Answer | **none** | ~0.2 s | ✅ real (subset) |
| `wikidata` | Wikidata SPARQL | **none** | **8.6 s** | ⚠️ opt-in, NOT on the reply path |

**The old weather stub returned `"sunny and about seventy-two degrees"` unconditionally.** It
would have invented conditions during a thunderstorm. Replaced with a real call that returns
**None** on missing key/network/rate-limit, so the dispatcher can say it cannot check.

### 13.6 THE ROUTING BUG -- and why the obvious fix failed
*"who won the Oscar for best actor in 2019"* returned **"Óscar Isaac is an American actor"**.

First fix attempted: require lexical overlap between query and returned title. **It was
defeated by the very case that motivated it** -- the page really is titled "Oscar Isaac" and
genuinely shares the token "oscar". No word-overlap heuristic separates the award from the
first name.

The real error was architectural: **a QUESTION was sent to an ENTITY-lookup endpoint.**
`_h_lookup` now routes on shape:

```
definitional ("what is X", "who is X")  -> DDG, then Wikipedia on the stripped entity
specific-fact ("who won X in 2019")     -> DDG only; if it cannot answer, say so
entity ("Jetson Orin Nano")             -> Wikipedia, then DDG
```

| query | result |
|---|---|
| who won the Oscar for best actor in 2019 | **None -- honest** |
| what is a transformer neural network | ✅ *"In deep learning, the transformer is a family of…"* |
| who is the prime minister of the UK | ✅ real |
| 92nd Academy Awards / Jetson Orin Nano / Rami Malek | ✅ real |

**None is a first-class outcome.** Every fabrication this project has shipped -- the 72-degree
weather, "Oscar Isaac", "closet queen", "Finn" -- came from a code path that preferred
*something* over *nothing*.

### 13.7 Wikidata SPARQL -- correct, and too slow to use inline
SPARQL is a query language for graph data; Wikidata's public endpoint is free and keyless, and
stores facts as relations with date qualifiers rather than prose. It answers the question both
other tools failed: **"Rami Malek"**.

But measured **8,591 ms**, and a retry a minute later returned
`HTTP 429: Aggressively rate-limiting to 1 req / min - this rule was created during active
wdqs outage`. That is ~5x BMO's entire tick, with a cap that fails the second question in a
conversation. Registered as a separate opt-in tool; using it needs a background thread plus a
`thinking_filler` backchannel (already built) and a local cache.

Deliberately **template-based**, not general NL->SPARQL: that is an open research problem, and
a wrong structured query returns a confidently wrong fact. Includes `wdt:P31 wd:Q5` to
restrict to humans -- without it the 2019 query also returned the film *Bohemian Rhapsody*.

---

## 14. Thinker v4 rejected, and the verifier lesson — 2026-08-15

### 14.1 Thinker v4: REJECTED, not deployed
| | |
|---|---|
| rows | 324 |
| best_val_loss | **2.0811** (v3 was 1.95 -- worse) |
| **Adventure Time references** | **175 / 324 = 54%** |
| emoji | 21 (6%) |

**Root cause: the `HARD_RULES` block was added to the SPEAKER corpus generator and never to
the THINKER's.** One generator was fixed, the other was not, and nothing checked.

This matters MORE in the thinker, not less. The thinker's reasoning is what the speaker then
paraphrases, so contaminated reasoning propagates downstream into a model that was itself
cleaned. Visible directly in a `perception_social` row -- the feature works and the setting
leaks in through the same sentence:

> *"...that splash of color feels warm and friendly, like the glow of a sunrise **over the
> treehouse**."*

Cartoon contamination by category: reasoning 53, companion 48, grounded 37, orchestration 21,
perception_social 16.

### 14.2 `perception_social` WORKED
72 rows, joint-largest class, and it produced the requested behaviour verbatim:

> *"Hey there! I really love your red jumper - so bright and cozy!"*

So the generalisation holds: **(identity state, perception tags, memory) -> a social move**,
learned as a policy rather than hard-coded to jumpers. The class survives into v5 unchanged;
only the generator's rules changed around it.

### 14.3 THE VERIFIER LESSON -- closed sets are checkable, open sets are not
`scripts/expand_name_stranger.py` verified generated rows two ways:

| check | set type | rule-checkable? | outcome |
|---|---|---|---|
| never asks to BE named | **closed** (a fixed list of bad phrasings) | yes | **9 correct catches** |
| user already gave a name | **closed** | yes | **3 correct catches** |
| **does ask for their name** | **OPEN** (unbounded good phrasings) | **no** | **93 FALSE rejects** |

First version rejected **77 of ~126** generated rows. Broadening the regex did not fix it --
it produced **93** false rejects against **9** real catches. That ratio was the signal that
the approach was wrong, not the pattern too narrow.

**Measured consequence -- the filter removed exactly the personality the corpus exists to
preserve:**

| | rows | BMO-idiom rate |
|---|---|---|
| accepted into corpus | 92 | **39.1%** |
| rejected (visible sample) | 6 | **100%** |

Rows the filter discarded, all of which DO ask for the name -- in BMO's own idiom:

> *"what's the **player tag** for you?"* · *"want to **save your name in the file**?"* ·
> *"who's the **new character joining the game**?"* · *"what should I call the **co-op
> partner**?"*

**Honest caveat:** only 6 of 105 rejects were logged (`reject_log[:6]`), so 100% vs 39.1% is
directionally clear but a small sample. **The 93-vs-9 split is the solid number.**

**Fix:** keep only the closed-set checks, trust the generator prompt for the open-set
requirement, and keep a `would_have_cut` counter so the cost of the old rule stays measurable
rather than becoming folklore.

### 14.4 `name_stranger` -- read the rows before assuming scarcity
On-device v3 said *"Could you name me?"* -- asking to BE named. Reading all 39 rows found
three distinct problems, in order of importance:

1. **Contamination (9 rows).** The user had ALREADY given their name (*"I'm Carlos."*). Those
   are `name_just_told` filed as `name_stranger`, so the class taught two behaviours at once.
2. **A malformed exemplar (1 row):** *"Thanks! I'm BMO. **Who should I call?**"* -- the exact
   shape of the bad output. It was being imitated.
3. **Scarcity.** 39 rows, versus 77 for `name_just_told`, for the class that fires FIRST in
   every real interaction.

**Fixing 1 and 2 matters more than 3.** Adding examples on top of a confused class teaches the
confusion harder.

### 14.5 Speaker v4 -- trained, NOT deployed
Corpus v10d: **3,703 rows, `name_stranger` 92** (gate required >=90, was 78 under the strict
verifier). `best_val_loss` **0.7286** at epoch 3 -- **marginally worse than v3's 0.7093**.

Held-out hostility handling did improve, less deflecting and more genuinely hurt:

| user | v3 | v4 |
|---|---|---|
| *"Shut up, nobody likes you."* | *"Did Beemo do something wrong?"* | *"Beemo feels small and sad when you say that."* |

Tool calling and personality intact. **v3 remains the deployed speaker**; v4 is not on the
Jetson.

### 14.6 THE CHAIN BUG -- third occurrence of a class already in the ledger
A chain script waited on a PROCESS NAME:

```bash
while ps -eo args | grep -q "[e]xpand_name_stranger"; do sleep 30; done
```

That matched **the Claude Code shell whose argv contained the heredoc that WROTE the script**.
Result: **28 minutes of idle GPUs**. The ledger already records this exact bug from the M2 era
(`pgrep -f "query_predictor_v1"` matching its own argv, 4 GPUs idle for hours).

**The `[e]` bracket trick does not help here.** It prevents grep matching *itself*; it cannot
prevent matching a genuinely different process whose command line contains the literal text.

**Fix -- never wait on process names.** Either:
* run stages **sequentially in one script** (no coordination needed at all), or
* wait on **MARKER FILES**, which are written *after* work completes, whereas processes exist
  *before* it.

**Related failure, same day:** a chain that waits on a gate can be KILLED by that gate
aborting, leaving nothing queued behind the retry. Thinker v5's chain died silently this way
when the `name_stranger` gate failed, and had to be re-armed by hand.

### 14.7 In flight
* **Thinker v5** -- same scenarios, generator now carrying the SAME `ABSOLUTE RULES` block as
  the speaker's. Gated to abort if **>16 of ~324** rows contain cartoon references (v4 had
  175). Capped at **4 epochs** because v3 and v4 both overfit past epoch 2-3.
* **Speaker v5** -- rebuilt on the closed-set-only verifier, reporting an `idiom_pct` for
  `name_stranger` so the result is directly comparable against v4's **39.1%**.
