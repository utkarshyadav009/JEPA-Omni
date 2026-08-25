# Making perception GENERIC — closing the retrieval-vs-generation gap

**The problem, stated by the user (2026-08-14) and confirmed by measurement:**
> "we won't have the same scenarios when doing live testing at all… shouldn't we have a
> universal/general system that sees an embedding and then understands what it means
> without the databank? I mean maybe I had that before when I plugged the worldstate into
> the llm during m3 and the latency was bad… but now its bigger and non generic"

This is exactly right, and the live run proves it. Pointed at a real room, the retrieval
system answered *"a tuning fork being tapped"* and *"a smoke detector beeping in an
office"* — the nearest neighbours in a bank of **VGGSound sound-event clips**, which
contains no caption for "a dim bedroom with a desk chair and posters". **Retrieval cannot
say anything its bank does not already contain.** More bank helps at the margin; it does
not fix the class of problem.

---

## 1. The real tension, with this project's own numbers

| approach | open-vocabulary? | measured latency | status |
|---|---|---|---|
| **M3 connector → Qwen2.5-1.5B** (soft-prompt world-state into an LLM) | **YES** | perceive 2.5-3.5 s + **generate 1-6 s** | dropped as too slow + confidently wrong |
| **Query predictor → retrieval** (current) | **NO** (bank-bound) | predictor 5 ms + lookup 2.6 ms | fast, but says nearest-bank-thing |

The user's framing — *"the latency was bad so I tried to get a faster system but now its
bigger and non generic"* — is precisely the trade that was made.

**The key insight this plan rests on: M3's latency problem was mostly a MODEL-SIZE problem,
and that problem has since been solved for unrelated reasons.**

| | at M3 time | now |
|---|---|---|
| generation model | Qwen2.5-1.5B | **LFM2.5-350M** |
| measured Jetson decode | **203 ms/token** | **9.4 ms/token** (226 ms / 24 tok) |
| 30-token description | ~6 s | **~280 ms** |
| embedding-prefix mechanism | needed a soft-prompt connector | **`llama_batch.embd` PROVEN** (byte-identical to the token path, no C++ fork) |

**A 21x faster decoder plus a proven embedding-input path means the generic approach that
was correctly rejected in July is now affordable.** That is the fix.

---

## 2. Proposal — "M3 done right": embedding prefix into the 350M fast tier

```
camera ─► ViT-L + WavJEPA ─► M2 ─► QueryPredictor(query) ─► z_q (1536-d)
                                                             │
                                              NEW: Projector (z_q → k × 1024)
                                                             │
                                    llama_batch.embd prefix into LFM2.5-350M
                                                             │
                                              free-form generated description
```

* **Trainable:** one small projector (z_q or the M2 pre-pool tokens → k soft tokens in the
  LLM's embedding space). Everything else frozen.
* **Objective:** next-token cross-entropy on the caption, conditioned on the prefix — i.e.
  exactly M3's objective, but into a 350M decoder through the proven `embd` path.
* **Data already on disk:** 188k VGGSound clips × 6 rich-caption granularities + 400k
  Action100M clips × 2. No new collection.
* **Expected latency:** ~280 ms for a 30-token sentence + ~5 ms projector, on top of
  perception. Comparable to the current retrieval path's *total* answer time (measured
  403 ms) because the query encoder (263 ms) can be dropped from the generation path.

### Why this is not just re-running the thing that failed
1. **21x faster decoder** (the actual cause of "latency was bad").
2. **No soft-prompt plumbing to invent** — `scripts/prototype_llama_embd_input.py` already
   proved llama.cpp accepts a raw embedding prefix byte-identically.
3. **Better conditioning input.** M3 consumed the M2 world-state. The query predictor's
   output is *query-conditioned* and measured stronger (unified `m2+vision+ambient`, R@1
   0.715), so the prefix carries more, and carries what was ASKED.
4. **The "confidently wrong" failure is addressable now** — a retrieval score exists as a
   confidence signal (§3), which M3 never had.

### The honest risk, stated up front
The Ultravox track closed with *"the 350M decoder is a hard wall"* (WER plateaued 0.903).
That result is real but **does not transfer directly**: ASR demands exact token-level
fidelity from a dense audio signal, while captioning demands semantic gist. Still, capacity
is the top risk, and the cheap way to find out is a small mercury-side run before any
Jetson work. If 350M is insufficient, the Qwen3-0.6B thinker (already deployed, 805 MB) is
the next rung and still ~10x faster than M3's Qwen2.5-1.5B.

---

## 3. Hybrid routing — keep the speed, add the generality

Retrieval and generation are complementary, and the router is already half-built
(`PerceptionQueryEngine.ask()` already returns a score, and `min_score` already gates it):

```
score = max cosine(z_q, bank)
if score >= τ:  speak the retrieved caption      (~8 ms — the common, in-distribution case)
else:           generate from the prefix         (~280 ms — novel scenes, the user's room)
```

τ is calibrated exactly like the identity memory's threshold (§Phase 3) — on held-out
in-distribution vs out-of-distribution scenes, choosing the point where retrieval stops
being trustworthy. This directly fixes the observed failure: a dim bedroom scores low
against a sound-event bank, so it routes to generation instead of confidently saying
"tuning fork".

---

## 4. Answers to the other two questions

### 4a. Why did int8 on the embedding table break it? (measured, not guessed)
Every cheap proxy said it was safe — per-token cosine **0.99991**, random-embedding unit
test **0.999975** — yet encoder-output cosine collapsed to **0.31-0.58** and retrieval
correct-field went **0.939 → 0.272**.

Three scale granularities were tested to find the mechanism, and **all three fail
identically**, which rules out the obvious explanation:

| variant | encoder-output cosine |
|---|---|
| per-row scale | 0.49 / 0.49 / 0.31 |
| per-column scale | 0.49 / 0.50 / 0.31 |
| two-sided (row+col, smoothquant-style) | 0.51 / 0.48 / 0.36 |
| **int8 LINEARS only (for contrast)** | **0.9998** |

So it is not outlier placement — it is **precision at the signal source**. The embedding IS
the model's entire input; there is nothing else to recover information from. Measured on the
real table: a typical dimension gets only **~22 of 127 levels** (row median |w| = 0.042
against a row max of 0.242), Gemma then multiplies by `sqrt(768) = 27.71`, and 24 residual
layers compound it. Quantizing *linear weights* perturbs transformations that the residual
stream partly routes around; quantizing the *input embedding* perturbs the signal itself at
layer 0.

**Practical rule this yields: quantize the linears (safe, 0.9998, ~105 MiB), never the input
embedding table. And never trust weight-space cosine — validate end-to-end.**

### 4b. Lazy-loading the bank — yes, partly, and worth doing
The bank has two parts with very different requirements:
* **caption TEXT** (121k Python strings, ~30-60 MB): only the WINNER is ever read. This can
  be lazy-loaded trivially — keep byte offsets, `seek()` the one winning line. **Pure win.**
* **caption EMBEDDINGS** (355 MiB fp16): needed in full for the arg-max matmul. Options:
  `torch.load(..., mmap=True)` and let the OS page it (lookup is only 2.6 ms, so there is
  latency budget); or a two-stage coarse-to-fine search over k-means centroids, which cuts
  the resident set by ~10x at a small recall cost.

But note: **if §2 lands, the bank stops being the primary path**, so lazy-loading it becomes
an optimization of the fallback rather than of the main road. Sequence accordingly.

---

## 5. Recommended sequence
1. **Mercury capacity probe (cheap, decisive):** train the z_q → LFM2.5-350M prefix
   projector on VGGSound rich captions; measure caption quality on held-out clips vs the
   M3-era baseline (word-overlap F1 0.317 / semantic cosine). Answers the 350M-capacity
   risk before any Jetson work.
2. **If it passes:** export, run through the proven `embd` path, measure on-device latency.
3. **Wire the hybrid router** with a calibrated τ; keep retrieval for the fast in-distribution
   case.
4. **Then** revisit bank lazy-loading / size, which by then is the fallback path only.


---

# RESULTS (2026-08-14) — the hypothesis was WRONG, and the real bottleneck is upstream

## What was run
`models/perception_prefix.py` + `train_perception_prefix.py`: a 31.3M projector → 16 soft
tokens → **frozen BMO Qwen3-0.6B thinker v3** (the same weights deployed as GGUF). Full
corpus: 172,593 VGGSound (×6 granularities) + 345,754 Action100M (×2). 8,000 steps, 4 GPUs.

## Result 1 — capacity is NOT the wall, but it did not beat M3
F1 trajectory: 0.155 → 0.230 → 0.246 → 0.257 → 0.260 → 0.265 → **0.269 (best)** → 0.268.
Against **M3's 0.317** on the same captions. Query-sensitivity reached **1.00** (every
question changes the output), so the text-question design works.

0.269 with a **0.6B** decoder vs M3's **1.5B** is respectable, and the curve plateaued
rather than collapsing — so the Ultravox "350M is a hard wall" verdict did NOT simply repeat.
But it did not beat the baseline either.

## Result 2 — retrieval BEATS generation, even out-of-distribution

| condition | word-F1 | semantic cos |
|---|---|---|
| retrieval, bank contains the clip | **0.5656** | **0.7481** |
| generation, same clips | 0.2657 | 0.5033 |
| retrieval, clip REMOVED from bank (leave-one-out) | **0.3773** | **0.6654** |
| generation, same | 0.2657 | 0.5033 |

**My leave-one-out OOD test was too weak, and I should say so plainly:** removing one clip
from a 2,000-clip VGGSound bank still leaves ~1,999 near-identical ones (military-parade
clips are plentiful), so retrieval still finds a genuinely good caption. That test does not
simulate "a room the bank has never seen".

Qualitatively generation is semantically RIGHT but loses on word-overlap because it
paraphrases — and it invents specifics:
> ref: *"A man wearing glasses laughs heartily…"*
> gen: *"a man laughing loudly as he stands in front of a wall. He is wearing a red shirt
> and has a white face mask."* ← the shirt and mask are hallucinated.

## Result 3 — THE DECISIVE TEST: both fail on the real room
Ran both paths on an actual frame from BMO's CSI camera (a dim bedroom: chairs, posters,
a door, a desk) against the full 121,104-caption bank:

| question | retrieval | generation |
|---|---|---|
| "Describe the room and setting in detail." | *"a person interacting with their environment inside a building… air conditioning unit mounted on the ceiling"* (0.747) | *"a person is engaged in a process involving a vacuum cleaner… near a window"* |
| "What is happening?" | *"Opening a microwave oven."* (0.734) | *"A person is operating a car door."* |
| "What do you see around you?" | *"The microwave door being opened and closed."* (0.667) | *"A person is operating a vacuum cleaner."* |

**Both are wrong.** There is no microwave, no vacuum cleaner, no car door. Generation is
arguably worse — it is confidently specific about objects that are absent, which is exactly
the failure mode that got M3 dropped.

## THE CONCLUSION — the genericity problem is UPSTREAM of the output head
Swapping retrieval → generation does not fix it, because **both consume the same
representation**, and that representation is what is out of distribution. M2/V-JEPA2 were
trained on VGGSound + Ego4D — *action and sound events in video*. A static, dim bedroom
lands outside that distribution, so the world-state encodes "some indoor activity" and every
downstream head, however expressive, can only produce VGGSound-flavoured answers.

**This reframes the fix.** The lever is the perception representation, not the decoder:

1. **Add a general image-text encoder for scene description.** This project's own M1 result
   is the evidence: the **SigLIP2 baseline scored R@1 32.5 vs the V-JEPA2 spine's 22.5**.
   V-JEPA2 is an *action-video* model; SigLIP/CLIP-family models are trained on broad
   web images and cover rooms, furniture and objects far better. For "what does this room
   look like", the wrong encoder is being asked the question.
2. **Or adapt the domain** — fine-tune/extend M2 on indoor-scene data.
3. **Keep retrieval as the fast path** regardless; it is measurably better than generation
   in-domain (0.5656 vs 0.2657) and even under weak-OOD (0.3773 vs 0.2657).

**What NOT to do:** spend more compute on the decoder side (bigger LLM, longer training,
more prefix tokens). The evidence says the ceiling is set before the decoder sees anything.


---

# SigLIP2 TEST (2026-08-14) — the encoder IS the bottleneck, confirmed

Held everything fixed and swapped only the image encoder: same room frame, same
121,104-caption bank. `scripts/siglip2_room_test.py`,
`checkpoints/SIGLIP2_ROOM_TEST.json`. SigLIP2 is zero-shot here — no training — because it
is natively an image-text model and can score captions directly.

## Probe: room sentences vs event sentences, on a photo of a room

| rank | kind | score | sentence |
|---|---|---|---|
| 1 | **ROOM** | +0.1281 | an empty dim room with furniture and pictures on the wall |
| 2 | **ROOM** | +0.0853 | a bedroom with a desk, an office chair and posters on the wall |
| 3 | **ROOM** | +0.0835 | an indoor room with a door, shelves and a chair |
| 4 | **ROOM** | +0.0642 | a home office with a desk and a computer chair |
| 5 | event | +0.0619 | a person playing an accordion indoors |
| 7 | event | −0.0191 | a person opening a microwave oven in a kitchen |
| 9 | event | −0.0410 | a car door being opened in a parking lot |
| 10 | event | −0.0425 | soldiers marching in a military parade |

**Room sentences take 4/4 of the top slots, and every event sentence scores at or below
zero.** The exact distractors the current stack fell for — *microwave*, *car door*,
*vacuum cleaner* — are ranked at the BOTTOM, several with negative similarity.

## Same bank, better encoder
SigLIP2's top-5 from the identical 121k bank are all **indoor-room** captions
("someone seated on a black leather couch inside a room adorned with wall art…", "a man
seated inside a room… the setting appears to be a home"), where the V-JEPA2/M2 path returned
"opening a microwave oven".

## Verdict
The failure was **never** retrieval-vs-generation. Both heads were reading a representation
that had no idea it was looking at a room. Given the same room and the same candidate
sentences, an encoder trained on broad web images ranks rooms first and events last, while
V-JEPA2+M2 — trained on VGGSound/Ego4D action-and-sound video — does the opposite.

This also matches this repo's own M1 result (SigLIP2 zero-shot R@1 **32.5** vs the V-JEPA2
spine's **22.5**), which was recorded a year ago and points the same way.

## Recommended architecture change
Add SigLIP2 as a **scene stream** alongside V-JEPA2 rather than replacing it — they are good
at different things, and this project has already measured that complementary streams beat
either alone (the query-predictor ablation: `m2+vision` 0.478 > `m2` 0.385 > `vision` 0.447):
* **V-JEPA2 + WavJEPA + M2** — motion, actions, audio events, AV congruence. Keep.
* **SigLIP2** — static scene content: rooms, objects, furniture, people, text. Add.
Feed both into the query predictor's existing multi-source `source_dims` interface (it
already takes arbitrary named streams, so this is a config change plus a retrain, not new
architecture). Jetson cost must be measured before committing — so400m is a large encoder;
a smaller SigLIP2 variant (base/large) should be benchmarked at the same time.
