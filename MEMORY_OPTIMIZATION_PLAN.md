# BMO Jetson memory plan — fitting identity + emotion TTS + streaming STT in 7.6 GB

Written 2026-08-15. **Every "measured" number came off the device**
(`jetson_artifacts/benchmarks/fit_2026-08-15/`). Anything not measured is labelled ESTIMATE,
and the estimate column is where this plan can be wrong.

TTS/STT model choice is **open**. This document gives the envelope to shop within and prices
the candidates; it does not assume the current models.

---

## 1. Measured baseline

Fresh reboot + `PREFLIGHT: PASS`, MAXN_SUPER, face engine + Xorg up, no TTS/STT, identity on:

| component | Δ MiB | note |
|---|---|---|
| baseline (face engine 223 + Xorg 65 + tailscaled + CUDA ctx) | ~1,750 | not reclaimable |
| speaker LFM2.5-350M Q8 | 253–885 | first GGUF load also absorbs the llama.cpp CUDA context |
| thinker Qwen3-0.6B v3 Q8 | 780–1,091 | |
| ViT-L int8 | 629–896 | |
| WavJEPA base | 618–699 | nat dropped — cost +701 MiB and +326 ms for nothing |
| M2 | ~50–88 | |
| **identity head (8.15 M)** | **+96** | **FITS. 4–9 ms/query.** |
| SigLIP2 (vision tower only) | 519–1,170 | largest single line |
| pre-encoded queries | **0** | replaced EmbeddingGemma's 578 |
| query predictor + 1,372 tags | 134–360 | replaced the 355 MiB caption bank |
| camera | 18–255 | |
| **avail after full load** | **1,306** | |
| runtime growth over 3 rounds | ~1,040 | activations, KV growth, camera buffers |
| **avail at end of run** | **266** | ← the real budget |

**Read totals, not lines.** Per-component deltas swing by hundreds of MiB between runs
because the caching allocator recharges freed blocks elsewhere. §3 L1 is a worked example of
being fooled by a single line.

## 2. The envelope

**~266 MiB free today**, and every lever in §3 adds to it. That is what TTS + STT must fit in.

## 3. Levers

### L1 — SigLIP2 CPU-first load · **MEASURED: NO NET CHANGE. Not a lever.**
Hypothesis: `AutoModel.from_pretrained(...).to(device)` then `del text_model` makes the GPU
pay the full 716 MiB peak, so loading on CPU, deleting, then `.to(device)` should save the
538 MiB text tower. **Applied and measured: the SigLIP2 step dropped +773 → +519 MiB, but
available-at-camera was 1,251 vs 1,256 — identical.** The allocator simply recharged the
difference to the query predictor (+360 vs +171) and camera (+255 vs +50). Kept anyway (it is
strictly more correct, and lowers peak), but **do not count it as savings.**

*Lesson: a single mem-log line improving is not a saving. Only end-of-run available is.*

### L2 — `flash_attn=True` · **MEASURED −228 MiB · LOW risk · DO THIS FIRST**

**Measured on the fast tier**, same model, same prompt, only the flag changing:

| config | MiB | mean_neg_logprob |
|---|---|---|
| `logits_all=True`, no FA (**current**) | **253.0** | 0.7932 (works) |
| `logits_all=False` + FA | 4.0 | **11.0904 — BROKEN** |
| **`logits_all=True` + FA** | **25.0** | 0.6043 (works) |

**~228 MiB, with the confidence router intact.** The boot log named this itself:
`FA is not enabled - padding V cache to 512`.

**TRAP — do not set `logits_all=False`.** It looks like the bigger win (253 → 4 MiB) but it
silently destroys confidence routing, which decides when to escalate to the thinker.
`mean_neg_logprob` pins to **11.0904 in every configuration — exactly ln(65536), the vocab
size**, i.e. a uniform distribution over an all-zero buffer. Every Python-level accessor was
tried (`scores[n_tokens-1]`, `scores[0]`, `scores[-1]`, `eval_logits[-1]`) and all return
zeros; `_ctx.get_logits_ith(-1)` raises a ctypes buffer-format error. llama-cpp-python 0.3.34
simply does not populate `scores` in the `generate()` path without `logits_all`.

**`n_ctx` is a red herring.** It is already **512** (lowered from 2048 in 2026-08-07 after a
real OOM) — there is no 8192→4096 reduction available. With FA on, 512 → 192 changed memory
by 0 MiB and output not at all. The real fast-tier prompt is **34 tokens**.

`type_k`/`type_v` = q8_0 measured *slightly worse* than FA alone (7 vs 4 MiB in the
logits_all=False sweep) — KV at n_ctx 512 is already small enough that quantizing it is
noise. q4_0 additionally **changed the generated text**. Not worth it.

### L2b — (superseded, kept for the record) KV-cache quantization · ESTIMATE 100–250 MiB
Two concrete, evidenced items:
* The boot log says **`the V embeddings have different sizes across layers and FA is not
  enabled - padding V cache to 512`** — we pay padding for want of flash attention.
  `llama-cpp-python 0.3.34` exposes `flash_attn` (verified present).
* It also exposes `type_k` / `type_v` (verified). `q8_0` halves KV at negligible quality
  cost; `q4_0` quarters it.
* `n_ctx` is the 512 default for both tiers. The thinker emits up to 320 CoT tokens so it
  needs real context; the **speaker** emits ≤48 and does not.

### L3 — Thinker Q8 → Q4_K_M · ESTIMATE ~370 MiB · MEDIUM risk
`bmo_thinker_qwen3_v3_Q8_0.gguf` is 767.5 MB; Q4_K_M lands ~400 MB. The thinker does real
`<think>` CoT, the most quantization-sensitive thing in the stack. **Gate on a reasoning A/B,
never on file size.**

### L4 — Speaker Q8 → Q4_K_M · ESTIMATE ~180 MiB · MEDIUM-HIGH risk
`bmo_lfm25_350m_v2` Q8 is 379 MB. At 350 M params Q4 degradation is proportionally worse than
on big models, and this model is **already the weakest link** (it needed the thought
sanitiser to stay grounded, and it cannot yet use a name it is handed). Do this last, if ever.

### L5 — Lazy-load TTS · ESTIMATE up to 765 MB while idle · HIGH risk
**The documented load order is load-bearing**: llama.cpp GGUFs must load before the
torch/int8 perception stack or they fail to allocate. Late-loading a GGUF into a fragmented
heap is exactly the NvMap error-12 failure this project already fought. Would need
`_load_gguf_retry` + `_compact_memory` and real soak testing.

### L6 — torchao int8 on the perception stack · **NOT AVAILABLE**
No-op on this device: torch 2.8.0 < the 2.11.0 its cpp extensions require. The 105 MiB saving
measured on mercury does **not** transfer. Do not re-derive this.

---

## 4. Choosing the STT — streaming is the requirement

| option | size (int8) | streaming? | notes |
|---|---|---|---|
| **sherpa-onnx-streaming-zipformer-en-2023-06-26** | **~70 MB** (enc 68 + dec 1.3 + join 0.25) | **yes — frame-synchronous transducer** | **recommended** |
| sherpa-onnx-streaming-zipformer-en-2023-06-21 | ~181 MB | yes | larger, likely better WER |
| sherpa-onnx-streaming-zipformer bilingual zh-en | ~190 MB | yes | only if Chinese is wanted |
| SenseVoice-Small (current) | 228 MB | **no — offline/chunked** | multilingual, RTF 0.075 |
| Moonshine-base (already integrated) | — | short-segment | already used for the turn-taking head |

**Recommendation: the 70 MB English streaming Zipformer.** It is ~3x smaller than the current
SenseVoice *and* it is the only option that actually satisfies "streaming" — SenseVoice is an
offline model we chunk, which is why turn-taking needed a separate Moonshine head. A true
transducer emits partial hypotheses per frame, which is also what the Tier-1 speculative
prefetcher (`SPECULATIVE_TURNTAKING.md`, measured ~811 ms removed on a hit) needs to run
mid-turn.

Trade to accept: English-only, and WER should be A/B'd against SenseVoice before switching.
Keep Moonshine for turn-taking or re-derive it from the transducer's partials.

## 5. Choosing the TTS — emotion is the requirement, and BMO's voice is already trained

**The decisive constraint is not size, it is that BMO's voice already exists.**
`bmo_neutts_emotion` is a fine-tune of BMO's own voice with **12 emotion tokens that match
`homeostatic_to_mood_state()` exactly**, so mood → voice needs zero translation, and it is
already wired end-to-end (`FastTierResult.mood` → `StreamingVoice.speak(emotion=...)`).
Switching backbone means redoing that.

| option | size | emotion | BMO's voice | verdict |
|---|---|---|---|---|
| **NeuTTS Nano + emotion fine-tune** | **241 MB** | 12 tokens (retrain) | **yes** | **recommended** |
| NeuTTS-Air emotion (current) | 568 MB | 12 tokens | yes | works; 327 MB more |
| NeuTTS-Air v5 (current default) | 765 MB | none | yes | no emotion |
| Chatterbox-Turbo (~350 M) | ~700 MB+ | emotion-exaggeration control | needs cloning/retrain | loses the fine-tune |
| Kokoro (82 M, Apache-2.0) | ~330 MB | **no real emotion control** | no | fails the requirement |
| Orpheus-3B | ~4 GB Q8 | inline tags | no | far too large |

**Recommendation: re-run the existing emotion fine-tune on the NeuTTS Nano backbone.** Same
recipe, same 1,421 Fish-rendered clips, same 12 tokens, same scripts
(`prep_bmo_emotion_neutts_dataset.py`, `finetune_bmo_emotion_neutts.py`) — only the base
checkpoint changes. **Saves 327 MB against the Air emotion voice and keeps both the voice
identity and the emotion control.** Nano's viability is already established: the old "Nano
can't emit EOS" verdict was retracted as a greedy-decode testing bug.

Fix the two known data defects in the same run: `lonely` (19 clips) and `happy` (39) are
under-represented after the text filter, and `<|LONELY|>` ran to the length cap.

### The codec is the hidden cost
| piece | size |
|---|---|
| NeuTTS Nano backbone | 241 MB |
| **NeuCodec ONNX int8 decoder** | **~298 MB** |

**The decoder is larger than the TTS model.** It is also the component that is currently
**broken** on this device — `import onnxruntime` aborts (`cpuid_info warning: Unknown CPU
vendor` → out-of-bounds assertion) with 5 GB free. sherpa-onnx is unaffected because it ships
its own `libonnxruntime.so.1.18.1`.

**This is the highest-leverage open question in the whole plan**: routing NeuCodec through
sherpa's bundled runtime would fix the crash *and* is the natural place to look for a smaller
decoder. Freeing memory will not fix TTS on its own.

## 6. The budget, with the recommended stack

| | MB |
|---|---|
| available today (measured) | **266** |
| + L2 KV cache (estimate) | +100…250 |
| **envelope** | **366…516** |
| — streaming Zipformer en int8 | −70 |
| — NeuTTS Nano emotion | −241 |
| — NeuCodec ONNX int8 decoder | −298 |
| **balance** | **−243 … −93** |

**Still short by roughly 100–250 MB**, and the shortfall is dominated by the codec. Order of
attack:
1. **L2** (KV cache) — cheap, reversible, do it first.
2. **Shrink or replace the codec** — it is 298 MB *and* broken; fixing it is required anyway.
3. **L3** (thinker Q4) with a reasoning A/B — only if still short.
4. **L4** (speaker Q4) last, and only with a behaviour check.

## 7. What NOT to do

* Don't count L1 as savings — measured no net change.
* Don't ship SenseVoice fp32 (894 MB) when a 70 MB streaming model does the job better.
* Don't re-attempt torchao int8 on the Jetson (L6).
* Don't reintroduce WavJEPA-nat: **+701 MiB, +326 ms, zero measured gain** in the current
  target space (audio-following 0.608 base-only vs 0.609 base+nat).
* Don't reintroduce EmbeddingGemma or a caption bank: pre-encoded queries cost **0 MiB**,
  tags cost **2 MiB**, together replacing 933 MiB — and tags beat captions on the real room.
* Don't assume freeing memory fixes TTS. The `onnxruntime` abort is a separate bug.


---

## 8. TTS reality check — 2026-08-15 (measured, and it corrects §5)

### 8.1 The runtime crash is GONE
`import onnxruntime` works: **ORT 1.23.2**, only a benign GPU-discovery warning. The recorded
blocker (`cpuid_info` "Unknown CPU vendor" -> out-of-bounds assertion) is **stale** — ORT was
upgraded at some point since it was written. The decoder imports (11.7 s), loads (1.3 s), and
decodes correctly.

**TTS is not blocked on a runtime bug. That item is closed.**

### 8.2 The codec is ALREADY int8 — the planned saving does not exist
§5 assumed converting the decoder fp32 -> int8 would take it 298 MB -> ~75 MB. The deployed
decoder is already `neuphonic/neucodec-onnx-decoder-int8`. **There is nothing to convert.**

### 8.3 What it actually costs (one session per FRESH process)
| config | total resident | warm decode |
|---|---|---|
| default ORT session | **401 MiB** | 108 ms |
| `enable_cpu_mem_arena=False` | **331 MiB** | 106 ms |

**Saving: 70 MiB, free of latency cost.** Applied in `models/m5_streaming_voice.py`, which now
owns the session directly (`NeuCodecOnnxDecoder` builds its own without the flag, and freeing
it afterwards returns only ~80 MiB because glibc does not hand arena pages back).
`decode_code` is `session.run(None, {"codes": codes})[0].astype(np.float32)` plus shape
validation, so owning it costs nothing behaviourally.

**RETRACTED — a 413 MiB claim that was a measurement artifact.** An earlier bench ran four
sessions *sequentially in one process* and reported default 414 MiB vs arena-off 1 MiB. Only
the FIRST session in a process pays; the rest reuse pages glibc never returned. Measuring one
session per fresh process gives 401 vs 331. **Benchmark allocation in a fresh process, or the
first measurement absorbs the cost of all the others.**

Decode speed is fine either way: 106 ms for 0.98 s of audio, **RTF 0.11**.

### 8.4 Corrected budget
| | MiB |
|---|---|
| available today | +266 |
| `flash_attn=True` (measured) | +228 |
| **envelope** | **494** |
| codec (arena off) | −331 |
| NeuTTS Nano emotion | −241 |
| streaming Zipformer int8 | −70 |
| **balance** | **−148** |

**Still ~148 MiB short**, and the codec is now the single largest line at 331 MiB with no
conversion available. The remaining levers are L3 (thinker Q8 -> Q6_K, ~240 MiB, needs a
reasoning A/B) and C1 (perception amortization, which frees runtime rather than load memory).


---

## 9. The biggest win was not a model — 2026-08-15

### 9.1 Where SigLIP2's memory actually goes (`scripts/jetson_memory_forensics.py`)
SigLIP2 measured 519-1,170 MiB resident against a 177 MiB fp16 vision tower. Per-stage:

| stage | Δ MiB |
|---|---|
| `import torch` | +200 |
| CUDA context | +33 |
| `import transformers` | +22 |
| **`AutoProcessor.from_pretrained`** | **+327-500** |
| SigLIP2 both towers -> CPU | +615 |
| drop text tower | −370 |
| vision tower -> GPU | cuda_alloc **185** |
| forward pass | +243 (cuda_alloc only +10) |

**Both of my hypotheses were wrong.** The caching allocator is not the problem
(`reserved − allocated` = 7 MiB; `empty_cache()` returned 20 MiB reserved and **zero**
system). Batch size is not the problem (4-at-once vs 1x4 sequential differ by 1 MiB).

**The largest single line is an image resize.** `AutoProcessor` costs more than the model it
feeds, and 4-5x more than perfect int8 of that model could ever save (89 MiB).

### 9.2 Replacing it — measured equivalent
Its config is trivial: 224x224, rescale 1/255, mean 0.5, std 0.5, `resample=2`
(PIL **BILINEAR** — bicubic drops pixel cosine to 0.992).
`models/m5_motion_crop.py::siglip2_preprocess` reproduces it in ~6 lines:

| | |
|---|---|
| pixel cosine vs AutoProcessor | 0.99994 |
| **EMBEDDING cosine vs AutoProcessor** | **0.99993 – 0.99996** |
| cosine between two DIFFERENT images | **0.9941** ← the discrimination scale |

The error is an order of magnitude below what the model distinguishes at. This mattered:
587,303 cached scene features were extracted with AutoProcessor on mercury, so the on-device
path had to stay in that distribution.

**End-to-end on the full stack:**

| | before | after |
|---|---|---|
| SigLIP2 load step | 519–1,170 MiB | **262 MiB** |
| avail at camera | 1,251–1,392 | **1,527** |
| **avail at end of 3 rounds** | 191–323 | **611** |
| `siglip_ms` | 95 | **95** (unchanged) |

### 9.3 Would a C++ runtime help? Mostly no.
The forensics answer this directly. A full C++/TensorRT rewrite of perception removes only
the `import torch` line — **200 MiB**. CUDA context (33), weights (185) and cuBLAS/cuDNN
workspace (243) are paid in any language, and the rest of the stack is already C++ where it
counts (llama.cpp for both LLMs, sherpa-onnx for STT, ONNX Runtime for the codec).

**Deleting one `AutoProcessor` call bought more than a C++ rewrite would, for six lines** —
and kept the ability to swap checkpoints.

### 9.4 The generalisable lesson, now seen twice in one day
* ONNX Runtime pre-allocated **~400 MiB** of CPU arena it never used (§8).
* `transformers` spent **~500 MiB** on a resize and a normalise (§9.1).

Neither appears in any parameter count, and both dwarf what quantizing the corresponding
model would return. **Measure what the FRAMEWORK is doing before compressing the MODEL.**

### 9.5 On quantizing SigLIP2 / S2D — not needed
int8 on the vision tower saves at most **89 MiB** (177 -> 88.6) and risks moving the
embeddings away from 587k cached features. S2D (CVPR 2026) is real and well-motivated —
activation outliers are AdamW artifacts, not features — but it is a **training-time**
method: using it means fine-tuning SigLIP2, re-extracting every cached feature, and
retraining the query predictor, to chase 89 MiB. **Not worth it now.** Revisit only if the
budget is still short after the framework-level savings.


---

## 10. FINAL BUDGET — it fits (2026-08-15)

Measured on the full stack, both fixes deployed, fresh reboot + preflight:

| | original | now |
|---|---|---|
| avail at end of 3 rounds | 266 MiB | **710 MiB** |

**+444 MiB recovered, none of it from quantizing a model:**

| lever | MiB | how |
|---|---|---|
| drop `AutoProcessor` | ~330 | 6 lines of PIL+numpy, embedding cosine 0.99993 |
| `flash_attn=True` | ~228 | one kwarg, confidence router intact |
| *(measured overlap/variance)* | −114 | per-run allocator variance |

| budget | MiB |
|---|---|
| available | **710** |
| codec (arena off) | −331 |
| NeuTTS Nano emotion | −241 |
| streaming Zipformer int8 | −70 |
| **balance** | **+68** |

**TTS + STT fit without thinker Q6_K or speaker Q4.** Both stay in reserve rather than being
spent — which matters, because the speaker is already the weakest link in the stack and the
thinker does real chain-of-thought.

**Run-to-run variance is large** (SigLIP2's own line measured +262 and +952 MiB in two
otherwise identical runs). Only end-of-run available is trustworthy; never size a decision
off a single component line.


---

## 11. Whole-pipeline forensics (2026-08-15) — and a measurement trap, third occurrence

`scripts/jetson_pipeline_forensics.py` stages every component and splits each delta into
MODEL / FRAMEWORK / RUNTIME, because the fix differs completely by category:

| category | MiB | fixability |
|---|---|---|
| MODEL | 472 | irreducible without quantization |
| **FRAMEWORK** | **383** | usually replaceable |
| RUNTIME | 349 | paid in any language |
| **TOTAL** | **1,204** | |

### 11.1 What is NOT worth fixing (measured, so we stop guessing)
| component | in-pipeline | **in isolation** | verdict |
|---|---|---|---|
| `AutoVideoProcessor` (V-JEPA2) | +158 MiB | **1.0 MiB** | **leave it** |
| `AutoFeatureExtractor` (WavJEPA) | +1 MiB | 1 MiB | leave it |
| `siglip2_preprocess` (ours) | +0.0 MiB | — | already free |

**THE TRAP, now hit three times in one day: attribution depends on ORDER.** The first
component to touch a shared dependency is charged for all of it. `AutoVideoProcessor`
looked like a 158 MiB problem purely because it was the first thing to pull in
torchvision/PIL machinery that something else would have loaded anyway. Measured on its own,
after those imports exist, it is 1 MiB.

The same trap produced the retracted 413 MiB ORT-arena claim (§8.3): four sessions in one
process, only the first paying. **Only end-to-end available memory, measured across a whole
run, is trustworthy.**

For completeness, V-JEPA2's processor IS replicable if it ever matters (resize shortest-edge
292 bilinear -> center-crop 256 -> /255 -> ImageNet mean/std, **embedding cosine 0.999995**
against a 0.8247 discrimination scale) -- but at 1 MiB there is no reason to.

### 11.2 What WAS worth fixing
SigLIP2's `AutoProcessor` -- confirmed not by a component line but by the end-to-end number:
**end-of-run available 266 -> 611 MiB** with it removed (then 710 MiB once `flash_attn`
shipped too).

### 11.3 TRAINING SIDE — the bigger number nobody was looking at
`scripts/extract_siglip2_scene.py` and `_vgg.py` construct `AutoProcessor` **per shard**, and
the measured-optimal extraction config is **64 shards** (64 x 4 threads = 256 = core count):

| per-shard processor | x64 shards |
|---|---|
| 327 MiB | **20.4 GB** |
| 500 MiB | **31.2 GB** |

...on a run that was already CPU-bound at 249 clips/s. `siglip2_preprocess` is a drop-in for
both scripts and needs no processor at all. **Not urgent** (mercury has 1.5 TiB and the
current caches are complete), but any future re-extraction should use it -- and if extraction
ever needs to run at higher shard counts, this is the constraint that will bite first.
