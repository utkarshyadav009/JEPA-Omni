# What's left to a finished BMO pipeline

Written 2026-08-15, updated through the speaker-v3 deploy and the tools work. Everything
marked MEASURED was run on the device; everything else is labelled.

---

## 0. What actually works right now, on-device

| capability | state | measured |
|---|---|---|
| perception → tags | ✅ | R@1 **0.737**, query **25 ms**, perception leg **651 ms** |
| identity head | ✅ | **+96 MiB**, 4–9 ms cached / 218–239 ms with crop |
| face crops | ✅ heuristic | motion centroid, no detector, no second camera client |
| eye tracking + perception together | ✅ | previously **impossible** — CSI sensor is exclusive |
| capture | ✅ | 905 → **6 ms** |
| two brains | ✅ | speaker 9.4 ms/tok, thinker 18.2 ms/tok |
| memory budget | ✅ | **710 MiB free**, +444 recovered; TTS+STT fit with +68 to spare |
| end-to-end tick | ✅ | **1,775 ms** |
| **persistent memory** | ✅ | reboot-verified, `memory_ms=0`, both halves persist |
| **speaker v3** | ✅ | deployed; uses a handed name, no invented names |
| **TTS runtime** | ✅ | never broken — ORT 1.23.2, RTF 0.11 |
| **tools** | ✅ | time/date/weather real; Wikipedia + DuckDuckGo keyless |

**BMO can see, understand, recognise, remember across reboots, reason, decide what to say, and
look things up. It cannot yet hear continuously, and none of it is wired into production.**

---

## A. BLOCKING — there is no working companion without these

### A1. ~~TTS is broken~~ — **CLOSED 2026-08-15: it was never broken**
The recorded blocker was STALE. `import onnxruntime` works (ORT 1.23.2), the decoder loads and
decodes at RTF 0.11. It is also already int8, so the planned 298->75 MB conversion does not
exist. Real cost 401 MiB, or 331 with `enable_cpu_mem_arena=False`. **The single biggest item
on this list resolved by reading rather than fixing.**

### A2. Corpus → retrain → deploy · **speaker v3 SHIPPED, thinker still open**

| | state |
|---|---|
| **speaker v3** | ✅ **deployed and verified on-device** — corpus v10c (3,641 rows, 0 cartoon refs), val_loss 0.7093, uses a handed name, no invented names, personality intact |
| speaker v4 | trained, **NOT deployed** — v10d 3,703 rows, `name_stranger` 92, val_loss **0.7286** (marginally worse than v3); better hostility handling |
| **thinker v4** | ❌ **REJECTED** — **175/324 rows (54%) cartoon-contaminated**; the `HARD_RULES` fix was applied to the speaker generator and never the thinker's |
| thinker v5 | in flight — same scenarios, generator now carrying the speaker's `ABSOLUTE RULES`, gated to abort at >16/~324 cartoon rows, 4 epochs |
| speaker v5 | in flight — closed-set-only verifier (the old open-set rule caused **93 false rejects vs 9 real catches** and cut BMO-idiom rows), reporting `idiom_pct` vs v4's 39.1% |

Remaining after v5 lands: pick the better speaker, thinker GGUF convert, listen-test, deploy.
See `ARCHITECTURE.md` §14 for the verifier lesson (closed sets are rule-checkable, open sets
are not) — it applies to any future corpus filtering.

### A3. Production wiring
**Everything measured today lives in `scripts/jetson_core_pipeline_test.py`, not in
production.** `build_bmo_stack()` has `perception_query` but **no identity head, no motion
centroid, no identity schedule** — and still constructs the **dead** `m3_connector` and the
**retired** `wavjepa_nat` (+701 MiB, zero measured benefit). `m5_streaming_loop` likewise has
no identity path.

### A4. Enrolment + persistence
`JepaMemory` has `enroll` / `save` / `load` / `calibrate_threshold`, but there is **no flow**:
BMO asks your name → hears it (needs A5) → enrols → **persists across reboot**. Today the
memory is empty every boot, so "I don't know you" is the only reachable branch in real use.

### A5. STT → streaming
SenseVoice is offline/chunked, which is *why* turn-taking needed a separate Moonshine head.
`sherpa-onnx-streaming-zipformer-en-2023-06-26` int8 is **~70 MB** (vs 228) and genuinely
frame-synchronous. Its per-frame partials are also what A6 needs. A/B the WER first.

---

## B. QUALITY — works, not good enough

| # | item | current | note |
|---|---|---|---|
| B1 | identity threshold | 0.5 default | the 0.765 TAR@FAR1% was calibrated on VoxCeleb2, **not on this camera/room**. Recalibrate on real enrolments. |
| B2 | face localisation | motion centroid | a **still person produces no crop at all**; centroid is chest-height, biased up 22% by heuristic. A real detector is the fix. |
| B3 | query sensitivity | within-clip 0.811 | vs 0.883 for the EmbeddingGemma reference — the one metric the SigLIP2 move did not recover |
| B4 | emotion voice coverage | `lonely` 19 clips, `happy` 39 | thin after the text filter; `<\|LONELY\|>` ran to the length cap |
| B5 | tag vocabulary | **1,482** phrases (+110 appearance) | still a hard ceiling on what BMO can report. See D1. |

---

## C. OPTIMIZATION — the phase that matters most

The identity schedule (`models/m5_identity_schedule.py`) is the template: **gate expensive
work behind a cheap always-on signal, reuse otherwise.** This is temporal amortization from
real-time rendering, and the same argument applies to almost everything below.

| # | item | current | idea |
|---|---|---|---|
| C1 | **perception leg** | **651 ms every tick** | the dominant cost, recomputed whether or not anything changed. Gate on the motion signal that already exists; reuse the last world-state when the scene is static. Biggest single win available. |
| C2 | SigLIP2 scene | 90 ms/tick, 4 frames | per-frame and incremental by nature — reuse frames across ticks instead of re-encoding all 4 |
| C3 | ~~codec int8~~ | — | **DEAD: the decoder is already int8.** `enable_cpu_mem_arena=False` gives 70 MiB and is applied. Nothing further here. |
| C4 | speculative prefetch | built, **unwired** | benchmarked at **~811 ms** of perceived latency removed on a hit; needs A5's partials |
| C5 | thinker Q6_K | Q8, 1,091 MiB | ~240 MB, only if still short; gate on a reasoning A/B |
| C6 | KV quantization | n/a | **measured worthless at n_ctx=512** — do not spend time here |

**Already settled, do not reopen:** WavJEPA-nat (+701 MiB, +326 ms, zero gain), torchao int8
(no-op, torch 2.8 < 2.11), SigLIP2 CPU-first load (moves cost, saves nothing),
`logits_all=False` (breaks confidence routing silently — `ln(65536)`).

---

## D. RESEARCH — parallel, not blocking

### D1. `llama_batch.embd` — the thinker reading perception directly
The mechanism is **proven and banked** (`prototype_llama_embd_input.py`, byte-identical to
the token path) and unused. It would remove the 1,482-tag ceiling (B5) entirely.

Tried twice, lost twice: M3 (F1 0.317, 1–6 s) and perception-prefix (F1 **0.269**). **But
both read the OLD encoder** — the one measured blind to scenes. Three things have changed:
the scene stream exists, the predictor went 0.458 → 0.737, and the target space is a proper
learned projection. So those results falsify *that encoder*, not the mechanism.

**Gate: must beat R@1 0.737 on the same eval before it displaces retrieval.** Run it as a
parallel track.

---

## Suggested order (revised 2026-08-15)

1. ~~A1 TTS~~ — **closed, was never broken.**
2. **A2 finish** — thinker v4 + speaker v4 are chained and running. Then GGUF + deploy.
3. **A5 streaming STT** — the last blocker on BMO *hearing*, and it unlocks both A4 and C4.
4. **A4 enrolment flow** — needs A5 to hear a name; everything else (enrol, persist, recall)
   is already built and reboot-verified.
5. **A3 production wiring** — move it out of the test harness, delete the dead `m3_connector`
   and retired `wavjepa_nat` while you are in there.
6. **C1 perception amortization** — largest remaining latency win; `m5_identity_schedule.py`
   already proves the pattern (218–240 ms → 43 ms amortized).
7. Then B and D as capacity allows.

**Do not reopen:** WavJEPA-nat, torchao int8, SigLIP2 CPU-first load, `logits_all=False`,
codec int8 conversion, KV quantization at n_ctx=512, SigLIP2 weight quantization (89 MiB max
vs ~330 MiB already taken from the framework).

---

# REVISION 2026-08-16 — reordered by what the live run actually showed

The ordering above was written when **perception was believed to be the dominant latency leg**.
That is no longer true. Enabling real CoT on the thinker (it had been silently disabled — see
CLAUDE.md, "Live-pipeline defects found 2026-08-16") moved it to **1,749–3,509 ms, median ~2.4 s**,
against perception's 650–1,400 ms. The thinker is now the pipeline's biggest cost.

## Closed this session
* **Audio branch was fed `torch.zeros`** — WavJEPA + M2-audio ran on silence. Live mic now wired.
* **`wearing` question silently skipped** — deployed candidates had no `appearance` category.
  `candidates_siglip2_v2.pt` deployed; empty categories now raise.
* **Thinker never emitted `<think>`** — `enable_thinking` inherited as `False`.
* **CoT was discarded** — speaker got the answer, not the reasoning, so it could only paraphrase.
* **`who: a person lying down (+0.71)`** — camera mounted upside down; `--rotate 180` (user fix).
  Perception now reads correctly: glasses / sitting / headphones / watching a screen / home office.

## New queue
| # | task | why now |
|---|---|---|
| 11 | **Speaker ignores the thinker** | with the thinker fixed, this is the last thing between us and a coherent turn. Test the *corpus-format gap* hypothesis first — if no speaker row is instruction-conditioned, this was never a v1→v5 regression |
| 12 | **`bmo-power` tools in the thinker** | user shipped the CLI + python module; unlocks battery→homeostatic-state coupling **and** gives #15 its reference signal |
| 13/14 | **GLR latent reasoning** | the 2.4 s median is the justification; `llama_batch.embd` hook already proven byte-identical |
| 15 | **NLMS fan cancellation** | spectral subtraction measurably failed; the tach from #12 is the reference input ANC actually needs |

## Still pending from the previous revision
A2 corpus/retrain (now folded into #11 — do the diagnosis before another blind retrain; **five
corpus regenerations have not fixed this**), A5 streaming Zipformer STT, A4 enrolment flow
(blocked on A5), A3 production wiring into `build_bmo_stack`, C1 perception amortization.

**Note on A3:** production still runs the *old* candidates file and the *old*
`GGUFReasoningTier` behaviour is now fixed in `models/` but `build_bmo_stack` has not been
re-verified against it. Do not consider A3 done until a full stack boot is measured with
`enable_thinking=True` — the thinker's memory profile changes when CoT actually generates.
