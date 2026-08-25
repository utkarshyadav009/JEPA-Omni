# Overnight status — 2026-08-08 (BMO voice + Tier 1 + tool-calling)

TL;DR: emotion voice **v3** is retrained and deployed (both defects you heard are fixed at
the root); **Tier 1** speculative prefetch is built + benchmarked (~893ms hidden per hit);
the **tool-calling execution layer** is built and working. Three things wait for you (audio
approval, an LLM-retrain review, and the full-stack benchmark that needs preflight).

## 🔊 Listen first (same voice, three views)
- **http://100.87.60.100:8901/v3/** — v3, mood-appropriate lines, includes "Beemo" words
  (this is the one to judge pronunciation + emotion on).
- **http://100.87.60.100:8901/v3jetson/** — v3 rendered on the actual Jetson.
- Older for comparison: `/v2/` (had the 7s garble), `/pronunciation/` (Fish A/B that proved
  the mispronunciation was in the training audio).

## ✅ Emotion voice v3 — deployed (task #185)
Both v2 defects fixed at the root:
- **Beemo mispronunciation** → the *Fish training audio* was spelling "B-M-O" as letters
  (your A/B confirmed it). Regenerated all 338 BMO-line audios with "Beemo" spelling so the
  sound matches the text. → *please confirm by ear on `/v3/`*.
- **7s intelligible-garble** → EOS over-run: happy/content had the longest training clips
  (max 431 codes), so those tokens kept generating past the words. Fixed with a >320-code
  length filter (in prep + finetune) and a length-aware inference cap. On the Jetson v3 now
  renders clean ~4s clips (no 7s). `content`/`lonely` are still a touch long on very long
  sentences — probably legitimate slow delivery, but worth your ear.
- **Jetson numbers (v3, TTS-only, no perception):** loads 4.8s, **TTFA 454ms warm / 581ms
  first**, durations clean. eval_loss 0.526.
- **Jetson numbers (v3 in the FULL production stack):** whole stack (2 LLMs + emotion TTS +
  perception) loads and speaks — **VOICE TTFA 723ms without preflight / 582ms after preflight**
  (contention with resident models is the difference), fast-tier generation confirmed working
  ("Let's make a list of small, manageable tasks..."), SMOKE_TEST_DONE. NOTE: even after
  preflight, the perception `.to(device)` hit the NVML assertion once and the retry (#183)
  recovered it — so the retry hardening is essential, not just a preflight substitute.
  Preflight run used your granted access; **all services restarted afterward** (bmo_app,
  bmo_tunnel, burningtruth_app/tunnel, jtop all confirmed active) — Jetson left as found.
- **On-device Beemo pronunciation test:** http://100.87.60.100:8901/beemo/ (6 moods, every
  line says "Beemo" — the definitive pronunciation check on the real device).
- Deployed as `bmo_neutts_emotion_Q8_0.gguf` on the Jetson but **opt-in** (`BMO_TTS_EMOTION=1`);
  v5 is still the default until you approve v3 by ear — then I flip the default.

## ✅ Full-stack load hardened — no preflight needed (task #183)
The full stack used to crash at a RANDOM load point (M3 or MiniCPM5) without preflight,
because it runs at the memory edge. Added the TTS load's proven 5x-retry+compaction to the
LLM-tier loads and the perception `.to(device)` calls (`_load_gguf_retry` / `_to_device_retry`
in `bmo_jetson_startup.py`). Verified live: MiniCPM5 load failed once → retry compacted →
succeeded → whole stack up. **The full stack now loads reliably without stopping any
services.** (Preflight is still nice-to-have when services are heavy, but no longer required.)

## ✅ Tier 1 speculative prefetch — built + benchmarked (task #182)
Your "predict the user / Moshi" idea, Jetson-feasible version. `models/m5_speculative.py`
speculatively runs the fast tier (+ pre-decodes first audio) on partial transcripts during
the user's turn; commits instantly if the final transcript matches, else silently falls back
(a wrong guess is never spoken).
- **Jetson benchmark (real models):** baseline response path = fast-tier 430ms + first-audio
  463ms = **893ms**; on a HIT → **~0ms perceived** (~811ms removed per hit). Miss → safe
  fallback (verified). `scripts/bench_tier1_speculative.py`.
- Not yet wired into the live loop — that needs mid-turn STT + a mic (Jetson has none), so
  it's deferred; the component + benchmark stand alone.

## ✅ Tool-calling — the missing execution layer, built (task #184)
The LLM already emits `<tool_call name=weather day=today/>` but nothing ran it. Now:
`models/m5_tools.py::ToolDispatcher` parses → executes → folds the result into BMO's line.
Tested end-to-end (`<tool_call name=time/>` → "It's 9:38 AM right now!"; weather/timer/
reminder too). time/date are real; weather/search/timer/reminder are stubs with obvious
real-API injection points. `assets/tool_call.gbnf` guarantees valid tag form.

## ⏳ Waiting for YOUR review (I did NOT do these autonomously — by design)
1. **Approve voice v3 by ear** (`/v3/`) → I flip the emotion voice to the production default.
2. **Expanded corpus draft** (`data/bmo_companion_tools_v8_DRAFT.jsonl`, GPT-OSS-120B) —
   generated: 132 mood + **48 companion** lines (these came out lovely — "remember when you
   snoozed during the boss fight? I kept the high score safe for you!") + 24 open-question.
   The tool_use slice (recovered via small batches after an n=72 truncation) added 60 lines
   covering all 6 tools evenly (weather/search/time/date/timer/reminder). Draft = 264 lines
   total. It's an *unverified draft*; review it, then it's a
   straightforward LoRA-retrain of LFM2+MiniCPM5 → new tools/companion behavior go live. I did
   NOT retrain/deploy LLM weights on unreviewed data.
3. ~~Full-stack Jetson benchmark~~ **DONE** — got it without preflight via the retry
   hardening (VOICE TTFA 723ms in the full stack, above). Nothing needed from you here.

## Notes
- Jetson access dropped mid-session (Tailscale SSH re-check) — you re-approved it; fine now.
- Training-box disk hit 100% mid-retrain (full NeuTTS checkpoints are 3.1G each). Freed ~70G
  by pruning `checkpoints/*/checkpoint-*` intermediates — every `/best`, GGUF, and the
  deployed v5 preserved. Watch disk on future retrains.
- LLMs do NOT need retraining for Beemo (TTS-side) or personality (already fine-tuned in) —
  verified there's no personality system prompt, only the live `[energy mood]` state.
- Everything is in `CLAUDE.md` (deep detail), `SPECULATIVE_TURNTAKING.md` (Tier 1), and the
  task list.
