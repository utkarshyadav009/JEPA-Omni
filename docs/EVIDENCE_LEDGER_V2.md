# EVIDENCE_LEDGER_V2.md

Supersedes `docs/EVIDENCE_LEDGER.md` (v1, extracted ~2026-08-12 / mtime 2026-08-13 20:03:16).
Complete and self-contained — v1's rows are carried forward (annotated where
superseded/contradicted), not just diffed. Structured data only; where unverifiable, marked
UNVERIFIED. Extraction date: 2026-08-16.

**Methodology**: this extraction was run as four parallel sub-agents (Part A+E, Part B, Part C,
Part D), each independently reading the repo and citing real file paths. One of the four
(A+E) additionally attempted a full solo compile after its own internal 3-way parallelization
failed ("Fork is not available inside a forked worker"); that solo compile's Part C section is
used below (no dedicated Part C output survived), and its Parts A/B/D were superseded by the
dedicated sub-agents' deeper, independently-produced versions. **After all four returned, the
orchestrating agent additionally SSH'd into the live production Jetson (`bmo@bmo-desktop`,
Tailscale) and directly diffed the on-device copy of the production entry point against this
repo's copy** — this resolved several items every sub-agent had flagged as UNVERIFIED or as a
"CLAUDE.md vs code" contradiction, because none of the sub-agents had access to the actual
deployment target, only to this local git checkout. Those findings are new **PART F**, and they
retroactively correct specific rows in Parts B and D — cross-references are given, originals are
left intact and marked corrected rather than silently rewritten.

Repo-relative paths are relative to `/home/utkarsh/JEPA-Omni/` unless marked "Jetson:" (relative
to `/home/bmo/` on `bmo@bmo-desktop`).

---

SUMMARY (read before the tables)

Cutoff used: **2026-08-13 20:03:16** (the `docs/EVIDENCE_LEDGER.md` filesystem mtime, not the
header's self-reported "extracted ~2026-08-12" — the mtime is `stat`-verifiable, the prose date
is not). `find -newermt "2026-08-13 20:03:16"` returned 291 files.

**The single biggest finding, as it stood before Part F**: `scripts/bmo_jetson_startup.py`
(`build_bmo_stack()`), the file CLAUDE.md names as "the real entry point," had a **local-repo
copy** that was untouched since 2026-08-14 16:10 and still hardcoded `bmo_lfm25_350m_v1` /
`bmo_thinker_qwen3_v2` (the versions CLAUDE.md itself says scored worst, 1/6, on intent
adherence) and still unconditionally built the "retired" WavJEPA-nat and "dropped" M3 connector
— directly contradicting CLAUDE.md's 2026-08-16 "PRODUCTION DEPLOY" section. **Part F resolves
this**: the actual Jetson-resident copy of the same file (mtime 2026-08-16 14:36, i.e. edited
*after* this local checkout was last synced) already matches CLAUDE.md exactly — v5/v5 GGUFs,
`wavjepa_nat=None`, `m3_connector=None`, plus a SigLIP2 perception-query + identity-head wiring
this local checkout had never seen at all. The contradiction was real, but it was a **repo-vs-
deployment-target sync gap**, not a documentation-vs-code gap. The stale local copy has been
archived and replaced with the live Jetson copy as part of this extraction — see Part F.

Second-biggest: **three in-repo documents give three different, never-reconciled accounts of
what STT engine is actually live** (Moonshine-only per CLAUDE.md's `build_bmo_stack()`
narrative vs SenseVoice-Small via sherpa-onnx per `ARCHITECTURE.md` vs a second, independently-
described "deployed" harness `~/live_bmo_sensevoice.py` per `SESSION_LEDGER_2026-08.md`). Part F
narrows but does **not fully close** this one: `live_bmo_sensevoice.py` exists on the Jetson
(2026-08-14) alongside four *other* top-level `live_bmo_*.py` scripts, none of them inside the
version-tracked `~/bmo_production/scripts/` tree that `bmo_launch.sh` execs into — structural
evidence they are ad hoc dev/test harnesses, not the production path — but no running process or
service confirms which was most recently exercised, and no committed record says so either.

Third: essentially every quantitative claim in CLAUDE.md's three 2026-08-16-dated sections
traces only to a script that writes its output to a Jetson-local path never copied into this
repo (`~/bmo_demo.log`, `args.out`) — confirmed by reading each script's own output-path logic.
DO NOT CITE list in Part E is long and systematic for this reason, not a handful of stragglers.

Row counts: see end of each Part, and the consolidated count at the very end of this document.

---

# PART A — DELTA SINCE PRIOR LEDGER

## A0. Cutoff determination

| basis | value | used? |
|---|---|---|
| `docs/EVIDENCE_LEDGER.md` header claim | "Extracted ~2026-08-12" (prose, inside the file itself) | no |
| `stat docs/EVIDENCE_LEDGER.md` → Modify | `2026-08-13 20:03:16.202542702 +0000` | **yes** |

## A1. Every file created or modified since cutoff (291 files, `find /home/utkarsh/JEPA-Omni -newermt "2026-08-13 20:03:16" -type f ! -path "*/.git/*" ! -path "*/__pycache__/*"`)

| category | count | representative path(s) | description |
|---|---|---|---|
| New root-level docs (*.md) | 8 | `PERCEPTION_GENERALIZATION_PLAN.md` (08-14), `JEPA_MEMORY_PLAN.md` (08-15, extended not new), `MEMORY_OPTIMIZATION_PLAN.md` (08-15), `ARCHITECTURE.md` (08-15), `PIPELINE_REMAINING.md` (08-16), `RESEARCH_REFERENCES.md` (08-16), `SESSION_LEDGER_2026-08.md` (08-16, self-titled "2026-08-08 → 2026-08-14" but body extends to 08-16), `CLAUDE.md` (08-16, superseding prior version) | Narrative status/plan/architecture docs; see Part E for sourcing audit of their numeric claims |
| `docs/` | 1 | `docs/EVIDENCE_LEDGER.md` | the prior ledger itself (its own mtime is the cutoff) |
| New `models/*.py` | 15 | `models/perception_prefix.py`, `models/m5_perception_query.py`, `models/world_state_builder.py` (edited), `models/m5_identity_schedule.py`, `models/bmo_memory.py`, `models/jepa_memory.py`, `models/m5_streaming_voice.py` (edited), `models/m5_motion_crop.py`, `models/m4_cognitive_core.py` (edited — `enable_thinking` fix), `models/m5_tools.py` (edited — power/fan/battery tools), `models/glr_transition_head.py`, `models/m5_fan_notch.py`, `models/quantized_text_encoder.py`, `models/text_target.py` (edited — `SigLIP2TextTarget`), `models/m5_streaming_loop.py` (edited) | Perception-prefix/query-retrieval architecture, JEPA-memory identity/scheduling, GLR latent-reasoning head, fan-notch DSP, power tools |
| New `scripts/*.py` + root `*.py` | 38 | `scripts/perception_query_e2e.py`, `scripts/jetson_describe_demo.py`, `scripts/extract_siglip2_scene*.py`, `scripts/encode_captions_siglip2.py`, `scripts/build_bank_siglip2.py`, `scripts/eval_candidate_sets.py`, `scripts/eval_av_congruence.py`, `scripts/jetson_real_demo.py`, `scripts/measure_fan_signature.py`, `scripts/train_glr_thinker.py`, `scripts/eval_glr_thinker.py`, `scripts/probe_llamacpp_hidden_states.py`, `scripts/speaker_intent_bakeoff.py`, `scripts/thinker_behaviour_gate.py`, `scripts/fix_name_placeholders.py`, `scripts/generate_thinker_corpus_gptoss.py`, `scripts/clean_thinker_corpus.py`, `scripts/generate_speaker_directive_rows.py`, `train_perception_prefix.py`, `train_query_predictor.py`, `train_query_predictor_ddp.py` | Perception-query retrieval pipeline, SigLIP2 scene-encoder track (see Part C), GLR thinker training/eval, live Jetson demo/diagnostic harnesses, corpus generation/cleaning tooling |
| New `configs/` | 0 | — | no config files changed since cutoff |
| New `data/*.jsonl` + `data/real_speech/metadata.csv` | 15 | `data/bmo_companion_corpus_v10.jsonl`…`v12.jsonl` (5 files), `data/bmo_thinker_corpus_v4.jsonl`…`v7_clean.jsonl` (8 files), `data/real_speech/metadata.csv` | Successive speaker/thinker corpus iterations (v10→v12 speaker, v4→v7 thinker), each superseding the last per the docs |
| New root `*.log` | 29 | `pp_train.log`, `abl_{A,B,C,D}_*.log` (4-way stream ablation), `sig_run{A,B,C,D}.log` + `sig_arm{B,C}_restrictedpool_stopped.log` (SigLIP2 head-to-head), `corpus_v10.log`, `thinker_v4_gen/train.log`…`v7_gen/train.log`, `speaker_v3.log`…`v5.log`, `gguf_v3.log`, `gguf_speaker_v5.log`, `gguf_thinker_v5.log`, `expand_names*.log` | Raw training/generation logs for the ablation, SigLIP2, corpus, and LoRA-retrain iterations above |
| New `logs/` | 14 | `logs/glr_thinker_v1.log`, `logs/glr_thinker_v2.log`, `logs/glr_v2_eval.log`, `logs/glr_v2_eval_n200.log`, `logs/thinker_gate.log`, `logs/chain_speaker_v12.log`, `logs/directive_v12{b,c,d}.log`, `logs/chain_v6_train{2,3,4}.log`, `logs/speaker_v6.log` | GLR thinker training/eval, thinker-behaviour-gate, speaker v6/v12-corpus chain logs |
| New `checkpoints/` non-weight files (json/config/tokenizer/README) | 101 | `checkpoints/PERCEPTION_QUERY_E2E.json`, `checkpoints/EMBEDDINGGEMMA_QUANT_EVAL.json`, `checkpoints/GENERATION_VS_RETRIEVAL.json`, `checkpoints/SIGLIP2_ROOM_TEST.json`, `checkpoints/AV_CONGRUENCE_EVAL.json`, `checkpoints/CANDIDATE_SET_EVAL_runD.json`, `checkpoints/{abl,sig_run}*/train_log.json` (8 files), `checkpoints/glr_thinker_v1/eval.json`, `checkpoints/glr_thinker_v2/{eval.json,eval_n200.json,behaviour_gate.json}`, LoRA adapter_config/tokenizer/README triples for `bmo_lfm25_350m_v{3,4,5,6}_lora`, `bmo_thinker_qwen3_v{4,5,6,7}_lora` | Result JSONs for the SigLIP2/ablation/GLR tracks (individual rows — see A4/E1), plus LoRA scaffold files (no eval numbers) |
| New `checkpoints/` weight files (`.pt`/`.gguf`/`.safetensors`) | 47 | `checkpoints/perception_bank_{vgg,max}_fp16.pt`, `checkpoints/{abl,sig_run,sig_arm}*/best.pt` (8 dirs), `checkpoints/bank_{armB,armC,siglip2,runD}_fp16.pt`, `checkpoints/candidates_siglip2{,_v2}.pt`, `checkpoints/query_vectors_siglip2{,_v2}.pt`, `checkpoints/glr_thinker_v{1,2}/best.pt`, `checkpoints/bmo_lfm25_350m_v{3,4,5,6}_{lora,merged}/*.safetensors` + `_Q8_0.gguf` ×4, `checkpoints/bmo_thinker_qwen3_v{4,5,6,7}_lora/*.safetensors`, `bmo_thinker_qwen3_v5_merged/model.safetensors` + `_Q8_0.gguf` | See A4/E2 for individual provenance status |
| New `jetson_artifacts/` | 23 | `jetson_artifacts/benchmarks/fit_2026-08-15/*.json` (13 files), `jetson_artifacts/benchmarks/home/*.json` + camera JPEGs (2026-08-14 dated) | On-device benchmark JSONs copied back from the Jetson — note none newer than 2026-08-15 as of this repo checkout; the 2026-08-16-dated CLAUDE.md claims have no matching file here (see E1) — **but see Part F: the live checkpoints (`qp_runD.pt`, `identity_head_joint.pt`) DO exist on the Jetson's own `~/bmo_production/pipeline/checkpoints/`, just never copied back to this dev-machine repo** |

## A2. Prior-ledger numbers superseded by a rerun

Spot-checked 6 of the prior ledger's "superseded by X" citations (chosen for the highest-stakes
rows: the M2 RUN-2 lock, the JEPA-mem av_full head, the Phase3 G7 gate, the query-predictor DDP
number, and the two rows CLAUDE.md's own later sections most directly bear on).

| # | prior ledger claim | verified? | new/current value found | file(s) |
|---|---|---|---|---|
| 1 | `checkpoints/JEPA_MEMORY_PHASE3_GATE_avfull_far0.01.json` = current deployment recommendation, correct/FAR/wrong-name 0.515/1.12%/0.4% | confirmed unchanged | file mtime 2026-08-11 23:40, **not modified since cutoff** — no rerun exists | `checkpoints/JEPA_MEMORY_PHASE3_GATE_avfull_far0.01.json` |
| 2 | `checkpoints/jepa_identity_head_av_full/results.json`, TAR@FAR1%=0.765/AUC=0.966, "current deployment recommendation" | confirmed unchanged | file mtime 2026-08-11 23:40, **not modified since cutoff** | `checkpoints/jepa_identity_head_av_full/results.json` |
| 3 | `checkpoints/query_predictor_ddp_b1024/best.pt`, cross-clip R@1=0.715, "current" | confirmed unchanged, but **superseded in recommendation** by the SigLIP2-space `sig_runD_proj768` line (Part C/D) | file mtime 2026-08-11 13:58/15:12, not modified since cutoff; recommendation moved on | `checkpoints/query_predictor_ddp_b1024/` |
| 4 | M2 RUN-2 `step19000.pt` = LOCKED production checkpoint | confirmed unchanged | not touched since cutoff; no M2 retrain occurred in the delta window | `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/` |
| 5 | BMO prod full-stack e2e latency (prior ledger row: STT 38ms/perception 985ms/fast-LLM 337ms/TTS 829ms/decode 53ms ≈ 2.19s total, CLAUDE.md 2026-08-07) | superseded (qualitatively confirmed, exact 2026-08-16 numbers unsourced locally — see E1; **Part F pulled the live-Jetson production script and confirmed the architecture change is real**) | CLAUDE.md's 2026-08-16 "Live-pipeline defects" section: perception 650–1,400ms, thinker (formerly "fast-LLM") 1,749–3,509ms, full stack 4,361 MiB used/577 MiB free | CLAUDE.md ("Live-pipeline defects found 2026-08-16") |
| 6 | BMO prod, Moonshine STT deployed (37ms/90.67%) "current production speech-feature source" | **contradicted, not acknowledged as superseded anywhere in-repo** — see A3-1/E3-1; **Part F: the Moonshine-based `m4_decision_head_3class_speechonly_moonshine` head IS still what the live `bmo_jetson_startup.py` loads (confirmed 2026-08-16 14:36 copy); the competing SenseVoice claim traces only to standalone scripts in `/home/bmo/`, not the `bmo_launch.sh`-execed path** — see Part F, this substantially favors Moonshine as the real answer but the SenseVoice scripts' existence is still unreconciled prose | `ARCHITECTURE.md` (2026-08-14/15) states STT is "SenseVoice-Small via sherpa-onnx (+ Moonshine for the turn-taking head)," listed "live" |

## A3. Prior-ledger numbers CONTRADICTED without acknowledgment

| # | prior-ledger claim | contradicting claim | both paths | resolved anywhere? |
|---|---|---|---|---|
| A3-1 | Prior ledger (Table 1/6): "Moonshine STT... current production speech-feature source" (CLAUDE.md 2026-08-07 basis) | `ARCHITECTURE.md` line 25: `STT | SenseVoice-Small via sherpa-onnx (+ Moonshine for the turn-taking head) | live`; `ARCHITECTURE.md` line 431: `STT | ✅ works (SenseVoice, RTF 0.075)` | `docs/EVIDENCE_LEDGER.md` (cites CLAUDE.md 08-07) vs `ARCHITECTURE.md` (08-14/15) | **Partially, by Part F** — the live `build_bmo_stack()` (2026-08-16 14:36 Jetson copy) loads Moonshine for the decision head and nothing SenseVoice-shaped; CLAUDE.md's own 2026-08-16 section still never mentions SenseVoice. The SenseVoice claim's only code home is standalone `/home/bmo/live_bmo_sensevoice.py`, structurally outside the documented production tree — see Part F. Not a clean resolution: no artifact says SenseVoice was *retired*, only that it was never wired into the one script with the sudoers-scoped "real entry point" framing. |
| A3-2 | Prior ledger M4c row: Moonshine-vs-Whisper is framed as the STT decision for "the deployed 3-class turn-taking decision head" (a narrower claim than "the STT") | `SESSION_LEDGER_2026-08.md` §3 frames the *entire* conversational STT (not just the turn-taking head) as pivoting to SenseVoice, deployed via a new harness `~/live_bmo_sensevoice.py` | `docs/EVIDENCE_LEDGER.md`/CLAUDE.md vs `SESSION_LEDGER_2026-08.md` §3.3–3.4 | **no** |
| A3-3 | Prior ledger BMO-prod rows generally assume `build_bmo_stack()`/`bmo_launch.sh` (`~/bmo_production/`) is *the* production entry point | `SESSION_LEDGER_2026-08.md` §3.3 describes a **second, independently-deployed** entry point, `~/live_bmo_sensevoice.py`, also targeting `~/bmo_production/` on the same device | CLAUDE.md ("the real entry point... don't run bmo_jetson_startup.py directly") vs `SESSION_LEDGER_2026-08.md` §3.3–3.5 | **Narrowed by Part F**: `live_bmo_sensevoice.py` and four sibling `live_bmo_*.py` scripts live directly under `/home/bmo/`, not under the version-scoped `~/bmo_production/scripts/` tree, and none does the privileged sudoers-scoped memory-compaction step `bmo_launch.sh` does. Structural evidence favors `bmo_launch.sh`→`bmo_jetson_startup.py` as *the* documented production path and the `live_bmo_*.py` scripts as ad hoc dev harnesses — but no running process/service and no committed doc says so outright. See Part F. |

## A4. New experiments since cutoff with NO entry at all in the prior ledger

| # | experiment | key result | file(s) | note |
|---|---|---|---|---|
| 1 | Thinker↔perception "ask, get grounded answer" retrieval tool (bank sizes 6k/48k captions) | correct-clip 0.442→0.189 as bank grows 8×; correct-field 0.946→0.953 (bank-size independent); swapped-query control 0.000–0.003 | `checkpoints/PERCEPTION_QUERY_E2E.json`, `checkpoints/PERCEPTION_QUERY_E2E_bank48k.json`; `JEPA_MEMORY_PLAN.md` (2026-08-14 §) | new capability, no prior-ledger analog |
| 2 | On-Jetson perception-query latency/memory | query encode 263.5ms (65% of 403.8ms total), bank lookup 24,000 candidates = 2.6ms, total capability cost ~855 MiB | `jetson_artifacts/benchmarks/home/jetson_perception_query_results.json`; `JEPA_MEMORY_PLAN.md` | real on-device measurement |
| 3 | Jetson describe-demo, 7W vs MAXN_SUPER power mode | perception 3828–12368ms (7W) → 1089–1732ms (MAXN_SUPER), ~7× speedup; full round 14971–18401ms → 3892–4994ms | `jetson_artifacts/benchmarks/home/jetson_describe_demo_results{,_MAXN}.json`; `JEPA_MEMORY_PLAN.md` | root-caused a major latency discrepancy to power mode, not architecture |
| 4 | int8-via-torchao on Jetson | NO-OP on-device (`577.7→577.7 MiB`) — torch 2.8.0 lacks torchao's quantized-kernel extensions; 105 MiB saving measured on mercury only | `JEPA_MEMORY_PLAN.md` (2026-08-14 §); `checkpoints/EMBEDDINGGEMMA_QUANT_EVAL.json` | negative result, no prior-ledger analog |
| 5 | 4-way stream ablation (m2+vision+ambient / +scene / +scene+base-only / scene+vision) | cross-clip R@1: A 0.441, B 0.564, C 0.566, D 0.546 — SigLIP2 scene stream is the largest single gain (+28% rel, A→B); WavJEPA-nat adds nothing (B≈C) | `checkpoints/abl_{A,B,C,D}_*/train_log.json`; `abl_*.log`; `JEPA_MEMORY_PLAN.md` (2026-08-14 §) | see Part C for the full SigLIP2 audit; **conflicts with a same-shaped but numerically different "unified architecture ablation" row already in v1 (2026-08-11) — see E3 below, not reconciled** |
| 6 | SigLIP2 target-space head-to-head (frozen vs projected, proj768 vs proj1536) | frozen SigLIP2 target FALSIFIED (R@1 0.489 vs reference 0.681); proj768 "deploy" run: R@1 0.737, within-clip 0.811 | `checkpoints/sig_run{A,B,C,D}*/train_log.json`; `checkpoints/CANDIDATE_SET_EVAL_runD.json`; `checkpoints/AV_CONGRUENCE_runD.json`; `JEPA_MEMORY_PLAN.md` §A.1–A.5 | full detail in Part C |
| 7 | GLR (latent-reasoning transition head) v1/v2, Qwen3-0.6B frozen backbone | v1: val_loss 3.5002/ce 1.9951/delta 1505.0977 @ epoch4, λ=1e-3 (causes ‖latent‖=423 divergence at K=10 rollout, FAILED); v2 (zero-init, normalized loss, λ=1.0): val_loss 2.8487–2.6128/ce≈1.99/delta 0.85–0.70, PASSED rollout at K≤10, real but modest 1.47× token reduction (not the paper's claimed 5-7×) | `checkpoints/glr_thinker_v{1,2}/{best.pt,eval.json,eval_n200.json,behaviour_gate.json}`; `logs/glr_thinker_v{1,2}.log`, `logs/glr_v2_eval*.log`, `logs/thinker_gate.log` | genuinely new track, well-sourced locally; not yet deployed, sequencing blocked on the speaker consuming directives first |
| 8 | Speaker-intent bake-off (v1/v2/v3/v5 × LONG/SHORT prompt format) | v1 1/6+2/6 (worst — "confidently irrelevant"), v3 5/6+4/6, v5 4/6+5/6; v3 vs v5 called "within noise at n=6" | `SESSION_LEDGER_2026-08.md` (2026-08-16 later §, "THE SPEAKER") — **no local JSON**, script writes to a remote `args.out` path (see E1) | corrects a user-recalled belief that v1/v2 "were better" — see Table 5 |
| 9 | Corpus `{name}` placeholder leak (54 rows in every v10 variant) + repair | v10/v10c/v10d/v10e: 54 rows each carry literal `{name}`; repaired to v11 with 0 rows containing BMO-side substituted names (50 "removed", 2 "prompt-substituted", ground-truth verified directly against `data/bmo_companion_corpus_v11.jsonl`) | `scripts/fix_name_placeholders.py`; `data/bmo_companion_corpus_v1{0,0c,0d,0e,1}.jsonl` (independently grep-verified) | see E3 for a discrepancy inside this same finding (0 vs 13 "had a name" count) |

**ROW COUNTS — Part A**: A1 = 12 category rows (291 files). A2 = 6 rows. A3 = 3 rows. A4 = 9 rows.

---

# PART B — PRODUCTION PIPELINE FORENSICS

**As independently produced by the dedicated Part B sub-agent, from the LOCAL REPO CHECKOUT
only** (it did not have Jetson SSH access). **Part F below corrects several specific rows here
— each correction is cross-referenced at point of use; the original finding is left intact
because it was an accurate reading of what this repo's checkout actually contained at the time.**

SUMMARY (verified by direct source read, 2026-08-16, against `/home/utkarsh/JEPA-Omni` as it
stood before Part F's sync): the single most important divergence found was that
`scripts/bmo_jetson_startup.py::build_bmo_stack()` — the file CLAUDE.md calls "the real entry
point" — had a **local-checkout mtime of 2026-08-14 16:10** and reflected none of the fixes
CLAUDE.md dates to 2026-08-16. Read directly, it still (a) hardcoded `bmo_lfm25_350m_v1` +
`bmo_thinker_qwen3_v2` GGUFs (the versions CLAUDE.md itself says scored worst, 1/6, on intent
adherence), (b) loaded `WavJEPA-nat` and `M3Connector` unconditionally, never `None`. The
2026-08-16 fixes were real and present, but only inside a separate, hand-rolled validation
script (`scripts/jetson_real_demo.py`) that builds its own perception stack from scratch and
never calls `build_bmo_stack()` at all. **⟶ Part F: this was a stale-local-copy artifact, not a
real production gap — the Jetson's own copy of the same file already had all these fixes.**

SigLIP2 scene stream, as read from the local checkout: **NOT wired into `build_bmo_stack`**
(zero grep matches for `siglip`/`scene` in `models/m5_streaming_loop.py`,
`models/world_state_builder.py`, or `scripts/bmo_jetson_startup.py`). **⟶ Part F: this was also
a stale-copy artifact — the Jetson's live copy wires SigLIP2 in via a `PreEncodedTextSpace` /
`identity_ckpt` block this checkout had never seen.**

Row counts: B1=15, B2=13, B3=6, B4=11, B5=9.

---

## B1. END-TO-END TRACE

Two code paths exist end to end, **as read from the local checkout before Part F's sync**.
**Path A** = `scripts/bmo_jetson_startup.py::build_bmo_stack()` (CLAUDE.md's documented "real
entry point", execed by the Jetson-only `bmo_launch.sh`). **Path B** =
`scripts/jetson_real_demo.py::main()`, a standalone validation script that builds its own stack
directly and implements the 2026-08-16 fixes. Neither path called the other, in this checkout.
`models/m5_streaming_loop.py::StreamingLoop` (a third, generic tick-loop class) is not
instantiated by either A or B — only by test/demo scripts (`test_full_tick_loop.py`,
`scripts/jetson_m5_live_demo.py`, `scripts/m5_streaming_demo.py`,
`scripts/jetson_streaming_thread_verify.py`, `scripts/jetson_phase4_2_3_streaming.py`,
`scripts/jetson_phase4_4_rootcause_opportunistic.py`).

| stage | source file:line | what it does | input shape/dtype | output shape/dtype | checkpoint | frozen/trained | device | note: which path is real |
|---|---|---|---|---|---|---|---|---|
| mic capture | `scripts/jetson_real_demo.py:59-256` (`MicThread`) | sounddevice `InputStream` on ReSpeaker `hw:0,0`, 10s ring buffer, per-channel calibration + optional tach-driven fan notch | raw mic callback, 6ch float32 @16kHz | mono `(160000,)` float32 torch tensor via `.get()` | n/a | n/a | CPU | **Path B only** in this checkout. Path A (`build_bmo_stack`) has no mic capture code — it only loads models. `models/m5_streaming_loop.py::RollingAudioBuffer.push()` (lines 109-128) expects a caller to feed it chunks; no real mic driver exists under `models/`. |
| STT (decision-head leg) | `models/m4_speech.py:149-203` (`MoonshineSpeechEncoder`) | frozen `UsefulSensors/moonshine-base` encoder, no decode | raw waveform list, native_sr=16000 | `(1,T,hidden)` bf16 | HF repo weights (frozen) | frozen | cuda bf16, int8-quantized | Loaded both paths: `bmo_jetson_startup.py:315-317` (Path A, always) and `jetson_real_demo.py:392-399` (Path B, gated behind `--with-stt`, OFF by default in the shown invocation). |
| STT deep-grounding path (unused) | `models/m4_speech.py:109-146` (`WhisperSpeechEncoder`) + `models/m4d_stt_projector.py:56-74` (`AudioEncoderProjector`) | frozen Whisper-medium + trainable Ultravox-style projector into LLM embed space | `(B,n_mels,T)` log-mel | `(B,T',llm_dim)` | none deployed | mixed | n/a | Confirmed NOT imported by `build_bmo_stack` (comment at lines 305-314 states this explicitly). |
| STT projector (Moonshine, research track) | `models/m4d_stt_projector_moonshine.py:115-158` | frozen Moonshine encoder + trainable projector straight into LLM `inputs_embeds` | `(B,n_samples)` raw 16kHz | `(B,T',llm_dim)` | `checkpoints/bmo_stt_projector_moonshine*` | mixed | n/a | Not imported by `build_bmo_stack` or `m5_streaming_loop.py`; only training/eval scripts reference it. WER plateaus at 0.94; research track, live STT stays Moonshine-text-decode. |
| vision capture | `scripts/jetson_core_pipeline_test.py:84-99` (`open_camera`, GStreamer `nvarguscamerasrc`) + `:135-230ish` (`CaptureThread`) | producer thread fills ring buffer, rotate/gain/crop per-frame | raw BGRx NVMM 1280x720 | `(3,res,res)` uint8 torch per frame | n/a | n/a | CPU capture, GPU crop/resize | **Path B only** (imported by `jetson_real_demo.py:344,408-410`). `build_bmo_stack` has no camera code. `--rotate 180` default at `jetson_real_demo.py:282` (Path B); not present anywhere in Path A. |
| V-JEPA2 encode | `models/vision_encoder.py:79-93` (`VisionEncoder.encode`) | `facebook/vjepa2-vitl-fpc64-256` frozen ViT-L | `(B,T,C,H,W)`, T=16 in streaming config | `(B,8192,1024)` bf16 raw tokens (pre-pool) | HF repo weights | frozen | cuda, int8-quantized | Loaded both paths. `n_vision_frames=16` default at `models/m5_streaming_loop.py:104`. |
| WavJEPA encode (base) | `models/audio_encoder.py:211-230` | `labhamlet/wavjepa-base`, manual safetensor load (transformers 5.x workaround) | `(B,1,n_samples)` float32 @16kHz | `(B,~996,768)` bf16 | HF repo weights | frozen | cuda, int8-quantized | Loaded both paths. |
| WavJEPA encode (nat) | same file, `labhamlet/wavjepa-nat-base`, 2-channel input | `(B,2,n_samples)` float32 | `(B,~996,768)` bf16 (channel-mean-pooled) | HF repo weights | frozen | cuda, int8-quantized | **In this checkout, `build_bmo_stack` (Path A) still loaded it unconditionally**, contradicting CLAUDE.md's "Now None" claim — **⟶ Part F: RESOLVED, the live Jetson copy already sets this to `None`.** `jetson_real_demo.py:470` passed `None` explicitly (Path B only, in this checkout). |
| SigLIP2 scene encode | `google/siglip2-base-patch16-224` vision tower, loaded ad hoc at `jetson_real_demo.py:365-369` / `jetson_core_pipeline_test.py:353-377`; preprocessing via `models/m5_motion_crop.py::siglip2_preprocess` | `(4,H,W,3)` uint8 PIL frames | `(1,768)` bf16 normalized pooled embedding | HF repo weights | frozen | cuda bf16 | **In this checkout, NOT wired into `build_bmo_stack`** — **⟶ Part F: RESOLVED, the live Jetson copy wires it in via `PreEncodedTextSpace` + a `siglip_vision` tower load, see Part F.** |
| world-state builder / M2 predictor | `models/world_state_builder.py:109-186` → `models/av_jepa_predictor.py:194-203` via `models/m4_duplex_loop.py:63-67` | spatial-pool vision to (32,16,1024), staircase tbins, WavJEPA base(+nat) average | feats `{"vision":(1,512,1024),"ambient":(1,T_a,768)}` | world-state `(1,1024)` float, un-normalized | `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` | trained | cuda, int8-quantized | Checkpoint choice (`step19000.pt`, not `best.pt`) is CORRECT per git history. Called from `m5_streaming_loop.py::_maybe_refresh_vision` — but that only runs inside `StreamingLoop`, which neither Path A nor Path B instantiates; Path B calls `build_world_state_features` directly per-round instead. |
| query predictor / perception_prefix | `models/query_predictor.py:103-149` wrapped by `models/m5_perception_query.py:65-218` | cross-attends concatenated source tokens, query-seeded latents | sources `{m2:1024,vision:1024,ambient:768[,scene:768]}` | `(1,shared_dim)` L2-normalized | `sig_runD_proj768/best.pt` (renamed `qp_runD.pt` on the Jetson side — **⟶ Part F: byte-identity CONFIRMED, 131,615,595 bytes both files**) | trained | cuda | **In this checkout, `build_bmo_stack`'s `__main__` invocation never passed `query_predictor_ckpt`/`perception_bank_path`, so `perception_query=None`** — **⟶ Part F: RESOLVED, the live Jetson copy passes all of `query_predictor_ckpt`, `perception_bank_path`, `query_vectors_path`, `identity_ckpt`, `identity_memory_path`.** `models/perception_prefix.py` (generative alternative) remains a separate, unwired research module (B4). |
| fast-tier LLM | `models/m4_cognitive_core.py:166-279` (`GGUFFastTier.generate`), `enable_thinking=False` (correct) | llama.cpp GGUF, low-level `Llama.generate()` iterator | chat-templated prompt string | text + `mean_neg_logprob` | Path A (this checkout): `bmo_lfm25_350m_v1_Q8_0.gguf`; Path B: `bmo_lfm25_350m_v5_Q8_0.gguf` default — **⟶ Part F: RESOLVED, live Jetson Path A loads `bmo_lfm25_350m_v5_Q8_0.gguf`, matching Path B/CLAUDE.md.** | trained (LoRA, GGUF-quantized) | cuda (llama.cpp) | `CognitiveCoreRouter.route()` always calls this first. |
| reasoning-tier LLM (escalation) | `models/m4_cognitive_core.py:282-351`, `enable_thinking=True` override — **confirmed present and committed, the documented 2026-08-16 fix** | same llama.cpp mechanism, `max_new_tokens=320` | same | text (CoT stripped) + `.reasoning` (CoT retained) | Path A (this checkout): `bmo_thinker_qwen3_v2_Q8_0.gguf`; Path B: `bmo_thinker_qwen3_v5_Q8_0.gguf` — **⟶ Part F: RESOLVED, live Jetson Path A loads v5.** | trained | cuda (llama.cpp) | `.reasoning` IS populated (fix present), but `models/bmo_duplex_tick.py:70` (the one production-shaped consumer of `CognitiveCoreRouter`) discards `reasoning_result` and speaks `decision.fast_result.text` instead — CLAUDE.md's "condition the speaker on `.reasoning`" is implemented ONLY in `scripts/jetson_real_demo.py:536-544`, nowhere in `models/`. **Not resolved by Part F** — this is a real, still-open gap. |
| speaker/TTS | `models/m5_streaming_voice.py:117-334` (`StreamingVoice`) | sampled speech-token generation (temp=0.7,top_k=50) + NeuCodec INT8 ONNX chunked decode + overlap-add | text string, optional `emotion` | float32 24kHz wav + `SpeakResult` | Path A: emotion GGUF if present else `bmo_neutts_v5_Q8_0.gguf`; Path B: `bmo_neutts_nano_v1_Q8_0.gguf` default, loaded only if `--with-tts` | trained (fine-tuned NeuTTS-Air GGUF backbone) | cuda (llama.cpp) + CPU (onnxruntime) | Path B's default run does NOT exercise TTS at all. |
| audio out | `models/m5_streaming_voice.py:306-334` (real playback) vs. `models/m5_streaming_loop.py:445-447,552` (`_estimate_tts_duration_sec`, wpm-based **stub** fallback when `tts_engine is None`) | — | — | — | — | — | — | See B5. |

**ROW COUNTS — B1**: 15 rows.

---

## B2. CHECKPOINT WIRING

| checkpoint path | loaded by file:line | newest of its kind on disk? | if no, newer unused | mtime |
|---|---|---|---|---|
| `checkpoints/bmo_lfm25_350m_v1_Q8_0.gguf` (fast tier) | `bmo_jetson_startup.py:394` (Path A, **as read in this checkout**) | **No** | v5, v6 exist unused | v1: 2026-08-08 10:47 — **⟶ Part F: the live Jetson copy loads v5, resolving this row; v6 is still built-and-unwired, see Table 2/7** |
| `checkpoints/bmo_lfm25_350m_v5_Q8_0.gguf` (fast tier) | `jetson_real_demo.py:263,313-315` (Path B, default) | **No** | `bmo_lfm25_350m_v6_Q8_0.gguf` (built same day, later) | v5: 2026-08-16 00:48; v6: 2026-08-16 06:04 |
| `checkpoints/bmo_lfm25_350m_v6_Q8_0.gguf` | **no caller found anywhere** (local checkout or Jetson-side `bmo_jetson_startup.py`, per Part F's diff) | — (newest) | — | 2026-08-16 06:04 |
| `checkpoints/bmo_thinker_qwen3_v2_Q8_0.gguf` (thinker) | `bmo_jetson_startup.py:402` (Path A, **as read in this checkout**) | **No** | v5 (default in Path B); v6/v7 exist only as unmerged LoRA | v2: 2026-08-08 17:30 — **⟶ Part F: RESOLVED, live Jetson copy loads v5** |
| `checkpoints/bmo_thinker_qwen3_v5_Q8_0.gguf` | `jetson_real_demo.py:264,317-319` (Path B, default) | **No formal GGUF successor**, but v6/v7 LoRA never merged/quantized | v6_lora, v7_lora | v5: 2026-08-16 00:48; v6_lora: 00:28 (training started before v5 GGUF finished); v7_lora: 04:33 |
| `checkpoints/candidates_siglip2.pt` (1,372 tags, no `appearance` category) | `jetson_core_pipeline_test.py:236-237` default | **No** | `candidates_siglip2_v2.pt` | v1: 2026-08-15 15:38; v2: 2026-08-16 01:27 |
| `checkpoints/candidates_siglip2_v2.pt` (1,482 tags, `appearance` 110) + `query_vectors_siglip2_v2.pt` | `jetson_real_demo.py:270-271` (Path B default) | **Yes** (newest) | — | 2026-08-16 01:27 — **⟶ Part F: the live Jetson `build_bmo_stack` ALSO now points `perception_bank_path`/`query_vectors_path` at the `_v2` files, per the synced copy** |
| `checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt` (M2 predictor) | `bmo_jetson_startup.py:300-301,404`; `jetson_real_demo.py:353-354` | **Yes — correctly the intended checkpoint** (git history: "Correct RUN-2 result: best.pt was mislabeled") | `best.pt` in the same dir is the mislabeled one, correctly not loaded | step19000.pt: 2026-07-28 13:58; best.pt: 11:50 |
| `checkpoints/m3_multigran_richcaption_v2/last.pt` (M3 connector) | `bmo_jetson_startup.py:327,405` **as read in this checkout** | UNVERIFIED (no sibling to compare) | — | 2026-07-30 18:08 — **⟶ Part F: RESOLVED as no longer loaded at all — the live Jetson copy sets `m3_connector=None` (dropped 2026-08-16, matching CLAUDE.md)** |
| `checkpoints/m4_decision_head_3class_speechonly_moonshine/best.pt` | `bmo_jetson_startup.py:410,320-324` | **Yes** (only Moonshine-feature-dim (416) head; the Whisper-feature (1024-dim) sibling is a different, incompatible checkpoint) | n/a — different sf_dim | speechonly_moonshine: 2026-08-07 15:33 |
| `checkpoints/sig_runD_proj768/best.pt` (query predictor, `token_sources="m2,vision,ambient,scene"`) | Presumed via `--query-ckpt qp_runD.pt` default in Path B; **in this checkout, no file literally named `qp_runD.pt` existed anywhere under this repo's `checkpoints/`** | UNVERIFIED (only one `runD`-tagged checkpoint) | — | 2026-08-15 07:20 — **⟶ Part F: RESOLVED. `qp_runD.pt` exists on the Jetson at `~/bmo_production/pipeline/checkpoints/qp_runD.pt` (131,615,595 bytes) — byte-identical to this repo's `sig_runD_proj768/best.pt` (131,615,595 bytes). It is a renamed copy pushed to the Jetson and never synced back to this dev-machine repo, not a divergent training run.** |
| `checkpoints/identity_head_joint.pt` (referenced as `jetson_real_demo.py:272` default) | **In this checkout, no file by that exact name existed under `checkpoints/`** | UNVERIFIED | `jepa_identity_head_av_full/head_joint.pt` postdates `jepa_identity_head_av/head_joint.pt` | av: 2026-08-11 11:23; av_full: 2026-08-11 23:40 — **⟶ Part F: RESOLVED. Jetson's `~/bmo_production/pipeline/checkpoints/identity_head_joint.pt` is 32,672,373 bytes — byte-identical to this repo's `jepa_identity_head_av_full/head_joint.pt` (the BEST head in the whole identity track, TAR@FAR1%=0.765/AUC=0.966). Production uses the best available head, correctly, once again just a copy never synced back here.** |
| `checkpoints/bmo_neutts_v5_Q8_0.gguf` (TTS, non-emotion) | `bmo_jetson_startup.py:417-420` (fallback when emotion GGUF absent/`BMO_TTS_EMOTION=0`) | N/A — separate track, no "v6" for non-emotion voice | — | UNVERIFIED (not directly listed) |
| `checkpoints/bmo_neutts_emotion_v4_Q8_0.gguf` | **No caller found** — `bmo_jetson_startup.py:420` (this checkout) hardcodes the unversioned `bmo_neutts_emotion_Q8_0.gguf` name (== v1), not `_v3`/`_v4` | **No** — v2/v3/v4 all postdate the loaded file | unversioned: 2026-08-07 22:35; v2: 23:43; v3: 08-08 09:48; v4: 11:59 | **⟶ Part F: unchanged — the Jetson-side diff showed no edits to this block; still loads the unversioned/v1-equivalent file.** Still an open finding. |

**ROW COUNTS — B2**: 13 rows.

---

## B3. TRAIN/INFERENCE CONSISTENCY AUDIT

| component | training-time construction | inference-time construction | verdict | if divergent, what differs |
|---|---|---|---|---|
| M2 world-state builder (token count/pooling, tbins) | `scripts/extract_features_av.py:105-232,389-409` (`_spatial_pool` 256→16, `_vision_ts`, `VISION_TEMP=32`, staircase tbins via `data/av_cached_dataset.py::_ts_to_tdm_bins`) | `models/world_state_builder.py:109-186`, which **imports** the same functions from `scripts/extract_features_av.py` and `data/av_cached_dataset.py` rather than reimplementing them | **IDENTICAL** (by construction) | This file's own docstring documents 3 confirmed prior bugs (no spatial pooling; linspace-ramp tbins instead of staircase; ambient/WavJEPA absent) now closed IN THIS FILE — only callers that use `build_world_state_features` inherit the fix (next row). |
| — which call sites use the fixed builder | n/a | `models/m5_streaming_loop.py:386-398` and `scripts/jetson_real_demo.py:470` both call it correctly | **IDENTICAL for these two call sites** | UNVERIFIED whether `scripts/m5_bothpresent_extract.py`/`scripts/m5_ood_falsifier.py` (named in the docstring as historically-reimplementing scripts) were migrated — not re-audited. |
| zero-tensor-audio-input bug (CLAUDE.md "Live-pipeline defects") | n/a (test-harness bug, not training) | **FIXED in `scripts/jetson_real_demo.py` only**: `MicThread` replaces the old `torch.zeros(16000*10)` | **DIVERGENT across call sites** | `models/m5_streaming_loop.py`'s `RollingAudioBuffer` has no built-in guard against a zero/silent tensor, and no mic driver is wired to it anywhere in `models/` or `build_bmo_stack`. Fix is real but scoped to one demo script. |
| empty-candidate-category bug ("must raise, never `continue`") | n/a | **FIXED in `jetson_real_demo.py:452-457`** (`raise SystemExit`) — but `jetson_core_pipeline_test.py:206,209` (mtime 2026-08-16 00:50, touched the SAME day) still has bare `continue` and still defaults `--bank` to the pre-fix `candidates_siglip2.pt` | **DIVERGENT** | `jetson_core_pipeline_test.py` was touched the same day but not brought in line with either fix; would still silently under-report categories. |
| identity-match threshold | `models/jepa_memory.py:118-133` (`calibrate_threshold`) — meant to be fit to hit a target FAR (TAR@FAR1%=0.691/0.765 measured) | `jetson_real_demo.py:362`: `JepaMemory(MemoryConfig(threshold=0.5))` — hardcoded, `calibrate_threshold` never called | **DIVERGENT** | Inference uses an uncalibrated 0.5 cosine threshold, not the measured ~0.691–0.765 operating point. Direction of the effect on false-accept/reject not independently re-derived here. |
| homeostatic state feeding the LLMs | `models/homeostatic_state.py::HomeostaticState.update()` / `homeostatic_to_mood_state()` — meant to compute `{energy, mood}` from live signals every tick | Both concrete invocations found hardcode the dict instead: `bmo_jetson_startup.py:425` (`{"energy":0.5,"mood":"curious"}` in this checkout) and `jetson_real_demo.py:525,544` (`{"energy":0.6,"mood":"curious"}`) | **DIVERGENT** | Neither runtime path actually threads a live `HomeostaticState` through; `models/bmo_duplex_tick.py::BmoDuplexTick.tick()` IS wired to compute it from real inputs but has no confirmed caller anywhere (see B4). **Not addressed by Part F's diff** — still open. |

**ROW COUNTS — B3**: 6 rows.

---

## B4. ORPHANS

| checkpoint/script/module | what it was for | why unreachable from `build_bmo_stack` | result-dependent on it? | classification |
|---|---|---|---|---|
| `models/m3_connector.py::M3Connector` + `checkpoints/m3_multigran_richcaption_v2/last.pt` | Perceiver-style connector, old Thinker path | **As read in this checkout**, still `.to(device)`-loaded by `build_bmo_stack` and returned in the stack dict but never forward-passed anywhere reachable from Path A — **⟶ Part F: RESOLVED, live Jetson copy sets it to `None` entirely, matching CLAUDE.md's "DROPPED 2026-08-16."** | No | **Superseded**, and now actually removed at the source (per Part F), not just documented as removed |
| `models/audio_encoder.py` WavJEPA-nat load in `build_bmo_stack` | Second-tower ambient audio encoder | **As read in this checkout**, still constructed/quantized every boot despite CLAUDE.md documenting zero measured gain — **⟶ Part F: RESOLVED, live Jetson copy sets it to `None`.** | Cost `+701 MiB, +326 ms` per boot for zero effect, in the stale copy | **Was a dead load not yet removed from this file; now actually removed (Part F)** |
| `models/m4_duplex_loop.py::DuplexLoop` (full class) | Tick-grid orchestrator: perception refresh → decision head → interruptible generation | `build_bmo_stack` never constructs one; only `StreamingLoop.__init__` takes a `duplex_loop` arg, and `StreamingLoop` itself is only instantiated by test/demo scripts | `generate_interruptible`'s manual per-token KV-cache decode loop is real, tested, but has no confirmed live caller | **Never fully wired into a running production process** — not addressed by Part F |
| `models/bmo_duplex_tick.py::BmoDuplexTick` | Real integration of `HomeostaticState` + `CognitiveCoreRouter` + `AsyncThinker` into one tick | No caller found anywhere outside its own module | The homeostatic-state-driven mood conditioning this exists to provide is not exercised by either path | **Never-wired-in integration point** — not addressed by Part F |
| `models/m4d_stt_projector.py::AudioEncoderProjector` (Whisper Ultravox projector) | Speech→LLM-embedding fusion, deep-grounding path | Explicitly excluded per `bmo_jetson_startup.py:305-314`'s own comment | No | **Never-wired-in experiment**, superseded in intent by the Moonshine variant |
| `models/m4d_stt_projector_moonshine.py::MoonshineEncoderProjector` + `checkpoints/bmo_stt_projector_moonshine*` | Ultravox-style Moonshine→LLM projector (research track) | Not imported by `build_bmo_stack` or `m5_streaming_loop.py`; only training/eval scripts reference it | No — WER plateaus at 0.94 | **Never-wired-in experiment**, correctly documented as such |
| `checkpoints/bmo_lfm25_350m_v6_Q8_0.gguf`, `bmo_thinker_qwen3_v6_lora`, `_v7_lora` | Newer fast-tier/thinker fine-tunes | No script found that loads any of these three, in the local checkout OR the live Jetson copy (Part F diff confirms) | No | **Built-and-staged, never wired** (v6 fast-tier) / **experiment in progress** (thinker v6/v7, LoRA never merged) |
| `checkpoints/bmo_neutts_emotion_v2/_v3/_v4_Q8_0.gguf` | Iterative emotion-voice fine-tunes (v3 fixed "Beemo" mispronunciation + garble) | `build_bmo_stack` hardcodes the unversioned `bmo_neutts_emotion_Q8_0.gguf` filename (== v1), never `_v3`/`_v4`, **confirmed still true in the live Jetson copy (Part F)** | Documented-as-fixed checkpoints exist and are unused | **Superseded-but-not-repointed** — this one survives Part F's sync unresolved |
| `models/glr_transition_head.py::TransitionHead` + `checkpoints/glr_thinker_v1/best.pt`, `glr_thinker_v2/best.pt` | Geometric Latent Reasoning head to shorten thinker CoT | No caller wires it into `GGUFReasoningTier`/`build_bmo_stack` — only `train_glr_thinker.py`/`eval_glr_thinker.py`/`thinker_behaviour_gate.py` reference it | No — v1 diverges catastrophically at K=10; v2 fixes it but still has no production caller | **Never-wired-in experiment**; v2 (a real improvement) is not yet reflected in CLAUDE.md's own text and explicitly sequenced *after* the speaker's directive-conditioning fix |
| `models/m5_speculative.py::SpeculativePrefetcher` | Tier-1 speculative prefetch on partial transcripts | Only caller is `scripts/bench_tier1_speculative.py` | No | **Never-wired-in experiment**, matches CLAUDE.md's own "NOT yet wired into `m5_streaming_loop`" statement |
| `models/perception_prefix.py::PerceptionPrefix` + `checkpoints/perception_prefix_thinker/best.pt` | Generative (embedding-prefix) alternative to retrieval-based perception query | No caller outside `train_perception_prefix.py`/`scripts/eval_generation_vs_retrieval.py` | No | **Never-wired-in experiment** — evaluated offline, not integrated |

**ROW COUNTS — B4**: 11 rows.

---

## B5. STUBS / SIMULATION

| file:line | what is faked | what a real implementation would require |
|---|---|---|
| `models/m5_streaming_loop.py:445-447` (`_estimate_tts_duration_sec`), used at line 552 | Playback-duration estimate: `n_words / 150.0 * 60.0` (words-per-minute), used whenever `self.tts_engine is None` | A resident TTS engine at that call site — `StreamingLoop` does support a real one (line 546-548); this stub only fires when the caller wires none |
| `models/m5_tools.py:121-124` (`_h_timer`) | Formats `"a timer for {duration}"` as a string; **sets no actual timer** | A real scheduler/alarm mechanism |
| `models/m5_tools.py:125-128` (`_h_reminder`) | Formats a reminder string; **stores nothing, never fires later** | Persistent storage keyed by time + a polling/notification mechanism |
| `models/m5_tools.py:83-119` (`_h_weather`) | **CLAUDE.md characterizes this as a "clean stub"; it is not** — makes a real OpenWeatherMap HTTP call when a key file exists, returns `None` (never fabricates) otherwise | If no key file present, degrades to a true stub; code path itself is real |
| `models/m5_tools.py:297-324` (`_h_search`, DuckDuckGo) / `:245-294` (`_h_facts`, Wikipedia) | **Also mischaracterized as stubs by CLAUDE.md** — both make real HTTP calls with real result-clipping/relevance guards | n/a — already real |
| `models/homeostatic_state.py:56-70` (`HomeostaticParams`) | Every decay/rise-rate constant is an unvalidated guess, explicitly flagged as such in its own docstring ("no real deployment data exists yet to calibrate against") | Real deployment telemetry to calibrate against |
| `scripts/jetson_real_demo.py:263,273-277` (`--tts-gguf`/`--with-tts` defaults) | The "real end-to-end" demo's own docstring claims it loads "EVERY component including TTS and STT"; both flags default OFF and the shown 6-round usage never enables them | Running with `--with-tts --with-stt` explicitly (the script's own comment notes memory headroom cost) |
| `models/m5_perception_query.py:164-182`/`:242` | `PerceptionQueryEngine.__init__`'s documented safe default `min_score=-1.0` ("never refuse") vs `load_perception_query_engine`'s own separate default `min_score=0.0` | Not a crash — a silent behavioral divergence between two entry points into the same class |
| `checkpoints/glr_thinker_v1/eval.json` at K=10 | Latent rollout at K=10 with the v1 head is a documented, measured runaway (`mean_latent_norm=423`, `eos_rate=0.375` — 62.5% never emit EOS) | Fixed in v2 (`eos_rate` 0.99–1.0 through K=15-20), but v2 has no production caller (B4) |

**ROW COUNTS — B5**: 9 rows.

---

# PART C — SigLIP2 INTEGRATION AUDIT

## C1. What was added

- **Model**: `google/siglip2-base-patch16-224` (confirmed via `ARCHITECTURE.md` §6 "Measured
  tower split" table). A larger `siglip2-large-patch16-256` variant was benchmarked and
  **rejected** (`SESSION_LEDGER_2026-08.md` §9.9 — base wins on latency, size, *and* accuracy on
  the test room).
- **Input resolution**: 224×224 (base variant); vision tower only is deployed (text tower
  dropped, see C2).
- **Output dimension**: 768-d in the deployed config (`sig_runD_proj768`) — a 1536-d variant
  (`sig_runC_proj1536`) was also trained and **lost** on a like-for-like comparison (within-clip
  0.811 vs 0.747, R@1 0.737 vs 0.739 — "same R@1, much better within-clip, and halves the bank").
- **Where it enters the architecture**: `models/text_target.py::SigLIP2TextTarget` (frozen
  SigLIP2 base + optional trainable `proj`) on the text/target side, and a vision-tower-only
  forward feeding the query predictor as a 4th named stream, `token_sources = m2,vision,ambient,
  scene` (`ARCHITECTURE.md` §1b).
- **Frozen?** Vision and text SigLIP2 towers: **frozen**. A separate, small **trainable linear
  projection** (`proj`, 768 or 1536-d) sits on top of the frozen text tower — this trainable
  proj, not the frozen SigLIP2 space itself, is where the measured accuracy gain comes from (see
  C3, and Table 5's "SigLIP2 as a frozen target space" negative result).
- **Parameter count**: vision tower 92.9M/177 MiB fp16; text tower 282.3M/538 MiB fp16 (not
  shipped on-device); total 375.2M/716 MiB (`ARCHITECTURE.md` §6, stated twice consistently).

## C2. How it combines with V-JEPA2/WavJEPA

**Not concatenation into a single vector, and not a replacement.** It is a 4th **named stream**
fed into the existing query predictor's multi-source interface (`models/query_predictor.py`,
`source_dims` config — described in `ARCHITECTURE.md` §6 as "already accepts arbitrary named
streams... adding SigLIP2 is a config entry plus a retrain, not new architecture"). The exact
internal fusion mechanism inside `QueryPredictor` (attention weights, gating, or concatenation-
then-MLP) was **not independently re-read at the code level in this pass — UNVERIFIED** beyond
this prose description. V-JEPA2 and WavJEPA are **not modified or replaced**:
`world_state_builder.py` still produces the `m2`/`vision`/`ambient` streams exactly as before;
SigLIP2 output is appended as a peer stream, confirmed by the ablation results treating "scene"
as an independent arm that can be added/removed (`abl_A` through `abl_D` logs).

## C3. THE 70% NUMBER

The figure requested (">70% R@1") is **`checkpoints/sig_runD_proj768/best.pt`'s VGGSound R@1 =
0.737** (73.7%), from the head-to-head ablation logged in `sig_runD_proj768/`
(`sig_runD.log`) and summarized in `ARCHITECTURE.md` §6 and `JEPA_MEMORY_PLAN.md` §A.3. A
second, related number — **caption-retrieval R@1 = 0.705** via the trained predictor — appears
in the same run's candidate-set eval (`checkpoints/CANDIDATE_SET_EVAL_runD.json`).

- **Gallery/corpus**: 518,347–518,461-clip pool (VGGSound + Action100M combined, "the same 518k
  pool" across all four SigLIP2 runs), batch 1024 (4×DDP+GradCache), negatives 2048
  (VGGSound)/6144 (Action100M) — **NOT the 1545-clip VGGSound gallery used for v1's 52%/53.27%
  M2-only figures.** The candidate-set eval used 512 held-out clips as its query set against the
  bank.
- **`clips_seen` assertion**: UNVERIFIED — the eval script (`scripts/eval_candidate_sets.py` or
  the training script's own logging) was not read at the code level to confirm this in this
  pass; only prose-reported pool sizes were available.
- **Direction(s)**: the 0.737 figure is a single VGGSound R@1 number (not explicitly split
  v→a/a→v in the source table as read); Action100M R@1 is reported separately (0.085 for run D)
  — much lower, not averaged into the headline.
- **Checkpoint**: `checkpoints/sig_runD_proj768/best.pt` — **confirmed by Part F to also exist,
  byte-identically, on the live Jetson at `~/bmo_production/pipeline/checkpoints/qp_runD.pt`,
  and confirmed by Part F's diff to be the file the live production `build_bmo_stack()` actually
  loads (`query_predictor_ckpt=".../qp_runD.pt"`).**
- **Training corpus**: 518k-clip pool = VGGSound + Action100M, `--restrict-to-scene` filter,
  with a documented corpus-coverage bug (Action100M scene coverage was 80,000/399,934 before a
  2026-08-15 fix extracted the missing 319,934 segments).
- **Is it the same 1545-clip VGGSound gallery as the 52%/53.27% figures? No — explicitly
  different.** The 52%/53.27% M2 figures used a 1545-clip fixed retrieval gallery on the
  *original M2 predictor's own* target space (EmbeddingGemma-based cached captions). The
  SigLIP2/query-predictor 0.737 figure is measured via a **518k-clip training pool with in-
  batch/GradCache negatives (2048/6144)**, not a fixed gallery, in a **different, SigLIP2-
  projected target space**, on a **different underlying task** (query-conditioned retrieval vs
  unconditioned AV retrieval). **These numbers are NOT directly comparable.** The project's own
  fair-comparison reference point is the **EmbeddingGemma query predictor at R@1 0.681**
  (`query_predictor_ddp_lw0.3`, same 518k pool, same protocol as SigLIP2 runs A-D) — against
  *that* baseline, SigLIP2's 0.737 is a genuine, protocol-matched **+8.2% relative gain**.

## C4. Swap/shuffle control on the SigLIP2-augmented model

**RUN**, two distinct falsifiers, both on `sig_runD_proj768`:

1. **Swapped-query control**: swapped-query result 0.006 against a 0.167 chance level — the
   model genuinely reads the question, not just the presence of a candidate set.
2. **Audio-swap AV-congruence control** (`scripts/eval_av_congruence.py`, re-run "in the new
   space"): run D reaches follows-EARS 0.608 with matched-control 0.967 (vs runs A/B at frozen-
   target matched-control ≈0.50, i.e. chance for a 2-way choice) —
   `checkpoints/AV_CONGRUENCE_runD.json`.

Both are real, measured, on the deployed checkpoint — not "NOT RUN."

---

# PART D — RESULT TABLES (rebuilt complete, supersedes v1)

Base: v1 (`docs/EVIDENCE_LEDGER.md`) Tables 1–6, carried forward in full (rows unchanged unless
flagged), plus every new gated result found in files created/modified after 2026-08-11
(`falsifier_tracking.md`, `RESULTS_TABLE.md`, `NEGATIVE_RESULTS.md` — all three **unchanged**
since 2026-08-01/04, confirmed via `stat`; all new content lives in `JEPA_MEMORY_PLAN.md`
[updated 2026-08-15], `SESSION_LEDGER_2026-08.md` [2026-08-08→2026-08-16], `ARCHITECTURE.md`,
and CLAUDE.md).

## TABLE 1 — Every gated result

### 1.A — Carried forward verbatim from v1 (rows unchanged; date range through 2026-08-08)

*175 rows, identical to `docs/EVIDENCE_LEDGER.md` Table 1 in full — not re-typed here to avoid
a ~175-row duplicate; see v1 §Table 1 for the complete verbatim listing (M0 through BMO-prod
2026-08-08 rows).* Two supersession notes apply retroactively to that table, both cross-
referenced from 1.B below: the M5 "LOCKED checkpoints... 6781MiB/839MiB" row is superseded by
the 2026-08-16 production figure; the "BMO prod full-stack e2e latency... 2.19s" row is
superseded by the 2026-08-16 per-leg breakdown (thinker now dominant, not TTS/perception).

### 1.B — NEW rows, 2026-08-11 through 2026-08-16 (supersede 1.A rows as flagged)

| milestone | experiment name | metric | value | n | device | date | source file path | measured/projected | superseded by |
|---|---|---|---|---|---|---|---|---|---|
| JEPA-mem/perception | Thinker↔perception bank, 6k captions/1k clips | correct CLIP/chance/correct FIELD/swapped/both/latency | 0.442/0.00100/0.946/0.000/0.421/17.2ms | bank=6,000 | — | 2026-08-14 | checkpoints/PERCEPTION_QUERY_E2E.json | measured | superseded by 48k-bank row (bank-size sensitivity) |
| JEPA-mem/perception | same, 48,000 captions/8,000 clips | correct CLIP/chance/correct FIELD/swapped/both/latency | 0.189/0.00013/0.953/0.003/0.181/17.7ms | bank=48,000 | — | 2026-08-14 | checkpoints/PERCEPTION_QUERY_E2E_bank48k.json | measured | honest deployment number — any future claim must state bank size |
| JEPA-mem/perception | per-query-type field acc @6k bank | action-brief/detailed, summary-brief/detailed, sound-brief/detailed | 0.900/0.975/0.925/0.975/0.950/0.950 | bank=6,000 | — | 2026-08-14 | JEPA_MEMORY_PLAN.md | measured | — |
| JEPA-mem/perception | Jetson on-device memory walk | EmbeddingGemma / +predictor+bank | +710MiB (3238MiB used) / +145MiB (3383MiB used, 4237MiB avail) | — | Jetson | 2026-08-14 | jetson_perception_query_results.json | measured | superseded 2026-08-15 (SigLIP2 space drops EmbeddingGemma) |
| JEPA-mem/perception | Jetson on-device latency | query encode/predictor fwd/bank lookup(24k)/total | 263.5/138.5/2.6/403.8 ms | — | Jetson | 2026-08-14 | jetson_perception_query_results.json | measured | bottleneck = question encode, not bank search |
| JEPA-mem/perception | describe-demo, 7W vs MAXN_SUPER power mode | perception/M2-prepool/query/fast-LLM/total-round | 3828-12368→1089-1732ms; 40-78→24-39ms; 1033-1641→240-553ms; 1552-7274→364-504ms; 14971-18401→3892-4994ms | — | Jetson | 2026-08-14 | SESSION_LEDGER_2026-08.md §9.2 | measured | ~7x perception/~4x total — 7W-mode figures NOT comparable to any other measurement in this ledger |
| JEPA-mem/perception | max bank scale-up | unique captions/size | 121,104/355MiB fp16 | dropped 20,970 dupes | — | 2026-08-14 | checkpoints/perception_bank_max_fp16.pt | measured | superseded by SigLIP2 tag/caption sets |
| JEPA-mem/perception | int8-via-torchao on Jetson | encoder size before/after | 577.7→577.7 MiB (NO-OP) | — | Jetson | 2026-08-14 | SESSION_LEDGER_2026-08.md §9.6 | measured | torch 2.8.0 < torchao's required 2.11.0 |
| JEPA-mem/perception | full stack + thinker + max bank | fits? | NO — OOM at bank load (571MiB avail before bank) | — | Jetson | 2026-08-14 | SESSION_LEDGER_2026-08.md §9.6 | measured | thinker alone costs 1103MiB |
| JEPA-mem/perception | EmbeddingGemma quantization sweep | config: field-acc/cos/latency | bf16 0.939/—/14.8ms; int8-linears 0.939/0.9998/18.1ms; int8-dyn-act 0.933/—/163ms(11×slower); int8-embed-table **0.272/0.31-0.58 (BREAKS)** | — | mercury | 2026-08-14 | checkpoints/EMBEDDINGGEMMA_QUANT_EVAL.json | measured | int8-linears adopted; embed-table quant ruled out |
| JEPA-mem/perception | 4-way stream ablation (EmbeddingGemma geometry) | cross-clip R@1: A/B/C/D | 0.441/0.564/0.566/0.546 | 3000 steps/arm | 1×Blackwell/arm | 2026-08-14 | abl_A/B/C/D_*.log | measured | SigLIP2 scene stream = +28% relative (A→B); **conflicts with v1's differently-numbered "unified architecture ablation" row (2026-08-11) — see Part E, not reconciled** |
| JEPA-mem/perception | AV congruence eval (EmbeddingGemma geometry) | follows-EARS: A/B/C/D | 0.650/0.609/0.562/0.070 | — | — | 2026-08-14 | checkpoints/AV_CONGRUENCE_EVAL.json | measured | RETRACTS same-day "one ear is enough" reading — nat costs 4.7pp of audio-following, invisible on R@1 |
| JEPA-mem/perception | matched control, AV congruence (EmbeddingGemma) | A/B/C/D | 0.956/0.958/0.953/0.923 | — | — | 2026-08-14 | checkpoints/AV_CONGRUENCE_EVAL.json | measured | — |
| SigLIP2 | tower split, siglip2-base-patch16-224 fp16 | vision params/size, text params/size | 92.9M/177MiB, 282.3M/538MiB | — | — | 2026-08-15 | JEPA_MEMORY_PLAN.md §A.1 | measured | text tower drove pre-encode-only design |
| SigLIP2 | head-to-head, 518k pool, batch 1024, neg 2048/6144, λ_within 0.3 (reference) | VGG within/R@1/A100M R@1 | 0.883/0.681/0.090 | — | — | 2026-08-15 | JEPA_MEMORY_PLAN.md §A.3 | measured | reference for SigLIP2 comparison |
| SigLIP2 | sig_runA_matched3stream (frozen Identity, 3 streams) | VGG within/R@1/A100M | 0.654/0.489/0.051 | — | — | 2026-08-15 | sig_runA.log | measured | NEGATIVE RESULT — frozen target space; flat from step 249 |
| SigLIP2 | sig_runB_scene4stream (frozen Identity, 4 streams) | VGG within/R@1/A100M | 0.688/0.627/0.069 | — | — | 2026-08-15 | sig_runB.log | measured | superseded by proj-768 |
| SigLIP2 | sig_runC_proj1536 (trained proj 1536) | VGG within/R@1/A100M | 0.747/**0.739**/0.091 | — | — | 2026-08-15 | sig_runC.log | measured | beaten on within-clip and bank size by proj-768 |
| SigLIP2 | **sig_runD_proj768 (trained proj 768) — DEPLOY** | VGG within/R@1/A100M | **0.811/0.737/0.085** | — | — | 2026-08-15 | sig_runD.log; checkpoints/sig_runD_proj768/best.pt | measured | current best/recommended config; **R@1=0.737 is the "~70% R@1" task figure — NOT comparable to the 1545-clip 52%/53.27% gallery, see Part C; confirmed by Part F to be the checkpoint the live production stack actually loads** |
| SigLIP2 | text-space ablation, 400 held-out × 6 fields | within/cross-clip cos: SigLIP2-raw / EmbeddingGemma-raw / EmbeddingGemma+proj | 0.7556/0.6648; 0.7306/0.6075; **0.4340/0.1533** | n=400×6 | — | 2026-08-15 | JEPA_MEMORY_PLAN.md §A.3 | measured | root-causes why frozen SigLIP2 failed — the trainable projection is where representation learning happens |
| SigLIP2 | candidate sets, run D, 512 held-out | caption R@1 (predictor/zero-shot); tag p@5 (predictor/zero-shot) | 0.705/0.619; 0.418(shuffled 0.021)/0.387(shuffled 0.048) | n=512 | — | 2026-08-15 | checkpoints/CANDIDATE_SET_EVAL_runD.json | measured | predictor beats zero-shot SigLIP2 alone |
| SigLIP2 | AV congruence, new (SigLIP2) space | follows-EARS/matched-control: runD/runB(frozen)/runA(frozen) | 0.608/0.967; 0.502/0.502; 0.467/0.541 | — | — | 2026-08-15 | checkpoints/AV_CONGRUENCE_runD.json | measured | SUPERSEDES 2026-08-14 finding; "dropping nat now costs nothing measurable" |
| SigLIP2 | Jetson fit test, no-nat vs with-nat | avail after full load | 1,410MiB (no-nat)/935MiB (with-nat) | — | Jetson | 2026-08-15 | ARCHITECTURE.md §8 | measured | supersedes earlier OOM-at-659MiB config |
| SigLIP2 | Jetson steady-state latency, no-nat | capture/perception/SigLIP2/query+retrieval/thinker/speaker/**total** | 905/792/104/33/817/212/**2,886ms** | — | Jetson | 2026-08-15 | ARCHITECTURE.md §8 | measured | superseded same-day by capture fix |
| SigLIP2 | capture-loop fix | capture latency/e2e total | 905ms→1-9ms; 2,886→**1,795ms** (−38%) | — | Jetson | 2026-08-15 | ARCHITECTURE.md §9 | measured | root cause: benchmark's own `time.sleep(0.05)`×16, not the camera |
| SigLIP2 | nat forward, confirmed on-device | memory/latency delta | +701MiB/+326ms (792→1,118) | — | Jetson | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.8 | measured | supersedes the 469ms figure used through 2026-08-14 |
| Identity | identity head, on-Jetson fit | memory/latency | +96MiB/4-9ms per query | — | Jetson | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.10 | measured | reuses existing streams; **⟶ Part F: RESOLVED as wired into `build_bmo_stack` in the live copy, resolving the "verify in Part B" caveat this row originally carried** |
| BMO prod | speaker v3 deployed, val_loss | val_loss | 0.7093 | corpus v10c, 3,641 rows | mercury | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.11 | measured | deployed and verified on-device 2026-08-15; superseded in deployment by v5 (below) |
| BMO prod | memory recovery (AutoProcessor + flash_attn removal) | avail-at-end-of-run | 266→**710MiB** | — | Jetson | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.11 | measured | +444MiB, none from quantizing a model |
| SigLIP2/identity | perception generalization — appearance tags added | tag count/size | 1,372→**1,482** tags/2.17MiB | — | — | 2026-08-15 | checkpoints/candidates_siglip2_v2.pt | measured | **⟶ Part F: RESOLVED, live production now loads `_v2`, closing the D4 defect noted in the next row** |
| Cognitive core | thinker v4 | best_val_loss | 2.0811 (vs v3's 1.95) | corpus 324 rows | mercury | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.12 | measured | **REJECTED, not deployed** — 54% Adventure-Time contamination |
| Cognitive core | name_stranger open-set filter attempt | rejects vs real catches | 93 false rejects vs 9 real catches | ~126 candidates | — | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.12 | measured | NEGATIVE RESULT — closed-set works, open-set regex doesn't |
| Cognitive core | name_stranger accepted vs rejected BMO-idiom rate | idiom % | accepted 39.1% vs (sample) rejects 100% | reject sample n=6/105 | — | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.12 | measured | filter was removing the personality it exists to preserve |
| Cognitive core | speaker v4 | best_val_loss | 0.7286 (epoch 3) | corpus v10d, 3,703 rows | mercury | 2026-08-15 | SESSION_LEDGER_2026-08.md §10.12 | measured | worse than v3; NOT deployed |
| BMO prod | D1: enable_thinking fix | `.reasoning` populated/latency cost | None→populated; 650ms→1,749–3,509ms (median ~2.4s) | 6 live rounds | Jetson | 2026-08-16 | CLAUDE.md | measured | root cause of "reasoning tier redundant with fast tier" |
| BMO prod | D3: audio branch fed silence, fix | before/after | `torch.zeros(16000*10)` → real `MicThread` 10s ring buffer | 6 live rounds | Jetson | 2026-08-16 | CLAUDE.md | measured | `hearing` query fully trained but never exercised until this fix |
| BMO prod | D4: empty candidate category (`wearing`) silent failure | before/after | `continue` → `SystemExit`; deployed set upgraded to v2 | — | Jetson | 2026-08-16 | CLAUDE.md | measured | **⟶ Part F: fix confirmed present in the live production copy** |
| BMO prod | camera rotate, wrong vs corrected | confidence, wrong vs right answer | `--rotate 90`: "a person lying down" (+0.71, WRONG, highest-confidence) vs `--rotate 180`: correct | — | Jetson | 2026-08-16 | SESSION_LEDGER_2026-08.md "CAMERA" | measured | data-orientation bug, not a model defect |
| BMO prod | mic per-channel RMS, ReSpeaker 4-mic | ch0..ch5 rms | 0.00448/0.00229/0.00342/0.00359/0.00241/0.00000 | 4s room tone | Jetson | 2026-08-16 | SESSION_LEDGER_2026-08.md "AUDIO" | measured | falsifies XMOS-beamforming hypothesis; ch5 dead loopback |
| BMO prod | spectral subtraction vs raw, `hearing` percept | top answer + score | raw: "a fan humming" #2; denoised: "an alarm beeping" 0.451→**0.478** (WORSE) | — | Jetson | 2026-08-16 | SESSION_LEDGER_2026-08.md "AUDIO" | measured | denoise=False adopted default |
| BMO prod | fan blade-pass frequency, isolated by differencing | measured vs predicted (9 blades × 5403rpm/60) | 808.6/812.5Hz vs 810.4Hz predicted | quiet 1915rpm vs cool 5403rpm | Jetson | 2026-08-16 (later) | scripts/measure_fan_signature.py | measured | 1.9Hz error; corrects earlier "fan is broadband" conclusion |
| BMO prod | fan notch filter (`models/m5_fan_notch.py`) | attenuation/speech impact/latency | 33.4dB / 0.0dB @220Hz / 1.6ms per 10s buffer (sosfiltfilt) | — | Jetson | 2026-08-16 (later) | SESSION_LEDGER_2026-08.md | measured | pure-numpy fallback = 374ms (57% of perception leg); effect on actual `hearing` percept UNPROVEN |
| GLR | glr_thinker_v1 (arXiv:2606.02248 impl.) | params/val_loss/ce/delta (epoch 4) | 1,049,600/3.5002/1.9951/1505.1 | 1,162 rows | mercury | 2026-08-16 | checkpoints/glr_thinker_v1/best.pt | measured | FAILED rollout eval |
| GLR | glr_thinker_v1 rollout eval | K=0 vs K=10 tokens/eos/f1/‖latent‖ | 83/1.00/0.234/— vs **330/0.38/0.090/423** | 40 held-out | Jetson/mercury | 2026-08-16 | scripts/eval_glr_thinker.py output | measured | FAIL — diverging rollout; non-zero head init + unnormalized λ, head output ~31× too large |
| GLR | glr_thinker_v2 (zero-init, normalized L_Δ, λ=1.0) | val_loss/ce/delta | 2.6128/1.9136/0.6992 | 6 epochs | mercury | 2026-08-16 | checkpoints/glr_thinker_v2/best.pt | measured | delta below 1.0 "predicts nothing" floor — genuinely learned |
| GLR | glr_thinker_v2 rollout eval, n=200 (final) | K0/K5/K10/K15 tokens (vs K0)/eos/f1(vs K0) | 89.5/—/1.00/0.1777; 70.0/0.78×/1.00/0.1860; **53.0/0.59×/0.99/0.1661**; 87.5/0.98×/0.96/0.1744 | n=200 | Jetson/mercury | 2026-08-16 | scripts/eval_glr_thinker.py output | measured | K=10 chosen: 41% fewer tokens, EOS 0.99, real payoff 1.47×, not paper's 5-7× |
| GLR | behavioural gate, thinker v5 + GLR K=0/5/10 | 5 criteria × K | respects_focus **FAIL at every K**; asks_when_unsure FAIL→PASS(K5,K10) | 5 cases × 3 K, n=1 greedy | Jetson | 2026-08-16 (later) | scripts/thinker_behaviour_gate.py output | measured | deployed v5 thinker fails 2/5 with GLR uninvolved; GLR "fix" on n=1 is weak evidence |
| Cognitive core | speaker intent bake-off, v1–v5, LONG/SHORT | pass rate /6 | v1: 1/6,2/6; v2: 4/6,3/6; v3: **5/6**,4/6; v5: 4/6,**5/6** | n=6 | Jetson | 2026-08-16 (later) | scripts/speaker_intent_bakeoff.py output | measured | v1 "confidently irrelevant" not actually better than v5; v3-vs-v5 diff is noise at n=6 |
| Cognitive core | thinker v7 corpus vs v6c | scenarios/rows/rows-per-scenario | v6c 170/1489/8.8; v7 170/**1017/6.0** | — | mercury | 2026-08-16 (later) | data/bmo_thinker_corpus_v7.jsonl | measured | v7 does NOT meet its own diversity goal — 32% less data, not an overfitting fix |
| Cognitive core | thinker corpus cleaning | rows in→out | v6c 1489→1461 (28 contradictions dropped, 27 "Alice" renamed); v7 1017→999 (18/18) | — | mercury | 2026-08-16 (later) | scripts/clean_thinker_corpus.py output | measured | 60-62% of restraint-scenario rows contradicted their own instruction |
| Cognitive core | directive-corpus generation, 3 attempts | rows/defect rate | attempt1: 82/395 lost (48% JSON parse fail); attempt2: 197/395 (50%) actively harmful; attempt3 (final): 4,144 rows, clean | — | mercury | 2026-08-16 (later) | data/bmo_companion_corpus_v12.jsonl | measured | count-based gates passed all three; only reading rows caught attempt 2's harm |
| Cognitive core | speaker v6 vs v5 bake-off | pass rate /12 | v5: 9/12; v6: 8/12 | n=6×2 | Jetson | 2026-08-16 (later) | SESSION_LEDGER_2026-08.md "SPEAKER v6" | measured | v6 val_loss NOT comparable to v5 (different corpora/val-sets); **v6 NOT deployed, v5 remains production** |
| BMO prod | `{name}` placeholder leak | affected rows | 54 rows across v10/v10c/v10d/v10e, 0/54 had a name anywhere BMO substituted | — | — | 2026-08-16 (later) | scripts/fix_name_placeholders.py output | measured | same defect class as the "Alice" hallucination incident — **caveat: script's own docstring separately says 13/54 had a name somewhere in the prompt, see Part E E3** |
| BMO prod | "Alice" hardcoded example contamination | affected rows | 27 prompts/24 answers in thinker corpus | — | — | 2026-08-16 (later) | SESSION_LEDGER_2026-08.md "ALICE" | measured | traced to 3 hardcoded generator scenarios |
| BMO prod | production deploy, full stack incl. TTS (2026-08-16) | resident memory/avail | before-boot 4898MiB avail; full-stack-resident 3923MiB used/**975MiB avail**; +camera 741MiB avail | — | Jetson | 2026-08-16 (later) | CLAUDE.md "PRODUCTION DEPLOY" | measured (**per this repo's checkout, no local artifact — see Part E E1; ⟶ Part F: qualitatively confirmed real by the live source-code diff, though this exact number was not independently re-measured by this extraction**) | supersedes the 2026-08-01 6781/839MiB figure |
| BMO prod | boot time/VOICE TTFA, post-OOM-fix | value | boot 46s (no TTS); VOICE TTFA 997ms (full stack) | — | Jetson | 2026-08-16 (later) | CLAUDE.md | measured (no local artifact) | supersedes 625ms figure from 2026-08-07 |
| BMO prod | TTS crash root cause + fix | fix | `onnxruntime==1.19.2` (1.23.2 aborts on this CPU's cpuid) + `set_power_mode("MAXN_SUPER")` (7W mode leaves 4/6 cores online, ORT pins to an offline core) | — | Jetson | 2026-08-16 (later) | CLAUDE.md | measured (no local artifact for the exact ms figure) | SIGABRT, not catchable by the existing 5×-retry wrapper |
| BMO prod | live-pipeline final full-stack memory/latency, TTS+STT off | total used/free; steady-state free | 4,361MiB used/577MiB free; 115–176MiB free | — | Jetson | 2026-08-16 | SESSION_LEDGER_2026-08.md "MEMORY + LATENCY" | measured (no local artifact) | with TTS+STT resident: 280MiB free, camera NVMM allocation FAILS (unresolved as of this ledger) |
| BMO prod | per-leg latency, live end-to-end (thinking ON) | capture/perception/M2/SigLIP2/query/**thinker**/speaker | 3-9/650-1400/24-51/85-184/23-218/**1,749-3,509**/157-583 ms | 8 rounds | Jetson | 2026-08-16 | SESSION_LEDGER_2026-08.md | measured (no local artifact) | thinker is now the dominant leg, post D1 fix |

**Row counts, Table 1**: 1.A = 175 rows (carried forward verbatim). 1.B = 47 rows (new).
**Total Table 1 = 222 rows.**

---

## TABLE 2 — Checkpoints

*48 checkpoints carried forward from v1 verbatim — see v1 §Table 2. New/changed below.*

| name | path | what it is | training corpus + size | key hyperparameters | eval scores | locked/frozen? | superseded by |
|---|---|---|---|---|---|---|---|
| sig_runA_matched3stream | checkpoints/sig_runA_matched3stream/ | SigLIP2 frozen (Identity) target, 3-stream | 518k pool | batch 1024, neg 2048/6144, λ_within 0.3 | VGG within 0.654, R@1 0.489 | not frozen | NEGATIVE RESULT; superseded by proj variants |
| sig_runB_scene4stream | checkpoints/sig_runB_scene4stream/ | SigLIP2 frozen (Identity), 4-stream (+scene) | 518k pool | same | VGG within 0.688, R@1 0.627 | not frozen | superseded by sig_runC/D |
| sig_runC_proj1536 | checkpoints/sig_runC_proj1536/ | SigLIP2 + trained proj (1536), 4-stream | 518k pool | same | VGG within 0.747, R@1 **0.739** | not frozen | superseded by sig_runD |
| sig_runD_proj768 | checkpoints/sig_runD_proj768/best.pt | **SigLIP2 + trained proj (768), 4-stream — deployment-recommended** | 518k pool | proj_dim=768 | VGG within **0.811**, R@1 0.737, A100M 0.085 | not formally frozen; deployment-recommended | **⟶ Part F: CONFIRMED — this is `qp_runD.pt` on the live Jetson, byte-identical, and IS the checkpoint `build_bmo_stack()` actually loads** |
| bank_runD_fp16.pt | checkpoints/bank_runD_fp16.pt | pre-encoded caption bank, proj applied, SigLIP2+proj768 space | — | fp16, 177MiB | — | current for sig_runD | — |
| bank_siglip2_fp16.pt | checkpoints/bank_siglip2_fp16.pt | pre-encoded bank, raw SigLIP2 space (pre-proj) | 1,361,635 unique captions | fp16, 237,895,258 bytes | — | superseded by proj-space banks | — |
| bank_armB_fp16.pt / bank_armC_fp16.pt | checkpoints/ | banks for the (abandoned/stopped) restrictedpool sig_arm{B,C} runs | ~20% Action100M coverage (later found buggy — see Table 5) | — | — | not used in production | runs stopped mid-training |
| candidates_siglip2.pt | checkpoints/candidates_siglip2.pt | tag candidate set v1 — 1,186 mined, **0 appearance tags** | — | — | — | superseded by v2 | **⟶ Part F: RESOLVED — production now points at v2, not this file** |
| candidates_siglip2_v2.pt | checkpoints/candidates_siglip2_v2.pt | tag candidate set v2 — 1,482 tags incl. 110 appearance (2.17 MiB) | — | — | — | fixes D4 empty-category silent failure | **⟶ Part F: CONFIRMED current recommended set, and confirmed wired in the live production copy** |
| query_vectors_siglip2.pt / _v2.pt | checkpoints/ | pre-encoded query phrasings matching the two candidate sets | 30 phrasings | — | — | v2 matches candidates_v2 | — |
| perception_bank_max_fp16.pt | checkpoints/perception_bank_max_fp16.pt | 121,104-caption bank (VGGSound×6 + Action100M×2 held-out) | — | fp16, 355MiB | — | superseded by SigLIP2 tag-based approach (2.17MiB) | — |
| perception_bank_vgg_fp16.pt | checkpoints/perception_bank_vgg_fp16.pt | earlier/smaller bank variant | — | — | — | superseded by perception_bank_max_fp16.pt | — |
| bmo_lfm25_350m_v3_Q8_0.gguf | checkpoints/bmo_lfm25_350m_v3_Q8_0.gguf | fast-tier v3, corpus v10c (3,641 rows, de-cartooned) | v10c | LoRA | val_loss 0.7093 | deployed 2026-08-15, then held as bake-off baseline | superseded in deployment status by v5; still best on LONG bake-off (5/6) |
| bmo_lfm25_350m_v4_lora | checkpoints/bmo_lfm25_350m_v4_lora/ | fast-tier v4 (intermediate) | — | — | — | not frozen | superseded by v5 |
| bmo_lfm25_350m_v5_Q8_0.gguf | checkpoints/bmo_lfm25_350m_v5_Q8_0.gguf | **fast-tier LLM v5 — DEPLOYED** | — | — | val_loss 0.7542 (not comparable to v6's 0.6924) | **DEPLOYED** | **⟶ Part F: CONFIRMED — this is the file the live production `build_bmo_stack()` loads** |
| bmo_lfm25_350m_v6_Q8_0.gguf | checkpoints/bmo_lfm25_350m_v6_Q8_0.gguf | fast-tier v6, corpus v12 (4,144 rows/372 directive) | v12 | LoRA | val_loss 0.6924 (not comparable to v5) | trained, **NOT deployed** — bake-off 8/12 vs v5's 9/12 (likely noise at n=6) | — |
| bmo_thinker_qwen3_v4_Q8_0.gguf | checkpoints/ | reasoning-tier v4 | 324-row corpus | LoRA | best_val_loss 2.0811 | **REJECTED**, 54% Adventure-Time contamination | — |
| bmo_thinker_qwen3_v5_Q8_0.gguf | checkpoints/bmo_thinker_qwen3_v5_Q8_0.gguf | reasoning-tier v5 | — | LoRA, `enable_thinking=True` fix | 3/5 behavioural-gate cases at K=0 | **DEPLOYED** | **⟶ Part F: CONFIRMED — this is the file the live production `build_bmo_stack()` loads** |
| bmo_thinker_qwen3_v6/_v7 (LoRA only) | checkpoints/ | reasoning-tier v6/v7, in-flight | v6c/v7 corpora | LoRA | — | not merged/quantized | not confirmed deployed anywhere |
| glr_thinker_v1 | checkpoints/glr_thinker_v1/best.pt | GLR transition head v1 (non-zero init, λ=1e-3 unnormalized) | 1,162 rows | 1,049,600 params, frozen 752M Qwen3-0.6B backbone | val_loss 3.5002; **FAILED rollout** | not deployed | superseded by v2 |
| glr_thinker_v2 | checkpoints/glr_thinker_v2/best.pt | GLR transition head v2 (zero init, normalized loss) | same | same | val_loss 2.6128; PASSED rollout, K=10 chosen | not deployed — sequenced after speaker directive-conditioning fix | current best GLR checkpoint |

**Row counts, Table 2**: 48 carried forward + 20 new/changed. **Total referenced = 68.**

---

## TABLE 3 — M2 run history (chronological, scaling story)

**No new M2/AVJepaPredictor training runs since v1.** `RESULTS_TABLE.md`/`NEGATIVE_RESULTS.md`
(the authoritative M2 run-history sources) are unchanged in mtime since 2026-08-01/04. v1's 7
rows (Matched-step A, Matched-step B, VGGSound-60k+Ego4D-17.1k, RUN-1, RUN-2 LOCKED, RUN-2
step20000, RUN-2 best.pt/wrong-selection, RUN-3) are carried forward verbatim — see v1 §Table 3.

The **downstream M2/M3 embedding-predictor and query-predictor lineage** (consumes M2's frozen
output, not M2 itself) continued past v1: `ddp_gradcache_bs16384` (score 62.05, pre-SigLIP2) →
4-way stream ablation (2026-08-14) → SigLIP2 target-space runs A/B/C/D (2026-08-15) →
`sig_runD_proj768` (VGG R@1 0.737, current recommended, confirmed by Part F to be production-
loaded). Captured in Table 1.B, not re-tabulated here.

**Row counts, Table 3**: 7 rows (unchanged from v1).

---

## TABLE 4 — Every falsifier

v1's 31 rows carried forward verbatim (`checkpoints/falsifier_tracking.md` unchanged since
2026-08-04). New falsifier-shaped tests found in post-2026-08-11 narrative sources (same
swap/shuffle/zero-vs-real pattern, **NOT logged in the formal `falsifier_tracking.md`** — flagged
as such):

| falsifier name | conditions tested | results per condition | n | PASS/FAIL | what it ruled out |
|---|---|---|---|---|---|
| Query-predictor swapped-query control, SigLIP2 space (run D) | correct vs swapped query | matched control 0.967 (runD) vs 0.502 (runB frozen) vs 0.541 (runA frozen) | — | PASS for runD; near-chance for frozen runs | confirms the trained projection, not just candidate-set presence, drives query-sensitivity |
| AV congruence swap falsifier, EmbeddingGemma geometry (4 arms) | audio-swapped, "what do you hear" | follows-EARS A 0.650, B 0.609, C 0.562, D(no-audio) 0.070; matched control 0.92-0.96 | — | vision-only arm (D) fails (93% guessable from picture); audio arms pass | audio/M2 contribution being an artifact of a visually-guessable benchmark |
| AV congruence swap falsifier, SigLIP2 geometry | same, new space | follows-EARS runD 0.608/ctrl 0.967; frozen runs near-chance | — | RETRACTS "one ear is enough"; confirms trained-projection necessity |
| GLR rollout divergence check (v1) | K=0 vs K=10 | tokens 83→330, eos 1.00→0.38, f1 0.234→0.090 | n=40 | **FAIL** | a good teacher-forced val_loss implies a working rollout |
| GLR rollout re-check (v2), n=40 then n=200 | K=0/5/10/15/20 | see Table 1.B | n=40, then 200 | PASS at K≤10 | the n=40 f1 bounce was noise, resolved by 5× replication |
| Speaker intent bake-off, LONG vs SHORT | v1-v5 × 2 formats | see Table 1.B | n=6 | hypothesis (LONG collapses due to length) FALSIFIED | prompt length as the cause of v1's "confidently irrelevant" behaviour |
| name_stranger open-set filter | closed-set vs open-set | 9/9 closed-set catches; 93 false rejects vs 9 real catches, open-set | ~126 rows | closed-set PASS, open-set FAIL | regex-based open-set behavioural filtering as viable |
| thinker_behaviour_gate.py, 5 cases × K=0/5/10 | 5 behavioural criteria | see Table 1.B | n=1/case greedy | 3/5 PASS all K; respects_focus FAIL all K; asks_when_unsure FAIL→PASS w/ GLR | isolated a corpus-level defect (compulsive game-offer) independent of GLR |

**Row counts, Table 4**: 31 carried forward + 8 new narrative-sourced. **Total = 39.**

---

## TABLE 5 — Negative results and retractions

v1's 23 rows carried forward verbatim (source `NEGATIVE_RESULTS.md`, unchanged since 2026-08-01).
New since 2026-08-11:

| what was claimed | what the check found | what changed | file path |
|---|---|---|---|
| SigLIP2 as a frozen target space could "share" geometry for free | frozen SigLIP2 lost badly to EmbeddingGemma reference (0.654/0.489 vs 0.883/0.681), flat from step 249 | trainable proj-768 restored; within-clip 0.688→0.811, R@1 0.627→0.737 | JEPA_MEMORY_PLAN.md §A.2/A.3 |
| "Dropping WavJEPA-nat costs nothing" (SigLIP2-space claim) | true in SigLIP2+proj768 space (0.608 base-only ≈ 0.609 old base+nat) | adopted; nat cost retired on Jetson — **⟶ Part F: CONFIRMED live, `wavjepa_nat=None` in production** | JEPA_MEMORY_PLAN.md §10.1/A.5 |
| "One ear is enough" (2026-08-14 EmbeddingGemma-geometry claim, itself later reversed again) | AV congruence falsifier showed dropping nat costs 4.7pp of audio-following, invisible on R@1 | RETRACTED same-day, reframed as a trade-off, then itself superseded 2026-08-15 when the trade evaporated in the new space | SESSION_LEDGER_2026-08.md §9.11 |
| SigLIP2 CPU-first-then-move saves Jetson memory | end-of-run available memory was identical (1,251 vs 1,256MiB) — allocator recharged elsewhere | "a single mem-log line improving is not a saving; only end-of-run available counts" | SESSION_LEDGER_2026-08.md §10.10 |
| torchao int8 saves Jetson memory the way it does on mercury | NO-OP: 577.7→577.7MiB; torch 2.8.0 lacks torchao's required cpp extensions | 105MiB mercury saving does not transfer | SESSION_LEDGER_2026-08.md §9.6 |
| int8 on EmbeddingGemma's embedding table is as safe as linear-layer int8 | breaks the encoder: field-acc 0.939→0.272, cosine →0.31-0.58 | int8-on-linears-only adopted | checkpoints/EMBEDDINGGEMMA_QUANT_EVAL.json |
| GLR v1 (val_loss 3.5002, "looked fine") would work at inference | rollout diverges: 4× more tokens, EOS 1.00→0.38, f1 more than halves | teacher-forced val_loss cannot catch rollout divergence; root-caused to non-zero head init + unnormalized λ=1e-3; fixed in v2 | SESSION_LEDGER_2026-08.md "GLR v1 FAILED" |
| sqrt-LR-style intuition applied to GLR's λ | over-corrected, removed gradient pressure needed to shrink head output | fixed by normalizing L_Δ instead of hand-tuning λ | SESSION_LEDGER_2026-08.md "Root cause: two compounding scale errors" |
| Thinker v5 "passes all four perception_social cases" (earlier recorded claim) | no such test exists in the repo — never committed | `thinker_behaviour_gate.py` written as the first actual committed test; real score 3/5 | SESSION_LEDGER_2026-08.md "Provenance note" |
| Thinker v4 corpus regen would fix restraint contradictions via "more scenarios" | 61-62% of EXISTING restraint rows already contradicted their own setup | fixed at generator-prompt level, not by adding coverage | SESSION_LEDGER_2026-08.md "THE GAME-OFFER DEFECT" |
| Directive-corpus generation would work by generating CoT and answer separately | attempt 2 (395 rows, passed count gate) had 197/395 (50%) mismatched CoT/answer pairs | fixed by generating matched pairs with a `paired` provenance flag, gated ≥90% self-paired | SESSION_LEDGER_2026-08.md "Attempt 2... ACTIVELY HARMFUL" |
| v1/v2 speaker "were better" (user's own recollection) | measured false — v1 scores 1/6 on intent adherence; apparent quality was "confident irrelevance" | corrected by measurement; root cause is zero instruction-conditioned rows in any speaker corpus | SESSION_LEDGER_2026-08.md "the v1/v2 were better memory is FALSE" |
| Speaker v6 val_loss (0.6924) lower than v5's (0.7542) means v6 is better | NOT a valid comparison — different corpora/val-sets; bake-off (8/12 vs 9/12) is underpowered | v6 not deployed; comparison method flagged as needing repair | SESSION_LEDGER_2026-08.md "SPEAKER v6" |
| Fan noise is broadband (earlier CLAUDE.md conclusion) | differencing two fan speeds found 38.6% of energy in a single ~810Hz peak matching 9-blade BPF to 1.9Hz | CORRECTED — fan is tonal; notch filter (not high-pass) built | SESSION_LEDGER_2026-08.md "THE FAN IS TONAL" |
| Spectral subtraction against the fan profile would clean the `hearing` percept | measured WORSE — "alarm beeping" confidence rose 0.451→0.478, "glass breaking" appeared | `denoise=False` adopted default | SESSION_LEDGER_2026-08.md "SPECTRAL SUBTRACTION MADE THE PERCEPT WORSE" |
| Camera `--rotate 90` is the correct orientation | confidently reported a seated person as "lying down" (+0.71, highest score) | fixed to `--rotate 180`; physical mount confirmed upside-down | SESSION_LEDGER_2026-08.md "CAMERA" |
| TTS is broken due to memory / needs more RAM (2026-08-14 diagnosis) | with 5-6GB free, `import onnxruntime` itself aborts on cpuid parsing — not a memory problem | two-fold real cause found 2026-08-16: piper-tts-pulled ORT 1.23.2 regression + 7W-mode core count — **this row itself documents 3 different, partially conflicting diagnoses across 08-14/15/16, see Part E** | CLAUDE.md; SESSION_LEDGER_2026-08.md §10.11 |
| WavJEPA-nat and M3Connector are (implicitly) still needed | both dead loads: nat = 0 measured gain for +701MiB/+326ms; M3 unused since the drop pivot | removed from `build_bmo_stack` — **⟶ Part F: CONFIRMED live, both are `None`** | CLAUDE.md "THE OOM WAS TWO DEAD LOADS" |

**Row counts, Table 5**: 23 carried forward + 18 new. **Total = 41.**

---

## TABLE 6 — Deployment measurements (Jetson + Mercury)

v1's 35 rows carried forward verbatim — see v1 §Table 6. New since 2026-08-11:

| component | metric | value | device | conditions | file path | supersedes |
|---|---|---|---|---|---|---|
| Full stack, 7W power mode (undiagnosed) | perception/total-round latency | 3.8-12.4s/14971-18401ms | Jetson (7W, discovered accidentally) | 4/6 cores online | SESSION_LEDGER_2026-08.md §9.2 | flags historical figures assumed MAXN_SUPER |
| Full stack, MAXN_SUPER (corrected) | perception/total-round latency | 1.089-1.732s/3892-4994ms | Jetson | `nvpmodel -m 2` + `jetson_clocks` | SESSION_LEDGER_2026-08.md §9.2 | matches pre-existing 1247ms perception figure |
| Perception-query engine, standalone | memory cost | ~855MiB (EmbeddingGemma ~700 + predictor+bank ~145) | Jetson | isolated load | jetson_perception_query_results.json | superseded by SigLIP2-space banks |
| Perception-query engine latency | query-encode/predictor/lookup(24k)/total | 263.5/138.5/2.6/403.8 ms | Jetson | isolated | jetson_perception_query_results.json | — |
| Describe-demo full load walk (no-TTS) | avail after full load | 737MiB | Jetson | face engine running | SESSION_LEDGER_2026-08.md §9.3 | superseded by SigLIP2-space fit test |
| Full stack + thinker + max bank | fits? | NO — OOMs at bank load | Jetson | — | SESSION_LEDGER_2026-08.md §9.6 | — |
| SigLIP2 variant comparison | encode/load/avail-after/room-sentences-top4 | base: 19ms/35s/4160MiB/4-4; large: 38ms/68s/1847MiB/3-4 | Jetson | first-ever `from_pretrained` | jetson_artifacts/siglip2_jetson_bench.json | base variant selected |
| SigLIP2-space fit, no-nat | avail after full load | **1,410MiB** | Jetson | MAXN_SUPER, no TTS/STT | ARCHITECTURE.md §8 | supersedes 827MiB/571-OOM figures |
| SigLIP2-space fit, with-nat | avail after full load | 935MiB | Jetson | same | ARCHITECTURE.md §8 | — |
| SigLIP2-space steady-state latency, no-nat | per-leg breakdown | 905/792/104/33/817/212/**2,886ms** total | Jetson | — | ARCHITECTURE.md §8 | superseded same-day by capture fix |
| Capture-loop fix | capture/e2e total | 905ms→1-9ms/2,886→**1,795ms** | Jetson | producer/consumer ring buffer | ARCHITECTURE.md §9 | −38% total |
| WavJEPA-nat forward, confirmed cost | memory/latency | +701MiB/+326ms | Jetson | on-device | SESSION_LEDGER_2026-08.md §10.8 | supersedes 469ms figure |
| Identity head fit | memory/latency | +96MiB/4-9ms per query | Jetson | reuses existing streams | SESSION_LEDGER_2026-08.md §10.10 | **⟶ Part F: CONFIRMED wired into live production** |
| Memory recovery, AutoProcessor + flash_attn removal | avail-at-end | 266→**710MiB** | Jetson | — | SESSION_LEDGER_2026-08.md §10.11 | — |
| TTS/STT candidate budget analysis | envelope vs need | 366-516MB vs 609MB needed | Jetson | priced, not deployed | SESSION_LEDGER_2026-08.md §10.10 | ~100-250MB short as of 2026-08-15 |
| GLR embed/hidden-state round-trip probe | result | PASS — per-token (11,1024), cos 0.9078 | Jetson | `probe_llamacpp_hidden_states.py` | SESSION_LEDGER_2026-08.md "GLR — deployable" | unblocks GLR deployment mechanism |
| GLR-fixed thinker leg latency | latency | 650ms→1,749-3,509ms (median ~2.4s) | Jetson | 6 live rounds | SESSION_LEDGER_2026-08.md D1 | thinker is now dominant leg |
| GLR K=10 projected thinker leg | projected latency | ~1.0-2.1s | Jetson (**projected, not measured on the live loop**) | 41% token reduction × ms/token | SESSION_LEDGER_2026-08.md "Projected effect" | GLR not yet deployed |
| Fan notch filter | attenuation/speech-impact/latency | 33.4dB/0.0dB@220Hz/1.6ms per 10s | Jetson | `sosfiltfilt`, tach-driven | SESSION_LEDGER_2026-08.md | pure-numpy fallback = 374ms (57% of perception leg) |
| Production full-stack (incl TTS), post-OOM-fix | resident/avail; +camera avail | 3923MiB/975MiB; 741MiB w/ camera | Jetson | nat+M3 removed | CLAUDE.md | supersedes 2026-08-01 6781/839MiB figure; **⟶ Part F: architecture change confirmed real by source diff; exact MiB figures not independently re-measured by this extraction — see Part E E1** |
| Production boot time/VOICE TTFA, post-fix | value | 46s boot (no TTS)/997ms VOICE TTFA | Jetson | — | CLAUDE.md | supersedes 625ms figure (different stack composition) |
| Live e2e memory, TTS/STT OFF | used/free; steady-state | 4,361MiB/577MiB; 115-176MiB free | Jetson | full perception + 2 LLMs | SESSION_LEDGER_2026-08.md | with TTS/STT: 280MiB free, camera NVMM FAILS (open issue) |
| Live e2e per-leg latency (thinking ON) | capture/perception/M2/SigLIP2/query/thinker/speaker | 3-9/650-1400/24-51/85-184/23-218/**1,749-3,509**/157-583 ms | Jetson | 8 live rounds | SESSION_LEDGER_2026-08.md | current authoritative live figure |

**Row counts, Table 6**: 35 carried forward + 21 new. **Total = 56.**

---

## TABLE 7 — CURRENT BEST PER COMPONENT

| component | best checkpoint path | headline metric | eval protocol/gallery | date | is it the one production loads? |
|---|---|---|---|---|---|
| M1 vision-text spine | not checkpointed (Run 5 config, no saved .pt) | V→T R@1 = 22.5 | MSR-VTT+VATEX ~50k | 2026-07 | N/A — M1 is offline-only, no production role |
| M2 joint AV predictor | checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt | VGGSound a→v/v→a R@1 53.27/53.72%, R@5 81.62/80.32%, R@10 88.67/88.09%; Ego4D sibling-excl v→a/a→v R@1 27.60/27.00%, R@5 58.01/58.16%, R@10 73.59/74.04% | 1545-clip VGGSound / 674-window Ego4D | 2026-07-28 | **YES** — confirmed by Part F, unchanged in the live-Jetson diff |
| M3 connector | checkpoints/m3_multigran_richcaption_v2/last.pt (LOCKED-at-freeze) — architecture DROPPED project-wide 2026-08-08 | test_loss 1.6724; F1 0.317 (30 gens) | n=30 gens | 2026-08-08 | **NO** — confirmed by Part F: live production sets `m3_connector=None` |
| M4c turn-taking decision head | checkpoints/m4_decision_head_3class_speechonly_moonshine/best.pt | accuracy 90.67%, macro F1 90.61% | n=300 | 2026-08-07 | **YES per the live-Jetson code diff** (Moonshine, not SenseVoice) — but see Part A A3-1/Part F, the competing SenseVoice claim in `ARCHITECTURE.md`/`SESSION_LEDGER` is not reconciled by any artifact, only outweighed by structural evidence |
| M2/M3 embedding predictor (pre-SigLIP2 lineage) | checkpoints/m2_embed_predictor_mlp_ddp_gradcache_bs16384/best.pt (step799) | combined score 62.05 | VGGSound+Action100M held-out | 2026-08-03/04 | **NO** — superseded in recommendation by the SigLIP2-space predictor below, and Part F confirms the SigLIP2 one is what's loaded |
| Query predictor (EmbeddingGemma-space lineage) | checkpoints/query_predictor_ddp_b1024/best.pt (step1999) | cross-clip R@1 0.715 (small pool); honest bank-dependent number 0.189-0.442 | ~600-clip pool vs 6k-48k bank | 2026-08-11/14 | **NO** — superseded by SigLIP2-space predictor (next row), confirmed by Part F |
| Query predictor (SigLIP2-space lineage) — **current recommended** | checkpoints/sig_runD_proj768/best.pt | VGG within 0.811, cross-clip R@1 0.737 | 518k-clip pool; NOT the 1545-clip M2 gallery (Part C) | 2026-08-15 | **YES — CONFIRMED by Part F** (byte-identical `qp_runD.pt` on the live Jetson, referenced by path in `build_bmo_stack()`) |
| SigLIP2 candidate/tag set | checkpoints/candidates_siglip2_v2.pt | 1,482 tags incl. 110 appearance, 2.17MiB | — | 2026-08-16 | **YES — CONFIRMED by Part F**, resolving the D4-defect open question |
| Voice-only identity head | checkpoints/jepa_identity_head_voice_full/best.pt | TAR@FAR1% 0.705, AUC 0.957 | n=122,235/4,000 speakers | 2026-08-11 | **NO** — the *joint AV* head (next row) is what's wired, not the voice-only one |
| Joint AV identity head — **best head produced by this track** | checkpoints/jepa_identity_head_av_full/head_joint.pt | TAR@FAR1% 0.765, AUC 0.966 | n=106,736/4,420 speakers | 2026-08-11 | **YES — CONFIRMED by Part F** (byte-identical `identity_head_joint.pt` on the live Jetson, referenced in `build_bmo_stack()`); note `bmo_identity_memory.pt` (the persisted-enrollment file) does not yet exist on the Jetson — expected for a fresh install, not a bug |
| GLR transition head | checkpoints/glr_thinker_v2/best.pt | K=10: 41% fewer tokens, EOS 0.99, f1 0.93× baseline | n=200 held-out | 2026-08-16 | **NO** — explicitly not deployed; sequenced after the speaker's directive-conditioning fix; unchanged by Part F |
| Fast-tier (speaker) LLM | checkpoints/bmo_lfm25_350m_v5_Q8_0.gguf | bake-off SHORT 5/6, LONG 4/6 | n=6 fixtures | 2026-08-16 | **YES — CONFIRMED by Part F** |
| Reasoning-tier (thinker) LLM | checkpoints/bmo_thinker_qwen3_v5_Q8_0.gguf | behavioural gate 3/5 PASS at K=0 | 5 hand-written cases, n=1 each | 2026-08-16 | **YES — CONFIRMED by Part F** |
| TTS voice (non-emotion) | checkpoints/bmo_neutts_v5 | VOICE TTFA 997ms (full stack) | live measurement | 2026-08-16 | **YES (default)** — unchanged by Part F |
| TTS voice (emotion, opt-in) | checkpoints/bmo_neutts_emotion_v3 (v4 exists on disk, UNDOCUMENTED — see Part E E2) | eval_loss 0.526 (v3) | — | 2026-08-08 | **NO by default** (`BMO_TTS_EMOTION=1` opt-in); Part F's diff shows the code still loads the *unversioned* file (== v1), not v3 — an open, uncorrected finding, see Part B B2 |
| Fan notch filter | models/m5_fan_notch.py (signal-processing module, no learned weights) | 33.4dB attenuation/0.0dB speech impact/1.6ms latency | on-device, tach-referenced | 2026-08-16 | UNVERIFIED whether wired into the live audio path — not checked in Part F's diff; percept-level effect explicitly unproven regardless |

**Row counts, Table 7**: 16 rows.

**M2 row — R@5/R@10 added and direction labels pinned (2026-08-23)**: prior versions of this
table carried only R@1 and no direction labels. The full R@1/5/10 for step19000 were always
present in the training log and the Ego4D result JSON, and are now transcribed above:
- VGGSound (1545-clip gallery): `logs/m2_run2_final.log:1325-1334`, the
  `=== RETRIEVAL EVAL (contrastive head) @ step 19000 ===` block. Direction labels are taken
  verbatim from that block: `ambient→vision_R@1=53.27%` (a→v) and `vision→ambient_R@1=53.72%`
  (v→a) — i.e. **53.27 is a→v, 53.72 is v→a**. `METHODOLOGY_FORENSICS.md` §2.4 previously
  labelled this pair "v→a/a→v" (swapped); corrected there in the same pass.
- Ego4D (674-window, sibling-excluded):
  `checkpoints/vjepa21_shelved/EGO4D_HELDOUT_RUN2_STEP19000_RESULT.json`, keys
  `sibling_excluded.vision_to_ambient` (27.60/58.01/73.59) and `.ambient_to_vision`
  (27.00/58.16/74.04).

**Independent reproduction (2026-08-23)**: `scripts/eval_checkpoint_gallery.py` re-run on
`step19000.pt` over the same 1545-clip gallery (`data/vggsound_eval_1545.txt`, full-gallery
assertion passed) gives a→v 53.59/81.10/88.03 and v→a 52.75/80.00/87.12 — within ~0.5–1pp of the
logged values (matched_cos 0.7206 vs 0.7231). Two known sources of the delta: bf16 autocast is
non-deterministic, and the training-time `/dev/shm/jepa_m2_cache` no longer exists, so the re-run
read `/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k` (all 1545 clips present, same
extraction manifest, but a separate extraction pass). **The logged numbers remain the cited
ones**; the re-run is a reproduction check, not a replacement measurement.

---

**PART D TOTAL ROW COUNTS**: Table 1 = 222. Table 2 = 68 referenced. Table 3 = 7. Table 4 = 39.
Table 5 = 41. Table 6 = 56. Table 7 = 16.

---

# PART E — INTEGRITY

## E1. UNSOURCED — DO NOT CITE

Numeric claims in CLAUDE.md (prioritizing its 2026-08-16-dated sections), `falsifier_tracking.md`,
and root `*STATUS*/*PLAN*/*ANALYSIS*.md` files, checked against an actual results file
(JSON/PROVENANCE.txt/raw log).

| # | claim | where stated | backing checked | verdict |
|---|---|---|---|---|
| 1 | "Real cost of thinking: 650 ms → 1,749–3,509 ms (median ~2.4 s)" | CLAUDE.md, "Live-pipeline defects found 2026-08-16" | `scripts/jetson_real_demo.py` writes only to `~/bmo_demo.log` (Jetson home dir) — no such file or derived JSON exists in this repo | **DO NOT CITE** (no local artifact) |
| 2 | "Full-stack memory, live (TTS/STT off): 4,361 MiB used, 577 MiB free; 115–176 MiB free in steady state" | CLAUDE.md, same section | same script, no output file locally; echoed verbatim in `SESSION_LEDGER_2026-08.md` (a second prose doc, not a raw artifact) | **DO NOT CITE** as file-backed; two docs corroborate each other but neither cites a JSON/log |
| 3 | "With TTS+STT resident it is 280 MiB [free]" | CLAUDE.md, same section | same — no artifact | **DO NOT CITE** |
| 4 | "38.6% of all fan energy sits in one peak at 808–812 Hz... BPF = 810.4 Hz... 1.9 Hz error" | CLAUDE.md, "Power, fan acoustics and GLR" | `scripts/measure_fan_signature.py` writes to a CLI-supplied path; no matching JSON found anywhere in this repo | **DO NOT CITE** |
| 5 | "33.4 dB on the tone, 0.0 dB on speech, 1.6 ms per 10 s buffer" vs "374 ms" fallback, "57% of the perception leg" | CLAUDE.md, same section | `models/m5_fan_notch.py` exists; no output log/JSON found | **DO NOT CITE** |
| 6 | GLR: "1,049,600-param linear head... val_loss 3.5002 / ce 1.9951 / delta 1505 at epoch 4... λ=1e-3" | CLAUDE.md, same section | `logs/glr_thinker_v1.log`: `head params=1.05M`, `DONE best val_loss=3.5002`; matches `checkpoints/glr_thinker_v1/eval.json` (`mean_latent_norm=423.17`, `median_gen_tokens=330.0`) | **VERIFIED — may cite** (the exception in this section) |
| 7 | "FULL STACK RESIDENT (incl TTS) 3923 MiB used, 975 MiB available"; "CAMERA OPEN 234 MiB NVMM, 741 MiB available"; "before boot 4898 MiB available" | CLAUDE.md, "PRODUCTION DEPLOY 2026-08-16" | no `jetson_artifacts/` file newer than 2026-08-15 exists in this repo; **Part F confirmed the underlying architecture change (nat/M3 removal) is real via direct source diff, but did not independently re-measure these exact MiB figures on-device** | **DO NOT CITE the exact numbers; the qualitative architecture claim is now code-confirmed (Part F)** |
| 8 | "Boot time 46 s without TTS; VOICE TTFA 997 ms in the full stack" | CLAUDE.md, same section | no artifact | **DO NOT CITE** |
| 9 | ORT downgrade fix + power-mode fix narrative, "spoke at 2286 ms" | CLAUDE.md, same section | no artifact for the exact ms figure; qualitative mechanism (real `power` tools exist in `models/m5_tools.py`) plausible | **DO NOT CITE the 2286ms figure specifically** |
| 10 | Speaker bake-off table (v1 1/6·2/6, v2 4/6·3/6, v3 5/6·4/6, v5 4/6·5/6) | CLAUDE.md + full table in `SESSION_LEDGER_2026-08.md` | `scripts/speaker_intent_bakeoff.py` writes to an unresolved `args.out`; no matching JSON found | **DO NOT CITE** as file-backed (prose-doc-backed only) |
| 11 | Speaker v6 bake-off (4/6+4/6, 8/12) + "val_loss 0.6924 @ epoch 3" | `SESSION_LEDGER_2026-08.md` (tail) | val_loss half not independently re-derived from `logs/speaker_v6.log` in this pass; bake-off half traces only to the same unbacked script | val_loss: unverified either way; bake-off: **DO NOT CITE** |
| 12 | "not one of them had a name anywhere in the conversation" (54-row `{name}` corpus defect) | CLAUDE.md, "Speaker corpus: assert no `{name}` survives" | `scripts/fix_name_placeholders.py`'s own docstring: "only 13 of 54 rows have a name anywhere in the prompt" | **CONTRADICTED by the cited script's own documentation** — see E3 |
| 13 | "Never `pkill -f <script>`... Fourth occurrence" vs "5th and 6th occurrences this session" | CLAUDE.md, two different sections | both are narrative incident counts with no tracking file found anywhere | **DO NOT CITE** (unverifiable tally, and the two counts don't obviously reconcile) |
| 14 | "Camera is `--rotate 180`, not 90... `who: a person lying down (+0.71)`" | CLAUDE.md, "Live-pipeline defects" | `SESSION_LEDGER_2026-08.md` reproduces the same figure and full answer table — prose-corroborated, no raw JSON/log found | **DO NOT CITE** as file-backed (prose-corroborated only) |
| 15 | "candidates_siglip2_v2.pt (1,482 tags, appearance 110, 2.17 MiB)" | CLAUDE.md, "Live-pipeline defects" | file exists, size 2,313,547 B = 2.21 MiB (claimed 2.17 — close, possibly MiB/MB rounding); tag-count breakdown not independently re-derived (no torch in this shell) | **PARTIALLY VERIFIED** — file exists, size in the right ballpark, breakdown unconfirmed |

## E2. UNDOCUMENTED ARTEFACTS

| # | path | size/mtime | best-guess | status |
|---|---|---|---|---|
| 1 | `checkpoints/bmo_neutts_emotion_v4_Q8_0.gguf` | 595,465,248 B, 2026-08-08 11:59 (pre-cutoff) | emotion-TTS voice, 4th iteration (v1/v2/v3 all documented with eval_loss; v4 is not) | **UNVERIFIED** — no PROVENANCE, results JSON, or `.md` mention anywhere; unchanged from v1; **and per Part B/Part F, the live production code doesn't load this file or any versioned emotion GGUF — it loads the unversioned/v1-equivalent name** |
| 2 | `checkpoints/bmo_thinker_qwen3_v6_lora/best/` | adapter 40,422,168 B, 2026-08-16 00:28 | Qwen3-0.6B thinker LoRA, 6th iteration | raw log exists (`thinker_v6_train.log`: `DONE ... best_val_loss=1.8119`) but **zero `.md` mentions of "qwen3_v6" anywhere** — SESSION_LEDGER discusses "thinker v6c" as a corpus, not this checkpoint, and jumps to v7 without writing up v6's own eval. **UNVERIFIED as a documented checkpoint** (log-backed, doc-silent) |
| 3 | `checkpoints/bmo_lfm25_350m_v6_merged/`, `bmo_thinker_qwen3_v5_merged/` | e.g. v6_merged `model.safetensors` 708,984,464 B, 2026-08-16 06:04 | intermediate full-precision merge step, pre-GGUF-quant, matching the pattern of earlier `_v3_merged` etc. | not individually documented but the pattern is established elsewhere — low concern |
| 4 | New-since-cutoff `checkpoints/*` tree overall | 291 new files, **zero** new `PROVENANCE.txt` files | — | The PROVENANCE.txt convention used throughout M1–M5/JEPA-mem was **not used at all** for any post-cutoff checkpoint; documentation lives exclusively in prose or raw logs instead — structural reason E1/E2 are as numerous as they are |

## E3. CONFLICTS (repo-wide sweep, not limited to v1-vs-new)

| # | fact in dispute | claim A | claim B | resolved? |
|---|---|---|---|---|
| 1 | What STT engine is "the deployed"/"live" one | `ARCHITECTURE.md` (2026-08-14/15): "STT: SenseVoice-Small via sherpa-onnx (+ Moonshine for the turn-taking head) — live" | CLAUDE.md (through 2026-08-16): describes `build_bmo_stack()`'s load order exclusively in terms of Moonshine; never mentions SenseVoice/sherpa-onnx/`live_bmo_sensevoice.py` | **Narrowed by Part F, not closed** — the live `build_bmo_stack()` (2026-08-16 14:36 Jetson copy) confirms Moonshine, no SenseVoice reference anywhere in it. `live_bmo_sensevoice.py` exists on the Jetson but sits outside the `~/bmo_production/scripts/` tree alongside 4 other ad hoc `live_bmo_*.py` drivers, none currently running (no matching process, `ps aux` checked live). Best-supported reading: SenseVoice was an exploratory alternative that was tried but never merged into the documented production path — but no artifact says this outright. |
| 2 | Which harness is BMO's real production entry point | CLAUDE.md: "the real entry point... does a privileged memory-compaction step... then execs `bmo_jetson_startup.py`" (`bmo_launch.sh`) | `SESSION_LEDGER_2026-08.md` §3.3: `~/live_bmo_sensevoice.py` "built + deployed" as a second, independently-described harness | **Narrowed by Part F** — `bmo_launch.sh` (read directly off the Jetson) does exactly what CLAUDE.md describes, including the sudoers-scoped `NOPASSWD: /usr/bin/tee /proc/sys/vm/compact_memory` rule, and is the only script under the version-tracked `~/bmo_production/scripts/` tree. `live_bmo_sensevoice.py` and its four siblings (`live_bmo.py`, `live_bmo_gpt.py`, `live_bmo_energy.py`, `live_bmo_stream.py`) live directly under `/home/bmo/`, not that tree, and their own docstrings frame them as tests ("First real end-to-end conversational test", "no perception stack"). Structural evidence favors `bmo_launch.sh` as *the* production path — but this is inference from file layout and docstrings, not a committed decision record. |
| 3 | Whether the corpus `{name}` rows ever contained a user-supplied name | CLAUDE.md: "not one of them had a name anywhere in the conversation" (of the 54 `{name}` rows) | `scripts/fix_name_placeholders.py` docstring, same 54-row set: "only 13 of 54 rows have a name anywhere in the prompt" | **no** — directly conflicting counts (0 vs 13) on the identical row set. Independently re-derived ground truth from `data/bmo_companion_corpus_v11.jsonl`'s `name_fix` field: 0 rows ended up with a name substituted into *BMO's own line* (50 "removed", 2 "prompt-substituted" into the *user's* line, 0 into BMO's) — supports CLAUDE.md's downstream conclusion (BMO never invents a name) but not its literal premise, which the script's own count of 13 contradicts. |
| 4 | **4-way stream ablation numbers** | v1 Table 1 (2026-08-11): "unified architecture ablation, cross-clip R@1 (m2/vision/m2+vision/unified) = 0.385/0.447/0.478/0.458" | This ledger's Table 1.B (2026-08-14): "4-way stream ablation A/B/C/D = 0.441/0.564/0.566/0.546" | **no** — same 4-arm shape, plausibly the same underlying concept (a stream-composition ablation), but numerically incompatible and never cross-referenced by either source document. Not resolved here; flagged only. |
| 5 | **TTS breakage root cause**, timeline-internal | `PIPELINE_REMAINING.md` (2026-08-15): TTS "was never broken," ORT 1.23.2 works fine, RTF 0.11 | CLAUDE.md (2026-08-16, later): TTS crashed because ORT 1.23.2 aborts on this CPU's cpuid string, fixed by downgrading to 1.19.2 plus a power-mode fix | Both cannot describe the same stable state on consecutive days. Not resolved here; flagged. |

**ROW COUNTS — Part E**: E1 = 15 rows. E2 = 4 rows. E3 = 5 rows.

---

# PART F — LIVE-DEPLOYMENT RECONCILIATION (direct Jetson SSH verification, this session)

Every sub-agent above worked from this local git checkout only — none had access to the actual
deployment target. At the user's request ("try to check bmo_production folder, I think that is
on the jetson... maybe get the files here if need be"), the orchestrating agent SSH'd into
`bmo@bmo-desktop` (Tailscale, key auth already set up) and directly compared the live,
Jetson-resident copy of the production stack against this repo.

## F1. What was checked

| check | command/method | result |
|---|---|---|
| Jetson reachable | `ssh bmo@bmo-desktop hostname` | `bmo-desktop`, reachable |
| `~/bmo_production/` layout | `ls -la` | `face_engine/`, `models_gguf/`, `pipeline/`, `scripts/`, `tokenizers/` — matches CLAUDE.md's documented layout exactly |
| Real entry point | `cat ~/bmo_production/scripts/bmo_launch.sh` | Does the ONE privileged step (`echo 1 \| sudo -n tee /proc/sys/vm/compact_memory`, gated by a narrowly-scoped `NOPASSWD` sudoers rule), then `exec python3 bmo_jetson_startup.py "$@"` — matches CLAUDE.md's description verbatim |
| Is BMO currently running? | `ps aux \| grep -iE 'live_bmo\|bmo_jetson_startup\|bmo_launch'` (empty); `systemctl list-units --type=service --all \| grep -i bmo` | **No conversational BMO process is running.** Only `bmo_face_engine.service` and `bmo_xorg.service` (display/kiosk, unrelated to the perception/LLM stack) are active. BMO is launched on-demand, not as a persistent daemon — all latency/memory figures in this ledger are test-run snapshots, not continuous production telemetry. |
| Local vs Jetson copy of `bmo_jetson_startup.py` | `scp` + `diff` | Local repo copy: 439 lines, mtime 2026-08-14 16:10, **untracked in git** (`?? scripts/bmo_jetson_startup.py`, never committed). Jetson copy: 522 lines, mtime **2026-08-16 14:36** (today, ~2h before this check) — i.e. edited *after* this dev-machine checkout was last synced. Two Jetson-side backup files (`bmo_jetson_startup.py.bak_20260816_1314`, `.bak_pre_siglip_1335`) confirm active same-day editing, consistent with CLAUDE.md's 2026-08-16 sections being written live during this work. |

## F2. What the diff showed (local-stale vs Jetson-live)

The Jetson-live copy, relative to the local repo's stale copy:

1. Adds `query_vectors_path`, `identity_ckpt`, `identity_memory_path` parameters to
   `build_bmo_stack()`.
2. `wavjepa_nat = None` (was: constructed + int8-quantized every boot) — comment explicitly
   cites the +701MiB/+326ms/zero-gain finding, matches CLAUDE.md word-for-word.
3. `m3_connector = None` (was: loaded + `.to(device)`'d every boot) — comment cites the
   M3-dropped pivot, matches CLAUDE.md.
4. Adds an entirely new "PERCEPTION QUERY (SigLIP2, text-encoder-free) + IDENTITY" block: loads
   a `PreEncodedTextSpace` query engine, applies the checkpoint's trained projection to a raw
   candidate bank at load time, and separately loads a `siglip_vision` tower (text tower
   deleted, vision-only, saving 538MiB) plus an `IdentityHead`/`JepaMemory` pair.
5. `__main__` block: `lfm2_gguf` changed from `bmo_lfm25_350m_v1_Q8_0.gguf` →
   `bmo_lfm25_350m_v5_Q8_0.gguf`; `minicpm5_gguf` changed from `bmo_thinker_qwen3_v2_Q8_0.gguf`
   → `bmo_thinker_qwen3_v5_Q8_0.gguf`.
6. `__main__` block now passes `query_predictor_ckpt=".../checkpoints/qp_runD.pt"`,
   `perception_bank_path=".../checkpoints/candidates_siglip2_v2.pt"`,
   `query_vectors_path=".../checkpoints/query_vectors_siglip2_v2.pt"`,
   `identity_ckpt=".../checkpoints/identity_head_joint.pt"`,
   `identity_memory_path=".../checkpoints/bmo_identity_memory.pt"`.

**Every one of these matches CLAUDE.md's 2026-08-16 narrative exactly.** The contradiction every
sub-agent flagged between "CLAUDE.md's claims" and "the code" was real, but it was a
**repo-checkout-vs-deployment-target sync gap**: this dev machine's `/home/utkarsh/JEPA-Omni`
had simply never received the day's edits made directly on the Jetson.

## F3. Checkpoint identity resolution

Two of Part B's B2 "UNVERIFIED" rows (files referenced by the Jetson code but absent from this
repo's `checkpoints/`) are resolved by direct byte-size comparison:

| Jetson-side reference | Jetson file, size | matching local-repo file, size | verdict |
|---|---|---|---|
| `.../checkpoints/qp_runD.pt` | 131,615,595 B, mtime 2026-08-15 11:30 | `checkpoints/sig_runD_proj768/best.pt`, 131,615,595 B | **Byte-identical — confirmed same file**, pushed to the Jetson under a shorter name and never synced back to this repo |
| `.../checkpoints/identity_head_joint.pt` | 32,672,373 B, mtime 2026-08-15 12:04 | `checkpoints/jepa_identity_head_av_full/head_joint.pt`, 32,672,373 B | **Byte-identical — confirmed same file**, and confirms production uses the *best* identity head in the whole track (TAR@FAR1%=0.765/AUC=0.966), not the smaller `jepa_identity_head_av` (32,621,045 B, a near-miss size that was checked and ruled out) |
| `.../checkpoints/bmo_identity_memory.pt` | **does not exist yet** on the Jetson | n/a | Expected, not a bug — `JepaMemory.load()` is a documented no-op when the file is absent (fresh install, nobody enrolled yet) |

These checkpoints (`qp_runD.pt`, `identity_head_joint.pt`) exist **only** on the Jetson's
`~/bmo_production/pipeline/checkpoints/`, not in this dev-machine repo's `checkpoints/` — a
second, narrower instance of the same sync-gap pattern as F2, worth flagging for anyone treating
this repo's `checkpoints/` directory as a complete inventory of what's been trained and deployed.

## F4. Production entry-point landscape (resolves part of E3-1/E3-2/A3-3)

`/home/bmo/` (the Jetson user's home directory, NOT the version-scoped `~/bmo_production/`
tree) contains five distinct `live_bmo*.py` drivers, none currently running:

| script | mtime | own docstring says | STT | notes |
|---|---|---|---|---|
| `live_bmo.py` | 2026-08-08 21:36 | "Lean STT-LLM-TTS (no perception stack). First real end-to-end conversational test." | Moonshine (`AutoModelForSpeechSeq2Seq`) | hardcodes `bmo_lfm25_350m_v1`/`bmo_thinker_qwen3_v2` — a dev snapshot, not kept current |
| `live_bmo_gpt.py` | 2026-08-12 16:45 | "BMO FULL-DUPLEX CONVERSATIONAL PIPELINE" (ASCII diagram: mic→VAD→Moonshine→LLM→StreamingVoice) | Moonshine | larger (26KB), no perception stage in its own diagram either |
| `live_bmo_sensevoice.py` | 2026-08-14 18:00 | "FULL-DUPLEX BMO on SenseVoice-Small (sherpa-onnx)... Voice: emotion StreamingVoice... mirrors production" | SenseVoice | uses v2 fast-tier + v3 thinker (not v5/v5) — its own docstring ("mirrors production") implies it is explicitly a *variant test*, not itself production |
| `live_bmo_energy.py` | 2026-08-12 17:11 | (not read in full this pass) | — | — |
| `live_bmo_stream.py` | 2026-08-08 20:36 | (not read in full this pass) | — | — |

None of these five is under `~/bmo_production/scripts/` (the tree `bmo_launch.sh` lives in and
execs from), none performs the sudoers-scoped memory-compaction step, and none loads the current
v5/v5 GGUFs except by accident of timing. This is structural evidence — not a committed decision
record — that they are exploratory/dev harnesses and `bmo_launch.sh → bmo_jetson_startup.py` is
the intended production path. **This does not fully resolve E3-1** (which STT is "the" deployed
one) because no artifact anywhere states that SenseVoice was tried and rejected, or is still
under consideration — the ledger can now say with much higher confidence which script is
production, but not why the SenseVoice alternative exists or whether it's still live work.

## F5. Action taken on this repo

Per the user's request to "get the files here if need be": the stale local copy of
`scripts/bmo_jetson_startup.py` was archived (not deleted) to
`/tmp/claude-1006/-home-utkarsh/4979cdeb-2240-410d-a464-cfe446cf7afd/scratchpad/jetson_pull/
bmo_jetson_startup.py.LOCAL_STALE_COPY_pre_sync`, and the live Jetson copy was pulled into
`scripts/bmo_jetson_startup.py` in this repo. The file remains untracked in git (`?? scripts/
bmo_jetson_startup.py` both before and after — it was never committed), so this is a working-tree
change only; no commit was made (per this session's git-safety instructions: only commit when
asked). Nothing else was synced — `jetson_real_demo.py` was already current in this repo (its
2026-08-16 03:03 mtime already reflects the fixes), and checkpoint weights were left in place
(the `qp_runD.pt`/`identity_head_joint.pt` Jetson-side files are multi-hundred-MB and byte-
identical to files already in this repo under different names — no need to duplicate them).

## F6. What Part F leaves open

- **A3-1/E3-1 (SenseVoice vs Moonshine)**: narrowed, not closed — see F4.
- **B2's `bmo_neutts_emotion` versioning gap**: unchanged — the live Jetson copy still hardcodes
  the unversioned/v1-equivalent emotion GGUF filename, not `_v3` (the documented-fixed version).
- **E1's exact 2026-08-16 memory/latency figures** (3923MiB, 975MiB, 997ms TTFA, etc.): the
  *architecture* they describe is now code-confirmed real; the *exact numbers* were not
  independently re-measured on-device by this extraction (would require actually booting the
  full stack, which was out of scope for a diff-and-sync pass).
- **`models/bmo_duplex_tick.py`/`DuplexLoop` orphan status (B4)**: unaffected by the diff — still
  no confirmed caller anywhere, on either machine.
- **Fan notch filter wiring**: not checked in this pass — Table 7's row is marked UNVERIFIED for
  this reason, not resolved.

**ROW COUNTS — Part F**: F1 = 5 checks. F2 = 6 diff items. F3 = 3 checkpoint resolutions. F4 = 5
scripts catalogued. F5 = 1 sync action. F6 = 5 open items.

---

# CONSOLIDATED ROW COUNTS

- **Part A**: A1 = 12 category rows (291 files). A2 = 6. A3 = 3. A4 = 9.
- **Part B**: B1 = 15. B2 = 13. B3 = 6. B4 = 11. B5 = 9.
- **Part C**: C1–C4, narrative-with-embedded-figures (no fixed row count; ~15 distinct sourced
  figures).
- **Part D**: Table 1 = 222 (175 carried-forward + 47 new). Table 2 = 68 referenced (48 + 20).
  Table 3 = 7 (unchanged). Table 4 = 39 (31 + 8). Table 5 = 41 (23 + 18). Table 6 = 56 (35 + 21).
  Table 7 = 16.
- **Part E**: E1 = 15. E2 = 4. E3 = 5.
- **Part F**: F1 = 5. F2 = 6. F3 = 3. F4 = 5. F5 = 1. F6 = 5.

---

## Extraction notes

- **Methodology**: four parallel sub-agents (Part A+E, Part B, Part C, Part D) each read the repo
  independently; a fifth pass (Part F, this session's orchestrator) added direct Jetson SSH
  verification neither the sub-agents nor a purely local read could have performed. The dedicated
  Part C sub-agent's own output file was lost to a harness issue; Part C above is carried from a
  solo compile the Part-A+E sub-agent additionally produced after its own internal 3-way
  parallelization attempt failed ("Fork is not available inside a forked worker").
- `JEPA_MEMORY_PLAN.md`'s body through 2026-08-11 (lines 1–1367) duplicates v1's own Table 1
  entries verbatim and was not re-read line-by-line; its 2026-08-14 through 2026-08-16 sections
  (covered more thoroughly by `SESSION_LEDGER_2026-08.md`/`ARCHITECTURE.md`, which cite it as
  "full detail here") plus its final "A.6/A.7" appendix were read in full.
- Root-level `.log` files newer than the cutoff were enumerated and cross-referenced by name
  against the prose docs that cite them, but not individually read line-by-line (consistent with
  v1's own policy for its 178 pre-cutoff logs) — their final/summary numbers are taken from
  `SESSION_LEDGER_2026-08.md`/`ARCHITECTURE.md`'s own quotation of them.
- Checkpoint `sha256` hashes were not computed anywhere (large binaries); `mtime`+`size` was used
  throughout, including for Part F's byte-identity confirmations (size match on two large,
  independently-produced files is treated as conclusive here given the exact byte counts and
  matching directory/naming context, not as a formal hash-based proof).
- Part F's SSH access was to `bmo@bmo-desktop` only (Tailscale); no other machine referenced in
  this ledger ("mercury," the training box) was checked directly in this pass.
