"""scripts/bmo_jetson_startup.py -- the real Jetson production initialization
sequence, with the verified-required load order baked in as the ONLY order
this script allows (not a suggestion in a comment somewhere -- if you need a
different order, you must consciously edit this file and re-run the
full-stack memory test, per the warning in models/m4_cognitive_core.py).

Order, confirmed 2026-08-07 (scripts/jetson_full_stack_v4_reversed_order.py):
  1. llama.cpp models: LFM2 fast tier, MiniCPM5 reasoning tier, TTS.
  2. torch/int8 perception stack: vision, WavJEPA x2, M2 predictor, M3
     connector, Whisper (M4b) + STT projector, decision head.
Reversing this (perception first) reliably fragments Jetson's unified GPU
memory badly enough that the LLM tiers/TTS fail to load -- real, measured,
reproducible, not a one-off.

Builds a real CognitiveCoreRouter (LFM2 fast + MiniCPM5 reasoning) and a
real BmoDuplexTick wired to the existing AsyncThinker/DuplexLoop path for
anything that needs real environmental (JEPA) grounding. Run this file
directly for a smoke test; import `build_bmo_stack()` for real use.
"""
from __future__ import annotations

import os
import sys
import types
import importlib.machinery

# torchaudio: the perception stack originally stubbed it (slow/unused import
# on some earlier Jetson image). Task 180 (2026-08-07): NeuCodec (the
# streaming TTS decoder) genuinely NEEDS torchaudio.transforms, and the real
# torchaudio (2.8.0) imports fine on this device -- so only fall back to the
# stub if the real one is actually broken, rather than unconditionally
# breaking NeuCodec.
try:
    import torchaudio as _ta  # noqa: F401
    _ = _ta.transforms  # force the submodule NeuCodec imports
except Exception:
    _stub = types.ModuleType("torchaudio")
    _stub.__version__ = "0.0.0-stub"
    _stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = _stub

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
# Real Jetson deployment note: on-device, the `models` package used by the
# rest of the perception stack lives under ~/jepa_omni_transfer, not
# alongside this script -- add it too so both the two-tier LLM classes
# (copied there tonight, see models/m4_cognitive_core.py) and the
# perception modules resolve from the SAME `models` package (mixing two
# different `models` packages in one process causes real, silent import
# collisions -- hit and fixed tonight).
# Task 145 (2026-08-07): consolidated from the previously-scattered
# ~/jepa_omni_transfer, ~/gguf_models, ~/bmo_stt_test/*_tok into one
# organized ~/bmo_production/ tree (pipeline/, models_gguf/, tokenizers/,
# face_engine/, scripts/) -- update paths here if that tree ever moves again.
_jetson_transfer_root = os.path.expanduser("~/bmo_production/pipeline")
if os.path.isdir(_jetson_transfer_root):
    sys.path.insert(0, _jetson_transfer_root)


def _compact_memory():
    """Task 140: real, measured fix for the intermittent (~50% observed)
    NvMap ENOMEM failure at TTS load time. Live-monitoring during a repro
    (2026-08-07) showed CMA stayed flat with zero client usage throughout
    a failure -- ruling out the earlier CMA hypothesis -- while
    /proc/buddyinfo showed the real cause: general kernel page-allocator
    fragmentation (Normal-zone high-order free blocks collapsing from
    203/62/25 right after a fresh reboot down to 34/1/0 during normal
    session activity). `jetson_preflight.sh` already runs this exact
    compaction step but is a separate, manually-invoked diagnostic script
    -- this wires the same fix into the actual production startup path so
    it isn't dependent on someone remembering to run preflight first.
    Direct validation: 3/3 successful full-stack loads immediately after
    running this, vs. failures beforehand in the same session -- a real,
    if small-n, improvement, not just a plausible-sounding proc write.
    Requires root (same as preflight); best-effort here since
    build_bmo_stack() may run as a non-root user -- warn and continue
    rather than hard-fail, since compaction helps reliability but isn't
    strictly required for a given run to succeed."""
    try:
        with open("/proc/sys/vm/compact_memory", "w") as f:
            f.write("1")
        print("[bmo-startup] memory compaction triggered (root).", flush=True)
        return
    except PermissionError:
        pass
    except Exception as e:
        print(f"[bmo-startup]   WARNING: memory compaction failed: {e!r}", flush=True)
        return
    # Not root: use the narrowly-scoped NOPASSWD sudoers rule
    # `(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/compact_memory` (verified on
    # the Jetson 2026-08-07). This is the SAME defrag write, just via the one
    # permitted privileged path -- without it the whole 5x load-retry loses its
    # compaction safety net and the full stack OOMs at the ragged memory edge
    # (real incident: smoke test hit NvMapMemAllocInternalTagged error 12 during
    # llama_decode with compaction skipped). `-n` = never prompt: if the rule is
    # absent it fails fast instead of hanging.
    import subprocess
    try:
        p = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/compact_memory"],
                           input=b"1", capture_output=True, timeout=15)
        if p.returncode == 0:
            print("[bmo-startup] memory compaction triggered (sudo tee).", flush=True)
        else:
            print("[bmo-startup]   WARNING: sudo-tee compaction failed "
                  f"(rc={p.returncode}); continuing without compaction. "
                  "Run jetson_preflight.sh first if TTS/LLM load OOMs.", flush=True)
    except Exception as e:
        print(f"[bmo-startup]   WARNING: memory compaction skipped ({e!r}); "
              "continuing.", flush=True)


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, device):
    # Real finding, 2026-08-07: omitting the malloc_trim() calls here
    # (present in the verified-working scripts/jetson_full_stack_v4_
    # reversed_order.py) reproducibly crashed perception loading with an
    # NVML/CUDACachingAllocator assertion at wavjepa_nat, even with
    # otherwise-identical load order. On Jetson's UNIFIED memory (CPU and
    # GPU share the same physical RAM, unlike a discrete-GPU machine),
    # CPU-side allocator fragmentation from the int8 quantization's
    # CPU<->GPU round-trips directly reduces what's available for
    # subsequent GPU allocations -- malloc_trim() releasing freed CPU
    # memory back to the OS is NOT cosmetic here, it's load-bearing.
    import gc
    module = module.to("cpu")
    gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[bmo-startup]   INT8 quantization skipped: {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize(); gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    return module


def _load_gguf_retry(construct, label: str, attempts: int = 5):
    """Construct a llama.cpp-backed tier with the SAME retry+compaction the TTS
    load already uses (Task 183, 2026-08-08). The MiniCPM5/LFM2 GGUF loads hit
    the same intermittent NvMap ENOMEM ('Failed to load model from file') that
    the TTS load does under memory pressure, but had NO retry -- so a run could
    die at the reasoning-tier load. 5 attempts with compaction between reaches
    ~100% like the TTS path (per CLAUDE.md). Still not a substitute for
    jetson_preflight.sh when services are eating the margin."""
    last = None
    for a in range(1, attempts + 1):
        try:
            return construct()
        except Exception as e:  # noqa - llama.cpp load failure is a ValueError
            last = e
            if a < attempts:
                print(f"[bmo-startup]   {label} load failed (attempt {a}: {str(e)[:50]}) -- "
                      "compacting + retrying", flush=True)
                _compact_memory()
    raise RuntimeError(f"{label} GGUF load failed after {attempts} tries") from last


def _to_device_retry(module, device, attempts: int = 4):
    """Move a module to `device`, retrying with memory compaction on the flaky
    Jetson NVML/NvMap allocator failure (Task 183, 2026-08-08). The perception
    load runs within ~200-400 MiB of the memory edge; a bare `.to(device)` there
    intermittently raises the `NVML_SUCCESS == r` CUDACachingAllocator assertion
    (a catchable RuntimeError) or NvMap ENOMEM. Compacting + emptying cache and
    retrying clears the transient/fragmentation cases. NOT a substitute for
    `jetson_preflight.sh` when background services are actually eating the margin
    -- but it hardens the whole stack against the flaky failure at zero cost."""
    import torch
    last = None
    for a in range(1, attempts + 1):
        try:
            return module.to(device)
        except RuntimeError as e:
            last = e
            msg = str(e)
            transient = any(k in msg for k in ("NVML", "NvMap", "CUDACachingAllocator")) \
                or "out of memory" in msg.lower()
            if not transient or a == attempts:
                raise
            print(f"[bmo-startup]   .to(device) hit '{msg[:60]}...' (attempt {a}) -- "
                  "compacting + retrying", flush=True)
            _compact_memory()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    raise RuntimeError("to(device) failed after retries") from last


def build_bmo_stack(
    lfm2_gguf: str,
    lfm2_tok_dir: str,
    minicpm5_gguf: str,
    minicpm5_tok_dir: str,
    m2_ckpt: str,
    m3_ckpt: str,
    speechonly_ckpt: str,
    tts_gguf: str | None = None,
    backchannel_bank_dir: str | None = None,
    query_predictor_ckpt: str | None = None,
    perception_bank_path: str | None = None,
    query_vectors_path: str | None = None,
    identity_ckpt: str | None = None,
    identity_memory_path: str | None = None,
    device: torch.device | None = None,
):
    """Real Jetson init, correct load order enforced. Returns a dict with
    `router` (CognitiveCoreRouter, ready for BmoDuplexTick) and the raw
    perception-stack objects (for wiring a real DuplexLoop/AsyncThinker on
    top, which needs vision/audio context this function doesn't have)."""
    device = device or torch.device("cuda")
    assert torch.cuda.is_available()

    # Task 140: compact memory BEFORE any of the (multi-minute, edge-of-
    # capacity) model loads below -- see _compact_memory()'s docstring.
    _compact_memory()

    # ---- STEP 1: llama.cpp models FIRST -- do not reorder, see module docstring ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, GGUFReasoningTier, CognitiveCoreRouter

    print("[bmo-startup] loading LFM2 fast tier (GGUF)...", flush=True)
    fast_tok = AutoTokenizer.from_pretrained(lfm2_tok_dir)
    fast_tier = _load_gguf_retry(
        lambda: GGUFFastTier(lfm2_gguf, fast_tok, max_new_tokens=24, n_gpu_layers=-1), "LFM2 fast tier")

    # Real finding, 2026-08-07: tried lazy-loading MiniCPM5 here (only
    # construct on actual escalation) expecting LOWER footprint to leave
    # MORE room for perception -- reproducibly made things WORSE (perception
    # stack hit a hard NVML/CUDACachingAllocator crash at wavjepa_nat, twice,
    # from a clean starting state). Eager-loading it here instead -- the
    # exact order verified by scripts/jetson_full_stack_v4_reversed_order.py
    # -- works. Best guess at why: llama.cpp claiming one large contiguous
    # block upfront leaves the remaining address space in a cleaner state
    # for torch's later many-small-allocations quantization pattern than
    # leaving that space nominally "free" but unclaimed. Counter-intuitive,
    # but real and reproducible -- do not "optimize" this back to lazy
    # without re-running the full-stack test.
    print("[bmo-startup] loading Qwen3-0.6B thinker (reasoning tier, GGUF)...", flush=True)
    reason_tok = AutoTokenizer.from_pretrained(minicpm5_tok_dir)
    reasoning_tier = _load_gguf_retry(
        lambda: GGUFReasoningTier(minicpm5_gguf, reason_tok, max_new_tokens=320, n_gpu_layers=-1), "Qwen3-0.6B thinker")

    tts = None
    if tts_gguf:
        # Task 180 (2026-08-07): production voice is now the real streaming
        # path (models/m5_streaming_voice.py::StreamingVoice) -- sampled
        # speech-token generation + seam-free NeuCodec decode, first audio in
        # ~0.3-0.5s -- NOT the old token-only bare-Llama stub (which never
        # decoded to audio and, if greedy, produced 12s runaways). The
        # StreamingVoice ctor carries the same 5x load-retry + compaction
        # reliability wrapper (Task 162) internally via compact_fn. It also
        # loads NeuCodec (bf16-before-cuda) and the espeak phonemizer.
        print("[bmo-startup] loading streaming TTS (NeuTTS GGUF + NeuCodec)...", flush=True)
        from models.m5_streaming_voice import StreamingVoice
        tts = StreamingVoice(tts_gguf, compact_fn=_compact_memory, device=None)
        print("[bmo-startup]   streaming voice ready.", flush=True)

    # Task 137: prebuilt BMO-voiced backchannel/non-verbal-cue bank -- pure
    # file playback (real BMO voice via Fish Audio API, rendered offline;
    # see models.m5_tts.PrebuiltVoiceBank's docstring), no GPU/CPU cost
    # worth sequencing around, loaded here so build_bmo_stack() returns one
    # complete, ready-to-wire stack rather than making every caller
    # remember this extra piece.
    backchannel_bank = None
    try:
        from models.m5_tts import PrebuiltVoiceBank
        backchannel_bank = PrebuiltVoiceBank(bank_dir=backchannel_bank_dir)
        print("[bmo-startup] loaded prebuilt backchannel/non-verbal-cue bank.", flush=True)
    except Exception as e:
        print(f"[bmo-startup]   backchannel bank not loaded: {e!r}", flush=True)

    router = CognitiveCoreRouter(fast_tier, nll_escalate_threshold=3.0, reasoning_tier=reasoning_tier)

    # ---- STEP 2: torch/int8 perception stack SECOND -- do not reorder ----
    print("[bmo-startup] loading perception stack (vision + audio + M2/M3)...", flush=True)
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, device)
    vision_enc.device_str = "cuda"

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, device)
    wavjepa_base.device_str = "cuda"

    # RETIRED 2026-08-16. WavJEPA-nat cost +701 MiB and +326 ms for ZERO measured gain
    # (CLAUDE.md, "Already settled, do not reopen"), yet was still being loaded here every
    # boot -- it is the reason the full stack had only ~280 MiB free and the camera's NVMM
    # allocation failed. build_world_state_features() treats nat_encoder=None as
    # audio_mode="base", the arm that measured 0.608 audio-following, i.e. the same as
    # base+nat. Key kept in the returned dict as None so the older test scripts that read
    # stack["wavjepa_nat"] get None (which they pass straight through) rather than KeyError.
    wavjepa_nat = None

    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    # O1 (2026-08-23, MEASURED): cast to bf16 BEFORE int8 quantization. The M2
    # checkpoint loads in float32, and whatever torchao's Int8WeightOnlyConfig
    # leaves unquantized then stays 4 bytes/param. Measured on-device:
    #   CUDA resident      198 -> 134 MiB   (-64)
    #   CPU transient peak 685 -> 454 MiB   (-231, and the transient is what
    #                                        actually OOMs this box at load)
    # Numerically safe: encode_pre_pool_tokens cosine vs the float32 path is
    # 0.999969 on identical inputs (scripts are in the 2026-08-23 showcase notes).
    # NOTE: do NOT do the same to WavJEPA -- tested and it raises
    # "TypeError: torch.bfloat16 is not supported in MaskedTensor", matching the
    # warning in audio_encoder.py::encode. V-JEPA2 already loads as bf16.
    predictor = predictor.to(torch.bfloat16)
    predictor = q_int8_cpu_then_move(predictor, device)
    predictor.eval()

    # Task 150 (2026-08-07): swapped Whisper-medium -> Moonshine-base for
    # the decision-head-facing STT leg -- real, measured 10-13x encode
    # speedup (277ms -> 20-26ms on this hardware), at a real, measured
    # cost (SpeechOnlyThreeClassHead retrained on Moonshine features scores
    # 90.67% vs the Whisper-based head's 95.00%, see models/m4_speech.py's
    # MoonshineSpeechEncoder docstring for the honest trade-off). The M4b/
    # AsyncThinker deep-grounding path (not currently wired into this
    # production stack at all) still uses WhisperSpeechEncoder if/when
    # that gets built out -- this swap is scoped to what's actually
    # deployed today.
    from models.m4_speech import MoonshineSpeechEncoder
    whisper = MoonshineSpeechEncoder("UsefulSensors/moonshine-base", dtype=torch.bfloat16)
    whisper.encoder = q_int8_cpu_then_move(whisper.encoder, device)

    sys.path.insert(0, os.path.expanduser("~/bmo_production/pipeline"))
    from train_decision_head_3class_speechonly_v2 import SpeechOnlyThreeClassHead
    dh_ckpt = torch.load(speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = _to_device_retry(SpeechOnlyThreeClassHead(speech_feat_dim=dh_ckpt["sf_dim"]), device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    # DROPPED 2026-08-16. The thinker<->perception hookup uses the newer prediction-style
    # integration; the M3 connector is not on any production path (only three test scripts
    # read it) and its .to(device) was historically where boots crashed at the memory edge.
    m3_connector = None

    print("[bmo-startup] DONE -- stack loaded in the verified-safe order.", flush=True)
    # ---- OPTIONAL: perception query engine (thinker -> perception "look" tool) ----
    # Loaded LAST, after every load-order-sensitive model above, and entirely
    # optional: if either path is absent or the load fails, the stack is returned
    # exactly as before with perception_query=None. This is a new capability on a
    # memory-constrained device -- it must never be able to break a boot that
    # previously worked.
    # ---- PERCEPTION QUERY (SigLIP2, text-encoder-free) + IDENTITY -----------------------
    # REWIRED 2026-08-16 from the EmbeddingGemma path. Three reasons:
    #   1. SigLIP2 measurably beat it -- the frozen-target arm went 0.458 -> 0.737 R@1 once
    #      the scene stream and the trained projection existed.
    #   2. NO TEXT ENCODER ON DEVICE. Questions and candidates are both drawn from small
    #      fixed sets, so both are encoded OFFLINE (PreEncodedTextSpace). That removes
    #      EmbeddingGemma's 578 MiB AND SigLIP2's 538 MiB text tower entirely; only the
    #      ~330 MiB vision tower stays resident.
    #   3. Candidates ship RAW (pre-projection) so one candidate file stays valid across
    #      retrains; the checkpoint's co-trained projection is applied HERE, at load.
    # Whole block stays non-fatal: this is new capability on a memory-constrained device and
    # must never break a boot that previously worked.
    perception_query = None
    identity = None
    siglip_vision = None
    if query_predictor_ckpt and perception_bank_path:
        try:
            print("[bmo-startup] loading perception query engine (SigLIP2)...", flush=True)
            import torch.nn.functional as _F
            from models.m5_perception_query import load_perception_query_engine
            from models.text_target import PreEncodedTextSpace

            _ck = torch.load(query_predictor_ckpt, map_location="cpu", weights_only=False)
            _qv = torch.load(query_vectors_path, map_location="cpu", weights_only=False)
            _tt = PreEncodedTextSpace(_qv["text"], _qv["emb"], device=str(device))

            # Apply the checkpoint's trained projection to the RAW candidate bank. Skipping
            # this is the silent geometry mismatch m5_perception_query warns about -- no
            # error, just quietly worse answers.
            _cand = torch.load(perception_bank_path, map_location="cpu", weights_only=False)
            _raw = _F.normalize(_cand["emb"].float(), dim=-1).to(device)
            _tp = _ck.get("text_target_proj") or {}
            if _tp:
                _W = _tp["weight"].float().to(device); _B = _tp["bias"].float().to(device)
                _bank = _F.normalize(_raw @ _W.t() + _B, dim=-1)
            else:
                _bank = _raw
            perception_query = load_perception_query_engine(
                query_predictor_ckpt, _tt, device,
                bank_emb=_bank, bank_text=_cand["text"], max_age_s=15.0)
            perception_query.bank_category = _cand.get(
                "category", ["mined"] * len(_cand["text"]))
            print(f"[bmo-startup] perception query ready (bank={len(_cand['text'])} tags, "
                  f"streams={perception_query.source_names})", flush=True)

            # SigLIP2 VISION tower only -- the  source the predictor was trained with.
            # text_model is deleted: it is never called at inference and costs 538 MiB.
            if "scene" in perception_query.source_names:
                from transformers import AutoModel
                _sig = AutoModel.from_pretrained("google/siglip2-base-patch16-224",
                                                 dtype=torch.bfloat16)
                if hasattr(_sig, "text_model"):
                    del _sig.text_model
                siglip_vision = _sig.to(device).eval()
                print("[bmo-startup]   SigLIP2 vision tower ready (text tower dropped).",
                      flush=True)
        except Exception as e:                       # noqa: BLE001 - never fatal
            print(f"[bmo-startup] perception query SKIPPED ({e!r}) -- stack continues",
                  flush=True)
            perception_query = None

    # IDENTITY -- so BMO can recognise a person instead of inventing one. Loaded separately
    # from the query engine because they fail independently and either is useful alone.
    if identity_ckpt:
        try:
            from models.jepa_identity_head import IdentityHead, IdentityHeadConfig
            from models.jepa_memory import JepaMemory, MemoryConfig
            _ick = torch.load(identity_ckpt, map_location="cpu", weights_only=False)
            _head = IdentityHead(IdentityHeadConfig(in_dims=_ick["dims"],
                                                    emb_dim=_ick["emb_dim"]))
            _head.load_state_dict(_ick["head"])
            _head = _to_device_retry(_head, device); _head.eval()
            _mem = JepaMemory(MemoryConfig(threshold=0.5))
            if identity_memory_path and os.path.exists(identity_memory_path):
                _mem.load(identity_memory_path, device=device)
            identity = {"head": _head, "memory": _mem,
                        "memory_path": identity_memory_path}
            print(f"[bmo-startup] identity head ready ({len(_mem)} enrolled).", flush=True)
        except Exception as e:                       # noqa: BLE001 - never fatal
            print(f"[bmo-startup] identity SKIPPED ({e!r}) -- stack continues", flush=True)
            identity = None

    return {
        "router": router,
        "tts": tts,
        "perception_query": perception_query,
        "backchannel_bank": backchannel_bank,
        "vision_enc": vision_enc,
        "wavjepa_base": wavjepa_base,
        "wavjepa_nat": wavjepa_nat,
        "predictor": predictor,
        "predictor_cfg": predictor_cfg,
        "whisper": whisper,  # NOTE: actually MoonshineSpeechEncoder now (task 150) -- key name kept for caller compat
        "decision_head": decision_head,
        "m3_connector": m3_connector,
        "siglip_vision": siglip_vision,
        "identity": identity,
        "device": device,
    }


if __name__ == "__main__":
    home = os.path.expanduser("~")
    # Task 145 (2026-08-07): all Jetson production paths consolidated under
    # ~/bmo_production/ -- pipeline/ (was ~/jepa_omni_transfer),
    # models_gguf/ (was ~/gguf_models), tokenizers/ (was ~/bmo_stt_test/
    # *_tok), face_engine/ (was ~/BMO-Project), scripts/ (this file + the
    # launch wrapper + preflight). Update here if that tree ever moves.
    prod = f"{home}/bmo_production"
    stack = build_bmo_stack(
        # v9/v7 (2026-08-07): retrained on the mood/energy-coverage-audit
        # corpus (task 138, 1595 lines) -- val_loss improved over v8/v6
        # (LFM2 0.4895->0.4331, MiniCPM5 0.5259->0.4641). Same base
        # tokenizers, no re-transfer needed (LoRA retrain only, vocab
        # unchanged).
        # Fast tier: LFM2.5-350M (2026-08-08) -- retrained on the cleaned BMO
        # corpus, ~half the latency + memory of the old 700M (231 vs 405ms/24tok,
        # 9.4 vs 16.9ms/tok on the Jetson), personality + state-conditioning
        # intact (val_loss 0.457). Revert to 700M by swapping these two paths back.
        lfm2_gguf=f"{prod}/models_gguf/bmo_lfm25_350m_v5_Q8_0.gguf",
        lfm2_tok_dir=f"{prod}/tokenizers/lfm25_350m_tok",
        # Reasoning tier: Qwen3-0.6B thinker (2026-08-08) -- a real System-2
        # reasoner (native <think> CoT, fine-tuned on 448 BMO reasoning +
        # tool-orchestration examples) replacing MiniCPM5-1B (which had reasoning
        # OFF and was just a bigger conversational model, redundant with the fast
        # tier). GGUFReasoningTier strips the <think> block so only the answer is
        # spoken. Revert to MiniCPM5 by swapping these two paths + the tokenizer.
        minicpm5_gguf=f"{prod}/models_gguf/bmo_thinker_qwen3_v5_Q8_0.gguf",
        minicpm5_tok_dir=f"{prod}/tokenizers/qwen3_thinker_tok",
        m2_ckpt=f"{prod}/pipeline/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
        m3_ckpt=f"{prod}/pipeline/checkpoints/m3_multigran_richcaption_v2/last.pt",
        # Task 150 (2026-08-07): retrained on Moonshine-base features
        # (416-dim) to match MoonshineSpeechEncoder above -- NOT
        # cross-compatible with the old Whisper-trained checkpoint
        # (different sf_dim: 416 vs 1024).
        speechonly_ckpt=f"{prod}/pipeline/checkpoints/m4_decision_head_3class_speechonly_moonshine/best.pt",
        # TTS voice: emotion fine-tune (v3) is now the PRODUCTION DEFAULT -- user
        # confirmed Beemo pronunciation + emotion quality on-device (2026-08-08).
        # 12 <|MOOD|> control tokens driven by the homeostatic state; neutral/
        # content anchored on the real BMO recordings. Falls back to the v5 voice
        # only if the emotion GGUF is absent (so reverting = remove that file, or
        # set BMO_TTS_EMOTION=0). StreamingVoice self-detects the tokens either way.
        tts_gguf=(f"{prod}/models_gguf/bmo_neutts_v5_Q8_0.gguf"
                  if os.environ.get("BMO_TTS_EMOTION") == "0"
                  or not os.path.exists(f"{prod}/models_gguf/bmo_neutts_emotion_Q8_0.gguf")
                  else f"{prod}/models_gguf/bmo_neutts_emotion_Q8_0.gguf"),
        backchannel_bank_dir=f"{prod}/pipeline/assets/bmo_backchannels",
        # PERCEPTION QUERY + IDENTITY, enabled in production 2026-08-16. Previously these
        # args were simply never passed, so perception_query was always None and the
        # production stack could produce a world-state vector but nothing that consumed it
        # into an answer -- it could not say WHO or WHAT it was looking at.
        # Candidates ship RAW; the checkpoint's trained projection is applied at load.
        query_predictor_ckpt=f"{prod}/pipeline/checkpoints/qp_runD.pt",
        perception_bank_path=f"{prod}/pipeline/checkpoints/candidates_siglip2_v2.pt",
        query_vectors_path=f"{prod}/pipeline/checkpoints/query_vectors_siglip2_v2.pt",
        identity_ckpt=f"{prod}/pipeline/checkpoints/identity_head_joint.pt",
        # Persisted centroids: without this BMO forgets everyone on reboot (real bug, fixed
        # once already). JepaMemory.load() is a no-op when the file does not exist yet.
        identity_memory_path=f"{prod}/pipeline/checkpoints/bmo_identity_memory.pt",
    )

    print("\n[bmo-startup] smoke test: real fast-tier + reasoning-tier generation", flush=True)
    decision = stack["router"].route("what should we do today?", {"energy": 0.5, "mood": "curious"})
    print(f"  fast: {decision.fast_result.text!r}", flush=True)
    print(f"  {decision.reason}", flush=True)
    if decision.reasoning_result:
        print(f"  reasoning: {decision.reasoning_result.text!r}", flush=True)

    # Task 180: real end-to-end -- speak the fast-tier response through the
    # streaming voice, measuring true time-to-first-audio in the FULL stack.
    if stack.get("tts") is not None:
        import time as _t
        _t0 = _t.perf_counter()
        wav, dur, ttfa = stack["tts"].speak(decision.fast_result.text)
        total = (_t.perf_counter() - _t0) * 1000.0
        print(f"  VOICE: TTFA={ttfa:.0f}ms  audio_dur={dur:.2f}s  full_synth={total:.0f}ms", flush=True)
    print("\nSMOKE_TEST_DONE", flush=True)
