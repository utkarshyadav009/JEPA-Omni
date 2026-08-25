"""Full-pipeline memory test on real Jetson hardware, 2026-08-07.

Adapts the earlier jetson_phase4_full_stack_memory_v2_withqwen.py (which
measured perception + M3-grounded Qwen2.5-1.5B generation, peak 6766MiB /
7620MiB, only 854MiB headroom) to reflect what actually changed tonight:
the old single torch/int8 Qwen "Thinker" LLM is not what's being asked
about here -- tonight added a SEPARATE two-tier GGUF LLM system (LFM2 fast
+ MiniCPM5 reasoning, both via llama.cpp) plus a real Ultravox-style STT
projector. This script measures the REALISTIC combined footprint of:
perception (vision+audio+M2, still-active infra, unchanged tonight) + STT
projector (tonight's new addition) + the two-tier GGUF LLM (tonight's new
addition) + TTS (NeuTTS backbone GGUF). Explicitly does NOT load the old
M3-connector+Qwen Thinker path -- that's a separate, still-existing
fallback not the focus of tonight's work; noted as a scoping choice, not
hidden.

llama.cpp allocates its own CUDA memory, NOT tracked by
torch.cuda.memory_allocated() -- tegrastats (system-level) is the only
number that reflects the TRUE combined usage across both allocators.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import types
import importlib.machinery

_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import torch

sys.path.insert(0, "/home/bmo/jepa_omni_transfer")


def tegra_ram_mib() -> dict:
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB \(lfb (\d+)x(\d+)MB\)", line)
    if not m:
        return {"used_mib": None, "total_mib": None, "lfb_blocks": None, "lfb_block_size": None, "raw": line}
    return {"used_mib": int(m.group(1)), "total_mib": int(m.group(2)),
             "lfb_blocks": int(m.group(3)), "lfb_block_size": int(m.group(4)), "raw": line}


def torch_mem_mib() -> dict:
    return {"allocated_mib": torch.cuda.memory_allocated() / 1024**2,
            "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}


_LOG = []
_RESULTS_PATH = None


def _dump():
    if _RESULTS_PATH:
        with open(_RESULTS_PATH, "w") as f:
            json.dump({"log": _LOG, "status": "IN_PROGRESS"}, f, indent=2)


def snapshot(tag: str) -> dict:
    torch.cuda.synchronize()
    tegra = tegra_ram_mib()
    entry = {"tag": tag, "t": time.time(), "tegrastats": tegra, "torch": torch_mem_mib()}
    _LOG.append(entry)
    lfb_mib = (tegra["lfb_blocks"] * tegra["lfb_block_size"]) if tegra["lfb_blocks"] is not None else None
    print(f"[full-stack-v3] {tag:55s} tegrastats_used={tegra['used_mib']}MiB  "
          f"largest_contig_free~={lfb_mib}MiB  torch_alloc={entry['torch']['allocated_mib']:.0f}MiB", flush=True)
    _dump()
    return entry


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, tag, device):
    module = module.to("cpu")
    gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
        print(f"[full-stack-v3]   int8-quantized: {tag}", flush=True)
    except Exception as e:
        print(f"[full-stack-v3]   INT8 FAILED for {tag}: {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize(); gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/jetson_full_stack_v3_results.json"))
    args = p.parse_args()

    global _RESULTS_PATH
    _RESULTS_PATH = args.out

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[full-stack-v3] device={torch.cuda.get_device_name(0)}", flush=True)
    snapshot("00_start")

    _ = torch.zeros(1, device=device); torch.cuda.synchronize()
    snapshot("01_cuda_context")

    # ---- Perception stack: unchanged by tonight's work ----
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, "vjepa2_vitl", device)
    vision_enc.device_str = "cuda"
    snapshot("02_vitl_int8_on_gpu")

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, "wavjepa_base", device)
    wavjepa_base.device_str = "cuda"
    snapshot("03_wavjepa_base_int8_on_gpu")

    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, "wavjepa_nat", device)
    wavjepa_nat.device_str = "cuda"
    snapshot("04_wavjepa_nat_int8_on_gpu")

    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, "m2_predictor", device)
    predictor.eval()
    snapshot("05_m2_predictor_int8_on_gpu")

    # ---- Tonight's new addition #1: STT projector (Ultravox-style fusion) ----
    from transformers import WhisperFeatureExtractor
    from models.m4d_stt_projector import AudioEncoderProjector
    stt_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    stt_llm_hidden = 1536  # LFM2-700M hidden size, matches training
    stt_model = AudioEncoderProjector("openai/whisper-base", stt_llm_hidden, device="cuda")
    stt_model.projector = stt_model.projector.to(torch.bfloat16)
    snapshot("06_stt_projector_whisper_base_on_gpu")

    # ---- Tonight's new addition #2: two-tier GGUF LLM (llama.cpp, separate allocator) ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, LazyGGUFReasoningTier, CognitiveCoreRouter

    fast_tok = AutoTokenizer.from_pretrained("/home/bmo/bmo_stt_test/lfm2_v3_tok")
    fast_tier = GGUFFastTier("/home/bmo/gguf_models/bmo_lfm2_700m_v6_Q8_0.gguf", fast_tok,
                              max_new_tokens=24, n_gpu_layers=-1)
    snapshot("07_llm_fast_tier_lfm2_gguf_on_gpu")

    reason_tok = AutoTokenizer.from_pretrained("/home/bmo/bmo_stt_test/minicpm5_v3_tok")
    reasoning_tier = LazyGGUFReasoningTier("/home/bmo/gguf_models/bmo_minicpm5_v4_Q8_0.gguf", reason_tok,
                                        max_new_tokens=60, n_gpu_layers=-1)
    snapshot("08_llm_reasoning_tier_minicpm5_gguf_on_gpu")

    # ---- Tonight's new addition #3: TTS (NeuTTS backbone, GGUF) ----
    from llama_cpp import Llama
    tts = Llama(model_path="/home/bmo/gguf_models/neutts-nano-Q8_0.gguf", n_gpu_layers=-1,
                n_ctx=512, verbose=False)
    snapshot("09_FULL_STACK_PEAK_perception+stt+two_tier_llm+tts")

    # ---- Real tick: exercise generation on the two-tier system (fast + forced escalation) ----
    router = CognitiveCoreRouter(fast_tier, nll_escalate_threshold=0.3, reasoning_tier=reasoning_tier)
    t0 = time.time()
    decision = router.route("what should we do today?", {"energy": 0.5, "mood": "curious"})
    gen_s = time.time() - t0
    peak = snapshot("10_AFTER_REAL_GENERATION_fast_and_reasoning")
    peak["generation_latency_s"] = gen_s
    peak["fast_text"] = decision.fast_result.text
    peak["reasoning_text"] = decision.reasoning_result.text if decision.reasoning_result else None
    peak["escalated"] = decision.escalate_to_thinker

    total = peak["tegrastats"]["total_mib"]
    used = peak["tegrastats"]["used_mib"]
    headroom = (total - used) if used is not None else None
    lfb = peak["tegrastats"]["lfb_blocks"]
    lfb_size = peak["tegrastats"]["lfb_block_size"]
    lfb_mib = (lfb * lfb_size) if lfb is not None else None
    print(f"\n[full-stack-v3] *** FULL STACK (perception+STT-projector+two-tier-LLM+TTS): "
          f"peak={used}MiB / {total}MiB total, headroom={headroom}MiB, "
          f"largest_contiguous_free~={lfb_mib}MiB ***", flush=True)

    with open(args.out, "w") as f:
        json.dump({"log": _LOG, "status": "DONE", "peak_used_mib": used, "total_mib": total,
                   "headroom_mib": headroom, "largest_contig_free_mib": lfb_mib}, f, indent=2)
    print(f"[full-stack-v3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
