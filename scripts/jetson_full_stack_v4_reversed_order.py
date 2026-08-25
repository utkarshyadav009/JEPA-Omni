"""Variant of jetson_full_stack_v3: reversed load order (llama.cpp models
first -- LLM fast+reasoning tiers + TTS -- THEN the torch/int8 perception
stack) to test whether load ORDER affects fragmentation severity. Real
finding from v3: perception-first order left only single-digit-MiB
contiguous free blocks by the time llama.cpp tried to allocate its own
models."""
from __future__ import annotations

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


def snapshot(tag: str):
    torch.cuda.synchronize()
    tegra = tegra_ram_mib()
    lfb_mib = (tegra["lfb_blocks"] * tegra["lfb_block_size"]) if tegra["lfb_blocks"] is not None else None
    print(f"[v4-reversed] {tag:55s} used={tegra['used_mib']}MiB  lfb~={lfb_mib}MiB", flush=True)
    return tegra


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
    except Exception as e:
        print(f"[v4-reversed]   INT8 FAILED for {tag}: {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize(); gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    return module


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    _ = torch.zeros(1, device=device); torch.cuda.synchronize()
    snapshot("00_cuda_context")

    # ---- llama.cpp models FIRST (reversed order) ----
    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier, GGUFReasoningTier

    fast_tok = AutoTokenizer.from_pretrained("/home/bmo/bmo_stt_test/lfm2_v3_tok")
    fast_tier = GGUFFastTier("/home/bmo/gguf_models/bmo_lfm2_700m_v6_Q8_0.gguf", fast_tok,
                              max_new_tokens=24, n_gpu_layers=-1)
    snapshot("01_llm_fast_tier_lfm2_gguf")

    reason_tok = AutoTokenizer.from_pretrained("/home/bmo/bmo_stt_test/minicpm5_v3_tok")
    reasoning_tier = GGUFReasoningTier("/home/bmo/gguf_models/bmo_minicpm5_v4_Q8_0.gguf", reason_tok,
                                        max_new_tokens=60, n_gpu_layers=-1)
    snapshot("02_llm_reasoning_tier_minicpm5_gguf")

    from llama_cpp import Llama
    tts = Llama(model_path="/home/bmo/gguf_models/neutts-nano-Q8_0.gguf", n_gpu_layers=-1,
                n_ctx=512, verbose=False)
    snapshot("03_tts_neutts_gguf")

    # ---- perception stack (torch) LAST ----
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, "vjepa2_vitl", device)
    snapshot("04_vitl_int8")

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, "wavjepa_base", device)
    snapshot("05_wavjepa_base_int8")

    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, "wavjepa_nat", device)
    snapshot("06_wavjepa_nat_int8")

    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(os.path.expanduser(
        "~/jepa_omni_transfer/checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt"),
        map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, "m2_predictor", device)
    snapshot("07_m2_predictor_int8")

    from transformers import WhisperFeatureExtractor
    from models.m4d_stt_projector import AudioEncoderProjector
    stt_model = AudioEncoderProjector("openai/whisper-base", 1536, device="cuda")
    stt_model.projector = stt_model.projector.to(torch.bfloat16)
    snapshot("08_FULL_STACK_LOADED_reversed_order")

    # ---- real generation pass through BOTH LLM tiers, to catch KV-cache growth ----
    from models.m4_cognitive_core import CognitiveCoreRouter
    router = CognitiveCoreRouter(fast_tier, nll_escalate_threshold=0.3, reasoning_tier=reasoning_tier)
    decision = router.route("what should we do today?", {"energy": 0.5, "mood": "curious"})
    print(f"[v4-reversed] fast={decision.fast_result.text!r}", flush=True)
    if decision.reasoning_result:
        print(f"[v4-reversed] reasoning={decision.reasoning_result.text!r}", flush=True)
    out = tts("Hello there, this is a test.", max_tokens=20)
    print(f"[v4-reversed] tts_ok={out['choices'][0]['text']!r}", flush=True)
    final = snapshot("09_AFTER_REAL_GENERATION_ALL_TIERS_PLUS_TTS")

    total = final["total_mib"]
    used = final["used_mib"]
    print(f"\n[v4-reversed] *** peak={used}MiB / {total}MiB, headroom={(total-used) if used else None}MiB ***", flush=True)


if __name__ == "__main__":
    main()
