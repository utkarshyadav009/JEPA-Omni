"""scripts/jetson_phase4_full_stack_memory.py — Phase 4.1: profile the
stack ViT-L + WavJEPA-base + WavJEPA-nat + M2 predictor + Whisper +
speech-only decision head. Real tegrastats ground truth, same methodology
as Phase 0 (q_int8_cpu_then_move, malloc_trim, snapshot()).

CORRECTION (2026-07-26, diagnostic follow-up): this docstring previously
claimed the 644MiB Phase-0/clarification headroom figure was measured on a
stack "containing NEITHER WavJEPA model" -- FALSE, retracted. Direct
re-check of checkpoints/m5_jetson/PHASE0_CLARIFICATION_PROVENANCE.txt's own
steady-state table shows WavJEPA-base and WavJEPA-nat WERE both loaded in
that measurement. The actual component THIS script is missing relative to
that one is Qwen2.5-1.5B (the M3 connector's generation LLM, ~1310MiB
alone) plus its KV-cache growth from a real 60-token generation -- this
script never imports or loads an LLM at all. Its 4569MiB peak / 3051MiB
headroom is a real number for THIS (Qwen-less) stack, but is NOT a
like-for-like comparison against the 644MiB figure. See
checkpoints/falsifier_tracking.md's 2026-07-26 diagnostic entry.

Run ON the Jetson, after jetson_preflight.sh PASS.
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

# Side effect discovered this session: installing torchaudio (for the
# Phase 3.3 CPU-VAD fix) is ABI-mismatched with this Jetson's torch 2.8.0
# build, and BOTH silero_vad AND transformers (audio_utils.py) do an
# unconditional `import torchaudio` -- breaking transformers imports
# machine-wide until stubbed. Same workaround as jetson_vad_cpu_latency.py.
_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tegra_ram_mib() -> dict:
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if not m:
        return {"used_mib": None, "total_mib": None, "raw": line}
    return {"used_mib": int(m.group(1)), "total_mib": int(m.group(2)), "raw": line}


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
    entry = {"tag": tag, "t": time.time(), "tegrastats": tegra_ram_mib(), "torch": torch_mem_mib()}
    _LOG.append(entry)
    print(f"[phase4] {tag:45s} tegrastats_used={entry['tegrastats']['used_mib']}MiB  "
          f"torch_alloc={entry['torch']['allocated_mib']:.0f}MiB", flush=True)
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
        print(f"[phase4]   int8-quantized: {tag}", flush=True)
    except Exception as e:
        print(f"[phase4]   INT8 FAILED for {tag}: {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize(); gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_fusion_20k_best/step19000_peak.pt"))
    p.add_argument("--speechonly-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class_speechonly/best.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/jetson_phase4_memory_results.json"))
    args = p.parse_args()

    global _RESULTS_PATH
    _RESULTS_PATH = args.out

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[phase4] device={torch.cuda.get_device_name(0)}", flush=True)
    snapshot("00_start")

    _ = torch.zeros(1, device=device); torch.cuda.synchronize()
    snapshot("01_cuda_context")

    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, "vjepa2_vitl", device)
    vision_enc.device_str = "cuda"
    snapshot("02_vitl_int8_on_gpu")

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, "wavjepa_base", device)
    wavjepa_base.device_str = "cuda"
    snapshot("03_wavjepa_base_int8_on_gpu (NEVER MEASURED BEFORE)")

    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, "wavjepa_nat", device)
    wavjepa_nat.device_str = "cuda"
    snapshot("04_wavjepa_nat_int8_on_gpu (NEVER MEASURED BEFORE)")

    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, "m2_predictor", device)
    predictor.eval()
    snapshot("05_m2_predictor_int8_on_gpu")

    from models.m4_speech import WhisperSpeechEncoder
    whisper = WhisperSpeechEncoder("openai/whisper-medium", dtype=torch.bfloat16)
    whisper.encoder = q_int8_cpu_then_move(whisper.encoder, "whisper", device)
    snapshot("06_whisper_int8_on_gpu")

    sys.path.insert(0, os.path.expanduser("~/jepa_omni_transfer"))
    from train_decision_head_3class_speechonly import SpeechOnlyThreeClassHead
    ckpt = torch.load(args.speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = SpeechOnlyThreeClassHead(speech_feat_dim=ckpt["sf_dim"]).to(device)
    decision_head.load_state_dict(ckpt["state_dict"])
    decision_head.eval()
    snapshot("07_speechonly_decision_head_on_gpu")

    # ---- full refresh cost: ViT-L + BOTH WavJEPA + fusion, one real pass ----
    torch.cuda.reset_peak_memory_stats()
    frames = (torch.rand(64, 3, 256, 256) * 255).to(torch.uint8)
    audio = torch.randn(160000)
    from models.world_state_builder import build_world_state_features
    t0 = time.time()
    result = build_world_state_features(frames, audio, 10.0, vision_enc, wavjepa_base, wavjepa_nat,
                                         predictor_cfg.max_tdm_bins, device)
    with torch.no_grad():
        ws = predictor.encode_world_state(result.feats, result.tbins)
    full_refresh_s = time.time() - t0
    peak = snapshot("08_FULL_REFRESH_vitl+wavjepa_base+wavjepa_nat+fusion")
    peak["full_refresh_latency_s"] = full_refresh_s
    print(f"[phase4] *** FULL REFRESH (ViT-L+WavJEPA-base+WavJEPA-nat+fusion): "
          f"{full_refresh_s:.2f}s, tegrastats_peak={peak['tegrastats']['used_mib']}MiB / "
          f"{peak['tegrastats']['total_mib']}MiB total ***", flush=True)

    with open(args.out, "w") as f:
        json.dump({"log": _LOG, "status": "DONE"}, f, indent=2)
    print(f"[phase4] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
