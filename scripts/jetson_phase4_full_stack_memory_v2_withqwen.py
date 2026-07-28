"""scripts/jetson_phase4_full_stack_memory_v2_withqwen.py — item 4: the
LIKE-FOR-LIKE full-stack memory test, closing the gap flagged in
scripts/jetson_phase4_full_stack_memory.py's correction note. That script's
4569MiB peak / 3051MiB headroom is real for ITS stack, but is missing
Qwen2.5-1.5B-Instruct (the M3 connector's generation LLM, ~1310MiB alone)
and a real generation pass's KV-cache growth -- exactly what the ORIGINAL
644MiB PHASE0_CLARIFICATION figure included. Until THIS script runs, "fits
on Jetson" is unverified for the corrected (pooled, 512-token) World-State
path used together with real generation.

Stack: ViT-L + WavJEPA-base + WavJEPA-nat + M2 predictor + Whisper-medium +
Qwen2.5-1.5B-Instruct + M3 connector (checkpoints/m4_joint/best.pt) +
speech-only decision head, all int8 weight-only where the tooling allows
(WavJEPA stays fp32-internal per its MaskedTensor constraint, same as
Phase 0). One real tick: corrected pooled World-State refresh (build_
world_state_features) + M3-grounded soft prompt + a real 60-token greedy
generation through Qwen (DuplexLoop.generate_interruptible). Reports peak
tegrastats usage and headroom against the 8GB (7620MiB usable) device.

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
    print(f"[phase4-v2] {tag:55s} tegrastats_used={entry['tegrastats']['used_mib']}MiB  "
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
        print(f"[phase4-v2]   int8-quantized: {tag}", flush=True)
    except Exception as e:
        print(f"[phase4-v2]   INT8 FAILED for {tag}: {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize(); gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_fusion_20k_best/step19000_peak.pt"))
    p.add_argument("--m4-joint-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_joint/best.pt"))
    p.add_argument("--speechonly-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class_speechonly/best.pt"))
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--out", default=os.path.expanduser("~/jetson_phase4_memory_v2_withqwen_results.json"))
    args = p.parse_args()

    global _RESULTS_PATH
    _RESULTS_PATH = args.out

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[phase4-v2] device={torch.cuda.get_device_name(0)}", flush=True)
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

    # ---- Qwen2.5-1.5B-Instruct (the piece the prior measurement lacked) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)
    llm = q_int8_cpu_then_move(llm, "qwen2.5-1.5b-instruct", device)
    llm.eval()
    snapshot("08_qwen25_1p5b_int8_on_gpu")

    # ---- M3 connector (checkpoints/m4_joint/best.pt) ----
    from models.m3_connector import M3Connector, M3ConnectorConfig
    m4joint_ckpt = torch.load(args.m4_joint_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m4joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m4joint_ckpt["m3_connector"])
    m3_connector.eval()
    snapshot("09_m3_connector_on_gpu")

    from models.m4_duplex_loop import DuplexLoop
    duplex = DuplexLoop(predictor, m3_connector, None, whisper, decision_head, llm, tokenizer, device)

    # ---- one real tick: corrected pooled World-State + M3 soft prompt + real generation ----
    torch.cuda.reset_peak_memory_stats()
    frames = (torch.rand(64, 3, 256, 256) * 255).to(torch.uint8)
    audio = torch.randn(160000)  # 10s @ 16kHz
    from models.world_state_builder import build_world_state_features
    MAX_AMBIENT_T = 1024

    t0 = time.time()
    result = build_world_state_features(frames, audio, 10.0, vision_enc, wavjepa_base, wavjepa_nat,
                                         predictor_cfg.max_tdm_bins, device)
    with torch.no_grad():
        pre_pool = predictor.encode_pre_pool_tokens(result.feats, result.tbins)
        if pre_pool.shape[1] > MAX_AMBIENT_T:
            pre_pool = pre_pool[:, :MAX_AMBIENT_T]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            soft_prompt = m3_connector(pre_pool)  # (1, n_latents, llm_hidden)
    perception_s = time.time() - t0
    snapshot("10_PERCEPTION_vitl+wavjepa_base+wavjepa_nat+fusion+m3connector")

    attn = torch.ones(1, soft_prompt.shape[1], dtype=torch.long, device=device)
    # Memory-characterization measurement, not a quality test: the soft
    # prompt here is built from RANDOM dummy vision/audio (no real content
    # to ground on), so Qwen tends to hit EOS after only a few tokens --
    # that would understate real KV-cache growth relative to a genuine
    # full-length turn. Temporarily disable the EOS early-stop so this
    # specific measurement reflects the full max_new_tokens budget (the
    # ORIGINAL 644MiB PHASE0_CLARIFICATION figure was measured over a
    # genuine 60-token generation; this must be like-for-like on THAT
    # axis too, not just on which components are loaded).
    # duplex.tokenizer.eos_token_id is read fresh each generation step
    # (models/m4_duplex_loop.py); HF's tokenizer setter validates any
    # assigned id via convert_ids_to_tokens (overflows on an out-of-range
    # sentinel like -999999), so wrap in a thin proxy instead of mutating
    # the real tokenizer's property.
    class _NoEosProxy:
        def __init__(self, real):
            self._real = real
            self.eos_token_id = -1  # plain attribute, no validation, never equals a real token id
        def __getattr__(self, name):
            return getattr(self._real, name)
    duplex.tokenizer = _NoEosProxy(tokenizer)
    t1 = time.time()
    result_gen = duplex.generate_interruptible(soft_prompt.float().to(torch.bfloat16), attn,
                                                max_new_tokens=args.max_new_tokens)
    generation_s = time.time() - t1
    duplex.tokenizer = tokenizer
    peak = snapshot("11_FULL_TICK_PEAK_perception+generation")
    peak["perception_latency_s"] = perception_s
    peak["generation_latency_s"] = generation_s
    peak["n_tokens_generated"] = result_gen.n_tokens_generated
    peak["generated_text"] = result_gen.text

    total = 7620
    used = peak["tegrastats"]["used_mib"]
    headroom = (total - used) if used is not None else None
    print(f"\n[phase4-v2] *** LIKE-FOR-LIKE FULL STACK (incl. Qwen2.5-1.5B, real generation): "
          f"peak={used}MiB / {total}MiB total, headroom={headroom}MiB ***", flush=True)
    print(f"[phase4-v2] perception={perception_s:.2f}s  generation={generation_s:.2f}s "
          f"({result_gen.n_tokens_generated} tokens)", flush=True)

    with open(args.out, "w") as f:
        json.dump({"log": _LOG, "status": "DONE", "peak_used_mib": used, "total_mib": total,
                   "headroom_mib": headroom}, f, indent=2)
    print(f"[phase4-v2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
