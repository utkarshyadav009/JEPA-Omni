"""scripts/jetson_phase0_memory.py — M5 Phase 0: measure the Jetson Orin
Nano Super (8GB unified LPDDR5) memory wall BEFORE building anything.

Run ON the Jetson, after tools/jetson_preflight.sh (from the moshi_oracle
project, reused here — see PROVENANCE) has PASSED, so the baseline is a
known-clean, non-fragmented starting point.

Loads the full pipeline progressively (WavJEPA base+nat, V-JEPA2 ViT-L,
Whisper-medium, Qwen2.5-1.5B-Instruct, M2 fusion predictor, M3 connector,
M4b projector, 3-class decision head), applying torchao int8 WEIGHT-ONLY
quantization to each component after loading (verified functional on this
box despite the cpp-extension warning -- see PROVENANCE). Honest labeling:
this is weight-only int8 (activations stay bf16/fp32) -- standard, real,
and does NOT shrink activation-memory peaks, only resident weight memory.
WavJEPA is loaded fp32 (existing repo constraint: MaskedTensor doesn't
support bf16) and int8-quantized from fp32 weights.

Two memory signals are recorded at every step, per the same discipline as
jetson_preflight.sh: tegrastats RAM (ground truth -- Jetson is unified
memory, so this IS both "system" and "GPU" memory) AND torch's own
allocator stats (finer-grained, useful for isolating per-component cost,
but known to undercount driver/context/fragmentation overhead -- tegrastats
is the number that decides whether something actually fits).

Usage (on the Jetson, after preflight PASS):
    python3 jetson_phase0_memory.py --ckpt-dir /home/bmo/jepa_omni_transfer
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

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tegra_ram_mib() -> dict:
    """One tegrastats sample -> {'used_mib':..., 'total_mib':...}. Ground
    truth for this platform (unified memory, no separate VRAM query)."""
    out = subprocess.run(["timeout", "3", "tegrastats", "--interval", "300"],
                          capture_output=True, text=True, timeout=10).stdout
    line = out.strip().split("\n")[0] if out.strip() else ""
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if not m:
        return {"used_mib": None, "total_mib": None, "raw": line}
    return {"used_mib": int(m.group(1)), "total_mib": int(m.group(2)), "raw": line}


def torch_mem_mib() -> dict:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


_RESULTS_PATH = None


def _dump_incremental(log: list) -> None:
    """Write partial results after EVERY snapshot, not just at the end --
    a crash (e.g. a real NvMap OOM, which is exactly what this script is
    trying to find) must not destroy the data collected up to that point."""
    if _RESULTS_PATH is None:
        return
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump({"log": log, "status": "IN_PROGRESS"}, f, indent=2)


def snapshot(tag: str, log: list) -> dict:
    torch.cuda.synchronize()
    entry = {"tag": tag, "t": time.time(), "tegrastats": tegra_ram_mib(), "torch": torch_mem_mib()}
    log.append(entry)
    print(f"[phase0] {tag:40s} tegrastats_used={entry['tegrastats']['used_mib']}MiB  "
          f"torch_alloc={entry['torch']['allocated_mib']:.0f}MiB  "
          f"torch_max_alloc={entry['torch']['max_allocated_mib']:.0f}MiB", flush=True)
    _dump_incremental(log)
    return entry


def _malloc_trim():
    """Force glibc to actually return freed CPU pages to the OS.
    gc.collect() alone drops Python references, but glibc malloc keeps
    freed arenas around for reuse rather than calling munmap() -- on a
    process that loads/frees several ~1GB+ models in sequence (this
    script), that retained-but-unused memory is real, tegrastats-visible,
    and NOT reclaimed by gc.collect() alone. Confirmed necessary here:
    without this, the v1->v2 torchao fix's per-model savings (real and
    large in an isolated single-model test) mostly failed to show up in
    the full multi-model pipeline's tegrastats numbers."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module: nn.Module, tag: str, device) -> nn.Module:
    """Quantize on CPU BEFORE moving to GPU -- the memory-realistic strategy
    for an 8GB unified-memory device: the full bf16/fp32 footprint should
    never have to coexist with its int8 copy on the GPU. This is also the
    correct edge-deployment pattern, not just a measurement trick.
    Returns the module (now on `device`, int8-quantized where possible)."""
    module = module.to("cpu")
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        # version=2 is REQUIRED, not cosmetic: version=1 (the default) never
        # frees the pre-quantization fp32/bf16 tensor after replacing it --
        # confirmed directly (models/... diag, see PROVENANCE) -- a 20x
        # Linear(4096,4096) test stayed at the FULL fp32 footprint (+1281MiB)
        # after v1 quantize+gc, vs the correct ~309MiB (matching the int8
        # theoretical size) under v2. Every earlier Phase-0 "int8" number
        # was inflated by roughly 2x on the Linear-heavy portion because of
        # this.
        quantize_(module, Int8WeightOnlyConfig(version=2))
        print(f"[phase0]   int8-quantized on CPU: {tag}", flush=True)
    except Exception as e:
        print(f"[phase0]   INT8 QUANTIZATION FAILED for {tag}: {e!r} -- moving at original dtype", flush=True)
    gc.collect()
    _malloc_trim()
    module = module.to(device)
    torch.cuda.synchronize()
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints"))
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default=os.path.expanduser("~/jepa_omni_transfer/phase0_results.json"))
    p.add_argument("--skip-quant", action="store_true", help="skip int8 -- bf16/fp32 baseline only")
    args = p.parse_args()

    global _RESULTS_PATH
    _RESULTS_PATH = args.out

    assert torch.cuda.is_available(), "CUDA not available -- nothing to measure"
    device = torch.device("cuda")
    log = []

    print(f"[phase0] device={torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)
    snapshot("00_process_start_no_cuda_context", log)

    # Force CUDA context creation (this alone costs real memory on Jetson --
    # driver + context overhead, before a single model weight is loaded)
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize()
    snapshot("01_cuda_context_created", log)

    # ---------- WavJEPA base + nat (fp32 -- MaskedTensor constraint) ----------
    # Loaded on CPU, quantized on CPU, THEN moved to GPU -- the full fp32
    # footprint must never coexist with its int8 copy in the 7.4GB budget.
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    if not args.skip_quant:
        wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, "wavjepa_base", device)
    else:
        wavjepa_base.model = wavjepa_base.model.to(device)
    wavjepa_base.device_str = "cuda"
    snapshot("02_wavjepa_base_int8_on_gpu", log)

    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    if not args.skip_quant:
        wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, "wavjepa_nat", device)
    else:
        wavjepa_nat.model = wavjepa_nat.model.to(device)
    wavjepa_nat.device_str = "cuda"
    snapshot("03_wavjepa_nat_int8_on_gpu", log)

    # ---------- V-JEPA2 ViT-L (the flagged prime suspect) ----------
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    if not args.skip_quant:
        vision_enc.model = q_int8_cpu_then_move(vision_enc.model, "vjepa2_vitl", device)
    else:
        vision_enc.model = vision_enc.model.to(device)
    vision_enc.device_str = "cuda"
    snapshot("04_vjepa2_vitl_int8_on_gpu", log)

    # isolate: ONE 64f/256px forward pass, peak activation memory
    torch.cuda.reset_peak_memory_stats()
    B, T, C, H, W = 1, 64, 3, 256, 256
    dummy_video = (torch.rand(B, T, C, H, W) * 255).to(torch.uint8)
    t0 = time.time()
    with torch.no_grad():
        feats = vision_enc.encode(dummy_video)
    torch.cuda.synchronize()
    vit_forward_s = time.time() - t0
    vit_isolated_peak = snapshot("05_VJEPA2_VITL_ISOLATED_FORWARD_64f_256px", log)
    vit_isolated_peak["forward_latency_s"] = vit_forward_s
    vit_isolated_peak["output_shape"] = list(feats.shape)
    print(f"[phase0] *** V-JEPA2 ViT-L isolated forward: {vit_forward_s:.2f}s, "
          f"peak_torch_alloc={vit_isolated_peak['torch']['max_allocated_mib']:.0f}MiB, "
          f"tegrastats_used={vit_isolated_peak['tegrastats']['used_mib']}MiB ***", flush=True)
    del dummy_video, feats
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- Whisper-medium (speech-activity feature encoder) ----------
    # WhisperSpeechEncoder builds on CPU by default (no device arg) --
    # quantize BEFORE the caller's .to(device), same CPU-first strategy.
    from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16)
    if not args.skip_quant:
        whisper = q_int8_cpu_then_move(whisper, "whisper_medium", device)
    else:
        whisper = whisper.to(device)
    snapshot("06_whisper_medium_int8_on_gpu", log)

    # ---------- Qwen2.5-1.5B-Instruct (frozen LLM) ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)   # stays on CPU
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)
    if not args.skip_quant:
        llm = q_int8_cpu_then_move(llm, "qwen1.5b", device)
    else:
        llm = llm.to(device)
    snapshot("07_qwen1.5b_int8_on_gpu", log)

    # ---------- M2 fusion predictor (our trained checkpoint) ----------
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(os.path.join(args.ckpt_dir, "m2_fusion_20k_best/step19000_peak.pt"),
                         map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    del m2ckpt
    snapshot("08_m2_predictor_loaded_fp32", log)

    # ---------- M3 connector + M4b projector (our trained checkpoint) ----------
    from models.m3_connector import M3Connector, M3ConnectorConfig
    joint_ckpt = torch.load(os.path.join(args.ckpt_dir, "m4_joint/best.pt"),
                             map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(joint_ckpt["m3_connector"])
    m3_connector.eval()
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()
    del joint_ckpt
    snapshot("09_m3_connector_m4b_projector_loaded", log)

    # ---------- decision head (tiny) ----------
    from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig
    dh_ckpt = torch.load(os.path.join(args.ckpt_dir, "m4_decision_head_3class/best.pt"),
                          map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = ThreeClassHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()
    del dh_ckpt
    steady_state = snapshot("10_FULL_STACK_LOADED_STEADY_STATE", log)

    # ---------- ONE full inference tick: perception + decision + generation ----------
    # CONTINUOUS tegrastats sampling spans this whole block (not just a
    # before/after snapshot) so the reported peak is provably DURING
    # inference -- specifically during llm.generate() with the KV-cache
    # populated -- not a resident-after-load number. A single long-running
    # `tegrastats --interval` subprocess is read continuously in a
    # background thread; each line gets a wall-clock read timestamp so the
    # peak sample can be checked against explicit t_gen_start/t_gen_end
    # markers recorded around the generate() call itself.
    import threading

    tegra_samples = []   # list of (read_time, used_mib)
    stop_sampling = threading.Event()

    def _sampler():
        proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        try:
            while not stop_sampling.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                t_read = time.time()
                m = re.search(r"RAM (\d+)/(\d+)MB", line)
                if m:
                    tegra_samples.append((t_read, int(m.group(1))))
        finally:
            proc.terminate()

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()
    time.sleep(0.3)   # let the sampler get its first reading before the tick starts

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    t_gen_start = None
    t_gen_end = None
    with torch.no_grad():
        dummy_video = (torch.rand(1, 64, 3, 256, 256) * 255).to(torch.uint8)
        v_feats = vision_enc.encode(dummy_video)  # (1, N, 1024) -- ViT-L 64f forward, live

        wav_mono = torch.zeros(1, 1, 16000 * 2)
        wav_bin = torch.zeros(1, 2, 16000 * 2)
        a_base = wavjepa_base.encode(wav_mono)
        a_nat = wavjepa_nat.encode(wav_bin)

        import numpy as np
        speech_wav = [np.zeros(16000 * 2, dtype=np.float32)]
        hidden, valid_frames = whisper(speech_wav, [2.0], device)
        vf = int(valid_frames[0].item())
        pooled_speech = hidden[0, :vf].float().mean(dim=0, keepdim=True)

        # World-State needs feats/tbins in AVJepaPredictor's expected format --
        # use zero-dim placeholders sized to match a short real tick (mechanism
        # test only, not a real clip) to get a genuine forward-pass memory cost
        zero_ws = torch.zeros(1, dh_cfg.world_state_dim, device=device)
        logits = decision_head(zero_ws, pooled_speech)
        pred_label = int(logits.argmax(dim=-1).item())

        prompt_ids = tokenizer("Testing.", return_tensors="pt")["input_ids"].to(device)
        soft_prompt = llm.get_input_embeddings()(prompt_ids)
        attn = torch.ones(soft_prompt.shape[:2], dtype=torch.long, device=device)
        t_gen_start = time.time()
        # max_new_tokens=60 (not 20) so the KV-cache actually grows to a
        # meaningful size and the sampler has enough wall-clock time inside
        # the generation window to catch multiple readings, not just one
        gen_ids = llm.generate(inputs_embeds=soft_prompt, attention_mask=attn, max_new_tokens=60,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
        t_gen_end = time.time()
    torch.cuda.synchronize()
    tick_latency_s = time.time() - t0
    time.sleep(0.3)   # let the sampler grab a final reading past the tick
    stop_sampling.set()
    sampler_thread.join(timeout=2)

    tick_peak = snapshot("11_FULL_TICK_AFTER_perception+decision+generation", log)
    tick_peak["tick_latency_s"] = tick_latency_s

    # find the max CONTINUOUS-sampled reading and confirm it falls inside
    # [t_gen_start, t_gen_end] -- this is the actual proof the reported
    # peak occurred DURING live generation with the KV-cache populated,
    # not at some other point (load, or after the tick ended)
    if tegra_samples:
        peak_t, peak_used = max(tegra_samples, key=lambda x: x[1])
        gen_window_samples = [(t, u) for t, u in tegra_samples if t_gen_start <= t <= t_gen_end]
        peak_during_gen = max(gen_window_samples, key=lambda x: x[1]) if gen_window_samples else None
        continuous_report = {
            "n_samples": len(tegra_samples),
            "overall_peak_used_mib": peak_used,
            "overall_peak_at_t_minus_gen_start_s": peak_t - t_gen_start,
            "peak_is_inside_generation_window": t_gen_start <= peak_t <= t_gen_end,
            "generation_window_s": t_gen_end - t_gen_start,
            "n_samples_inside_generation_window": len(gen_window_samples),
            "peak_used_mib_WITHIN_generation_window": peak_during_gen[1] if peak_during_gen else None,
            "all_samples_used_mib": [u for _, u in tegra_samples],
        }
    else:
        continuous_report = {"error": "no tegrastats samples collected"}

    print(f"\n[phase0] === CONTINUOUS SAMPLING DURING FULL TICK (proves peak is live, not resident-after-load) ===")
    print(json.dumps({k: v for k, v in continuous_report.items() if k != "all_samples_used_mib"}, indent=2))
    print(f"[phase0] *** FULL TICK: {tick_latency_s:.2f}s total, generation alone={t_gen_end-t_gen_start:.2f}s, "
          f"OVERALL PEAK (continuous sampling)={continuous_report.get('overall_peak_used_mib')}MiB, "
          f"peak inside generation window: {continuous_report.get('peak_is_inside_generation_window')} ***", flush=True)
    _dump_incremental(log)
    with open(_RESULTS_PATH, "r") as f:
        _partial = json.load(f)
    _partial["continuous_tick_sampling"] = continuous_report
    with open(_RESULTS_PATH, "w") as f:
        json.dump(_partial, f, indent=2)

    # ---------- KV-cache growth headroom (multi-turn) ----------
    # Qwen2.5-1.5B: 28 layers, 2 KV heads (GQA), head_dim=128 -- per published config.
    # bytes/token = 2 (K+V) * n_layers * n_kv_heads * head_dim * dtype_bytes
    cfg = llm.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None))
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = 2  # bf16
    bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    mib_per_1k_tokens = bytes_per_token * 1000 / 1024**2

    steady_used = steady_state["tegrastats"]["used_mib"]
    total_mib = steady_state["tegrastats"]["total_mib"]
    headroom_mib = total_mib - steady_used

    kv_report = {
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "bytes_per_token": bytes_per_token, "mib_per_1k_tokens": mib_per_1k_tokens,
        "steady_state_used_mib": steady_used, "total_mib": total_mib,
        "raw_headroom_mib": headroom_mib,
        "max_kv_tokens_at_raw_headroom": int(headroom_mib / mib_per_1k_tokens * 1000) if mib_per_1k_tokens > 0 else None,
    }
    print(f"\n[phase0] === KV-CACHE HEADROOM ===")
    print(json.dumps(kv_report, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"log": log, "kv_cache_report": kv_report, "continuous_tick_sampling": continuous_report,
                   "quantized": not args.skip_quant,
                   "status": "COMPLETE"}, f, indent=2)
    print(f"\n[phase0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("[phase0] CRASHED -- partial results (if any snapshots were taken) are still in the "
              "output file from incremental dumps; this crash IS itself a Phase-0 data point.", flush=True)
        raise
