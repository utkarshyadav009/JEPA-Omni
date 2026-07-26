"""scripts/jetson_phase4_2_3_streaming.py — Phase 4.2 + 4.3: real Jetson
streaming-loop measurement with the CORRECTED World-State construction
(vision pooling + staircase tbins + real WavJEPA-base/nat ambient, all
verified in Phase 1.2/2.1), stride_vision_sec=window_vision_sec=10.0 (item
5's decision, licensed by the drift curve), and a priority CUDA stream for
the decision path (item 2b).

Run ON the Jetson, after jetson_preflight.sh PASS.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import types
import importlib.machinery

_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def q_int8_cpu_then_move(module, device):
    module = module.to("cpu")
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[verify] int8 quant failed: {e!r}", flush=True)
    gc.collect()
    module = module.to(device)
    torch.cuda.synchronize()
    return module


def stats(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(xs), "mean_ms": statistics.mean(xs), "max_ms": max(xs),
            "p50_ms": s[len(s) // 2], "p95_ms": s[int(len(s) * 0.95)]}


def run_ticks(stream, cfg, duration_sec, decision_stream=None):
    def make_dummy_frame():
        return (torch.rand(3, 256, 256) * 255).to(torch.uint8)

    audio = (np.random.randn(16000 * 20) * 0.01).astype(np.float32)
    tick_wall_ms = []
    n_ticks = int(duration_sec / cfg.tick_interval_sec)
    real_t0 = time.time()
    t_sim = 0.0
    audio_pos = 0
    samples_per_tick = int(cfg.tick_interval_sec * cfg.audio_sr)
    stream.start_vision_refresh_thread(hz=1.0 / cfg.stride_vision_sec)

    ws_ages = []
    for i in range(n_ticks):
        stream.ingest_video_frame(make_dummy_frame())
        chunk = audio[audio_pos: audio_pos + samples_per_tick]
        if len(chunk) < samples_per_tick:
            audio_pos = 0
            chunk = audio[:samples_per_tick]
        else:
            audio_pos += samples_per_tick
        stream.ingest_audio_chunk(chunk)

        speech_window = stream.audio_buf.get_window()
        speech_dur = len(speech_window) / cfg.audio_sr if speech_window is not None else 0.0

        with stream._ws_lock:
            last_t = stream._last_vision_refresh_t
        if last_t is not None:
            ws_ages.append(time.time() - last_t)

        t0 = time.perf_counter()
        if decision_stream is not None:
            with torch.cuda.stream(decision_stream):
                log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur, generate_fn=None)
            torch.cuda.current_stream().wait_stream(decision_stream)
        else:
            log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur, generate_fn=None)
        tick_wall_ms.append((time.perf_counter() - t0) * 1000.0)

        t_sim += cfg.tick_interval_sec
        real_elapsed = time.time() - real_t0
        sleep_left = t_sim - real_elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)

    stream.stop_vision_refresh_thread()
    return tick_wall_ms, ws_ages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=float, default=30.0)
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_fusion_20k_best/step19000_peak.pt"))
    p.add_argument("--speechonly-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class_speechonly/best.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/jetson_phase4_2_3_results.json"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[phase4.2-3] device={torch.cuda.get_device_name(0)}", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from models.m4_speech import WhisperSpeechEncoder
    from models.m4_duplex_loop import DuplexLoop
    from models.m5_streaming_loop import StreamingLoop, StreamingConfig
    from train_decision_head_3class_speechonly import SpeechOnlyThreeClassHead

    print("[phase4.2-3] loading real ViT-L, WavJEPA-base, WavJEPA-nat, M2, Whisper (all int8)...", flush=True)
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, device)
    vision_enc.device_str = "cuda"

    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    base_enc.model = q_int8_cpu_then_move(base_enc.model, device)
    base_enc.device_str = "cuda"

    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    nat_enc.model = q_int8_cpu_then_move(nat_enc.model, device)
    nat_enc.device_str = "cuda"

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, device)
    predictor.eval()

    whisper = WhisperSpeechEncoder("openai/whisper-medium", dtype=torch.bfloat16)
    whisper.encoder = q_int8_cpu_then_move(whisper.encoder, device)

    ckpt = torch.load(args.speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = SpeechOnlyThreeClassHead(speech_feat_dim=ckpt["sf_dim"]).to(device)
    decision_head.load_state_dict(ckpt["state_dict"])
    decision_head.eval()

    duplex = DuplexLoop(predictor, None, None, whisper, decision_head, None, None, device)

    all_results = {}

    # ---- 4.2: stride = window = 10.0s ----
    print("\n[phase4.2] stride_vision_sec = window_vision_sec = 10.0 ...", flush=True)
    cfg = StreamingConfig(stride_vision_sec=10.0, window_vision_sec=10.0, window_ambient_sec=10.0)
    stream = StreamingLoop(duplex, cfg, interruption_policy=None, vision_encoder=vision_enc,
                            max_tdm_bins=predictor_cfg.max_tdm_bins,
                            ambient_base_encoder=base_enc, ambient_nat_encoder=nat_enc)
    tick_wall_ms, ws_ages = run_ticks(stream, cfg, args.duration_sec)
    encoders_lats = [l["encoders_ms"] for l in stream.vision_logs]
    fusion_lats = [l["fusion_predictor_ms"] for l in stream.vision_logs]
    all_results["4.2_stride10_no_priority_stream"] = {
        "tick_wall_ms": stats(tick_wall_ms),
        "encoders_ms": stats(encoders_lats),
        "fusion_predictor_ms": stats(fusion_lats),
        "n_vision_refreshes": len(stream.vision_logs),
        "refresh_interval_target_s": cfg.stride_vision_sec,
        "staleness_distribution_s": stats(ws_ages),
    }
    print(json.dumps(all_results["4.2_stride10_no_priority_stream"], indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- 4.3: priority CUDA stream for the decision path ----
    print("\n[phase4.3] priority CUDA stream for decision path ...", flush=True)
    decision_stream = torch.cuda.Stream(priority=-1)
    stream2 = StreamingLoop(duplex, cfg, interruption_policy=None, vision_encoder=vision_enc,
                             max_tdm_bins=predictor_cfg.max_tdm_bins,
                             ambient_base_encoder=base_enc, ambient_nat_encoder=nat_enc)
    tick_wall_ms2, ws_ages2 = run_ticks(stream2, cfg, args.duration_sec, decision_stream=decision_stream)
    all_results["4.3_stride10_WITH_priority_stream"] = {
        "tick_wall_ms": stats(tick_wall_ms2),
        "n_vision_refreshes": len(stream2.vision_logs),
        "staleness_distribution_s": stats(ws_ages2),
    }
    print(json.dumps(all_results["4.3_stride10_WITH_priority_stream"], indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[phase4.2-3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
