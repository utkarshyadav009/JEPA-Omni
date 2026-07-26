"""scripts/jetson_streaming_thread_verify.py — item 6: re-verify the
threaded StreamingLoop (vision refresh on its own thread, decision path
never touches World-State) with the REAL V-JEPA2 ViT-L + M2 predictor +
Whisper on the Jetson, not the duck-typed sleep() harness. Vision and
generation/decision share ONE device here -- this is the real contention
test the mercury-only harness couldn't do.

Run ON the Jetson, after jetson_preflight.sh PASS.

Usage:
    python3 jetson_streaming_thread_verify.py --duration-sec 30
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def q_int8_cpu_then_move(module, device):
    import gc
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=float, default=30.0)
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_fusion_20k_best/step19000_peak.pt"))
    p.add_argument("--speechonly-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class_speechonly/best.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/jetson_streaming_thread_results.json"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[verify] device={torch.cuda.get_device_name(0)}", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from models.m4_speech import WhisperSpeechEncoder
    from models.m4_duplex_loop import DuplexLoop
    from models.m5_streaming_loop import StreamingLoop, StreamingConfig
    from train_decision_head_3class_speechonly import SpeechOnlyThreeClassHead

    print("[verify] loading real V-JEPA2 ViT-L (int8)...", flush=True)
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, device)
    vision_enc.device_str = "cuda"

    print("[verify] loading real M2 predictor (int8)...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, device)
    predictor.eval()

    print("[verify] loading real Whisper-medium (int8)...", flush=True)
    whisper = WhisperSpeechEncoder("openai/whisper-medium", dtype=torch.bfloat16)
    whisper.model = q_int8_cpu_then_move(whisper.model, device)

    print("[verify] loading speech-only decision head...", flush=True)
    ckpt = torch.load(args.speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = SpeechOnlyThreeClassHead(speech_feat_dim=ckpt["sf_dim"]).to(device)
    decision_head.load_state_dict(ckpt["state_dict"])
    decision_head.eval()

    duplex = DuplexLoop(predictor, None, None, whisper, decision_head, None, None, device)
    cfg = StreamingConfig()
    stream = StreamingLoop(duplex, cfg, interruption_policy=None,
                            vision_encoder=vision_enc, max_tdm_bins=predictor_cfg.max_tdm_bins)

    def make_dummy_frame():
        return (torch.rand(3, 256, 256) * 255).to(torch.uint8)

    audio = (np.random.randn(16000 * 20) * 0.01).astype(np.float32)

    tick_wall_ms = []
    n_ticks = int(args.duration_sec / cfg.tick_interval_sec)
    print(f"[verify] running {n_ticks} REAL ticks ({args.duration_sec:.0f}s, paced to real time), "
          f"vision thread running concurrently with REAL ViT-L forwards...", flush=True)

    real_t0 = time.time()
    t_sim = 0.0
    audio_pos = 0
    samples_per_tick = int(cfg.tick_interval_sec * cfg.audio_sr)
    stream.start_vision_refresh_thread(hz=0.3)

    ws_none_ticks = 0
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

        if stream.get_cached_world_state() is None:
            ws_none_ticks += 1

        t0 = time.perf_counter()
        log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur, generate_fn=None)
        tick_wall_ms.append((time.perf_counter() - t0) * 1000.0)

        t_sim += cfg.tick_interval_sec
        real_elapsed = time.time() - real_t0
        sleep_left = t_sim - real_elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)

        if (i + 1) % 40 == 0:
            print(f"[verify] tick {i+1}/{n_ticks}  last_wall={tick_wall_ms[-1]:.1f}ms  "
                  f"vision_refreshes_so_far={len(stream.vision_logs)}", flush=True)

    stream.stop_vision_refresh_thread()

    def stats(xs):
        if not xs:
            return None
        s = sorted(xs)
        return {"n": len(xs), "mean_ms": statistics.mean(xs), "max_ms": max(xs),
                "p50_ms": s[len(s)//2], "p95_ms": s[int(len(s)*0.95)]}

    vitl_lats = [l["vitl_forward_ms"] for l in stream.vision_logs]
    fusion_lats = [l["fusion_predictor_ms"] for l in stream.vision_logs]

    result = {
        "device": torch.cuda.get_device_name(0),
        "duration_sec": args.duration_sec,
        "n_ticks": n_ticks,
        "tick_wall_ms": stats(tick_wall_ms),
        "vitl_forward_ms": stats(vitl_lats),
        "fusion_predictor_ms": stats(fusion_lats),
        "n_vision_refreshes": len(stream.vision_logs),
        "ws_none_ticks": ws_none_ticks,
    }
    print("\n[verify] === RESULTS ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[verify] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
