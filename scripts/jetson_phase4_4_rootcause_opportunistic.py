"""scripts/jetson_phase4_4_rootcause_opportunistic.py — item 4: root-cause
the Phase 4.3 p95 tail (786.5ms even WITH the priority CUDA stream), then
test an opportunistic vision-refresh policy against it.

STATUS AS WRITTEN: NOT RUN. Blocked on Jetson SSH access (Tailscale
re-auth pending) as of 2026-07-26. Ready to run once access resumes --
do not report any number from this file as measured until a real
JETSON_PHASE4_4_RESULTS.json exists on disk.

Design (per your instructions):
  4a ROOT CAUSE FIRST: split every tick's wall-clock latency by
     `log.overlapped_vision_forward` (models/m5_streaming_loop.py's new
     per-tick instrumentation -- a threading.Event sampled at tick entry,
     True iff a vision/ambient forward pass was in flight on the vision
     thread at that instant). Reports mean/p95/n for each group under the
     CURRENT strided policy (stride=window=10.0s, priority stream ON, the
     Phase 4.3 configuration) -- confirms or refutes "the tail IS the
     ticks that land during a vision forward" directly, not by inference.
  4b OPPORTUNISTIC REFRESH: StreamingLoop.start_vision_refresh_thread_
     opportunistic() (new method) prefers refreshing during MicGate.
     is_playing windows (the robot is already gated for TTS playback, so a
     vision refresh landing there costs nothing extra to the decision
     path), falling back to a hard staleness deadline. Uses REAL EasyCom
     test-session audio (data.m4_easycom_turntaking.build_ticks()[1],
     first 120 ticks, same corpus/slice Day 1 used) to drive real
     decide3_speechonly() decisions -- when a real "speak" decision fires,
     mic-gating is triggered the SAME WAY Day 1 and the echo-cancellation
     gate already established as this project's disclosed methodology: a
     plausible fixed-length response text converted to an estimated TTS
     duration via _estimate_tts_duration_sec (150 wpm) -- NOT a full real
     Qwen/M3 generation pass (unnecessary for a scheduling-policy test,
     and this Jetson stack does not have Qwen resident here; see item
     4.1's retraction note on why Qwen isn't in this stack). Real ViT-L +
     WavJEPA-base/nat + M2 fusion + Whisper + decision-head forwards
     throughout -- only the TEXT that drives TTS-duration estimation is a
     placeholder, matching Day 1's disclosed simulation boundary exactly.

Reports p95 for BOTH policies (strided-with-priority-stream vs
opportunistic-with-priority-stream) so the comparison is apples-to-apples
(priority stream held constant, only the refresh-scheduling policy
varies).

Usage (on the Jetson, after jetson_preflight.sh PASS):
    python scripts/jetson_phase4_4_rootcause_opportunistic.py
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

SIMULATED_RESPONSE_TEXT = ("Sure, I can help with that. Let me take a look and get back to "
                            "you in just a moment with the details you need.")  # 26 words ~ 10.4s @ 150wpm


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


def build_real_audio_stream():
    from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset
    _, test_ticks = build_ticks()
    ds = EasyComTurnTakingDataset(test_ticks[:120])
    chunks = [ds[i]["waveform"] for i in range(len(ds))]
    full_audio = np.concatenate(chunks).astype(np.float32)
    print(f"[phase4.4] real EasyCom audio stream: {len(full_audio)/16000:.1f}s from {len(ds)} real ticks", flush=True)
    return full_audio


def run_ticks(stream, cfg, duration_sec, full_audio, decision_stream, refresh_mode: str,
              staleness_deadline_sec: float = 15.0):
    def make_dummy_frame():
        return (torch.rand(3, 256, 256) * 255).to(torch.uint8)

    if refresh_mode == "strided":
        stream.start_vision_refresh_thread(hz=1.0 / cfg.stride_vision_sec)
    elif refresh_mode == "opportunistic":
        stream.start_vision_refresh_thread_opportunistic(staleness_deadline_sec=staleness_deadline_sec,
                                                           poll_interval_sec=cfg.tick_interval_sec)
    else:
        raise ValueError(refresh_mode)

    def _gen():
        from models.m4_duplex_loop import GenerationResult
        return GenerationResult(token_ids=[], text=SIMULATED_RESPONSE_TEXT, interrupted=False,
                                 n_tokens_generated=0)

    tick_logs = []
    n_ticks = int(duration_sec / cfg.tick_interval_sec)
    real_t0 = time.time()
    t_sim = 0.0
    audio_pos = 0
    samples_per_tick = int(cfg.tick_interval_sec * cfg.audio_sr)

    for i in range(n_ticks):
        stream.ingest_video_frame(make_dummy_frame())
        chunk = full_audio[audio_pos: audio_pos + samples_per_tick]
        if len(chunk) < samples_per_tick:
            audio_pos = 0
            chunk = full_audio[:samples_per_tick]
        else:
            audio_pos += samples_per_tick
        stream.ingest_audio_chunk(chunk)

        speech_window = stream.audio_buf.get_window()
        speech_dur = len(speech_window) / cfg.audio_sr if speech_window is not None else 0.0

        t0 = time.perf_counter()
        if decision_stream is not None:
            with torch.cuda.stream(decision_stream):
                log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur,
                                   generate_fn=_gen)
            torch.cuda.current_stream().wait_stream(decision_stream)
        else:
            log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur,
                               generate_fn=_gen)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        tick_logs.append({"wall_ms": wall_ms, "action": log.action,
                          "overlapped_vision_forward": log.overlapped_vision_forward})

        t_sim += cfg.tick_interval_sec
        real_elapsed = time.time() - real_t0
        sleep_left = t_sim - real_elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)

    stream.stop_vision_refresh_thread()
    return tick_logs


def summarize(tick_logs):
    all_wall = [t["wall_ms"] for t in tick_logs]
    overlapped = [t["wall_ms"] for t in tick_logs if t["overlapped_vision_forward"]]
    not_overlapped = [t["wall_ms"] for t in tick_logs if not t["overlapped_vision_forward"]]
    n_gated = sum(1 for t in tick_logs if t["action"] == "gated")
    return {
        "all_ticks": stats(all_wall),
        "overlapped_vision_forward_ticks": stats(overlapped),
        "not_overlapped_ticks": stats(not_overlapped),
        "n_gated_ticks": n_gated,
        "n_total_ticks": len(tick_logs),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=float, default=60.0)
    p.add_argument("--m2-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m2_fusion_20k_best/step19000_peak.pt"))
    p.add_argument("--speechonly-ckpt", default=os.path.expanduser("~/jepa_omni_transfer/checkpoints/m4_decision_head_3class_speechonly/best.pt"))
    p.add_argument("--out", default=os.path.expanduser("~/jetson_phase4_4_results.json"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[phase4.4] device={torch.cuda.get_device_name(0)}", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from models.m4_speech import WhisperSpeechEncoder
    from models.m4_duplex_loop import DuplexLoop
    from models.m5_streaming_loop import StreamingLoop, StreamingConfig
    from train_decision_head_3class_speechonly import SpeechOnlyThreeClassHead

    print("[phase4.4] loading real ViT-L, WavJEPA-base, WavJEPA-nat, M2, Whisper (all int8)...", flush=True)
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
    full_audio = build_real_audio_stream()

    cfg = StreamingConfig(stride_vision_sec=10.0, window_vision_sec=10.0, window_ambient_sec=10.0)
    all_results = {}

    for mode in ["strided", "opportunistic"]:
        print(f"\n[phase4.4] === refresh_mode={mode}, priority stream ON ===", flush=True)
        decision_stream = torch.cuda.Stream(priority=-1)
        stream = StreamingLoop(duplex, cfg, interruption_policy=None, vision_encoder=vision_enc,
                                max_tdm_bins=predictor_cfg.max_tdm_bins,
                                ambient_base_encoder=base_enc, ambient_nat_encoder=nat_enc)
        tick_logs = run_ticks(stream, cfg, args.duration_sec, full_audio, decision_stream, mode)
        summary = summarize(tick_logs)
        all_results[mode] = summary
        print(json.dumps(summary, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n[phase4.4] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
