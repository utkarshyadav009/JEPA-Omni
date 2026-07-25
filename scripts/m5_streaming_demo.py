"""scripts/m5_streaming_demo.py — M5 Day 1: wire the full streaming loop
end-to-end and measure real tick latency on Blackwell (mercury).

rolling AV windows -> World-State refresh (strided) -> speech-activity
(every tick) -> 3-class decision head -> generate (M4b speech path,
reusing the already-verified backchannel-production generation pattern)
-> simulated TTS + mic-gating (echo self-interruption fix) -> VAD
interruption halt -> interruption policy (resume/re-plan/abandon).

Data: REAL EasyCom audio (concatenated test segments -- speak, silence,
backchannel -- advanced over simulated wall-clock time) drives the audio
side. Vision uses DUMMY random frames (same choice already made and
disclosed in Phase 0 -- no long real AV-paired stream exists yet; this
exercises ViT-L's real compute cost, not real content). HONEST CAVEAT
stated once, not hidden: feeding a real (dummy-content) World-State
simultaneously with real audio to the decision head is an input regime
the head was never trained on (training always zeroed exactly one
modality) -- this script is a WIRING/latency test of the mechanism, not a
validated production decision quality claim. That gap is the same
genuinely-paired-AV-data gap flagged earlier in this project.

Usage:
    python scripts/m5_streaming_demo.py --duration-sec 30
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
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig
from models.m4_duplex_loop import DuplexLoop
from models.m4_interruption_policy import InterruptionPolicy, InterruptionPolicyConfig
from models.m5_streaming_loop import StreamingLoop, StreamingConfig
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--decision-head-ckpt", default="checkpoints/m4_decision_head_3class/best.pt")
    p.add_argument("--duration-sec", type=float, default=30.0)
    p.add_argument("--out", default="checkpoints/m5_streaming/blackwell_streaming_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m5-stream] hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    print("[m5-stream] loading frozen LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[m5-stream] loading frozen V-JEPA2 ViT-L vision encoder...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)

    print("[m5-stream] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    print("[m5-stream] loading M3 connector + M4b projector...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(joint_ckpt["m3_connector"])
    m3_connector.eval()
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()

    print("[m5-stream] loading whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[m5-stream] loading 3-class decision head...", flush=True)
    dh_ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = ThreeClassHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    duplex = DuplexLoop(predictor, m3_connector, m4b_projector, whisper, decision_head,
                         llm, tokenizer, device)
    policy = InterruptionPolicy(InterruptionPolicyConfig())
    cfg = StreamingConfig()
    stream = StreamingLoop(duplex, cfg, interruption_policy=policy,
                            vision_encoder=vision_enc, max_tdm_bins=predictor_cfg.max_tdm_bins)

    print("[m5-stream] building real EasyCom audio stream (concatenated test ticks)...", flush=True)
    _, test_ticks = build_ticks()
    ds = EasyComTurnTakingDataset(test_ticks[:120])
    audio_chunks = []
    for i in range(len(ds)):
        item = ds[i]
        audio_chunks.append(item["waveform"])
    full_audio = np.concatenate(audio_chunks).astype(np.float32)
    print(f"[m5-stream] real audio stream: {len(full_audio)/16000:.1f}s from {len(ds)} real EasyCom ticks", flush=True)

    # dummy video frames (see module docstring -- exercises real ViT-L
    # compute, not real content; no long real AV-paired stream available)
    frame_shape = (3, 256, 256)

    def make_dummy_frame():
        return (torch.rand(*frame_shape) * 255).to(torch.uint8)

    def generate_fn(speech_wave, dur_sec):
        with torch.no_grad():
            hidden, valid_frames = whisper([speech_wave], [dur_sec], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                stoks, smask = m4b_projector(hidden.float(), valid_frames)
            sattn = (~smask).long()
            return duplex.generate_interruptible(stoks, sattn, max_new_tokens=40)

    t_sim = 0.0
    audio_pos = 0
    samples_per_tick = int(cfg.tick_interval_sec * cfg.audio_sr)
    tick_wall_latencies = []
    n_ticks = int(args.duration_sec / cfg.tick_interval_sec)

    print(f"[m5-stream] running {n_ticks} ticks ({args.duration_sec:.0f}s simulated) ...", flush=True)
    for tick_i in range(n_ticks):
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

        gen_called = {"used": False}

        def _gen():
            gen_called["used"] = True
            return generate_fn(speech_window, speech_dur)

        t0 = time.perf_counter()
        log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur,
                           generate_fn=_gen)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        tick_wall_latencies.append(wall_ms)
        t_sim += cfg.tick_interval_sec

        if (tick_i + 1) % 40 == 0 or log.action == "speak":
            print(f"[m5-stream] tick {tick_i+1}/{n_ticks} t={t_sim:.2f}s action={log.action:12s} "
                  f"vision_refreshed={log.vision_refreshed}  wall={wall_ms:.1f}ms  "
                  f"latencies={log.latencies_ms}", flush=True)

    action_counts = {}
    for log in stream.logs:
        action_counts[log.action] = action_counts.get(log.action, 0) + 1

    vitl_lats = [l.latencies_ms["vitl_forward_ms"] for l in stream.logs if "vitl_forward_ms" in l.latencies_ms]
    fusion_lats = [l.latencies_ms["fusion_predictor_ms"] for l in stream.logs if "fusion_predictor_ms" in l.latencies_ms]
    vision_refresh_lats = [a + b for a, b in zip(vitl_lats, fusion_lats)]
    speech_lats = [l.latencies_ms["speech_activity_ms"] for l in stream.logs if "speech_activity_ms" in l.latencies_ms]
    decision_lats = [l.latencies_ms["decision_ms"] for l in stream.logs if "decision_ms" in l.latencies_ms]
    gen_lats = [l.latencies_ms["generation_ms"] for l in stream.logs if "generation_ms" in l.latencies_ms]

    def stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean_ms": statistics.mean(xs), "max_ms": max(xs),
                "p95_ms": statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs)}

    results = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "n_ticks": n_ticks, "duration_sec": args.duration_sec,
        "config": cfg.__dict__,
        "action_counts": action_counts,
        "tick_wall_latency_ms": stats(tick_wall_latencies),
        "vision_refresh_latency_ms": stats(vision_refresh_lats),
        "vitl_forward_latency_ms": stats(vitl_lats),
        "fusion_predictor_latency_ms": stats(fusion_lats),
        "speech_activity_latency_ms": stats(speech_lats),
        "decision_latency_ms": stats(decision_lats),
        "generation_latency_ms": stats(gen_lats),
        "n_vision_refreshes": len(vision_refresh_lats),
        "n_generations": len(gen_lats),
    }
    print("\n[m5-stream] === RESULTS ===")
    print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m5-stream] wrote {args.out}")


if __name__ == "__main__":
    main()
