"""scripts/jetson_m5_live_demo.py — Phase D1: the full end-to-end loop on
real Jetson hardware, never done before this session. Live camera
(models/m5_live_capture.py's LiveCameraCapture, RealSense D435i, device
index 4 confirmed 2026-08-01) + live mic (LiveMicCapture, ReSpeaker 4 Mic
Array, device 24) -> StreamingLoop.tick() (rolling AV buffers -> strided
World-State refresh -> speech-activity -> decide3_speechonly -> on SPEAK,
build a soft prompt from BOTH M3 (vision grounding, recomputed fresh at
generation time from the current buffer windows -- the vision-refresh
thread only caches the POOLED World-State, not the pre-pool tokens M3
needs, so this recomputes rather than touching that caching path) AND M4b
(speech content, from the current speech window) -> real Qwen2.5-1.5B
generation -> real Piper TTS out through the ReSpeaker's speaker) -> real
MicGate suppresses the decision path while playing.

Checkpoints: the three LOCKED ones (git tag freeze-submission-v1) for
M2/M3/decision-head, plus M4b from checkpoints/m4_joint/best.pt (the only
real trained M4b projector in this project -- not separately locked by
the freeze tag, since the freeze only named M2/M3/head, but this is the
same choice already relied on for A1's memory-transferability argument).

Usage:
    python3 scripts/jetson_m5_live_demo.py --duration-sec 60
"""
from __future__ import annotations

import argparse
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

MAX_AMBIENT_T = 1024


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def q_int8_cpu_then_move(module, device):
    # Matches scripts/jetson_phase4_full_stack_memory_v2_withqwen.py's proven
    # sequence exactly. FOUND (2026-08-01): the version of this helper first
    # written for this script was missing the _malloc_trim() call between
    # quantize_() and module.to(device) -- without it, Qwen's move to GPU
    # crashed reproducibly (twice in a row, identical traceback) with
    # `NvMapMemAllocInternalTagged error 12` -> `NVML_SUCCESS == r INTERNAL
    # ASSERT FAILED`, even though A1's script (same models, same int8
    # approach) never showed this. On Jetson's unified memory, CPU and GPU
    # share one physical pool -- glibc's malloc arena holding onto large
    # freed-but-not-returned-to-OS blocks (normal glibc behavior) reduces the
    # CONTIGUOUS memory NvMap has available to satisfy a large GPU allocation,
    # which is exactly the "near its contiguity limit" risk already flagged
    # for this deployment. malloc_trim(0) forces glibc to actually return
    # freed memory to the OS, freeing up that contiguity.
    import gc
    module = module.to("cpu")
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(module, Int8WeightOnlyConfig(version=2))
    except Exception as e:
        print(f"[live-demo]   int8 quant failed (running fp/bf16 instead): {e!r}", flush=True)
    gc.collect(); _malloc_trim()
    module = module.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gc.collect(); _malloc_trim()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--m3-ckpt", default="checkpoints/m3_multigran_richcaption_v2/last.pt")
    p.add_argument("--m4-joint-ckpt", default="checkpoints/m4_joint/best.pt",
                   help="source of the M4b speech projector (not separately locked by the freeze tag)")
    p.add_argument("--speechonly-ckpt", default="checkpoints/m4_decision_head_3class_speechonly_v2/best.pt")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--camera-index", type=int, default=4)
    p.add_argument("--mic-device", type=int, default=24)
    p.add_argument("--tts-device", type=int, default=24)
    p.add_argument("--duration-sec", type=float, default=60.0)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--out", default=os.path.expanduser("~/jetson_m5_live_demo_results.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live-demo] device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)

    print("[live-demo] loading V-JEPA2 ViT-L (int8)...", flush=True)
    from models.vision_encoder import VisionEncoder
    vision_enc = VisionEncoder(device="cpu", dtype=torch.bfloat16)
    vision_enc.model = q_int8_cpu_then_move(vision_enc.model, device)
    vision_enc.device_str = str(device)

    print("[live-demo] loading WavJEPA-base/nat (int8)...", flush=True)
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    wavjepa_base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device="cpu")
    wavjepa_base.model = q_int8_cpu_then_move(wavjepa_base.model, device)
    wavjepa_base.device_str = str(device)
    wavjepa_nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device="cpu")
    wavjepa_nat.model = q_int8_cpu_then_move(wavjepa_nat.model, device)
    wavjepa_nat.device_str = str(device)

    print("[live-demo] loading locked M2 predictor (int8)...", flush=True)
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg)
    m2ckpt = torch.load(args.m2_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor = q_int8_cpu_then_move(predictor, device)
    predictor.eval()

    print("[live-demo] loading Whisper-medium (int8)...", flush=True)
    from models.m4_speech import WhisperSpeechEncoder
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16)
    whisper.encoder = q_int8_cpu_then_move(whisper.encoder, device)

    print("[live-demo] loading locked speech-only decision head...", flush=True)
    from train_decision_head_3class_speechonly_v2 import SpeechOnlyThreeClassHead
    dh_ckpt = torch.load(args.speechonly_ckpt, map_location=device, weights_only=False)
    decision_head = SpeechOnlyThreeClassHead(speech_feat_dim=dh_ckpt["sf_dim"]).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    print("[live-demo] loading Qwen2.5-1.5B-Instruct (int8)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16)
    llm = q_int8_cpu_then_move(llm, device)
    llm.eval()

    print("[live-demo] loading locked M3 connector...", flush=True)
    from models.m3_connector import M3Connector, M3ConnectorConfig
    m3ckpt = torch.load(args.m3_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**m3ckpt["connector_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(m3ckpt["connector"])
    m3_connector.eval()

    print("[live-demo] loading M4b speech projector (from m4_joint)...", flush=True)
    from models.m4_speech import UltravoxProjector, UltravoxProjectorConfig
    m4joint_ckpt = torch.load(args.m4_joint_ckpt, map_location=device, weights_only=False)
    m4b_cfg = UltravoxProjectorConfig(**m4joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(m4joint_ckpt["m4b_projector"])
    m4b_projector.eval()

    print("[live-demo] loading real TTS (Piper) + backchannel inventory...", flush=True)
    from models.m5_tts import TTSEngine, BackchannelInventory
    tts = TTSEngine(device=args.tts_device)
    backchannel_inv = BackchannelInventory(tts)

    from models.m4_duplex_loop import DuplexLoop
    from models.m4_interruption_policy import InterruptionPolicy, InterruptionPolicyConfig
    from models.m5_streaming_loop import StreamingLoop, StreamingConfig
    from models.world_state_builder import build_world_state_features

    duplex = DuplexLoop(predictor, m3_connector, m4b_projector, whisper, decision_head, llm, tokenizer, device)
    policy = InterruptionPolicy(InterruptionPolicyConfig())
    cfg = StreamingConfig()
    stream = StreamingLoop(duplex, cfg, interruption_policy=policy,
                            vision_encoder=vision_enc, max_tdm_bins=predictor_cfg.max_tdm_bins,
                            ambient_base_encoder=wavjepa_base, ambient_nat_encoder=wavjepa_nat,
                            tts_engine=tts, backchannel_inventory=backchannel_inv)

    print(f"[live-demo] starting live camera (index {args.camera_index}) + mic (device {args.mic_device})...", flush=True)
    from models.m5_live_capture import LiveCameraCapture, LiveMicCapture
    cam = LiveCameraCapture(stream, device_index=args.camera_index, target_fps=cfg.video_fps)
    mic = LiveMicCapture(stream, sample_rate=cfg.audio_sr, chunk_sec=cfg.tick_interval_sec, device=args.mic_device)
    cam.start()
    mic.start()

    def generate_fn():
        """Vision-grounded (M3, recomputed fresh from the current buffer
        windows) + speech-grounded (M4b, from the current speech window)
        soft prompt, concatenated, one real interruptible generation."""
        video_window = stream.video_buf.get_window()
        ambient_window = stream.ambient_buf.get_window()
        speech_window = stream.audio_buf.get_window()
        speech_dur = len(speech_window) / cfg.audio_sr if speech_window is not None else 0.0

        with torch.no_grad():
            if video_window is not None and ambient_window is not None:
                audio_t = torch.from_numpy(ambient_window).float()
                true_dur = audio_t.shape[0] / cfg.audio_sr
                result = build_world_state_features(video_window, audio_t, true_dur, vision_enc,
                                                      wavjepa_base, wavjepa_nat, predictor_cfg.max_tdm_bins, device)
                pre_pool = predictor.encode_pre_pool_tokens(result.feats, result.tbins)
                if pre_pool.shape[1] > MAX_AMBIENT_T:
                    pre_pool = pre_pool[:, :MAX_AMBIENT_T]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    soft_prompt_m3 = m3_connector(pre_pool)
                attn_m3 = torch.ones(1, soft_prompt_m3.shape[1], dtype=torch.long, device=device)
            else:
                soft_prompt_m3 = torch.zeros(1, 0, m3_cfg.llm_hidden, device=device)
                attn_m3 = torch.ones(1, 0, dtype=torch.long, device=device)

            if speech_window is not None and speech_dur > 0.05:
                hidden, valid_frames = whisper([speech_window.astype(np.float32)], [speech_dur], device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    stoks, smask = m4b_projector(hidden.float(), valid_frames)
                attn_m4b = (~smask).long()
            else:
                stoks = torch.zeros(1, 0, m3_cfg.llm_hidden, device=device)
                attn_m4b = torch.ones(1, 0, dtype=torch.long, device=device)

        soft_prompt = torch.cat([soft_prompt_m3.to(stoks.dtype), stoks], dim=1)
        attn = torch.cat([attn_m3, attn_m4b], dim=1)
        return duplex.generate_interruptible(soft_prompt, attn, max_new_tokens=args.max_new_tokens)

    # Startup race (found 2026-08-01, crashed the first real live-capture run):
    # models/m5_streaming_loop.py's _run_decision_path() calls
    # decide3_speechonly(sf) UNCONDITIONALLY even when speech_waveform was
    # None (sf stays None too) -- every prior caller of tick() pre-filled the
    # audio buffer with at least one synthetic chunk before the first tick,
    # so this path was never hit. Live mic capture is asynchronous (PortAudio
    # delivers its first callback after one full blocksize, ~tick_interval_sec
    # later) -- the buffer is genuinely empty for a brief moment at startup.
    # Wait here rather than patching the shared, already-tested streaming
    # loop under time pressure.
    print("[live-demo] waiting for first real mic chunk before ticking...", flush=True)
    wait_t0 = time.time()
    while stream.audio_buf.get_window() is None and time.time() - wait_t0 < 3.0:
        time.sleep(0.05)
    if stream.audio_buf.get_window() is None:
        print("[live-demo]   WARNING: no mic audio after 3s -- check LiveMicCapture/device index", flush=True)

    t_sim = 0.0
    real_t0 = time.time()
    n_ticks = int(args.duration_sec / cfg.tick_interval_sec)
    stream.start_vision_refresh_thread(hz=1.0 / cfg.stride_vision_sec)
    print(f"[live-demo] running {n_ticks} real ticks ({args.duration_sec:.0f}s) -- talk to it now", flush=True)

    tick_wall_ms = []
    try:
        for i in range(n_ticks):
            speech_window = stream.audio_buf.get_window()
            speech_dur = len(speech_window) / cfg.audio_sr if speech_window is not None else 0.0

            t0 = time.perf_counter()
            log = stream.tick(t_sim, speech_waveform=speech_window, speech_dur_sec=speech_dur,
                               generate_fn=generate_fn)
            tick_wall_ms.append((time.perf_counter() - t0) * 1000.0)

            if log.action == "speak" and log.generation_text:
                print(f"[live-demo] t={t_sim:.1f}s SPEAK -> {log.generation_text!r}", flush=True)
            elif log.action == "backchannel":
                print(f"[live-demo] t={t_sim:.1f}s BACKCHANNEL", flush=True)
            elif log.action not in ("silence", "gated"):
                print(f"[live-demo] t={t_sim:.1f}s action={log.action}", flush=True)

            t_sim += cfg.tick_interval_sec
            real_elapsed = time.time() - real_t0
            sleep_left = t_sim - real_elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        stream.stop_vision_refresh_thread()
        cam.stop()
        mic.stop()

    action_counts = {}
    for log in stream.logs:
        action_counts[log.action] = action_counts.get(log.action, 0) + 1

    def stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean_ms": statistics.mean(xs), "max_ms": max(xs),
                "p95_ms": statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs)}

    results = {
        "duration_sec": args.duration_sec,
        "n_ticks": n_ticks,
        "action_counts": action_counts,
        "tick_wall_latency_ms": stats(tick_wall_ms),
        "n_camera_frames_captured": cam.n_frames_captured,
        "n_mic_chunks_captured": mic.n_chunks_captured,
        "generations": [
            {"t": lg.t, "text": lg.generation_text}
            for lg in stream.logs if lg.action == "speak" and lg.generation_text
        ],
    }
    print("\n[live-demo] === RESULTS ===")
    print(json.dumps({k: v for k, v in results.items() if k != "generations"}, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[live-demo] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
