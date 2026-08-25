"""scripts/m5_live_vs_offline_gate.py — Phase C3: cosine gate between a
World-State built from LIVE camera+mic capture and one built by decoding a
simultaneously-recorded file of the exact same scene the "offline" way
(scripts/extract_features_av.py's own decode functions). PASS if mean
cosine >= 0.99.

Both paths converge on the SAME construction function
(models/world_state_builder.py's build_world_state_features) -- that
module already closed the "streaming reimplemented feature construction
and got it wrong" bug class (see its docstring). What C3 tests is
upstream of that: does live capture (models/m5_live_capture.py's
LiveCameraCapture/LiveMicCapture: cv2 grab -> center-crop -> resize -> RGB
uint8; sounddevice InputStream -> float32 mono chunks) produce frame/audio
tensors equivalent to _decode_video_raw/_decode_audio_raw's offline decode
of a file recording the same physical moment. If capture-side
preprocessing (resize interpolation, color channel order, sample-rate
mismatch, clipping) diverges from the offline decode path even slightly,
this gate is where it would show up -- NOT in build_world_state_features,
which is identical on both sides by construction.

Procedure:
  1. Open camera + mic, simultaneously (a) push captured frames/audio into
     a RollingVideoBuffer/RollingAudioBuffer exactly as the streaming loop
     would, AND (b) write raw frames to an MP4 (cv2.VideoWriter) and audio
     to a WAV (soundfile) for the SAME window duration.
  2. Build World-State from (a) directly (the "live" path).
  3. Decode the just-written MP4/WAV with _decode_video_raw/
     _decode_audio_raw (scripts/extract_features_av.py -- the exact
     functions training's cache used) and build World-State from that
     (the "offline" path).
  4. Cosine(live, offline) per window, report mean/min across
     --n-windows repeats, PASS/FAIL against 0.99.

Requires a camera + mic attached (same hardware dependency as
models/m5_live_capture.py) and a static-ish scene during each window (the
two recordings are sequential, not literally the same frames, since the
live buffer is consumed before the file is written -- see NOTE in main()
for why this is the intended design, not a shortcut).

Not runnable on the dev machine (no camera/mic here) -- run on the Jetson.

KNOWN BLOCKER (2026-08-01, real hardware): the OFFLINE decode path
(_decode_video_raw/_decode_audio_raw, imported from
scripts/extract_features_av.py) depends on torchcodec, which fails to
import on this Jetson regardless of LD_LIBRARY_PATH fixes -- its compiled
extension needs libnvrtc.so.13/libcudart.so.13 (present at
~/.local/lib/python3.10/site-packages/nvidia/cu13/lib/, torch itself is
built against CUDA 12.6) AND hits a separate libstdc++ ABI mismatch
(`undefined symbol: ..._M_replace_coldEPcmPKcmm`) even once the CUDA libs
are found -- a deeper incompatibility than the torchaudio stub trick
elsewhere in this project fixes, not resolved by adding cu13's lib dir to
LD_LIBRARY_PATH. Root cause not fully diagnosed (likely a torchcodec wheel
built against a newer libstdc++ than this JetPack R36.4/Ubuntu 22.04
image ships) -- fixing it would need a different torchcodec version, a
from-source build, or a matched ffmpeg version, none attempted yet given
this gate is explicitly lower priority than Phase D. The LIVE half of
this gate (capture_window, live_frames_to_tensor, world_state_of) has NO
torchcodec dependency and works standalone -- only the offline-decode
comparison side is blocked.

Usage:
    python3 scripts/m5_live_vs_offline_gate.py --n-windows 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
import importlib.machinery

# Jetson-only workaround (matches scripts/jetson_phase4_full_stack_memory_v2_withqwen.py):
# torchaudio's compiled extension needs libcudart.so.13, which doesn't exist on this
# Jetson's CUDA install -- importing transformers transitively imports torchaudio and
# crashes with OSError before any of our own code runs. Stub it out before importing torch.
_stub = types.ModuleType("torchaudio")
_stub.__version__ = "0.0.0-stub"
_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def capture_window(camera_index: int, mic_device, window_sec: float, fps: float,
                    mic_sr: int, video_out_path: str, audio_out_path: str):
    """Simultaneously (1) records window_sec of camera+mic into raw
    in-memory buffers for the live path, and (2) writes the SAME captured
    frames/audio to video_out_path/audio_out_path so step 3 (offline
    decode) reads literally the frames/samples the live path saw -- not a
    second, separately-timed capture. This is what makes the comparison a
    genuine equivalence check on preprocessing rather than a comparison of
    two different moments in time."""
    import cv2
    import sounddevice as sd
    import soundfile as sf

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    frames_raw = []   # list of (H,W,C) BGR uint8, native camera resolution -- NOT yet resized
    n_frames_target = int(window_sec * fps)
    period = 1.0 / fps

    rec_audio = sd.rec(int(window_sec * mic_sr), samplerate=mic_sr, channels=1,
                        dtype="float32", device=mic_device)

    t0 = time.perf_counter()
    for _ in range(n_frames_target):
        tf0 = time.perf_counter()
        ok, frame = cap.read()
        if ok:
            frames_raw.append(frame)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
            writer.write(frame)
        elapsed = time.perf_counter() - tf0
        sleep_left = period - elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)
    cap.release()
    if writer is not None:
        writer.release()

    sd.wait()
    audio = rec_audio[:, 0].astype(np.float32)
    sf.write(audio_out_path, audio, mic_sr)

    return frames_raw, audio


def live_frames_to_tensor(frames_raw, n_frames_out: int, resolution: int) -> torch.Tensor:
    """Same preprocessing as models/m5_live_capture.py's LiveCameraCapture
    (center-crop to square, INTER_AREA resize, BGR->RGB, CHW) -- imported
    logic duplicated inline rather than instantiating the capture class,
    since that class pushes straight into ingest_video_frame with no
    return value; kept in exact lockstep with it deliberately."""
    import cv2
    from data.video_text_dataset import _uniform_frame_indices

    n = len(frames_raw)
    idx = _uniform_frame_indices(n, n_frames_out)
    out = []
    for i in idx:
        frame_bgr = frames_raw[i]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        cropped = frame_rgb[top:top + side, left:left + side]
        resized = cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(resized).permute(2, 0, 1).contiguous())
    return torch.stack(out, dim=0)   # (n_frames_out, C, H, W) uint8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera-index", type=int, default=4,
                   help="on the deployed Jetson (RealSense D435i), index 4 is the "
                        "1920x1080 RGB color stream -- confirmed by direct capture "
                        "(2026-08-01); indices 0/1/3/5 open() but read() always fails "
                        "(other RealSense sub-streams needing a specific pixel format), "
                        "index 2 is a single-channel 1280x720 stream (infrared, not RGB)")
    p.add_argument("--mic-device", type=int, default=24,
                   help="ReSpeaker 4 Mic Array (UAC1.0) -- confirmed via sounddevice.query_devices() "
                        "(2026-08-01); also the playback device (out=2), same index works for both")
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=6.4)
    p.add_argument("--mic-sr", type=int, default=16000)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
                   help="locked M2 checkpoint (freeze-submission-v1)")
    p.add_argument("--max-tdm-bins", type=int, default=512)
    p.add_argument("--pass-threshold", type=float, default=0.99)
    p.add_argument("--out", default="checkpoints/m5_jetson/PHASE_C3_LIVE_VS_OFFLINE_RESULTS.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[c3-gate] loading real V-JEPA2 ViT-L, WavJEPA-base/nat, M2 predictor "
          "(locked checkpoint)...", flush=True)
    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
    from models.world_state_builder import build_world_state_features
    from scripts.extract_features_av import _decode_video_raw, _decode_audio_raw, CLIP_DURATION_S

    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))

    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                  max_tdm_bins=args.max_tdm_bins, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    def world_state_of(frames_u8, audio_f32, true_dur):
        r = build_world_state_features(frames_u8, audio_f32, true_dur, vision_enc,
                                        base_enc, nat_enc, args.max_tdm_bins, device)
        with torch.no_grad():
            return predictor.encode_world_state(
                {k: v.float() for k, v in r.feats.items()}, r.tbins).float()

    results = {"per_window": [], "config": vars(args)}
    tmp_dir = tempfile.mkdtemp(prefix="c3_gate_")

    for w in range(args.n_windows):
        print(f"\n[c3-gate] window {w+1}/{args.n_windows}: recording {args.window_sec:.0f}s "
              f"live (camera+mic), writing to disk simultaneously...", flush=True)
        video_path = os.path.join(tmp_dir, f"win{w}.mp4")
        audio_path = os.path.join(tmp_dir, f"win{w}.wav")
        frames_raw, live_audio = capture_window(args.camera_index, args.mic_device,
                                                  args.window_sec, args.fps, args.mic_sr,
                                                  video_path, audio_path)

        # --- live path: preprocess the in-memory frames/audio directly ---
        live_frames = live_frames_to_tensor(frames_raw, n_frames_out=64, resolution=256)
        live_audio_t = torch.from_numpy(live_audio)
        ws_live = world_state_of(live_frames, live_audio_t, args.window_sec)

        # --- offline path: decode the just-written files the training way ---
        off_frames = _decode_video_raw(video_path, num_frames=64, resolution=256)
        off_audio = _decode_audio_raw(audio_path, target_sr=args.mic_sr)
        ws_off = world_state_of(off_frames, off_audio, args.window_sec)

        cos = torch.nn.functional.cosine_similarity(ws_live, ws_off, dim=-1).item()
        print(f"[c3-gate]   cosine(live, offline) = {cos:.4f}", flush=True)
        results["per_window"].append({"window": w, "cosine": cos,
                                       "n_frames_captured": len(frames_raw)})

    cosines = [r["cosine"] for r in results["per_window"]]
    mean_cos = float(np.mean(cosines))
    min_cos = float(np.min(cosines))
    verdict = "PASS" if mean_cos >= args.pass_threshold else "FAIL"
    results["mean_cosine"] = mean_cos
    results["min_cosine"] = min_cos
    results["pass_threshold"] = args.pass_threshold
    results["verdict"] = verdict

    print(f"\n[c3-gate] mean cosine = {mean_cos:.4f}  min = {min_cos:.4f}  "
          f"threshold = {args.pass_threshold}  VERDICT = {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[c3-gate] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
