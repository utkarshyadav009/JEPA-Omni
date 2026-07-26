"""scripts/phase3_easycom_frozen_eval.py — Phase 3.2: FROZEN EasyCom
retrieval eval, gallery drawn from sessions 10/11/12 ONLY (M2 will train
on sessions 1-9 only, per the session-disjoint split already verified
safe for A1). Non-overlapping 10s windows from every Video_Compressed
chunk in those 3 sessions: 77 chunks x 6 windows/chunk = 462 windows.
Uses world_state_builder (Phase 1.2-verified) for feature construction,
train_m2.py's own pool_and_project + contrastive_retrieval_eval pattern
(imported, not reimplemented) for the retrieval metric itself.

clips_seen assertion in the style of the 1545 VGGSound gallery: this
script hard-asserts n==462 before reporting any number.

Usage:
    python scripts/phase3_easycom_frozen_eval.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.world_state_builder import build_world_state_features
from data.m4_speech_dataset import EASYCOM_ROOT

WINDOW_SEC = 10.0
CHUNK_DUR_SEC = 60.0
TEST_SESSIONS = [10, 11, 12]
EXPECTED_GALLERY_SIZE = 462
M2_CKPT = "checkpoints/m2_fusion_20k_best/step19000_peak.pt"
AUDIO_SR = 16000


def build_gallery_manifest():
    """Deterministic, frozen: every Video_Compressed chunk in sessions
    10/11/12, 6 non-overlapping 10s windows each (60/10), sorted by
    (session, chunk, window_idx) for reproducibility."""
    manifest = []
    for sess in TEST_SESSIONS:
        chunk_dir = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{sess}")
        chunks = sorted(glob.glob(os.path.join(chunk_dir, "*.mp4")))
        for chunk_path in chunks:
            chunk_stem = os.path.splitext(os.path.basename(chunk_path))[0]
            for w in range(6):
                center = w * WINDOW_SEC + WINDOW_SEC / 2.0
                manifest.append({"session": sess, "chunk": chunk_stem, "video_path": chunk_path,
                                  "window_idx": w, "center_sec": center})
    return manifest


def decode_window(video_path, center_sec, device):
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices
    VIDEO_FPS = 20.0  # EasyCom Video_Compressed native fps (confirmed via ffprobe earlier this session)
    t0 = max(0.0, center_sec - WINDOW_SEC / 2)
    t1 = min(CHUNK_DUR_SEC, t0 + WINDOW_SEC)
    t0 = max(0.0, t1 - WINDOW_SEC)
    dec = VideoDecoder(video_path, device="cpu")
    f0 = int(round(t0 * VIDEO_FPS)); f1 = int(round(t1 * VIDEO_FPS))
    n_total = getattr(dec.metadata, "num_frames", None) or len(dec)
    f1 = min(f1, n_total); f0 = min(f0, f1 - 1)
    idx = _uniform_frame_indices(f1 - f0, 64)
    abs_idx = sorted(set(i + f0 for i in idx))
    batch = dec.get_frames_at(indices=abs_idx)
    decoded = batch.data
    remap = {o: i for i, o in enumerate(abs_idx)}
    gi = torch.tensor([remap[i + f0] for i in idx], dtype=torch.long)
    frames = decoded.index_select(0, gi)
    if frames.shape[-2] != 256 or frames.shape[-1] != 256:
        x = frames.float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames, t0, t1


def decode_audio(video_path, t0, t1):
    import subprocess, io, soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
           "-i", video_path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    assert sr == AUDIO_SR
    return torch.from_numpy(audio), audio.shape[0] / AUDIO_SR


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[easycom-eval] device={device}", flush=True)

    manifest = build_gallery_manifest()
    print(f"[easycom-eval] gallery manifest: {len(manifest)} windows "
          f"({ {s: sum(1 for m in manifest if m['session']==s) for s in TEST_SESSIONS} })", flush=True)
    assert len(manifest) == EXPECTED_GALLERY_SIZE, \
        f"clips_seen assertion FAILED: expected {EXPECTED_GALLERY_SIZE}, got {len(manifest)}"
    print(f"[easycom-eval] clips_seen assertion PASSED: {len(manifest)} == {EXPECTED_GALLERY_SIZE}", flush=True)

    with open("checkpoints/vjepa21_shelved/EASYCOM_FROZEN_GALLERY_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[easycom-eval] loading real encoders + M2 (frozen, checkpoints/m2_fusion_20k_best)...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(M2_CKPT, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    vision_proj = nn.Linear(1024, 256).to(device)
    vision_proj.load_state_dict(ckpt["vision_proj"])
    vision_proj.eval()
    ambient_proj = nn.Linear(1024, 256).to(device)
    ambient_proj.load_state_dict(ckpt["ambient_proj"])
    ambient_proj.eval()

    cache_path = "checkpoints/vjepa21_shelved/easycom_frozen_eval_zvza_cache.pt"
    zv_list, za_list = [], []
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        zv_list, za_list = cached["zv"], cached["za"]
        print(f"[easycom-eval] RESUMING: {len(zv_list)} already encoded", flush=True)

    for i, m in enumerate(manifest):
        if i < len(zv_list):
            continue
        try:
            frames, t0, t1 = decode_window(m["video_path"], m["center_sec"], device)
            audio, true_dur = decode_audio(m["video_path"], t0, t1)
            result = build_world_state_features(frames, audio, true_dur, vision_enc, base_enc, nat_enc,
                                                 predictor_cfg.max_tdm_bins, device)
            with torch.no_grad():
                src_tokens = predictor.encode_source_tokens(result.feats, result.tbins)
                z_v = F.normalize(vision_proj(src_tokens["vision"].mean(1)).float(), dim=-1)
                z_a = F.normalize(ambient_proj(src_tokens["ambient"].mean(1)).float(), dim=-1)
            zv_list.append(z_v[0].cpu())
            za_list.append(z_a[0].cpu())
        except Exception as e:
            print(f"[easycom-eval] window {i} FAILED: {e!r}", flush=True)
            zv_list.append(None)
            za_list.append(None)
        if (i + 1) % 50 == 0 or i == len(manifest) - 1:
            torch.save({"zv": zv_list, "za": za_list}, cache_path)
            print(f"[easycom-eval] {i+1}/{len(manifest)} encoded", flush=True)

    torch.save({"zv": zv_list, "za": za_list}, cache_path)

    valid_idx = [i for i in range(len(zv_list)) if zv_list[i] is not None]
    n_valid = len(valid_idx)
    print(f"[easycom-eval] {n_valid}/{len(manifest)} windows successfully encoded", flush=True)

    z_v = torch.stack([zv_list[i] for i in valid_idx], 0)
    z_a = torch.stack([za_list[i] for i in valid_idx], 0)
    N = z_v.shape[0]
    gt = torch.arange(N)
    sim = z_v @ z_a.T
    results = {"gallery_size_attempted": len(manifest), "gallery_size_scored": N,
               "clips_seen_assertion": len(manifest) == EXPECTED_GALLERY_SIZE}
    for name, ranked in [("vision_to_ambient", (-sim).argsort(1)), ("ambient_to_vision", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
            results[f"{name}_R@{k}"] = round(hits * 100, 2)

    matched_sim = sim.diagonal().mean().item()
    perm = torch.roll(torch.arange(N), 1)
    shuffled_sim = sim[torch.arange(N), perm].mean().item()
    results["matched_cos_sim"] = round(matched_sim, 4)
    results["shuffled_cos_sim"] = round(shuffled_sim, 4)
    results["shuffle_sanity_gap"] = round(matched_sim - shuffled_sim, 4)

    print("\n[easycom-eval] === RESULTS (m2_fusion_20k_best baseline on frozen EasyCom eval) ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    out_path = "checkpoints/vjepa21_shelved/EASYCOM_FROZEN_EVAL_BASELINE.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[easycom-eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
