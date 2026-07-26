"""scripts/phase_ego4d_heldout_gallery_score.py — item 2b: score
m2_fusion_20k_best on the FROZEN, source-file-disjoint Ego4D held-out
gallery (scripts/phase_ego4d_heldout_gallery_build.py,
EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json, n=1542, matching the 1545-clip
VGGSound gallery's scale). This REPLACES the EasyCom retrieval eval as the
M2 gate (EasyCom retired: ~84% speech-dominant, see falsifier_tracking.md
2026-07-26 diagnostic entry -- wrong domain for a vision<->ambient
congruency gate).

VIDEO_FPS=30.0 confirmed via direct ffprobe on 4 sampled files (2x
video_540ss, 2x clips) -- both Ego4D sources are exactly 30/1.

Uses world_state_builder (Phase 1.2-verified) + train_m2.py's own
pool_and_project pattern (imported, not reimplemented) -- identical
methodology to the retired EasyCom eval, so this number is a fair,
like-for-like replacement.

clips_seen assertion against the frozen manifest size before reporting.

Usage:
    python scripts/phase_ego4d_heldout_gallery_score.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
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

HELDOUT_MANIFEST = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json"
EXPECTED_GALLERY_SIZE = 1542
WINDOW_SEC = 10.0
VIDEO_FPS = 30.0  # confirmed via ffprobe: both video_540ss and clips sources are exactly 30/1
AUDIO_SR = 16000
M2_CKPT = "checkpoints/m2_fusion_20k_best/step19000_peak.pt"
CACHE_PATH = "checkpoints/vjepa21_shelved/ego4d_heldout_zvza_cache.pt"
OUT_PATH = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE.json"


def decode_window(video_path, start_sec, device):
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices
    t0 = start_sec
    t1 = start_sec + WINDOW_SEC
    dec = VideoDecoder(video_path, device="cpu")
    n_total = getattr(dec.metadata, "num_frames", None) or len(dec)
    f0 = int(round(t0 * VIDEO_FPS)); f1 = int(round(t1 * VIDEO_FPS))
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
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
           "-i", video_path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    assert sr == AUDIO_SR
    if audio.shape[0] < 1:
        audio = torch.zeros(int(WINDOW_SEC * AUDIO_SR))
        return audio, WINDOW_SEC
    return torch.from_numpy(audio), audio.shape[0] / AUDIO_SR


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ego4d-heldout-score] device={device}", flush=True)

    with open(HELDOUT_MANIFEST) as f:
        manifest = json.load(f)
    print(f"[ego4d-heldout-score] gallery manifest: {len(manifest)} windows", flush=True)
    assert len(manifest) == EXPECTED_GALLERY_SIZE, \
        f"clips_seen assertion FAILED: expected {EXPECTED_GALLERY_SIZE}, got {len(manifest)}"
    print(f"[ego4d-heldout-score] clips_seen assertion PASSED: {len(manifest)} == {EXPECTED_GALLERY_SIZE}", flush=True)

    print("[ego4d-heldout-score] loading real encoders + M2 (frozen, m2_fusion_20k_best)...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(M2_CKPT, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    vision_proj = nn.Linear(1024, 256).to(device)
    vision_proj.load_state_dict(ckpt["vision_proj"]); vision_proj.eval()
    ambient_proj = nn.Linear(1024, 256).to(device)
    ambient_proj.load_state_dict(ckpt["ambient_proj"]); ambient_proj.eval()

    zv_list, za_list = [], []
    if os.path.isfile(CACHE_PATH):
        cached = torch.load(CACHE_PATH, weights_only=False)
        zv_list, za_list = cached["zv"], cached["za"]
        print(f"[ego4d-heldout-score] RESUMING: {len(zv_list)} already encoded", flush=True)

    for i, m in enumerate(manifest):
        if i < len(zv_list):
            continue
        try:
            frames, t0, t1 = decode_window(m["path"], m["start_sec"], device)
            audio, true_dur = decode_audio(m["path"], t0, t1)
            result = build_world_state_features(frames, audio, true_dur, vision_enc, base_enc, nat_enc,
                                                 predictor_cfg.max_tdm_bins, device)
            with torch.no_grad():
                src_tokens = predictor.encode_source_tokens(result.feats, result.tbins)
                z_v = F.normalize(vision_proj(src_tokens["vision"].mean(1)).float(), dim=-1)
                z_a = F.normalize(ambient_proj(src_tokens["ambient"].mean(1)).float(), dim=-1)
            zv_list.append(z_v[0].cpu())
            za_list.append(z_a[0].cpu())
        except Exception as e:
            print(f"[ego4d-heldout-score] window {i} ({m['path']}@{m['start_sec']}) FAILED: {e!r}", flush=True)
            zv_list.append(None)
            za_list.append(None)
        if (i + 1) % 50 == 0 or i == len(manifest) - 1:
            torch.save({"zv": zv_list, "za": za_list}, CACHE_PATH)
            print(f"[ego4d-heldout-score] {i+1}/{len(manifest)} encoded", flush=True)

    torch.save({"zv": zv_list, "za": za_list}, CACHE_PATH)

    valid_idx = [i for i in range(len(zv_list)) if zv_list[i] is not None]
    n_valid = len(valid_idx)
    print(f"[ego4d-heldout-score] {n_valid}/{len(manifest)} windows successfully encoded", flush=True)

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

    def mean_offdiag_cosine(z):
        s = z @ z.T
        mask = ~torch.eye(z.shape[0], dtype=torch.bool)
        return round(s[mask].mean().item(), 4)
    results["vision_within_modality_mean_offdiag_cosine"] = mean_offdiag_cosine(z_v)
    results["ambient_within_modality_mean_offdiag_cosine"] = mean_offdiag_cosine(z_a)

    print("\n[ego4d-heldout-score] === RESULTS (new required M2 gate baseline) ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ego4d-heldout-score] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
