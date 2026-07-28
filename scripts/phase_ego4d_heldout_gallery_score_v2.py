"""scripts/phase_ego4d_heldout_gallery_score_v2.py — item 1d (score) + item
2c (fix): score m2_fusion_20k_best on the REBUILT Ego4D held-out gallery
(EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json, n=674, cap<=2 windows/file
across 350 files -- fixes the v1 gallery's ambiguity-bound near-duplicate
problem, see EGO4D_GATE_DIAGNOSTIC.json).

Item 2c fix included here: decode_audio() now RAISES on zero-length audio
instead of silently substituting zeros and returning WINDOW_SEC -- a failed
window is recorded as failed (None in the cache, excluded from all metrics),
never silently given a degenerate all-zero embedding that would corrupt
retrieval and inflate within-modality cosine. (2a's re-check found 0/1542
such failures in the v1 gallery, so this is a proactive fix, not evidence
of an actual corruption in v1's numbers -- but the code path stays fixed
going forward, including for this rebuild.)

Reports, in one pass: raw R@1/5/10 both directions, sibling-excluded R@1/5/10
(should barely differ from raw given cap<=2/file), file-level R@1/5/10 +
chance, shuffle-sanity gap, within-modality cosine.

Usage:
    python scripts/phase_ego4d_heldout_gallery_score_v2.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.world_state_builder import build_world_state_features

HELDOUT_MANIFEST = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT_V2.json"
EXPECTED_GALLERY_SIZE = 674
WINDOW_SEC = 10.0
VIDEO_FPS = 30.0  # confirmed via ffprobe on ALL 81 v1-gallery files (EGO4D_GATE_DIAGNOSTIC.json 2d);
                  # re-verified below for the v2 gallery's (partially overlapping) file set too
AUDIO_SR = 16000
M2_CKPT = "checkpoints/m2_fusion_20k_best/step19000_peak.pt"
CACHE_PATH = "checkpoints/vjepa21_shelved/ego4d_heldout_v2_zvza_cache.pt"
OUT_PATH = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_V2.json"


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
    """Item 2c FIX: raises on zero-length audio instead of silently
    substituting zeros. Callers must catch and mark the window as failed."""
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
           "-i", video_path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    if sr != AUDIO_SR:
        raise ValueError(f"sample rate mismatch: got {sr}, expected {AUDIO_SR}")
    if audio.shape[0] < 1:
        raise ValueError(f"zero-length audio decoded from {video_path} @ [{t0:.2f},{t1:.2f}]")
    return torch.from_numpy(audio), audio.shape[0] / AUDIO_SR


def sibling_excluded_metrics(sim: np.ndarray, source_ids: np.ndarray, ks=(1, 5, 10)):
    N = sim.shape[0]
    ranks = np.zeros(N, dtype=np.int64)
    n_candidates = np.zeros(N, dtype=np.int64)
    for i in range(N):
        mask = source_ids != source_ids[i]
        mask[i] = True
        cand_idx = np.nonzero(mask)[0]
        cand_scores = sim[i, cand_idx]
        order = np.argsort(-cand_scores)
        ranked_idx = cand_idx[order]
        rank = int(np.nonzero(ranked_idx == i)[0][0])
        ranks[i] = rank
        n_candidates[i] = mask.sum()
    results = {}
    for k in ks:
        results[f"R@{k}"] = round(float((ranks < k).mean() * 100), 2)
    results["mean_effective_gallery_size"] = round(float(n_candidates.mean()), 2)
    return results


def file_level_rk(sim: np.ndarray, source_ids: np.ndarray, ks=(1, 5, 10)):
    N = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    results = {}
    for k in ks:
        topk = order[:, :k]
        hit = np.array([np.any(source_ids[topk[i]] == source_ids[i]) for i in range(N)])
        results[f"R@{k}"] = round(float(hit.mean() * 100), 2)
    return results


def file_level_chance(source_ids: np.ndarray, ks=(1, 5, 10), n_sim=5000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(source_ids)
    chance = {}
    for k in ks:
        hits = 0
        for _ in range(n_sim):
            i = rng.integers(0, N)
            cand = rng.choice(N, size=k, replace=False)
            hits += int(np.any(source_ids[cand] == source_ids[i]))
        chance[f"R@{k}"] = round(hits / n_sim * 100, 2)
    return chance


def main() -> None:
    import argparse
    p_args = argparse.ArgumentParser()
    p_args.add_argument("--m2-ckpt", default=M2_CKPT)
    p_args.add_argument("--cache-path", default=CACHE_PATH)
    p_args.add_argument("--out-path", default=OUT_PATH)
    args = p_args.parse_args()
    m2_ckpt_path = args.m2_ckpt
    cache_path = args.cache_path
    out_path = args.out_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ego4d-heldout-v2] device={device}  m2_ckpt={m2_ckpt_path}", flush=True)

    with open(HELDOUT_MANIFEST) as f:
        manifest = json.load(f)
    print(f"[ego4d-heldout-v2] gallery manifest: {len(manifest)} windows", flush=True)
    assert len(manifest) == EXPECTED_GALLERY_SIZE, \
        f"clips_seen assertion FAILED: expected {EXPECTED_GALLERY_SIZE}, got {len(manifest)}"
    print(f"[ego4d-heldout-v2] clips_seen assertion PASSED: {len(manifest)} == {EXPECTED_GALLERY_SIZE}", flush=True)

    # item 2d re-verification for this (partially different) file set
    paths = sorted(set(m["path"] for m in manifest))
    fps_vals = Counter()
    for p in paths:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", p],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        fps_vals[out] += 1
    print(f"[ego4d-heldout-v2] fps check, {len(paths)} files: {dict(fps_vals)}", flush=True)
    assert fps_vals == Counter({"30/1": len(paths)}), "non-30fps file found -- VIDEO_FPS assumption violated"

    print("[ego4d-heldout-v2] loading real encoders + M2 (frozen, m2_fusion_20k_best)...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    ckpt = torch.load(m2_ckpt_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model"], strict=True)
    predictor.eval()
    vision_proj = nn.Linear(1024, 256).to(device)
    vision_proj.load_state_dict(ckpt["vision_proj"]); vision_proj.eval()
    ambient_proj = nn.Linear(1024, 256).to(device)
    ambient_proj.load_state_dict(ckpt["ambient_proj"]); ambient_proj.eval()

    zv_list, za_list = [], []
    n_failed = 0
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        zv_list, za_list = cached["zv"], cached["za"]
        print(f"[ego4d-heldout-v2] RESUMING: {len(zv_list)} already processed", flush=True)

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
            print(f"[ego4d-heldout-v2] window {i} ({m['path']}@{m['start_sec']}) FAILED (item 2c: "
                  f"recorded as failed, not zeroed): {e!r}", flush=True)
            zv_list.append(None)
            za_list.append(None)
        if (i + 1) % 50 == 0 or i == len(manifest) - 1:
            torch.save({"zv": zv_list, "za": za_list}, cache_path)
            print(f"[ego4d-heldout-v2] {i+1}/{len(manifest)} processed", flush=True)

    torch.save({"zv": zv_list, "za": za_list}, cache_path)

    valid_idx = [i for i in range(len(zv_list)) if zv_list[i] is not None]
    n_valid = len(valid_idx)
    n_failed = len(manifest) - n_valid
    print(f"[ego4d-heldout-v2] {n_valid}/{len(manifest)} windows successfully encoded "
          f"({n_failed} failed, EXCLUDED from all metrics below)", flush=True)

    valid_manifest = [manifest[i] for i in valid_idx]
    source_ids = np.array([m["source_id"] for m in valid_manifest])
    z_v = torch.stack([zv_list[i] for i in valid_idx], 0).numpy()
    z_a = torch.stack([za_list[i] for i in valid_idx], 0).numpy()
    N = z_v.shape[0]
    gt = np.arange(N)
    sim = z_v @ z_a.T

    results = {"gallery_size_attempted": len(manifest), "gallery_size_scored": N,
               "n_failed_excluded": n_failed,
               "clips_seen_assertion": len(manifest) == EXPECTED_GALLERY_SIZE}

    for name, ranked_dir in [("vision_to_ambient", sim), ("ambient_to_vision", sim.T)]:
        ranked = (-ranked_dir).argsort(1)
        for k in (1, 5, 10):
            hits = (ranked[:, :k] == gt[:, None]).any(1).mean()
            results[f"raw_{name}_R@{k}"] = round(float(hits * 100), 2)

    results["sibling_excluded"] = {
        "vision_to_ambient": sibling_excluded_metrics(sim, source_ids),
        "ambient_to_vision": sibling_excluded_metrics(sim.T, source_ids),
    }
    file_r_v2a = file_level_rk(sim, source_ids)
    file_r_a2v = file_level_rk(sim.T, source_ids)
    chance = file_level_chance(source_ids)
    results["file_level"] = {"vision_to_ambient": file_r_v2a, "ambient_to_vision": file_r_a2v,
                              "chance": chance}

    matched_sim = float(np.diagonal(sim).mean())
    perm = np.roll(np.arange(N), 1)
    shuffled_sim = float(sim[np.arange(N), perm].mean())
    results["matched_cos_sim"] = round(matched_sim, 4)
    results["shuffled_cos_sim"] = round(shuffled_sim, 4)
    results["shuffle_sanity_gap"] = round(matched_sim - shuffled_sim, 4)

    def mean_offdiag_cosine(z):
        s = z @ z.T
        mask = ~np.eye(z.shape[0], dtype=bool)
        return round(float(s[mask].mean()), 4)
    results["vision_within_modality_mean_offdiag_cosine"] = mean_offdiag_cosine(z_v)
    results["ambient_within_modality_mean_offdiag_cosine"] = mean_offdiag_cosine(z_a)

    print("\n[ego4d-heldout-v2] === RESULTS (rebuilt gate, m2_fusion_20k_best baseline) ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ego4d-heldout-v2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
