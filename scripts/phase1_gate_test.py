"""scripts/phase1_gate_test.py — Phase 1.2, the falsifier for the
world_state_builder fix itself. Take 20 VGGSound clips present in the
existing feature cache, decode raw video+audio from source (SAME decode
functions as scripts/extract_features_av.py, imported not reimplemented),
run models.world_state_builder.build_world_state_features(), and compare
against the cached (already-extracted) features.

LEVEL 1: elementwise max-abs-diff on vision/ambient tensors, exact-match
rate on tbins.
LEVEL 2: cosine similarity of compute_world_state() output, fresh vs
cached, per clip.

PASS = mean cosine >= 0.99 AND min cosine >= 0.98 AND tbins match exactly.
"""
from __future__ import annotations

import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import _decode_video_raw, _decode_audio_raw, NUM_FRAMES, RESOLUTION, AUDIO_SR
from models.world_state_builder import build_world_state_features, assert_staircase
from models.vision_encoder import VisionEncoder
from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor

CACHE_DIR = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k"
VIDEO_DIR = "/home/utkarsh/data/vggsound"
M2_CKPT = "checkpoints/m2_fusion_20k_best/step19000_peak.pt"
N_CLIPS = 20


def shard(vid: str) -> str:
    return vid[:2]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate] device={device}", flush=True)

    gallery = [l.strip() for l in open(os.path.join(PROJECT_ROOT, "data/vggsound_eval_1545.txt")) if l.strip()]
    clips = []
    for vid in gallery:
        cache_path = os.path.join(CACHE_DIR, shard(vid), vid + ".pt")
        video_path = os.path.join(VIDEO_DIR, vid + ".mp4")
        if os.path.isfile(cache_path) and os.path.isfile(video_path):
            clips.append((vid, cache_path, video_path))
        if len(clips) >= N_CLIPS:
            break
    print(f"[gate] {len(clips)} clips selected", flush=True)

    print("[gate] loading real encoders...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    base_enc = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat_enc = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(M2_CKPT, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    per_clip = []
    for i, (vid, cache_path, video_path) in enumerate(clips):
        try:
            cached = torch.load(cache_path, weights_only=False)
            frames = _decode_video_raw(video_path, NUM_FRAMES, RESOLUTION)   # (64,3,256,256) uint8
            audio = _decode_audio_raw(video_path, AUDIO_SR)                  # (n_samples,) float32
            true_dur = audio.shape[0] / AUDIO_SR

            result = build_world_state_features(frames, audio, true_dur, vision_enc, base_enc, nat_enc,
                                                  predictor_cfg.max_tdm_bins, device)

            # LEVEL 1: elementwise comparison
            fresh_vis = result.feats["vision"][0].float().cpu()
            cached_vis = cached["vision"].reshape(32 * 16, 1024).float()
            vis_maxdiff = (fresh_vis - cached_vis).abs().max().item()

            fresh_aud = result.feats["ambient"][0].float().cpu()
            # cached ambient: reconstruct the SAME mean(base,nat) the dataset does
            cached_base = cached["ambient_base"].float()
            cached_nat = cached["ambient_nat"].float()
            if cached_base.shape[0] == cached_nat.shape[0]:
                cached_aud = (cached_base + cached_nat) * 0.5
            else:
                cached_aud = cached_base
            aud_len = min(fresh_aud.shape[0], cached_aud.shape[0])
            aud_maxdiff = (fresh_aud[:aud_len] - cached_aud[:aud_len]).abs().max().item()

            # tbins exact match (recompute cached vis tbins the same way av_cached_dataset.py does)
            from data.av_cached_dataset import _ts_to_tdm_bins
            cached_vis_ts = cached["vision_ts"]
            cached_vis_ts_exp = cached_vis_ts.unsqueeze(1).expand(32, 16, 2).reshape(512, 2)
            cached_clip_dur = float(cached.get("clip_duration_s", true_dur))
            cached_vis_bins = _ts_to_tdm_bins(cached_vis_ts_exp, cached_clip_dur, predictor_cfg.max_tdm_bins)
            fresh_vis_bins = result.tbins["vision"][0].cpu()
            tbins_match = (cached_vis_bins == fresh_vis_bins).float().mean().item()

            # LEVEL 2: compute_world_state cosine similarity
            with torch.no_grad():
                ws_fresh = predictor.encode_world_state(result.feats, result.tbins)[0].float().cpu()

                cached_vis_b = cached["vision"].reshape(1, 512, 1024).float().to(device)
                cached_aud_b = cached_aud.unsqueeze(0).to(device)
                cached_feats = {"vision": cached_vis_b, "ambient": cached_aud_b}
                cached_tbins = {"vision": cached_vis_bins.unsqueeze(0).to(device),
                                 "ambient": _ts_to_tdm_bins(
                                     cached["ambient_base_ts"][:cached_aud.shape[0]], cached_clip_dur,
                                     predictor_cfg.max_tdm_bins).unsqueeze(0).to(device)}
                ws_cached = predictor.encode_world_state(cached_feats, cached_tbins)[0].float().cpu()

            cos = torch.nn.functional.cosine_similarity(ws_fresh.unsqueeze(0), ws_cached.unsqueeze(0)).item()

            per_clip.append({"vid": vid, "vision_maxdiff": vis_maxdiff, "ambient_maxdiff": aud_maxdiff,
                              "tbins_match_rate": tbins_match, "cosine_sim": cos})
            print(f"[gate] {i+1}/{len(clips)} {vid}: vis_maxdiff={vis_maxdiff:.4f} "
                  f"aud_maxdiff={aud_maxdiff:.4f} tbins_match={tbins_match:.3f} cos_sim={cos:.4f}", flush=True)
        except Exception as e:
            print(f"[gate] {vid} FAILED: {e!r}", flush=True)
            per_clip.append({"vid": vid, "error": str(e)})

    cos_vals = [c["cosine_sim"] for c in per_clip if "cosine_sim" in c]
    tbins_vals = [c["tbins_match_rate"] for c in per_clip if "tbins_match_rate" in c]
    result_summary = {
        "n_clips": len(clips),
        "n_ok": len(cos_vals),
        "mean_cosine": sum(cos_vals) / len(cos_vals) if cos_vals else None,
        "min_cosine": min(cos_vals) if cos_vals else None,
        "all_tbins_exact_match": all(v == 1.0 for v in tbins_vals) if tbins_vals else None,
        "per_clip": per_clip,
    }
    passed = (result_summary["mean_cosine"] is not None and result_summary["mean_cosine"] >= 0.99
              and result_summary["min_cosine"] >= 0.98
              and result_summary["all_tbins_exact_match"] is True)
    result_summary["PASS"] = passed

    out_path = "checkpoints/vjepa21_shelved/PHASE1_GATE_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    print(f"\n[gate] PASS={passed}  mean_cos={result_summary['mean_cosine']}  min_cos={result_summary['min_cosine']}  "
          f"tbins_exact={result_summary['all_tbins_exact_match']}", flush=True)
    print(f"[gate] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
