"""scripts/m5_ood_falsifier.py — M5 critical-path falsifier: separates the
three real divergences between the streaming loop and the decision head's
training regime, one variable at a time, on REAL paired EasyCom AV data
(Video_Compressed + Close_Microphone_Audio + Speech_Transcriptions, same
chunk IDs across all three -- confirmed genuinely time-aligned, not
assumed).

  D1 (worst): decision head trained with EXACTLY ONE modality zeroed
    (models/m4_decision_head.py docstring) -- VGGSound rows: real
    World-State / zero speech-feat. EasyCom rows: zero World-State / real
    speech-feat. The streaming loop feeds BOTH real simultaneously, an
    input regime the head never saw during training.
  D2: the Day-1 streaming DEMO used random-noise video frames. This
    falsifier uses REAL decoded EasyCom video throughout (see below), so
    D2 does not apply to conditions (a)/(b)/(c) themselves -- it is
    resolved by construction, not tested as a separate arm.
  D3: up to stride_vision_sec (2.0s, models/m5_streaming_loop.py) of
    World-State staleness from strided vision refresh.

Three conditions, real paired AV, real tick, one variable changed at a
time:
  (a) real AV, BOTH modalities present, FRESH World-State  -- isolates D1
  (b) real AV, World-State ZEROED as in training, FRESH     -- control,
      reproduces the exact regime the head was gated on
  (c) real AV, BOTH present, World-State from a window 2.0s
      EARLIER in the same 60s chunk (the streaming loop's actual stride)
      -- isolates D3 on top of D1

Video: Video_Compressed/Session_{s}/{chunk}.mp4, decoded via torchcodec
(same VideoDecoder + _uniform_frame_indices pattern as
scripts/extract_features_av.py / data/video_text_dataset.py), a 10s
window (WINDOW_VISION_SEC, matching CLIP_DURATION_S / the trained
distribution) at the tick's own timestamp for (a)/(b), shifted 2.0s
earlier for (c) -- both windows clipped to the chunk's real [0,60s] span.

Usage:
    python scripts/m5_ood_falsifier.py --n-per-class 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vision_encoder import VisionEncoder
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m4_speech import WhisperSpeechEncoder
from models.m4_decision_head import ThreeClassHead, DecisionHeadConfig, LABEL_TO_IDX, IDX_TO_LABEL
from data.m4_easycom_turntaking import build_ticks
from data.m4_speech_dataset import EASYCOM_ROOT, VIDEO_FPS, WHISPER_SR

WINDOW_VISION_SEC = 10.0     # matches CLIP_DURATION_S -- the trained distribution
STRIDE_VISION_SEC = 2.0      # matches StreamingConfig.stride_vision_sec
N_FRAMES = 64
VIDEO_CHUNK_DURATION_SEC = 60.0

_PID_RE = re.compile(r"(.+)_Participant_ID_\d+\.wav$")


def chunk_of(audio_path: str) -> str:
    base = os.path.basename(audio_path)
    m = _PID_RE.match(base)
    return m.group(1) if m else base[:-4]


def decode_video_window(video_path: str, center_sec: float, device) -> torch.Tensor:
    """Decode N_FRAMES uniformly from [center-W/2, center+W/2], clipped to
    the real [0, 60s] chunk span. Returns (T,C,H,W) uint8 on CPU (matching
    VisionEncoder.encode's expected input)."""
    from torchcodec.decoders import VideoDecoder
    from data.video_text_dataset import _uniform_frame_indices

    t0 = max(0.0, center_sec - WINDOW_VISION_SEC / 2)
    t1 = min(VIDEO_CHUNK_DURATION_SEC, t0 + WINDOW_VISION_SEC)
    t0 = max(0.0, t1 - WINDOW_VISION_SEC)   # keep full window length near chunk edges

    decoder = VideoDecoder(video_path, device="cpu")
    fps = VIDEO_FPS
    f0 = int(round(t0 * fps))
    f1 = max(f0 + 1, int(round(t1 * fps)))
    num_total_native = getattr(decoder.metadata, "num_frames", None) or len(decoder)
    f1 = min(f1, num_total_native)
    f0 = min(f0, f1 - 1)

    rel_idx = _uniform_frame_indices(f1 - f0, N_FRAMES)
    abs_idx = sorted(set(i + f0 for i in rel_idx))
    remap = {o: i for i, o in enumerate(abs_idx)}
    batch = decoder.get_frames_at(indices=abs_idx)
    decoded = batch.data
    gi = torch.tensor([remap[i + f0] for i in rel_idx], dtype=torch.long)
    frames = decoded.index_select(0, gi.to(decoded.device))

    if frames.shape[-2] != 256 or frames.shape[-1] != 256:
        x = frames.float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        frames = x.round_().clamp_(0, 255).to(torch.uint8)
    return frames


def build_xy(cache_batch, ws_dim, sf_dim, mode: str):
    """mode: 'fresh' (real ws), 'zero' (control), 'stale' (real ws from
    the shifted window). Returns (ws_tensor, sf_tensor, y_tensor)."""
    n = len(cache_batch)
    sf = torch.stack([c["sf"] for c in cache_batch], 0)
    y = torch.tensor([LABEL_TO_IDX[c["label3"]] for c in cache_batch], dtype=torch.long)
    if mode == "zero":
        ws = torch.zeros(n, ws_dim)
    elif mode == "fresh":
        ws = torch.stack([c["ws_fresh"] for c in cache_batch], 0)
    elif mode == "stale":
        ws = torch.stack([c["ws_stale"] for c in cache_batch], 0)
    else:
        raise ValueError(mode)
    return ws, sf, y


def eval_condition(decision_head, ws, sf, y, device):
    with torch.no_grad():
        logits = decision_head(ws.to(device), sf.to(device))
        preds = logits.argmax(dim=-1).cpu()
    n_classes = 3
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)   # rows=true, cols=pred
    for t, p in zip(y.tolist(), preds.tolist()):
        conf[t, p] += 1
    acc = (preds == y).float().mean().item()
    per_class_recall = {}
    per_class_f1 = {}
    for c in range(n_classes):
        tp = conf[c, c].item()
        n_true = conf[c].sum().item()
        n_pred = conf[:, c].sum().item()
        recall = tp / max(1, n_true)
        precision = tp / max(1, n_pred)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class_recall[IDX_TO_LABEL[c]] = recall
        per_class_f1[IDX_TO_LABEL[c]] = f1
    macro_f1 = sum(per_class_f1.values()) / n_classes
    return {
        "accuracy": acc, "macro_f1": macro_f1,
        "per_class_recall": per_class_recall, "per_class_f1": per_class_f1,
        "confusion_matrix_rows_true_cols_pred": conf.tolist(),
        "label_order": [IDX_TO_LABEL[i] for i in range(n_classes)],
    }, preds


def swap_control(decision_head, ws, sf, y, device):
    n = ws.shape[0]
    perm = torch.roll(torch.arange(n), shifts=1)
    ws_swapped = ws[perm]
    with torch.no_grad():
        preds_normal = decision_head(ws.to(device), sf.to(device)).argmax(-1).cpu()
        preds_swapped = decision_head(ws_swapped.to(device), sf.to(device)).argmax(-1).cpu()
    acc_normal = (preds_normal == y).float().mean().item()
    acc_swapped_vs_true = (preds_swapped == y).float().mean().item()
    frac_changed = (preds_normal != preds_swapped).float().mean().item()
    return {"acc_normal": acc_normal, "acc_swapped_vs_true_label": acc_swapped_vs_true,
            "fraction_decision_changed_under_ws_swap": frac_changed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--decision-head-ckpt", default="checkpoints/m4_decision_head_3class/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="checkpoints/m5_streaming/ood_falsifier_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"[ood-falsifier] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

    print("[ood-falsifier] loading real V-JEPA2 ViT-L, M2 predictor, Whisper, 3-class decision head...", flush=True)
    vision_enc = VisionEncoder(device=str(device), dtype=torch.bfloat16)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)
    dh_ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = ThreeClassHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()

    print("[ood-falsifier] selecting real EasyCom test ticks with a matching real video chunk...", flush=True)
    _, test_ticks = build_ticks()
    by_label = {"speak": [], "silence": [], "backchannel": []}
    for t in test_ticks:
        chunk = chunk_of(t.audio_path)
        video_path = os.path.join(EASYCOM_ROOT, "Video_Compressed", f"Session_{t.session}", chunk + ".mp4")
        if not os.path.isfile(video_path):
            continue
        by_label[t.label3].append((t, video_path))
    for lbl in by_label:
        rng.shuffle(by_label[lbl])
    sample = []
    for lbl in ["speak", "silence", "backchannel"]:
        sample.extend(by_label[lbl][:args.n_per_class])
    rng.shuffle(sample)
    print(f"[ood-falsifier] n={len(sample)} ticks with real matched video "
          f"({ {k: min(len(v), args.n_per_class) for k, v in by_label.items()} })", flush=True)

    cache_path = args.out.replace(".json", "_encode_cache.pt")
    cache_batch = []
    if os.path.isfile(cache_path):
        cache_batch = torch.load(cache_path, weights_only=False)
        print(f"[ood-falsifier] RESUMING: loaded {len(cache_batch)} already-encoded ticks from {cache_path}", flush=True)

    for i, (t, video_path) in enumerate(sample):
        if i < len(cache_batch):
            continue   # already encoded in a prior (interrupted) run -- sample order is deterministic (fixed seed)
        center = (t.start_sec + t.end_sec) / 2.0
        frames_fresh = decode_video_window(video_path, center, device)
        frames_stale = decode_video_window(video_path, center - STRIDE_VISION_SEC, device)

        with torch.no_grad():
            v_fresh = vision_enc.encode(frames_fresh.unsqueeze(0))
            v_stale = vision_enc.encode(frames_stale.unsqueeze(0))
            n_tok = v_fresh.shape[1]
            bin_idx = torch.linspace(0, predictor_cfg.max_tdm_bins - 1, n_tok, device=device).round().long()
            feats_fresh = {"vision": v_fresh.float()}
            feats_stale = {"vision": v_stale.float()}
            tbins = {"vision": bin_idx.unsqueeze(0)}
            ws_fresh = predictor.encode_world_state(feats_fresh, tbins)[0].cpu()
            ws_stale = predictor.encode_world_state(feats_stale, tbins)[0].cpu()

            import soundfile as sf_io
            audio, sr = sf_io.read(t.audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            i0, i1 = max(0, int(t.start_sec * sr)), min(len(audio), int(t.end_sec * sr))
            clip = audio[i0:i1]
            if clip.size == 0:
                clip = np.zeros(int(0.02 * sr), dtype=np.float32)
            if sr != WHISPER_SR:
                import librosa
                clip = librosa.resample(clip, orig_sr=sr, target_sr=WHISPER_SR)
                sr = WHISPER_SR
            dur = clip.shape[0] / sr
            hidden, valid_frames = whisper([clip.astype(np.float32)], [dur], device)
            vf = int(valid_frames[0].item())
            sf_feat = hidden[0, :vf].float().mean(dim=0).cpu()

        cache_batch.append({"ws_fresh": ws_fresh, "ws_stale": ws_stale, "sf": sf_feat, "label3": t.label3,
                             "text": t.text, "session": t.session})
        torch.save(cache_batch, cache_path)   # incremental -- a kill mid-run loses at most 1 tick's work
        if (i + 1) % 5 == 0:
            print(f"[ood-falsifier] encoded {i+1}/{len(sample)} real (video,audio) ticks", flush=True)

    ws_dim, sf_dim = dh_cfg.world_state_dim, dh_cfg.speech_feat_dim
    results = {}
    for mode, label in [("fresh", "a_both_present_fresh"), ("zero", "b_control_zeroed_fresh"),
                        ("stale", "c_both_present_stale")]:
        ws, sf, y = build_xy(cache_batch, ws_dim, sf_dim, mode)
        stats, preds = eval_condition(decision_head, ws, sf, y, device)
        print(f"\n[ood-falsifier] === condition {label} ===")
        print(json.dumps(stats, indent=2))
        entry = {"stats": stats}
        if mode != "zero":
            sc = swap_control(decision_head, ws, sf, y, device)
            print(f"[ood-falsifier] swap-control ({label}): {json.dumps(sc)}")
            entry["swap_control"] = sc
        else:
            entry["swap_control"] = "SKIPPED -- ws is permanently zero in this condition, " \
                                     "swapping an all-zero tensor with another all-zero tensor is vacuous"
        results[label] = entry

    results["n"] = len(cache_batch)
    results["label_counts"] = dict(Counter(c["label3"] for c in cache_batch))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ood-falsifier] wrote {args.out}")


if __name__ == "__main__":
    main()
