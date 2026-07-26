"""scripts/phase_diag_easycom_event_composition.py — Diagnostic 1b: what
fraction of the 462 frozen EasyCom eval windows (same manifest as
phase3_easycom_frozen_eval.py) contain any non-speech acoustic event above
the SAME 0.10 confidence floor used for the Ego4D AV-relevance filter
(scripts/ego4d_av_relevance_filter.py -- same AST model, same SPEECH_IDX /
NOISE_LIKE_IDX exclusion, top_event_prob metric)?

Hypothesis under test: EasyCom is a conversation corpus and M2's ambient
path (WavJEPA) is the NON-speech path -- if EasyCom windows are almost all
speech with near-zero non-speech event content, the near-chance retrieval
result is the correct answer to a task the ambient signal cannot solve on
this corpus, not evidence of collapse by itself.

Usage:
    python scripts/phase_diag_easycom_event_composition.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ego4d_av_relevance_filter import SPEECH_IDX, NOISE_LIKE_IDX

MANIFEST_PATH = "checkpoints/vjepa21_shelved/EASYCOM_FROZEN_GALLERY_MANIFEST.json"
WINDOW_SEC = 10.0
AUDIO_SR = 16000
FLOOR = 0.10


def extract_audio(path: str, start_sec: float) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_sec:.3f}", "-t", f"{WINDOW_SEC}",
           "-i", path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    import soundfile as sf_io
    audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    assert sr == AUDIO_SR
    if audio.shape[0] < AUDIO_SR:
        audio = np.pad(audio, (0, AUDIO_SR - audio.shape[0]))
    return audio


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[easycom-event-comp] device={device}", flush=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"[easycom-event-comp] manifest: {len(manifest)} windows", flush=True)

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_fe = AutoFeatureExtractor.from_pretrained(ast_name)
    ast_model = AutoModelForAudioClassification.from_pretrained(ast_name).to(device).eval()
    id2label = ast_model.config.id2label

    cache_path = "checkpoints/vjepa21_shelved/easycom_event_composition_scores.json"
    scored = []
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            scored = json.load(f)
        print(f"[easycom-event-comp] RESUMING: {len(scored)} already scored", flush=True)

    for i, m in enumerate(manifest):
        if i < len(scored):
            continue
        window_start = max(0.0, m["center_sec"] - WINDOW_SEC / 2.0)
        try:
            audio = extract_audio(m["video_path"], window_start)
            inputs = ast_fe(audio, sampling_rate=AUDIO_SR, return_tensors="pt")
            with torch.no_grad():
                logits = ast_model(**{k: v.to(device) for k, v in inputs.items()}).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            mask = np.ones(probs.shape[0], dtype=bool)
            mask[SPEECH_IDX] = False
            mask[NOISE_LIKE_IDX] = False
            top_event_idx = int(np.argmax(probs * mask))
            top_event_prob = float(probs[top_event_idx])
            speech_prob = float(probs[SPEECH_IDX].sum())
            scored.append({"session": m["session"], "chunk": m["chunk"], "window_idx": m["window_idx"],
                           "top_event_idx": top_event_idx, "top_event_label": id2label[top_event_idx],
                           "top_event_prob": top_event_prob, "speech_prob": speech_prob})
        except Exception as e:
            print(f"[easycom-event-comp] window {i} FAILED: {e!r}", flush=True)
            scored.append({"session": m["session"], "chunk": m["chunk"], "window_idx": m["window_idx"],
                           "failed": True})
        if (i + 1) % 50 == 0 or i == len(manifest) - 1:
            with open(cache_path, "w") as f:
                json.dump(scored, f, indent=2)
            print(f"[easycom-event-comp] {i+1}/{len(manifest)} scored", flush=True)

    with open(cache_path, "w") as f:
        json.dump(scored, f, indent=2)

    valid = [s for s in scored if not s.get("failed")]
    above_floor = [s for s in valid if s["top_event_prob"] >= FLOOR]
    from collections import Counter
    label_counts = Counter(s["top_event_label"] for s in above_floor)

    results = {
        "n_windows": len(manifest),
        "n_scored": len(valid),
        "n_failed": len(scored) - len(valid),
        "floor": FLOOR,
        "n_above_floor": len(above_floor),
        "frac_above_floor": round(len(above_floor) / max(1, len(valid)), 4),
        "mean_top_event_prob": round(float(np.mean([s["top_event_prob"] for s in valid])), 4),
        "median_top_event_prob": round(float(np.median([s["top_event_prob"] for s in valid])), 4),
        "mean_speech_prob": round(float(np.mean([s["speech_prob"] for s in valid])), 4),
        "top_event_labels_above_floor": dict(label_counts.most_common(15)),
    }
    print(json.dumps(results, indent=2), flush=True)

    out_path = "checkpoints/vjepa21_shelved/EASYCOM_EVENT_COMPOSITION.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[easycom-event-comp] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
