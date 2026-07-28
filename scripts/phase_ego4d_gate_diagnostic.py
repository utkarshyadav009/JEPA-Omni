"""scripts/phase_ego4d_gate_diagnostic.py — items 1b + 2a (+ 1a, 2d) for
the Ego4D held-out gate re-review: is the gate even usable as built?

1a: windows-per-file distribution in the gallery (min/median/max/histogram).
1b: FILE-LEVEL R@1/R@5/R@10 both directions (correct if retrieved window is
    from the same source file, using the FULL/unexcluded candidate pool),
    with the file-level CHANCE rate at each k reported alongside (chance is
    NOT 1/n_files since file sizes are skewed -- computed by simulation).
2a: re-run the ORIGINAL (buggy) audio-extraction path for all 1542 gallery
    windows -- no GPU, no re-encoding, just the ffmpeg call -- and count how
    many hit the silent zero-audio fallback (audio.shape[0] < 1).
2d: video fps via ffprobe for ALL 81 unique gallery files (not a 4-file
    sample) -- a wrong fps assumption would silently shift every window's
    frame sampling.

Does NOT yet decide whether to rebuild the gallery (item 1d) or exclude
zero-audio windows from the metrics (item 2b) -- those are follow-up steps
once 1b/2a's numbers are in.

Usage:
    python scripts/phase_ego4d_gate_diagnostic.py
"""
from __future__ import annotations

import io
import json
import subprocess
from collections import Counter

import numpy as np
import torch

MANIFEST_PATH = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json"
CACHE_PATH = "checkpoints/vjepa21_shelved/ego4d_heldout_zvza_cache.pt"
WINDOW_SEC = 10.0
AUDIO_SR = 16000
OUT_PATH = "checkpoints/vjepa21_shelved/EGO4D_GATE_DIAGNOSTIC.json"


# ---- 1a: windows-per-file distribution ----------------------------------
def windows_per_file(manifest):
    source_ids = [m["source_id"] for m in manifest]
    counts = Counter(source_ids)
    vals = sorted(counts.values())
    n = len(vals)
    hist = Counter(vals)
    return {
        "n_files": n,
        "min": vals[0],
        "median": vals[n // 2],
        "max": vals[-1],
        "mean": round(sum(vals) / n, 2),
        "histogram_windows_per_file": dict(sorted(hist.items())),
    }


# ---- 1b: file-level R@k both directions + chance rate -------------------
def file_level_rk(sim: np.ndarray, source_ids: np.ndarray, ks=(1, 5, 10)):
    N = sim.shape[0]
    order = np.argsort(-sim, axis=1)  # (N, N) ranked candidate indices per row
    results = {}
    for k in ks:
        topk = order[:, :k]  # (N, k)
        hit = np.array([np.any(source_ids[topk[i]] == source_ids[i]) for i in range(N)])
        results[f"R@{k}"] = round(float(hit.mean() * 100), 2)
    return results


def file_level_chance(source_ids: np.ndarray, ks=(1, 5, 10), n_sim=2000, seed=0):
    """Simulate: for each query, draw k DISTINCT random candidates (uniform,
    from all N items, self allowed as a candidate like the real metric) and
    check if any share the query's source file. Averaged over many trials."""
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


# ---- 2a: re-run the ORIGINAL (buggy) audio path, count zero-fallbacks ----
def decode_audio_original(video_path, t0, t1):
    """Exact copy of the buggy decode_audio() from
    scripts/phase_ego4d_heldout_gallery_score.py -- reproduced here
    (not imported) so re-running it cannot accidentally pick up a fix."""
    import soundfile as sf_io
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1-t0:.3f}",
           "-i", video_path, "-vn", "-ar", str(AUDIO_SR), "-ac", "1", "-f", "wav", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        audio, sr = sf_io.read(io.BytesIO(out.stdout), dtype="float32")
    except Exception as e:
        return None, str(e)
    if sr != AUDIO_SR:
        return None, f"sr mismatch: {sr}"
    if audio.shape[0] < 1:
        return "ZERO_FALLBACK", None
    return audio.shape[0], None


def check_zero_audio_fallback(manifest):
    n_zero = 0
    n_error = 0
    n_ok = 0
    zero_examples = []
    for i, m in enumerate(manifest):
        t0 = m["start_sec"]
        t1 = t0 + WINDOW_SEC
        result, err = decode_audio_original(m["path"], t0, t1)
        if result == "ZERO_FALLBACK":
            n_zero += 1
            if len(zero_examples) < 10:
                zero_examples.append({"idx": i, "path": m["path"], "start_sec": t0})
        elif result is None:
            n_error += 1
        else:
            n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"[gate-diag] 2a: {i+1}/{len(manifest)} checked "
                  f"(zero={n_zero}, error={n_error}, ok={n_ok})", flush=True)
    return {"n_zero_fallback": n_zero, "n_decode_error": n_error, "n_ok": n_ok,
            "zero_fallback_examples": zero_examples}


# ---- 2d: fps check for ALL unique gallery files --------------------------
def check_fps_all_files(manifest):
    paths = sorted(set(m["path"] for m in manifest))
    fps_values = {}
    for p in paths:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", p],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        fps_values[p] = out
    distinct = Counter(fps_values.values())
    return {"n_files_checked": len(paths), "distinct_fps_values": dict(distinct),
            "all_30fps": all(v == "30/1" for v in fps_values.values())}


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    cached = torch.load(CACHE_PATH, weights_only=False)
    zv_list, za_list = cached["zv"], cached["za"]
    assert len(manifest) == len(zv_list) == 1542

    source_ids = np.array([m["source_id"] for m in manifest])
    z_v = torch.stack(zv_list, 0).numpy()
    z_a = torch.stack(za_list, 0).numpy()
    sim = z_v @ z_a.T

    print("[gate-diag] === 1a: windows-per-file distribution ===", flush=True)
    wpf = windows_per_file(manifest)
    print(json.dumps(wpf, indent=2), flush=True)

    print("[gate-diag] === 1b: file-level R@k (full pool) + chance ===", flush=True)
    file_r_v2a = file_level_rk(sim, source_ids)
    file_r_a2v = file_level_rk(sim.T, source_ids)
    chance = file_level_chance(source_ids)
    print(f"vision->ambient: {file_r_v2a}", flush=True)
    print(f"ambient->vision: {file_r_a2v}", flush=True)
    print(f"chance: {chance}", flush=True)

    print("[gate-diag] === 2a: zero-audio-fallback re-check (ffmpeg, no GPU) ===", flush=True)
    zero_audio = check_zero_audio_fallback(manifest)
    print(json.dumps(zero_audio, indent=2), flush=True)

    print("[gate-diag] === 2d: fps check, all unique gallery files ===", flush=True)
    fps_check = check_fps_all_files(manifest)
    print(json.dumps(fps_check, indent=2), flush=True)

    results = {
        "1a_windows_per_file": wpf,
        "1b_file_level_Rk": {"vision_to_ambient": file_r_v2a, "ambient_to_vision": file_r_a2v,
                              "chance": chance},
        "2a_zero_audio_fallback": zero_audio,
        "2d_fps_check": fps_check,
        "gallery_size_scored_vs_attempted": {"scored": len(manifest), "attempted": 1542},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gate-diag] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
