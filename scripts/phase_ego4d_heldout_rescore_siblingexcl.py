"""scripts/phase_ego4d_heldout_rescore_siblingexcl.py — item 1 fix: rescore
the frozen Ego4D held-out gallery (checkpoints/vjepa21_shelved/
EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json, n=1542, embeddings already cached
in ego4d_heldout_zvza_cache.pt) with SAME-SOURCE-FILE windows excluded from
each query's distractor set.

Two flaws in the original gate being fixed here, per instruction:
  (a) vacuous threshold -- baseline R@1=0.71% is so close to chance that
      "beat the baseline" is a noise-clearable bar, not a real gate.
  (b) sibling contamination -- 1542 windows from only 81 files (~19/file)
      means each query's distractor pool contains ~18 near-duplicate
      windows from the SAME continuous footage, unlike VGGSound's 1545
      gallery (1545 DISTINCT videos). Not a like-for-like comparison.

Fix (no re-extraction -- embeddings already on disk): for query i, EXCLUDE
every OTHER window j (j != i) whose source_id equals query i's source_id
from the candidate pool before ranking; the true target (column i itself)
always stays in the pool. Report:
  - sibling-excluded R@1/5/10, both directions
  - mean effective gallery size (candidates remaining per query after
    exclusion)
  - FILE-LEVEL R@1 (secondary, lenient metric): using the FULL,
    UNEXCLUDED candidate pool, is the single top-ranked candidate from the
    correct source file (even if it's the wrong window within that file)?
  - shuffle-sanity gap + within-modality cosine, recomputed for
    completeness (should match the original EGO4D_HELDOUT_BASELINE.json
    since these don't depend on sibling exclusion)

Usage:
    python scripts/phase_ego4d_heldout_rescore_siblingexcl.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

MANIFEST_PATH = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_GALLERY_FILEDISJOINT.json"
CACHE_PATH = "checkpoints/vjepa21_shelved/ego4d_heldout_zvza_cache.pt"
OUT_PATH = "checkpoints/vjepa21_shelved/EGO4D_HELDOUT_BASELINE_SIBLINGEXCL.json"


def sibling_excluded_metrics(sim: np.ndarray, source_ids: np.ndarray, ks=(1, 5, 10)):
    """sim[i, j] = similarity of query i (one modality) to candidate j
    (other modality). Ground truth for row i is column i. Excludes every
    OTHER column j (j != i) sharing query i's source_id."""
    N = sim.shape[0]
    ranks = np.zeros(N, dtype=np.int64)
    n_candidates = np.zeros(N, dtype=np.int64)
    for i in range(N):
        mask = source_ids != source_ids[i]
        mask[i] = True  # always keep the true target
        cand_idx = np.nonzero(mask)[0]
        cand_scores = sim[i, cand_idx]
        order = np.argsort(-cand_scores)
        ranked_idx = cand_idx[order]
        rank = int(np.nonzero(ranked_idx == i)[0][0])  # 0-indexed
        ranks[i] = rank
        n_candidates[i] = mask.sum()
    results = {}
    for k in ks:
        results[f"R@{k}"] = round(float((ranks < k).mean() * 100), 2)
    results["mean_effective_gallery_size"] = round(float(n_candidates.mean()), 2)
    results["min_effective_gallery_size"] = int(n_candidates.min())
    results["max_effective_gallery_size"] = int(n_candidates.max())
    return results


def file_level_r1(sim: np.ndarray, source_ids: np.ndarray) -> float:
    """Lenient secondary metric on the FULL (unexcluded) candidate pool:
    is the single top-ranked candidate from the correct source FILE (not
    necessarily the exact matching window)?"""
    N = sim.shape[0]
    top1 = sim.argmax(axis=1)
    correct = (source_ids[top1] == source_ids[np.arange(N)])
    return round(float(correct.mean() * 100), 2)


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    cached = torch.load(CACHE_PATH, weights_only=False)
    zv_list, za_list = cached["zv"], cached["za"]
    assert len(manifest) == len(zv_list) == len(za_list) == 1542
    assert all(z is not None for z in zv_list) and all(z is not None for z in za_list)

    source_ids = np.array([m["source_id"] for m in manifest])
    _, counts = np.unique(source_ids, return_counts=True)
    n_files = len(counts)
    print(f"[rescore] n={len(manifest)} windows, {n_files} source files, "
          f"windows/file min={counts.min()} max={counts.max()}", flush=True)

    z_v = torch.stack(zv_list, 0).numpy()
    z_a = torch.stack(za_list, 0).numpy()
    sim = z_v @ z_a.T  # (N,N), row=vision query, col=ambient candidate

    print("[rescore] computing sibling-excluded vision->ambient metrics...", flush=True)
    v2a = sibling_excluded_metrics(sim, source_ids)
    print("[rescore] computing sibling-excluded ambient->vision metrics...", flush=True)
    a2v = sibling_excluded_metrics(sim.T, source_ids)

    file_r1_v2a = file_level_r1(sim, source_ids)
    file_r1_a2v = file_level_r1(sim.T, source_ids)

    N = sim.shape[0]
    matched_sim = float(np.diagonal(sim).mean())
    perm = np.roll(np.arange(N), 1)
    shuffled_sim = float(sim[np.arange(N), perm].mean())

    def mean_offdiag_cosine(z):
        s = z @ z.T
        mask = ~np.eye(z.shape[0], dtype=bool)
        return round(float(s[mask].mean()), 4)

    results = {
        "gallery_size": N,
        "n_source_files": n_files,
        "sibling_excluded": {
            "vision_to_ambient": v2a,
            "ambient_to_vision": a2v,
        },
        "file_level_R@1_full_pool_lenient": {
            "vision_to_ambient": file_r1_v2a,
            "ambient_to_vision": file_r1_a2v,
        },
        "matched_cos_sim": round(matched_sim, 4),
        "shuffled_cos_sim": round(shuffled_sim, 4),
        "shuffle_sanity_gap": round(matched_sim - shuffled_sim, 4),
        "vision_within_modality_mean_offdiag_cosine": mean_offdiag_cosine(z_v),
        "ambient_within_modality_mean_offdiag_cosine": mean_offdiag_cosine(z_a),
        "reference_vggsound_1545_in_domain": {
            "vision_within_modality_mean_offdiag_cosine": 0.0308,
            "ambient_within_modality_mean_offdiag_cosine": 0.0289,
            "shuffle_sanity_gap": 0.6604,
        },
        "reference_original_unfiltered_baseline": {
            "note": "EGO4D_HELDOUT_BASELINE.json, no sibling exclusion",
            "vision_to_ambient_R@1": 0.71,
            "ambient_to_vision_R@1": 0.91,
        },
    }
    print(json.dumps(results, indent=2), flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[rescore] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
