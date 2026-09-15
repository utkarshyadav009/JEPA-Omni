"""Temporal pairs for the RUN-5 v2 predictive-fusion objective (docs/RUN5_SPEC_V2.md).

Yields, for a window at time t:
    feats/tbins for t        -- the FULL token features, so the fusion runs and gradients flow
    dv          for t        -- target: mean-pooled V-JEPA2 features at t+delta MINUS at t

WHY THE TARGET IS dV AND NOT dW (each reason is a measurement, docs/FUSION_BOTTLENECK.md):
  * the arrow of time is VISUAL -- vision->dW predicts forward 0.0446 and backward -0.0009,
    exactly zero, while audio->dW is symmetric at 1.34x
  * W loses precisely this -- dV from pre-fusion vision is 0.2758, from W 0.1330
  * dW would be the wrong target -- W is already best at predicting it and least directional
    about it, so training on dW reinforces the symmetric component that IS the problem

The target comes from a FROZEN encoder, so the predictive term cannot be satisfied by collapse
and needs no stop-gradient or EMA teacher.

WHERE THE TWO HALVES COME FROM. Inputs are the 2 s-stride token cache (~3.93 MB/window);
targets are the 1 s-stride world-state files (~5.5 KB/window), which already store `vision_mean`
per window. Loading a second 3.93 MB token file just to mean-pool it would be ~20x the I/O for
the same number. Pairs are matched on start_s, never on array position.

The feature cache for Epic-Kitchens is sharded by `vid[:3].lower()` while AVCachedDataset's
`_shard()` is `vid[:2]`, so paths are resolved here by an explicit index rather than that
convention. The per-sample TRANSFORM is AVCachedDataset.transform, called directly -- restating
it is exactly how build_both drifted from the shared builder.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from data.av_cached_dataset import AVCachedDataset, av_collate_fn


class TemporalPairDataset(Dataset):
    def __init__(self, feat_dirs: Sequence[str], ws_dir: str, delta_s: float = 10.0,
                 video_ids: Optional[Sequence[str]] = None, max_tdm_bins: int = 512,
                 audio_mode: str = "mean", prefix: str = "ek_"):
        self.delta_s = float(delta_s)
        self.prefix = prefix
        self._tx = AVCachedDataset.__new__(AVCachedDataset)   # transform only; never indexed
        self._tx.max_tdm_bins = int(max_tdm_bins)
        self._tx.audio_mode = audio_mode

        # index every feature file by (video, window index)
        by_vid: Dict[str, Dict[int, str]] = {}
        for root in feat_dirs:
            if not os.path.isdir(root):
                continue
            for shard in sorted(os.listdir(root)):
                sd = os.path.join(root, shard)
                if not os.path.isdir(sd) or shard.startswith("."):
                    continue
                for fn in os.listdir(sd):
                    if not (fn.startswith(prefix) and fn.endswith(".pt") and "_w" in fn):
                        continue
                    stem = fn[:-3]
                    vid, w = stem[len(prefix):].rsplit("_w", 1)
                    by_vid.setdefault(vid, {})[int(w)] = os.path.join(sd, fn)

        keep = set(video_ids) if video_ids is not None else None
        self.items: List[Dict] = []
        self.dv_cache: Dict[str, torch.Tensor] = {}
        for vid, wmap in sorted(by_vid.items()):
            if keep is not None and vid not in keep:
                continue
            ws_path = os.path.join(ws_dir, f"{vid}.pt")
            if not os.path.exists(ws_path):
                continue
            ws = torch.load(ws_path, map_location="cpu", weights_only=True)
            starts = ws["start_s"].tolist()
            pos = {round(float(s), 3): k for k, s in enumerate(starts)}
            vm = ws["vision_mean"].float()          # (N, 1024), frozen V-JEPA2, mean-pooled
            self.dv_cache[vid] = vm
            for w, path in sorted(wmap.items()):
                t = float(w)                         # window index == start second (1 s stride)
                i = pos.get(round(t, 3))
                j = pos.get(round(t + self.delta_s, 3))
                if i is None or j is None:
                    continue
                self.items.append({"vid": vid, "path": path, "i": i, "j": j})
        print(f"[TemporalPairDataset] {len(self.items):,} pairs from {len(self.dv_cache)} videos "
              f"at delta={self.delta_s}s", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        try:
            d = torch.load(it["path"], map_location="cpu", weights_only=True)
        except (FileNotFoundError, RuntimeError):
            return self.__getitem__((idx + 1) % len(self.items))
        out = self._tx.transform(d, f"{it['vid']}_w{it['i']:05d}")
        vm = self.dv_cache[it["vid"]]
        out["dv"] = (vm[it["j"]] - vm[it["i"]])      # (1024,) float32
        return out


def temporal_pair_collate(batch: List[Dict]) -> Dict:
    out = av_collate_fn(batch)
    out["dv"] = torch.stack([b["dv"] for b in batch])
    return out
