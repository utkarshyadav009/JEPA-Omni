"""data/av_cached_dataset.py — M2 audio-visual cached feature dataset.

Loads THREE tensors + per-token timestamps from /dev/shm/jepa_m2_cache/ and
maps each modality onto a shared TDM (Temporal Discrete Map) axis of up to
max_tdm_bins integer bins.

Returns a dict matching the input contract of AVJepaPredictor:
    feats  : {
        "vision"  : (T_v, 1024)  bf16   — vision tokens on TDM axis
        "ambient" : (T_a, 768)   bf16   — audio tokens on TDM axis
                                          (average of base+nat, or base only)
    }
    tbins  : {
        "vision"  : (T_v,)  int64   — TDM bin per vision token
        "ambient" : (T_a,)  int64   — TDM bin per audio token
    }
    clip_id : str

Vision shape stored on disk: (32, 16, 1024) — we flatten to (512, 1024)
across the TDM axis, then map each of the 512 tokens to a TDM bin.

AVJepaPredictor input contract (from av_jepa_predictor.py):
    feats  : Dict[str, (B, T_m, dim_m)]
    tbins  : Dict[str, (B, T_m)]  integer bins in [0, max_tdm_bins)
    mask   : Dict[str, (B, T_m) bool]

So we return per-sample dicts; the DataLoader collation handles batching.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── manifest validation ────────────────────────────────────────────────────
REQUIRED_MANIFEST_KEYS = {
    "vision_repo",
    "wavjepa_base_repo",
    "wavjepa_nat_repo",
    "vision_spatial_pool",
    "video_fps",
    "video_n_frames",
    "video_resolution",
    "audio_sample_rate",
    "wavjepa_base_token_rate_hz",
    "wavjepa_nat_token_rate_hz",
}


def load_and_validate_manifest(cache_dir: str) -> Dict:
    """Load manifest.json and assert all required keys are present."""
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No manifest.json in {cache_dir!r}. "
            "Run scripts/extract_features_av.py first."
        )
    with open(path) as f:
        m = json.load(f)

    missing = REQUIRED_MANIFEST_KEYS - set(m.keys())
    if missing:
        raise ValueError(
            f"manifest.json is missing required keys: {sorted(missing)}\n"
            f"Re-run extraction to regenerate the manifest."
        )
    return m


# ── TDM bin mapping ───────────────────────────────────────────────────────
def _ts_to_tdm_bins(
    timestamps: Tensor,   # (T, 2) float  [start_s, end_s]
    clip_duration_s: float,
    max_bins: int,
) -> Tensor:
    """Map per-token start timestamps → integer TDM bins in [0, max_bins).

    Uses the start time of each token (timestamps[:, 0]).
    """
    starts = timestamps[:, 0]                             # (T,)
    bins   = (starts / clip_duration_s * max_bins).long() # (T,)
    bins   = bins.clamp(0, max_bins - 1)
    return bins


def _shard(vid: str) -> str:
    return vid[:2]


def _feat_path(cache_dir: str, vid: str) -> str:
    return os.path.join(cache_dir, _shard(vid), f"{vid}.pt")


# ── Dataset ───────────────────────────────────────────────────────────────
class AVCachedDataset(Dataset):
    """Audio-visual cached feature dataset for M2.

    Parameters
    ----------
    cache_dir : str
        Root of the AV feature cache (contains manifest.json and shards).
    clip_ids : List[str]
        List of video_ids to load. Falls back to scanning the cache dir if
        None (useful for quick tests).
    max_tdm_bins : int
        Shared TDM axis resolution (matches AVJepaConfig.max_tdm_bins = 512).
    audio_mode : str
        "base"   — use ambient_base only
        "nat"    — use ambient_nat only
        "mean"   — average base + nat (default; both encoders run at same rate)
    """

    def __init__(
        self,
        cache_dir: str,
        clip_ids: Optional[List[str]] = None,
        max_tdm_bins: int = 512,
        audio_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.cache_dir    = cache_dir
        self.max_tdm_bins = max_tdm_bins
        self.audio_mode   = audio_mode

        self.manifest = load_and_validate_manifest(cache_dir)
        self._rng     = random.Random(0)

        if clip_ids is not None:
            self.clip_ids = clip_ids
        else:
            # Auto-discover from cache shards
            self.clip_ids = self._discover_clip_ids()

        print(
            f"[AVCachedDataset] {len(self.clip_ids)} clips, "
            f"max_tdm_bins={max_tdm_bins}, audio_mode={audio_mode}",
            flush=True,
        )

    def _discover_clip_ids(self) -> List[str]:
        ids = []
        for shard in sorted(os.listdir(self.cache_dir)):
            shard_dir = os.path.join(self.cache_dir, shard)
            if not os.path.isdir(shard_dir) or shard.startswith("."):
                continue
            for fname in sorted(os.listdir(shard_dir)):
                if fname.endswith(".pt"):
                    ids.append(fname[:-3])
        return ids

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> Dict:
        vid  = self.clip_ids[idx]
        path = _feat_path(self.cache_dir, vid)
        try:
            d = torch.load(path, map_location="cpu", weights_only=True)
        except (FileNotFoundError, RuntimeError):
            # Fallback to a random other sample (mirrors M1 cached dataset)
            alt = self._rng.randrange(len(self.clip_ids))
            if alt == idx:
                alt = (idx + 1) % len(self.clip_ids)
            return self.__getitem__(alt)

        clip_dur = float(d.get("clip_duration_s", 10.0))

        # ── Vision: (32, 16, 1024) → flatten → (512, 1024) ─────────────
        vis  = d["vision"]          # (32, 16, 1024) bf16
        T_v  = vis.shape[0]
        S_v  = vis.shape[1]
        D_v  = vis.shape[2]
        # Flatten temporal × spatial → single sequence on TDM axis
        vis_flat = vis.reshape(T_v * S_v, D_v)   # (512, 1024)

        vis_ts = d["vision_ts"]     # (32, 2) — one timestamp per temporal group
        # Each temporal token covers S_v=16 spatial tokens → expand timestamps
        vis_ts_exp = vis_ts.unsqueeze(1).expand(T_v, S_v, 2).reshape(T_v * S_v, 2)

        vis_bins = _ts_to_tdm_bins(vis_ts_exp, clip_dur, self.max_tdm_bins)  # (512,)

        # ── Audio: combine base + nat ────────────────────────────────────
        base = d["ambient_base"]    # (T_b, 768) bf16
        nat  = d["ambient_nat"]     # (T_n, 768) bf16
        base_ts = d["ambient_base_ts"]   # (T_b, 2)
        nat_ts  = d["ambient_nat_ts"]    # (T_n, 2)

        if self.audio_mode == "base":
            aud      = base
            aud_ts   = base_ts
        elif self.audio_mode == "nat":
            aud      = nat
            aud_ts   = nat_ts
        else:  # "mean" — base and nat have same T (99.6 Hz each), so avg
            if base.shape[0] == nat.shape[0]:
                aud    = (base.float() + nat.float()).mul_(0.5).to(torch.bfloat16)
                aud_ts = base_ts
            else:
                # Fall back to base if shapes differ
                aud    = base
                aud_ts = base_ts

        aud_bins = _ts_to_tdm_bins(aud_ts, clip_dur, self.max_tdm_bins)   # (T_a,)

        return {
            "feats": {
                "vision":  vis_flat,   # (512, 1024) bf16
                "ambient": aud,        # (T_a, 768)  bf16
            },
            "tbins": {
                "vision":  vis_bins,   # (512,)  int64
                "ambient": aud_bins,   # (T_a,)  int64
            },
            "clip_id": vid,
        }


# ── Collation ──────────────────────────────────────────────────────────────
def av_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of per-sample dicts into a batched dict.

    Vision and ambient may have different T per sample, so we pad to the
    max length within the batch.  Padding tokens use bin=0 and zero feats;
    they are masked out in the trainer via a key_padding_mask if needed.
    """
    B = len(batch)

    # Determine max sequence lengths
    max_Tv = max(s["feats"]["vision"].shape[0]  for s in batch)
    max_Ta = max(s["feats"]["ambient"].shape[0] for s in batch)
    D_v    = batch[0]["feats"]["vision"].shape[1]
    D_a    = batch[0]["feats"]["ambient"].shape[1]

    vis_feats  = torch.zeros(B, max_Tv, D_v,  dtype=torch.bfloat16)
    aud_feats  = torch.zeros(B, max_Ta, D_a,  dtype=torch.bfloat16)
    vis_bins   = torch.zeros(B, max_Tv,       dtype=torch.long)
    aud_bins   = torch.zeros(B, max_Ta,       dtype=torch.long)
    vis_pad    = torch.ones (B, max_Tv,       dtype=torch.bool)  # True=padding
    aud_pad    = torch.ones (B, max_Ta,       dtype=torch.bool)

    clip_ids = []
    for i, s in enumerate(batch):
        Tv = s["feats"]["vision"].shape[0]
        Ta = s["feats"]["ambient"].shape[0]
        vis_feats[i, :Tv] = s["feats"]["vision"]
        aud_feats[i, :Ta] = s["feats"]["ambient"]
        vis_bins [i, :Tv] = s["tbins"]["vision"]
        aud_bins [i, :Ta] = s["tbins"]["ambient"]
        vis_pad  [i, :Tv] = False
        aud_pad  [i, :Ta] = False
        clip_ids.append(s["clip_id"])

    return {
        "feats": {"vision": vis_feats, "ambient": aud_feats},
        "tbins": {"vision": vis_bins,  "ambient": aud_bins},
        "padding_mask": {"vision": vis_pad, "ambient": aud_pad},
        "clip_ids": clip_ids,
    }


# ── Manifest validation helper (used by train_m2.py) ─────────────────────
def validate_av_manifest(cache_dir: str) -> Dict:
    """Validate manifest and return it.  Raises on mismatch."""
    return load_and_validate_manifest(cache_dir)


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/jepa_m2_cache"
    print(f"Smoke-testing AVCachedDataset from {cache}")
    ds = AVCachedDataset(cache, max_tdm_bins=512)
    s  = ds[0]
    print("feats shapes:")
    for k, v in s["feats"].items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
    print("tbins shapes:")
    for k, v in s["tbins"].items():
        print(f"  {k}: {tuple(v.shape)}  max_bin={v.max().item()}")
    print(f"clip_id: {s['clip_id']}")
