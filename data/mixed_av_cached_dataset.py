"""data/mixed_av_cached_dataset.py — combine multiple independent
AVCachedDataset corpora (each its own cache_dir, e.g. the persistent
VGGSound cache, a freshly-extracted Ego4D train split, a freshly-extracted
AudioSet-Strong draw) into ONE dataset for M2 training.

Deliberately NOT a single merged cache_dir with symlinks: independent
AVCachedDataset instances (one per cache_dir) are simpler, carry zero risk
of ever writing into or corrupting the persistent VGGSound cache, and
AVCachedDataset already validates its own manifest.json per cache_dir --
no need to reconcile N manifests into one.

__getitem__(idx) dispatches by a precomputed index map (which underlying
dataset + which local index) built at construction time from the
concatenated clip_id list -- so idx short-circuits which
AVCachedDataset.__getitem__ actually runs, no per-call path parsing.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from torch.utils.data import Dataset

from data.av_cached_dataset import AVCachedDataset


class MixedSource(NamedTuple):
    name: str
    cache_dir: str
    clip_ids: List[str]


class MixedAVCachedDataset(Dataset):
    def __init__(self, sources: List[MixedSource], max_tdm_bins: int = 512, audio_mode: str = "mean"):
        assert len(sources) >= 2, "MixedAVCachedDataset is for combining 2+ corpora; use AVCachedDataset directly for one"
        self._datasets = {
            src.name: AVCachedDataset(cache_dir=src.cache_dir, clip_ids=src.clip_ids,
                                       max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)
            for src in sources
        }
        self.clip_ids: List[str] = []
        self._index_map: List[Tuple[str, int]] = []
        for src in sources:
            ds = self._datasets[src.name]
            self.clip_ids.extend(ds.clip_ids)
            self._index_map.extend((src.name, i) for i in range(len(ds.clip_ids)))

        total = len(self.clip_ids)
        counts = ", ".join(f"{len(self._datasets[src.name].clip_ids)} {src.name}" for src in sources)
        shares = ", ".join(f"{len(self._datasets[src.name].clip_ids) / total * 100:.1f}% {src.name}" for src in sources)
        print(f"[MixedAVCachedDataset] {counts} = {total} total clips ({shares})", flush=True)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Dict:
        source, local_idx = self._index_map[idx]
        return self._datasets[source][local_idx]
