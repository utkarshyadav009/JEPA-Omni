"""models/jepa_memory.py — Phase 3: the memory BMO actually stores and looks up.

This is the piece the whole track exists for: enrol a person once, recognise them
later, and — the part that matters most in a home — say "I don't know you" instead of
confidently greeting a stranger by the maker's name.

DESIGN, each choice traceable to a measurement rather than taste:

* **Cosine on L2-normalised identity embeddings.** `IdentityHead` returns L2-normalised
  vectors and was trained with AAM-softmax, an ANGULAR margin — so the geometry the
  memory queries is the geometry the head was optimised for. Using anything else here
  would throw away the reason ArcFace was chosen.

* **Running centroid + spread per identity, not a list of every sighting.** Phase 1C
  measured that enrolment depth genuinely sharpens recognition (N=5 gallery, WavJEPA:
  1 clip 0.584 -> 8 clips 0.760), so repeated exposure must IMPROVE the entry. Storing a
  centroid gives that for free at O(1) memory per person, and the spread gives a
  per-identity confidence scale. `n` is kept so the centroid is a true running mean.

* **A calibrated rejection threshold, fit on held-out identities.** The operating point
  is chosen to hit a target false-accept rate on people the system has NEVER enrolled —
  not on a fixed cosine guess. Phase 2c measured TAR@FAR=1% = 0.691 for the joint head,
  so this threshold is where that number is actually cashed in.

* **Unknown by default.** `query()` returns `None` unless the best match clears the
  threshold AND beats the runner-up by a margin. The second condition matters: two
  enrolled people who look/sound alike should produce "not sure", not a coin flip.

Deliberately NOT here: any face-recognition database, image, or raw audio. The store
holds only abstract JEPA-space vectors, which is the north star's whole point.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryEntry:
    label: str
    centroid: Tensor          # (D,) L2-normalised running mean
    n: int = 0                # sightings folded in
    spread: float = 0.0       # mean cosine distance of sightings to the centroid


@dataclass
class MemoryConfig:
    threshold: float = 0.5        # min cosine to accept a match (calibrate_threshold)
    margin: float = 0.02          # best must beat runner-up by this much
    max_entries: int = 512


class JepaMemory:
    """Enrol / query / consolidate over identity embeddings."""

    def __init__(self, cfg: Optional[MemoryConfig] = None):
        self.cfg = cfg or MemoryConfig()
        self.entries: Dict[str, MemoryEntry] = {}

    # ── write path ────────────────────────────────────────────────────────
    def enroll(self, emb: Tensor, label: str) -> MemoryEntry:
        """Fold one sighting into `label`'s entry, creating it if new.

        The centroid is a true running mean over sightings (re-normalised), so
        enrolling the same person repeatedly tightens the estimate rather than
        overwriting it or growing memory."""
        z = F.normalize(emb.detach().float().flatten(), dim=0)
        e = self.entries.get(label)
        if e is None:
            if len(self.entries) >= self.cfg.max_entries:
                raise RuntimeError(f"memory full ({self.cfg.max_entries} identities)")
            self.entries[label] = MemoryEntry(label=label, centroid=z, n=1, spread=0.0)
            return self.entries[label]
        raw = e.centroid * e.n + z
        e.n += 1
        e.centroid = F.normalize(raw, dim=0)
        d = float(1.0 - torch.dot(z, e.centroid))
        e.spread += (d - e.spread) / e.n          # running mean of distance-to-centroid
        return e

    def forget(self, label: str) -> bool:
        return self.entries.pop(label, None) is not None

    # ── read path ─────────────────────────────────────────────────────────
    def scores(self, emb: Tensor) -> List[Tuple[str, float]]:
        if not self.entries:
            return []
        z = F.normalize(emb.detach().float().flatten(), dim=0)
        labels = list(self.entries)
        C = torch.stack([self.entries[k].centroid for k in labels])
        s = (C @ z).tolist()
        return sorted(zip(labels, s), key=lambda t: -t[1])

    def query(self, emb: Tensor) -> Tuple[Optional[str], float, str]:
        """-> (label or None, confidence, reason).

        Returns None on BOTH failure modes, and says which one it was:
          'below_threshold' -- nobody enrolled looks/sounds close enough (a stranger)
          'ambiguous'       -- two enrolled people are too close to call
        Confidently naming a stranger is the failure this whole method exists to avoid."""
        rank = self.scores(emb)
        if not rank:
            return None, 0.0, "empty_memory"
        label, best = rank[0]
        if best < self.cfg.threshold:
            return None, best, "below_threshold"
        if len(rank) > 1 and (best - rank[1][1]) < self.cfg.margin:
            return None, best, "ambiguous"
        return label, best, "match"

    # ── calibration ───────────────────────────────────────────────────────
    def calibrate_threshold(self, genuine: np.ndarray, impostor: np.ndarray,
                            target_far: float = 0.01) -> Dict[str, float]:
        """Set the threshold from measured score distributions.

        genuine  : cosine scores of sightings against their OWN enrolled entry
        impostor : cosine scores of people who are NOT enrolled, against every entry
        The threshold is the impostor distribution's (1 - target_far) quantile, i.e. the
        point where the false-accept rate is what you asked for. Reports the true-accept
        rate you get in exchange, so the trade is explicit rather than implied."""
        thr = float(np.quantile(impostor, 1.0 - target_far))
        self.cfg.threshold = thr
        return {"threshold": thr,
                "far": float((impostor >= thr).mean()),
                "tar": float((genuine >= thr).mean()),
                "target_far": target_far}

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"cfg": vars(self.cfg),
                    "entries": {k: {"label": e.label, "centroid": e.centroid,
                                    "n": e.n, "spread": e.spread}
                                for k, e in self.entries.items()}}, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "JepaMemory":
        """`device` moves the centroids after loading. Without it they stay on CPU
        (map_location="cpu" is deliberate -- it must load on a machine with no GPU) and the
        first query against a CUDA embedding dies with
            "Expected all tensors to be on the same device, but got mat is on cpu"
        which is exactly what happened on the Jetson the first time memory was persisted
        across processes (2026-08-15). Pass the device you will query from."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(MemoryConfig(**d["cfg"]))
        m.entries = {k: MemoryEntry(**v) for k, v in d["entries"].items()}
        if device is not None:
            m.to(device)
        return m

    def to(self, device) -> "JepaMemory":
        """Move every enrolled centroid onto `device`, in place."""
        for e in self.entries.values():
            e.centroid = e.centroid.to(device)
        return self

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (f"JepaMemory({len(self.entries)} identities, "
                f"thr={self.cfg.threshold:.3f}, margin={self.cfg.margin:.3f})")


if __name__ == "__main__":
    torch.manual_seed(0)
    D = 256
    mem = JepaMemory()
    # three synthetic identities, each a tight cluster on the sphere
    people = {p: F.normalize(torch.randn(D), dim=0) for p in ("maker", "alex", "sam")}
    for p, base in people.items():
        for _ in range(5):
            mem.enroll(F.normalize(base + 0.25 * torch.randn(D), dim=0), p)
    print(mem)
    for p, base in people.items():
        e = mem.entries[p]
        print(f"  {p:6s} n={e.n} spread={e.spread:.4f}")

    gen = np.array([float(torch.dot(mem.entries[p].centroid,
                                    F.normalize(b + 0.25 * torch.randn(D), dim=0)))
                    for p, b in people.items() for _ in range(200)])
    imp = np.array([max(s for _, s in mem.scores(F.normalize(torch.randn(D), dim=0)))
                    for _ in range(400)])
    print("calibration:", {k: round(v, 4) for k, v in
                           mem.calibrate_threshold(gen, imp, target_far=0.01).items()})

    known = F.normalize(people["maker"] + 0.25 * torch.randn(D), dim=0)
    stranger = F.normalize(torch.randn(D), dim=0)
    print("known    ->", mem.query(known))
    print("stranger ->", mem.query(stranger))
    mem.save("/tmp/_jepa_mem_test.pt")
    print("reloaded ->", JepaMemory.load("/tmp/_jepa_mem_test.pt"))
