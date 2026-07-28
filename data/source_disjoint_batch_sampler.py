"""data/source_disjoint_batch_sampler.py — item 4: prevent same-source-video
windows from landing in the same contrastive batch as false negatives.

VGGSound has no risk (each clip IS a unique source video, 1:1). Ego4D and
EasyCom extraction produce MULTIPLE 10s windows per source video/session-
chunk -- two windows from the same continuous recording are often near-
duplicate content, and InfoNCE/GradCache treat every other item in the
batch as a negative. Without this, those near-duplicates become false
negatives, actively penalising the model for saying two clearly-related
clips are similar.

Naming convention (defines source identity from clip_id alone, no extra
manifest lookup needed):
  VGGSound: clip_id unchanged (e.g. "OxPnZzn1_L8_000883") -- each clip is
            its own source.
  Ego4D:    "ego4d_{source_video_id}_w{window_idx:04d}"
  EasyCom:  "easycom_s{session}_{chunk}_w{window_idx:04d}"
source_key_from_clip_id() strips the trailing "_w<digits>" suffix for the
ego4d_/easycom_ prefixes; everything else (VGGSound) is its own key.
"""
from __future__ import annotations

import random
import re
from typing import Iterator, List, Optional

from torch.utils.data import Sampler

_WINDOW_SUFFIX_RE = re.compile(r"^(ego4d_.+|easycom_.+)_w\d+$")


def source_key_from_clip_id(clip_id: str) -> str:
    m = _WINDOW_SUFFIX_RE.match(clip_id)
    if m:
        return m.group(1)
    return clip_id  # VGGSound (or anything unrecognized): each clip is its own source


class SourceDisjointBatchSampler(Sampler[List[int]]):
    """Yields batches of `batch_size` indices such that no batch contains
    two indices sharing a source key. Greedy construction: shuffle index
    order each epoch, then fill each batch left-to-right, skipping (and
    deferring to later in the same pass) any index whose source is already
    present in the batch being filled. A handful of leftover indices whose
    source collides with every open batch at the end of a pass are
    dropped (logged, not reintroduced) rather than forming an undersized
    or colliding final batch.

    DISTRIBUTED (rank/num_replicas, added 2026-07-26): every rank computes
    the SAME global batch list (identical seed+epoch -> identical shuffle
    -> identical greedy construction, no rank-specific randomness), THEN
    slices it round-robin by rank (batch i goes to rank i % num_replicas).
    This is the key property a naive "just run torchrun on the existing
    sampler" approach would violate: without rank-aware slicing, every
    rank would iterate the FULL batch list independently and end up
    processing IDENTICAL batches (not complementary shards) -- each
    all-gather would concatenate 4 copies of the same 48 embeddings, not
    192 genuinely distinct negatives. Slicing after the fact guarantees
    (a) every yielded batch is still internally source-collision-free
    (the property was established globally before slicing), (b) different
    ranks get DIFFERENT batches, (c) every rank yields the exact same
    NUMBER of batches per epoch (required for DDP step-lockstep -- the
    global list is truncated to a multiple of num_replicas first, so no
    rank can run out of batches before another and hang on an all-reduce)."""

    def __init__(self, clip_ids: List[str], batch_size: int, drop_last: bool = True, seed: int = 0,
                 rank: int = 0, num_replicas: int = 1):
        self.clip_ids = clip_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.source_keys = [source_key_from_clip_id(c) for c in clip_ids]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_global_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.clip_ids)))
        rng.shuffle(order)

        pending = list(order)
        n_dropped = 0
        batches: List[List[int]] = []
        while pending:
            batch: List[int] = []
            batch_sources = set()
            leftover = []
            for idx in pending:
                key = self.source_keys[idx]
                if key in batch_sources:
                    leftover.append(idx)
                    continue
                batch.append(idx)
                batch_sources.add(key)
                if len(batch) == self.batch_size:
                    break
            else:
                # exhausted `pending` without filling the batch
                if self.drop_last or not batch:
                    n_dropped += len(batch)
                    pending = leftover
                    continue
            # whatever wasn't consumed into `batch` (including anything
            # after the break point) goes back into the pool
            consumed = set(batch)
            remaining_after_batch = [i for i in pending if i not in consumed]
            pending = remaining_after_batch
            if len(batch) == self.batch_size or (not self.drop_last and batch):
                batches.append(batch)
            else:
                n_dropped += len(batch)
        if n_dropped:
            import sys
            print(f"[SourceDisjointBatchSampler] epoch {self.epoch}: dropped {n_dropped} "
                  f"leftover indices that could not form a same-source-free batch", file=sys.stderr)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build_global_batches()
        if self.num_replicas > 1:
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            if usable < len(batches):
                import sys
                print(f"[SourceDisjointBatchSampler] epoch {self.epoch}: truncating "
                      f"{len(batches) - usable} trailing batch(es) so every rank sees "
                      f"the same count ({usable // self.num_replicas} batches/rank)", file=sys.stderr)
            batches = batches[:usable]
            batches = batches[self.rank::self.num_replicas]
        yield from batches

    def __len__(self) -> int:
        # approximate: assumes source collisions are rare enough not to
        # change the batch count materially (true for this corpus mix --
        # see PROVENANCE for the measured collision/drop rate)
        n = len(self.clip_ids)
        total = n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        return total // self.num_replicas
