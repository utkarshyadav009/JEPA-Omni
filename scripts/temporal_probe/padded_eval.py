"""scripts/temporal_probe/padded_eval.py — P0.1

The padding-corrected gallery retrieval eval.

WHY THIS IS A NEW FILE AND NOT AN EDIT TO train_m2.py
-----------------------------------------------------
Task rule 2 forbids modifying train_m2.py, and train_m2.pool_and_project is
ALSO the training-time contrastive head -- editing it in place would change the
training objective, which is out of scope and would invalidate the checkpoint's
provenance. This module provides masked equivalents used by EVAL ONLY. The
unmasked originals in train_m2.py are untouched and still produce every
previously published number bit-for-bit.

THE DEFECT
----------
data/av_cached_dataset.py:av_collate_fn pads ambient to the longest clip in the
batch, fills the pad slots with ZEROS, and returns a "padding_mask" naming them.
No consumer in the M2 eval path ever passed that mask on. Consequences, both of
which this module fixes:

  (1) ATTENTION LEAK. AVJepaPredictor._backbone accepts a key_padding_mask and
      every caller passed None, so zero-feature pad tokens (which still receive a
      modality embedding and the bin-0 temporal embedding, so they are NOT zero
      vectors by the time they reach attention) were attended to by every real
      token in all 8 layers.
  (2) POOLING LEAK. pool_and_project does src["ambient"].mean(1) over the PADDED
      length, so a clip in a batch with a longer neighbour has its representation
      divided by a larger denominator and averaged with junk.

Because the batch's pad length is set by the LONGEST clip in that batch, both
leaks make a clip's embedding a function of which other clips it was batched
with. Combined with train_m2.build_dataloader's shuffle=True on the eval path
(sampler is None => shuffle=True, train_m2.py:352-356), this is the mechanism
behind the measured run-to-run spread in the published gallery R@1.

Measured on the 1,545-clip VGGSound gallery: ambient T_a spans 400-1002 tokens,
1543/1545 clips are padded, ~0.83% of ambient slots per batch are pad
(docs/artifacts/temporal_probe/ambient_padding_stats.json).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def flat_padding_mask(pad: Dict[str, Tensor], feats: Dict[str, Tensor]) -> Tensor:
    """Concatenate per-modality pad masks in the SAME order AVJepaPredictor._embed
    concatenates tokens (iteration order of feats), so index i of the flat mask
    names index i of the token sequence. Asserted, not assumed."""
    parts, names = [], []
    for m in feats:
        assert m in pad, f"no padding mask for modality '{m}'"
        assert pad[m].shape == feats[m].shape[:2], (
            f"pad['{m}'] {tuple(pad[m].shape)} does not match feats['{m}'] "
            f"{tuple(feats[m].shape[:2])}")
        parts.append(pad[m]); names.append(m)
    flat = torch.cat(parts, 1)
    # nn.MultiheadAttention emits NaN for a row that is entirely padding.
    assert not flat.all(1).any(), "a sample is entirely padding -- attention would emit NaN"
    return flat


def masked_mean(x: Tensor, pad: Tensor) -> Tensor:
    """Mean over non-pad positions only. x (B,T,D), pad (B,T) True=PAD."""
    keep = (~pad).unsqueeze(-1).to(x.dtype)                 # (B,T,1)
    n = keep.sum(1).clamp_min(1.0)                          # (B,1)
    return (x * keep).sum(1) / n


def pool_and_project_masked(
    predictor, vision_proj, ambient_proj,
    feats: Dict[str, Tensor], tbins: Dict[str, Tensor], pad: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Padding-corrected twin of train_m2.pool_and_project. Identical in every
    other respect: same encode_source_tokens masked-pass semantics, same two
    linear heads, same L2 normalisation."""
    flat = flat_padding_mask(pad, feats)
    src = predictor.encode_source_tokens(feats, tbins, key_padding_mask=flat)
    z_v = F.normalize(vision_proj(masked_mean(src["vision"], pad["vision"])).float(), dim=-1)
    z_a = F.normalize(ambient_proj(masked_mean(src["ambient"], pad["ambient"])).float(), dim=-1)
    return z_v, z_a


def encode_world_state_masked(
    predictor, feats: Dict[str, Tensor], tbins: Dict[str, Tensor], pad: Dict[str, Tensor],
) -> Tensor:
    """Padding-corrected world-state: pad tokens excluded from both the backbone's
    self-attention and the single-query attentive pool."""
    flat = flat_padding_mask(pad, feats)
    return predictor.encode_world_state(feats, tbins, key_padding_mask=flat)
