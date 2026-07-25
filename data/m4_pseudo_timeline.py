"""data/m4_pseudo_timeline.py — Ego4D-independent synthetic turn-taking data.

Builds (clip, tick) examples out of VGGSound + our existing captions, no new
data collection required. Simulates a streaming observer inside each 10s
clip via the same causal token-cutoff trick used in
scripts/m4_shift_metric_eval.py: filtering feats/tbins to tokens with
tbin < cutoff approximates "what a live system would have seen so far."

Label rule (deliberately simple, matches the M4 design doc's "narrate at
scene boundaries" proposal):
  - the FIRST tick of a clip (25% of its TDM axis observed) = SPEAK,
    target = the clip's caption text (a plausible "something new just
    happened, describe it" moment)
  - later ticks of the SAME clip (50/75/100%) = SILENCE, target = the
    <silence> control token (the scene hasn't changed, nothing new to say)

This teaches the MECHANISM (when do we emit text vs. <silence>, control-
token plumbing, tick-conditioned soft-prompt refresh) on data we already
have. It does NOT teach real turn-taking dynamics (no second speaker, no
interruption events, no gaze/attention cues) -- that gap is exactly why
Ego4D LAM/TTM (or a fallback, see scripts/m4_fallback_scoping.md) is still
needed before this becomes a real turn-taking model, not just a plumbing
validation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

SPEAK_FRACTION = 0.25
SILENCE_FRACTIONS = (0.5, 0.75, 1.0)


class M4PseudoTimelineDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], cache_dir: str, tokenizer,
                 silence_token_id: int, max_tdm_bins: int = 512):
        from data.av_cached_dataset import AVCachedDataset
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.silence_token_id = silence_token_id
        self.max_tdm_bins = max_tdm_bins
        clip_ids = [cid for cid, _ in pairs]
        self.av_ds = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                                      max_tdm_bins=max_tdm_bins, audio_mode="mean")
        self.caption_by_id = dict(pairs)

        self.index: List[Tuple[int, float, bool]] = []
        for i in range(len(self.av_ds)):
            self.index.append((i, SPEAK_FRACTION, True))
            for frac in SILENCE_FRACTIONS:
                self.index.append((i, frac, False))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        clip_idx, frac, is_speak = self.index[idx]
        item = self.av_ds[clip_idx]
        cutoff = int(frac * self.max_tdm_bins)

        feats_f, tbins_f = {}, {}
        for m in ("vision", "ambient"):
            tb = item["tbins"][m]
            keep = tb < cutoff
            if keep.sum().item() < 1:
                keep = torch.ones_like(tb, dtype=torch.bool)   # degenerate cutoff: keep everything
            feats_f[m] = item["feats"][m][keep]
            tbins_f[m] = tb[keep]

        if is_speak:
            text = self.caption_by_id[item["clip_id"]]
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        else:
            text = "<silence>"
            ids = [self.silence_token_id, self.tokenizer.eos_token_id]

        return {
            "feats": feats_f, "tbins": tbins_f,
            "target_ids": torch.tensor(ids, dtype=torch.long),
            "label": "speak" if is_speak else "silence",
            "text": text, "clip_id": item["clip_id"], "tick_fraction": frac,
        }


def m4_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    B = len(batch)
    max_Tv = max(s["feats"]["vision"].shape[0] for s in batch)
    max_Ta = max(s["feats"]["ambient"].shape[0] for s in batch)
    D_v = batch[0]["feats"]["vision"].shape[1]
    D_a = batch[0]["feats"]["ambient"].shape[1]

    vis_feats = torch.zeros(B, max_Tv, D_v, dtype=torch.bfloat16)
    aud_feats = torch.zeros(B, max_Ta, D_a, dtype=torch.bfloat16)
    vis_bins = torch.zeros(B, max_Tv, dtype=torch.long)
    aud_bins = torch.zeros(B, max_Ta, dtype=torch.long)
    vis_pad = torch.ones(B, max_Tv, dtype=torch.bool)
    aud_pad = torch.ones(B, max_Ta, dtype=torch.bool)

    max_L = max(s["target_ids"].shape[0] for s in batch)
    tgt_ids = torch.full((B, max_L), pad_token_id, dtype=torch.long)
    tgt_mask = torch.zeros(B, max_L, dtype=torch.long)

    labels_str, texts, clip_ids, fracs = [], [], [], []
    for i, s in enumerate(batch):
        Tv, Ta = s["feats"]["vision"].shape[0], s["feats"]["ambient"].shape[0]
        vis_feats[i, :Tv] = s["feats"]["vision"]; aud_feats[i, :Ta] = s["feats"]["ambient"]
        vis_bins[i, :Tv] = s["tbins"]["vision"]; aud_bins[i, :Ta] = s["tbins"]["ambient"]
        vis_pad[i, :Tv] = False; aud_pad[i, :Ta] = False
        L = s["target_ids"].shape[0]
        tgt_ids[i, :L] = s["target_ids"]; tgt_mask[i, :L] = 1
        labels_str.append(s["label"]); texts.append(s["text"])
        clip_ids.append(s["clip_id"]); fracs.append(s["tick_fraction"])

    return {
        "feats": {"vision": vis_feats, "ambient": aud_feats},
        "tbins": {"vision": vis_bins, "ambient": aud_bins},
        "padding_mask": {"vision": vis_pad, "ambient": aud_pad},
        "target_ids": tgt_ids, "target_mask": tgt_mask,
        "labels": labels_str, "texts": texts, "clip_ids": clip_ids, "tick_fractions": fracs,
    }
