"""models/world_state_builder.py — the ONE shared World-State construction
path, built 2026-07-26 to close the item-0 divergence: the streaming loop,
scripts/m5_bothpresent_extract.py, and scripts/m5_ood_falsifier.py had each
independently reimplemented (and each got wrong, the same way) the
vision/ambient feature construction that scripts/extract_features_av.py
and data/av_cached_dataset.py use for M2's actual training data.

Three confirmed bugs this file fixes, all verified by direct source reads
(not by description, including any of my own prior descriptions):
  1. No spatial pooling -- models/vision_encoder.py's VisionEncoder.encode()
     returns (1, 8192, 1024) (confirmed by direct execution); training
     cached (32, 16, 1024) = 512 tokens via scripts/extract_features_av.py's
     _spatial_pool(). Streaming fed the raw 8192.
  2. Vision tbins used a linspace RAMP (512 or 8192 distinct values);
     training's _ts_to_tdm_bins (data/av_cached_dataset.py) produces a
     STAIRCASE -- 32 distinct group-timestamps, each shared by 16 spatial
     tokens, floor-mapped into max_bins.
  3. Ambient/WavJEPA was absent entirely -- feats had only a "vision" key.
     RollingAudioBuffer fed Whisper (for the decision head's speech-
     activity feature), never WavJEPA (M2's actual audio encoder).

This module IMPORTS the real training functions rather than
reimplementing them: _spatial_pool and _decode_audio_raw's resample/mono-
mix logic from scripts/extract_features_av.py, _ts_to_tdm_bins from
data/av_cached_dataset.py. Only the "compute real per-token timestamps
for a window whose duration is measured, not assumed" logic is new (a
direct generalisation of extract_features_av.py's _video_ts/_audio_ts,
parameterised on the TRUE captured duration instead of CLIP_DURATION_S).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_features_av import (
    _spatial_pool,
    _vision_ts,
    VISION_SPAT,
    VISION_TEMP,
    VISION_DIM,
    AUDIO_SR,
    CLIP_DURATION_S,
)
from data.av_cached_dataset import _ts_to_tdm_bins


def _group_timestamps(n_temp: int, n_raw_frames: int, true_window_dur_sec: float) -> torch.Tensor:
    """FINDING (Phase 1.2 gate, 2026-07-26): training's OWN extraction
    script has an internal inconsistency -- scripts/extract_features_av.py
    calls `vis_ts = _vision_ts()` with NO args, i.e. using _vision_ts's
    default `dur=CLIP_DURATION_S` (the ASSUMED 10.0), not the real per-clip
    duration -- while data/av_cached_dataset.py's _ts_to_tdm_bins call
    later divides by the REAL saved `clip_duration_s` (derived from actual
    decoded audio length). A internally-consistent implementation (this
    function's first version, using true_window_dur_sec throughout) gave
    high cosine similarity (mean 0.997) but did NOT bit-for-bit reproduce
    the cached tbins, because training itself doesn't cancel duration the
    way a clean derivation would.
    To EXACTLY reproduce training (the Phase 1.2 gate's requirement), this
    function now calls the REAL _vision_ts() (imported, not reimplemented)
    with its default dur=CLIP_DURATION_S for the timestamp CONSTRUCTION,
    while the caller still passes the true captured duration to
    _ts_to_tdm_bins for the bin DIVISION -- replicating training's
    inconsistency deliberately, not fixing it, because "replicate exactly"
    was the instruction. Returns (n_temp, 2) [start_s, end_s]."""
    assert n_temp == VISION_TEMP and n_raw_frames == 64, \
        "this path only replicates _vision_ts's exact defaults; a genuinely different frame/temporal config needs a fresh reproduction check"
    return _vision_ts()   # dur=CLIP_DURATION_S (10.0), matching training's literal call exactly


def _uniform_token_timestamps(n_tokens: int, true_window_dur_sec: float) -> torch.Tensor:
    """Direct generalisation of extract_features_av.py's _audio_ts,
    parameterised on the true captured duration. Returns (n_tokens, 2)."""
    td = true_window_dur_sec / n_tokens
    ts = torch.zeros(n_tokens, 2)
    for t in range(n_tokens):
        ts[t, 0] = t * td
        ts[t, 1] = (t + 1) * td
    return ts


@dataclass
class WorldStateBuildResult:
    feats: Dict[str, torch.Tensor]   # {"vision": (1,512,1024), "ambient": (1,T_a,768)}
    tbins: Dict[str, torch.Tensor]   # matching (1,512) / (1,T_a)
    vision_raw_ntok: int
    ambient_ntok: int


def build_world_state_features(
    frames: torch.Tensor,          # (T_raw, C, H, W) uint8, T_raw==64 (V-JEPA2's trained frame count)
    audio: torch.Tensor,           # (n_samples,) float32, 16kHz mono (or multichannel -- mixed down here)
    true_window_dur_sec: float,    # the REAL captured duration of this window, not assumed
    vision_encoder,                # models.vision_encoder.VisionEncoder
    base_encoder,                  # models.audio_encoder.AudioEncoder (WavJEPA-base, 1ch)
    nat_encoder,                   # models.audio_encoder.AudioEncoder (WavJEPA-nat, 2ch)
    max_tdm_bins: int,
    device: torch.device,
) -> WorldStateBuildResult:
    """The one shared construction. Returns feats/tbins for BOTH modalities
    together, from the same true_window_dur_sec -- there is no code path
    here that returns one without the other (closes bug 3 structurally,
    not just by convention)."""

    # ---- VISION: encode -> reshape -> pool (bug 1 + bug 2 fix) ----------
    with torch.no_grad():
        raw = vision_encoder.encode(frames.unsqueeze(0).to(device))   # (1, 8192, 1024) -- confirmed by direct execution
    vision_raw_ntok = raw.shape[1]
    raw = raw[0]  # (8192, 1024)
    assert raw.shape[0] == VISION_TEMP * 256, \
        f"expected {VISION_TEMP*256} raw vision tokens, got {raw.shape[0]} -- ViT-L frame/patch config changed?"
    vis_full = raw.view(VISION_TEMP, 256, VISION_DIM)
    vis_pooled = _spatial_pool(vis_full).to(torch.bfloat16)   # (32, 16, 1024) -- SAME function as training

    vis_ts = _group_timestamps(VISION_TEMP, n_raw_frames=64, true_window_dur_sec=true_window_dur_sec)  # (32,2)
    vis_ts_exp = vis_ts.unsqueeze(1).expand(VISION_TEMP, VISION_SPAT, 2).reshape(VISION_TEMP * VISION_SPAT, 2)
    vis_bins = _ts_to_tdm_bins(vis_ts_exp, true_window_dur_sec, max_tdm_bins)  # (512,) -- SAME function as training

    vis_flat = vis_pooled.reshape(VISION_TEMP * VISION_SPAT, VISION_DIM)  # (512, 1024)

    # ---- AMBIENT: WavJEPA-base + WavJEPA-nat, averaged (bug 3 fix) ------
    wav = audio
    if wav.dim() > 1:
        wav = wav.mean(0)   # mix to mono EXACTLY as _decode_audio_raw does -- never feed real
                             # multichannel to the nat model, training used duplicated mono
    wav1 = wav.unsqueeze(0).unsqueeze(0)          # (1, 1, n_s) -- WavJEPA-base input, per 1b
    wav2 = wav.unsqueeze(0).expand(2, -1).unsqueeze(0)   # (1, 2, n_s) -- duplicated mono, per 1b
    with torch.no_grad():
        base_feat = base_encoder.encode(wav1.to(device))[0]   # (T_b, 768)
        nat_feat = nat_encoder.encode(wav2.to(device))[0]      # (T_n, 768)
    if base_feat.shape[0] == nat_feat.shape[0]:
        aud = (base_feat.float() + nat_feat.float()).mul_(0.5).to(torch.bfloat16)   # audio_mode="mean", confirmed logs/m2_week.log
    else:
        aud = base_feat.to(torch.bfloat16)   # fallback, matches data/av_cached_dataset.py's own fallback

    aud_ts = _uniform_token_timestamps(aud.shape[0], true_window_dur_sec)   # (T_a, 2)
    aud_bins = _ts_to_tdm_bins(aud_ts, true_window_dur_sec, max_tdm_bins)   # (T_a,) -- SAME function, no linspace

    feats = {
        "vision": vis_flat.float().unsqueeze(0).to(device),
        "ambient": aud.float().unsqueeze(0).to(device),
    }
    tbins = {
        "vision": vis_bins.unsqueeze(0).to(device),
        "ambient": aud_bins.unsqueeze(0).to(device),
    }
    return WorldStateBuildResult(feats=feats, tbins=tbins,
                                  vision_raw_ntok=vision_raw_ntok, ambient_ntok=aud.shape[0])


def assert_staircase(vis_bins: torch.Tensor, max_tdm_bins: int, n_temp: int = VISION_TEMP, n_spat: int = VISION_SPAT) -> None:
    """Gate check (Phase 1.1b): 32 distinct values, each repeated 16x,
    {0,16,...,496} for a 10.0s window at max_bins=512. Fails loudly, not
    silently, if the pooling/binning wiring regresses."""
    vals = vis_bins.view(n_temp, n_spat)
    distinct_per_group = [int(vals[t].unique().numel()) for t in range(n_temp)]
    assert all(d == 1 for d in distinct_per_group), \
        f"expected each of the {n_temp} temporal groups to share ONE bin value across its {n_spat} spatial tokens, got {distinct_per_group}"
    group_vals = [int(vals[t, 0].item()) for t in range(n_temp)]
    assert len(set(group_vals)) == n_temp, \
        f"expected {n_temp} DISTINCT group bin values (a staircase), got {len(set(group_vals))} distinct: {sorted(set(group_vals))}"
