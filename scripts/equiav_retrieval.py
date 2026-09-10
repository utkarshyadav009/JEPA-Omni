"""scripts/equiav_retrieval.py — EquiAV audio<->visual retrieval baseline on the fixed 1545-clip
VGGSound gallery (data/vggsound_eval_1545.txt).

Model: EquiAV (ICML 2024), https://github.com/JongSuk1/EquiAV
Checkpoint: official pretrained release (Google Drive id 1QCvBcu-CAXFLKqfk0G7niO2JO5kf74K6, linked
from the repo README under "Pre-training"), pretrained on AudioSet-2M. Downloaded to
/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/equiav/equiav_pretrained.pth.

Preprocessing (read directly from the official repo, NOT reused from our own cavmae_retrieval.py):
  - Audio: torchcodec AudioDecoder resampled to 16kHz, mono-mixed (== kaldi.fbank's default
    channel=-1 channel-averaging behaviour), mean-subtracted, then
    torchaudio.compliance.kaldi.fbank(htk_compat=True, sample_frequency=16000, use_energy=False,
    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) exactly matching
    datasets/AudioVisual.py:_wav2fbank. Padded with trailing zeros or HEAD-truncated (not
    center-cropped -- matches `fbank[0:target_length, :]` in the same function) to target_length
    = 1024 frames. Normalised as (fbank + 4.346) / 4.332 -- the exact audio_conf constants
    EquiAV's own retrieval.py hardcodes for zero-shot retrieval. Reshaped to (3, 1024, 128) by
    channel-repeating (matches MainDataset.__getitem__'s `fbank.unsqueeze(0).repeat(3,1,1)`).
  - Video: single frame (we use the true middle frame of the clip -- the repo's own retrieval
    path assumes a pre-extracted `frame_{frame_use}/{video_id}.jpg` directory we don't have, so
    this is a documented approximation of the same "one representative frame" idea, consistent
    with frame_use pointing near the clip's middle in the paper's convention). Resize(224,
    bicubic, shorter side) -> CenterCrop(224) -> ToTensor -> Normalize(mean=[0.4850,0.4560,0.4060],
    std=[0.2290,0.2240,0.2250]) -- matches MainDataset.preprocess exactly.
  - Model forward: MainModel.forward_feat(audio, video) (models/pt_EquiAV.py), both outputs
    L2-normalised -- exactly what retrieval.py does for zero-shot retrieval.

Compatibility shims applied (both documented, neither changes model math):
  - `timm.models.vision_transformer.Attention` is patched with the classic (timm<=0.4.x /
    MAE-era) signature accepting `qk_scale`, which timm>=1.0 (installed in this env) removed.
    The replacement is the standard scaled-dot-product ViT self-attention formula -- numerically
    identical to what old timm did.
  - `numpy.float` (removed in modern numpy) is restored as an alias for the builtin `float`,
    needed by the repo's models/pos_embed.py.

Note on determinism: MainModel.forward_feat internally samples ONE random augmentation-parameter
vector per forward call (via self.sample_t_a / self.sample_t_v) and cross-attends the patch
tokens against it -- this is EquiAV's actual released design (aggregating over augmentation
draws), not a bug introduced here. A fixed seed is set for reproducibility; re-running with a
different seed will shift results slightly.

Usage:
    conda run -n jepa-omni python scripts/equiav_retrieval.py
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
from typing import Dict, List, Tuple

import numpy as np
if not hasattr(np, "float"):
    np.float = float  # noqa: A003 -- shim for models/pos_embed.py on modern numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUIAV_REPO = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/equiav/repo"
sys.path.insert(0, EQUIAV_REPO)

torch.manual_seed(0)

# ── timm compatibility shim (see module docstring) ─────────────────────────
class _CompatAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

import timm.models.vision_transformer as _tvt
_tvt.Attention = _CompatAttention

from models.pt_EquiAV import MainModel  # noqa: E402

EMBED_DIM = 512  # forward_feat output dim (pred_a_inter / pred_v_inter out_dim)
AUDIO_NORM_MEAN = -4.346
AUDIO_NORM_STD = 4.332
TARGET_FRAMES = 1024
MELBINS = 128
IM_RES = 224
IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD = [0.2290, 0.2240, 0.2250]

_video_preprocess = T.Compose([
    T.Resize(IM_RES, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IM_RES),
])


def wav_to_fbank(wav: Tensor) -> Tensor:
    """wav: (n_samples,) float32 @ 16kHz, mono -> (1024, 128) normalised fbank."""
    wav = wav.float().unsqueeze(0)  # (1, n_samples)
    wav = wav - wav.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        wav, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type="hanning", num_mel_bins=MELBINS, dither=0.0, frame_shift=10,
    )  # (T, 128)
    n_frames = fbank.shape[0]
    p = TARGET_FRAMES - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[0:TARGET_FRAMES, :]  # HEAD crop, matches _wav2fbank exactly
    fbank = (fbank - AUDIO_NORM_MEAN) / AUDIO_NORM_STD
    return fbank


def decode_audio_video(video_path: str) -> Tuple[Tensor, Tensor]:
    """Returns (audio_input (3,1024,128), video_input (3,224,224)) or raises."""
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    adec = AudioDecoder(video_path, sample_rate=16000)
    w = adec.get_all_samples().data
    w = w.mean(0) if w.shape[0] > 1 else w[0]
    fbank = wav_to_fbank(w)
    audio_input = fbank.unsqueeze(0).repeat(3, 1, 1)  # (3, 1024, 128)

    vdec = VideoDecoder(video_path, device="cpu")
    n = int(getattr(vdec.metadata, "num_frames", None) or len(vdec))
    idx = n // 2
    frame = vdec.get_frames_at(indices=[idx]).data[0].float() / 255.0  # (3,H,W) in [0,1]
    frame = _video_preprocess(frame)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    video_input = (frame - mean) / std

    return audio_input, video_input


def recall_at_k(sim: Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/equiav/equiav_pretrained.pth")
    parser.add_argument("--video-dir", default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    parser.add_argument("--eval-list", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}")
    N_requested = len(clip_ids)

    # ── load model ──────────────────────────────────────────────────────
    model = MainModel(gpu=1)  # gpu=1 suppresses the repo's own print spam
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd_clean = {k.replace("__M__.", ""): v for k, v in sd.items()}
    result = model.load_state_dict(sd_clean, strict=False)
    print(f"EquiAV: loaded state_dict, missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    if result.missing_keys:
        print("  missing:", result.missing_keys[:10])
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── extract embeddings ──────────────────────────────────────────────
    audio_embs, video_embs, ok_ids, failed = [], [], [], []
    for vid in clip_ids:
        vpath = os.path.join(args.video_dir, vid + ".mp4")
        try:
            a_in, v_in = decode_audio_video(vpath)
            audio_embs.append(a_in)
            video_embs.append(v_in)
            ok_ids.append(vid)
        except Exception as e:
            failed.append((vid, str(e)))

    N = len(ok_ids)
    print(f"Evaluating {N} clips on {args.device}", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} clips failed to decode and were EXCLUDED (evaluating on the {N}-clip intersection):")
        for vid, err in failed[:20]:
            print(f"  {vid}: {err}")

    a_embs_out = torch.zeros(N, EMBED_DIM)
    v_embs_out = torch.zeros(N, EMBED_DIM)
    bs = args.batch_size
    for i in range(0, N, bs):
        a_batch = torch.stack(audio_embs[i:i + bs]).to(args.device)
        v_batch = torch.stack(video_embs[i:i + bs]).to(args.device)
        with torch.no_grad():
            a_out, v_out = model.forward_feat(a_batch, v_batch)
            a_out = F.normalize(a_out, dim=-1)
            v_out = F.normalize(v_out, dim=-1)
        a_embs_out[i:i + a_out.shape[0]] = a_out.cpu()
        v_embs_out[i:i + v_out.shape[0]] = v_out.cpu()
        if (i // bs) % 5 == 0:
            print(f"  {i + a_out.shape[0]}/{N}", flush=True)

    if N == N_requested == 1545:
        print(f"clips_seen={N}  (full-gallery OK, matches required 1545)")
    else:
        print(f"clips_seen={N}  requested={N_requested}  -- NOT the full 1545 gallery, evaluating on intersection")

    sim_av = a_embs_out @ v_embs_out.T
    sim_va = v_embs_out @ a_embs_out.T
    res_av = recall_at_k(sim_av)
    res_va = recall_at_k(sim_va)

    print("\n" + "=" * 50)
    print(f"EquiAV retrieval  N={N}")
    print("-" * 50)
    print("audio→visual:", " ".join(f"{k}={v:.2f}%" for k, v in res_av.items()))
    print("visual→audio:", " ".join(f"{k}={v:.2f}%" for k, v in res_va.items()))
    print("=" * 50)

    sha256 = hashlib.sha256()
    with open(args.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    params_m = sum(v.numel() for v in sd_clean.values()) / 1e6

    results = {
        "n_clips": N,
        "clips_requested": N_requested,
        "eval_list": "data/vggsound_eval_1545.txt",
        "audio_to_visual": res_av,
        "visual_to_audio": res_va,
        "ckpt": args.ckpt,
        "ckpt_sha256": sha256.hexdigest(),
        "repo": "https://github.com/JongSuk1/EquiAV",
        "params_millions": round(params_m, 2),
        "pretrain_corpus": "AudioSet-2M (this is the officially released pt_main.py --dataset AudioSet_2M checkpoint; VGGSound is a separate, fine-tuning-only dataset for this repo, not used to pretrain this checkpoint)",
        "contamination_flag": "HELD-OUT",
        "preprocessing": (
            "Audio: torchcodec AudioDecoder resampled to 16kHz, mono-mixed, mean-subtracted, "
            "torchaudio.compliance.kaldi.fbank(htk_compat=True, sample_frequency=16000, "
            "use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, "
            "frame_shift=10) matching datasets/AudioVisual.py:_wav2fbank; padded/head-truncated "
            "to 1024 frames; normalised (fbank+4.346)/4.332 per retrieval.py's audio_conf; "
            "reshaped to (3,1024,128) by channel-repeat. Video: true middle frame (approximates "
            "the repo's pre-extracted frame_use=10 convention, which assumes a frame directory we "
            "don't have), Resize(224,bicubic)->CenterCrop(224)->Normalize(mean=[0.4850,0.4560,"
            "0.4060],std=[0.2290,0.2240,0.2250]) matching MainDataset.preprocess. Model: "
            "MainModel.forward_feat(), both outputs L2-normalised, matching retrieval.py. Two "
            "environment compatibility shims applied (timm Attention qk_scale arg restored; "
            "numpy.float alias restored) -- neither changes model math, both documented in the "
            "module docstring. forward_feat samples one random augmentation vector internally "
            "per call (EquiAV's actual released design); torch.manual_seed(0) fixes this run."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda run -n jepa-omni python scripts/equiav_retrieval.py",
    }
    out_path = os.path.join(PROJECT_ROOT, "data", "equiav_retrieval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
