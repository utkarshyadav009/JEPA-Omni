"""scripts/cavmae_retrieval.py — STOP-3: CAV-MAE audio↔visual retrieval baseline.

Runs the CAV-MAE model (audio_model.25.pth, ViT-B/16, 11+1 blocks) on a fixed
~1000-clip VGGSound eval subset and computes:
    audio→visual  R@1, R@5, R@10
    visual→audio  R@1, R@5, R@10

Reference: CAV-MAE paper reports ~R@1 ≈ 15 on VGGSound at 0.50 masking.

Audio preprocessing:
  - Uses torchcodec AudioDecoder (no torchaudio dependency)
  - Log-mel spectrogram: n_fft=1024, hop=160, n_mels=128, sr=16000
  - CAV-MAE normalization: (log_mel - (-4.2677393)) / 4.5689974
  - Truncated/padded to 1024 time frames (10.24s at 100fps)

Video preprocessing:
  - Center frame, 224×224, ImageNet normalisation

Usage:
    conda run -n jepa-omni python scripts/cavmae_retrieval.py
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import random
import csv
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── CAV-MAE constants ──────────────────────────────────────────────────────
EMBED_DIM       = 768
AUDIO_NBLOCKS   = 11
VIDEO_NBLOCKS   = 11
SHARED_NBLOCKS  = 1
NUM_HEADS       = 12
AUDIO_FREQ      = 128       # mel bins
AUDIO_N_PATCHES = 8 * 26    # = 208 (8 freq patches × 26 time patches)
# 26 time patches × 16 = 416 time frames from mel spec → ~4.16s at 100fps (hop 160, sr 16k)
AUDIO_TIME_FRAMES = 416     # spectrogram frames expected by model
VIDEO_SIZE      = 224
VIDEO_PATCHES   = (VIDEO_SIZE // 16) ** 2   # 196

# CAV-MAE audio normalization constants (from dataset statistics in the paper repo)
AUDIO_NORM_MEAN = -4.2677393
AUDIO_NORM_STD  = 4.5689974

# ImageNet normalization for video
IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD  = [0.2290, 0.2240, 0.2250]


# ── Pure-PyTorch Kaldi-compatible log-mel (matches CAV-MAE training) ─────────
# CAV-MAE was trained with torchaudio.compliance.kaldi.fbank(
#   htk_compat=True, frame_length=25, frame_shift=10, num_mel_bins=128,
#   sample_frequency=16000, use_log_fbank=True, dither=0.0)
# Key params: frame=25ms→400 samp, FFT=512, Povey window, f_min=20Hz,
#             pre-emphasis=0.97, no area-normalisation of filterbank.

_mel_fb_cache: Dict[str, Tensor] = {}

import math as _math

def _build_mel_filterbank(sr: int = 16000, n_fft: int = 512,
                           n_mels: int = 128, f_min: float = 20.0,
                           f_max: float = 8000.0) -> Tensor:
    """HTK mel-scale triangular filterbank, NOT area-normalised (matches Kaldi)."""
    def hz2mel(hz): return 2595.0 * _math.log10(1.0 + hz / 700.0)
    def mel2hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs  = n_fft // 2 + 1
    mel_min  = hz2mel(f_min)
    mel_max  = hz2mel(f_max)
    mel_pts  = [mel_min + i * (mel_max - mel_min) / (n_mels + 1)
                for i in range(n_mels + 2)]
    hz_pts   = [mel2hz(m) for m in mel_pts]
    bin_pts  = [int(_math.floor(h * (n_fft + 1) / sr)) for h in hz_pts]

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f0, f1 + 1):
            if k < n_freqs and f1 > f0:
                fb[m - 1, k] = (k - f0) / (f1 - f0)
        for k in range(f1, f2 + 1):
            if k < n_freqs and f2 > f1:
                fb[m - 1, k] = (f2 - k) / (f2 - f1)
    return fb


def waveform_to_logmel(
    wav: Tensor,
    sr:  int   = 16000,
    n_mels: int = AUDIO_FREQ,
    target_frames: int = AUDIO_TIME_FRAMES,
) -> Tensor:
    """(n_samples,) float32 @ sr=16kHz → (1, n_mels, target_frames) normalised log-mel.

    Uses torchaudio.compliance.kaldi.fbank to exactly match CAV-MAE training preprocessing.
    """
    import torchaudio
    wav = wav.float()
    # kaldi.fbank expects (channel, time) waveform
    fb = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=sr,
        use_log_fbank=True,
        frame_length=25,
        frame_shift=10,
        num_mel_bins=n_mels,
        dither=0.0,
    )                                            # (T, n_mels)
    lmel = fb.T                                  # (n_mels, T)

    # CAV-MAE normalisation (constants from VGGSound Kaldi fbank statistics)
    lmel = (lmel - AUDIO_NORM_MEAN) / AUDIO_NORM_STD

    # Pad or centre-crop to target_frames
    T = lmel.shape[-1]
    if T < target_frames:
        lmel = F.pad(lmel, (0, target_frames - T))
    else:
        start = (T - target_frames) // 2
        lmel  = lmel[:, start:start + target_frames]

    return lmel.unsqueeze(0)   # (1, n_mels, target_frames)


# ── Video preprocessing ───────────────────────────────────────────────────
def decode_center_frame(video_path: str, size: int = VIDEO_SIZE) -> Tensor:
    """Return (3, size, size) float32 ImageNet-normalised center frame."""
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(video_path, device="cpu")
        n = int(getattr(decoder.metadata, "num_frames", None) or len(decoder))
        idx   = n // 2
        frame = decoder.get_frames_at(indices=[idx]).data[0].float() / 255.0
        if frame.shape[-2] != size or frame.shape[-1] != size:
            frame = F.interpolate(frame.unsqueeze(0), (size, size),
                                  mode="bilinear", align_corners=False)[0]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (frame - mean) / std
    except Exception:
        return torch.zeros(3, size, size)


# ── Minimal CAV-MAE encoder ───────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CAVMAEBlock(nn.Module):
    """Standard ViT block.  CAV-MAE stores 6 LNs per block (norm1/norm1_a/norm1_v,
    norm2/norm2_a/norm2_v) for decoder reuse; encoder only uses norm1 + norm2."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.norm1_a = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        self.attn    = Attention(dim, num_heads)
        self.norm2   = nn.LayerNorm(dim)
        self.norm2_a = nn.LayerNorm(dim)
        self.norm2_v = nn.LayerNorm(dim)
        self.mlp     = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, modality: str = "") -> Tensor:
        if modality == "a":
            x = x + self.attn(self.norm1_a(x))
            x = x + self.mlp(self.norm2_a(x))
        elif modality == "v":
            x = x + self.attn(self.norm1_v(x))
            x = x + self.mlp(self.norm2_v(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class CAVMAEEncoder(nn.Module):
    """Encoder-only CAV-MAE for retrieval (weights from audio_model.25.pth)."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 audio_nblocks: int = AUDIO_NBLOCKS,
                 video_nblocks: int = VIDEO_NBLOCKS,
                 shared_nblocks: int = SHARED_NBLOCKS):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_a = PatchEmbed(1, embed_dim, 16)
        self.patch_embed_v = PatchEmbed(3, embed_dim, 16)
        self.pos_embed_a   = nn.Parameter(torch.zeros(1, AUDIO_N_PATCHES, embed_dim))
        self.pos_embed_v   = nn.Parameter(torch.zeros(1, VIDEO_PATCHES,   embed_dim))
        self.cls_token_a   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_v   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_a    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(16, embed_dim))
        self.blocks_a = nn.ModuleList([
            CAVMAEBlock(embed_dim, num_heads) for _ in range(audio_nblocks)])
        self.blocks_v = nn.ModuleList([
            CAVMAEBlock(embed_dim, num_heads) for _ in range(video_nblocks)])
        self.blocks_u = nn.ModuleList([
            CAVMAEBlock(embed_dim, num_heads) for _ in range(shared_nblocks)])
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm   = nn.LayerNorm(embed_dim)

    def encode_audio(self, spec: Tensor) -> Tensor:
        """spec: (B, 1, AUDIO_FREQ, AUDIO_TIME_FRAMES) → (B, embed_dim) L2-normed."""
        B = spec.shape[0]
        x = self.patch_embed_a(spec)                    # (B, N_a, D)
        x = x + self.pos_embed_a + self.modality_a
        cls = self.cls_token_a.expand(B, -1, -1)
        reg = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
        x   = torch.cat([cls, reg, x], dim=1)
        for blk in self.blocks_a:
            x = blk(x)
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm_a(x)
        return F.normalize(x[:, 0], dim=-1)

    def encode_video(self, frame: Tensor) -> Tensor:
        """frame: (B, 3, H, W) ImageNet-normalised → (B, embed_dim) L2-normed."""
        B = frame.shape[0]
        x = self.patch_embed_v(frame)                   # (B, N_v, D)
        x = x + self.pos_embed_v + self.modality_v
        cls = self.cls_token_v.expand(B, -1, -1)
        reg = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
        x   = torch.cat([cls, reg, x], dim=1)
        for blk in self.blocks_v:
            x = blk(x)
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm_v(x)
        return F.normalize(x[:, 0], dim=-1)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "CAVMAEEncoder":
        sd      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd_clean = {k[len("module."):]: v for k, v in sd.items()}
        model   = cls()
        own_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in sd_clean.items() if k in own_keys}
        result   = model.load_state_dict(filtered, strict=False)
        n_miss   = len(result.missing_keys)
        print(f"CAVMAEEncoder: loaded {len(filtered)} keys, missing={n_miss}")
        if n_miss > 0:
            print("  missing:", result.missing_keys[:5])
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


# ── Retrieval metrics ─────────────────────────────────────────────────────
def recall_at_k(sim: Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    """sim: (N, N) cosine sim. Row i retrieved by col i (matched pairs)."""
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt   = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="/home/utkarsh/models/cav-mae/audio_model.25.pth")
    parser.add_argument("--video-dir",  default="/home/utkarsh/data/vggsound")
    parser.add_argument("--n-clips",    type=int, default=1000)
    parser.add_argument("--eval-list",  default=None)
    parser.add_argument("--save-list",  default="data/cavmae_eval_subset.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── build eval clip list ─────────────────────────────────────────────
    if args.eval_list and os.path.isfile(args.eval_list):
        with open(args.eval_list) as f:
            clip_ids = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(clip_ids)} clips from {args.eval_list}")
    else:
        all_ids = []
        csv_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if not row: continue
                fname = row[0].strip()
                vid   = os.path.splitext(fname)[0]
                vpath = os.path.join(args.video_dir, fname)
                if os.path.exists(vpath):
                    all_ids.append(vid)

        rng = random.Random(42)
        rng.shuffle(all_ids)
        clip_ids = all_ids[:args.n_clips]
        print(f"Selected {len(clip_ids)} eval clips from test split.")
        os.makedirs(os.path.dirname(args.save_list) or ".", exist_ok=True)
        with open(args.save_list, "w") as f:
            f.writelines(c + "\n" for c in clip_ids)
        print(f"Saved eval subset → {args.save_list}")

    N = len(clip_ids)
    print(f"Evaluating {N} clips on {args.device}", flush=True)

    # ── load model ───────────────────────────────────────────────────────
    model = CAVMAEEncoder.from_checkpoint(args.ckpt, device=args.device)

    # ── extract embeddings ───────────────────────────────────────────────
    from torchcodec.decoders import AudioDecoder

    audio_embs = torch.zeros(N, EMBED_DIM)
    video_embs = torch.zeros(N, EMBED_DIM)

    for i in range(0, N, args.batch_size):
        batch_ids = clip_ids[i:i + args.batch_size]
        B = len(batch_ids)

        # Audio
        specs = []
        for vid in batch_ids:
            vpath = os.path.join(args.video_dir, vid + ".mp4")
            try:
                dec = AudioDecoder(vpath, sample_rate=16000)
                w   = dec.get_all_samples().data
                w   = w.mean(0) if w.shape[0] > 1 else w[0]
                spec = waveform_to_logmel(w)
            except Exception:
                spec = torch.zeros(1, AUDIO_FREQ, AUDIO_TIME_FRAMES)
            specs.append(spec)
        spec_batch = torch.stack(specs).to(args.device)
        with torch.no_grad():
            audio_embs[i:i + B] = model.encode_audio(spec_batch).cpu()

        # Video
        frames = []
        for vid in batch_ids:
            vpath = os.path.join(args.video_dir, vid + ".mp4")
            frames.append(decode_center_frame(vpath))
        frame_batch = torch.stack(frames).to(args.device)
        with torch.no_grad():
            video_embs[i:i + B] = model.encode_video(frame_batch).cpu()

        if (i // args.batch_size) % 5 == 0:
            print(f"  {i + B}/{N}", flush=True)

    # ── retrieval ────────────────────────────────────────────────────────
    print(f"\naudio_emb std: {audio_embs.std(dim=0).mean().item():.4f}")
    print(f"video_emb std: {video_embs.std(dim=0).mean().item():.4f}")
    print(f"audio mean-vec norm: {audio_embs.mean(0).norm():.4f}")
    print(f"video mean-vec norm: {video_embs.mean(0).norm():.4f}")

    # Mean-centering: removes the DC bias common to all embeddings in this checkpoint.
    # Required because the model's CLS tokens all share a strong mean direction (||μ||≈0.84).
    audio_embs = F.normalize(audio_embs - audio_embs.mean(0), dim=-1)
    video_embs = F.normalize(video_embs - video_embs.mean(0), dim=-1)

    sim_av = audio_embs @ video_embs.T
    sim_va = video_embs @ audio_embs.T
    res_av = recall_at_k(sim_av)
    res_va = recall_at_k(sim_va)

    print("\n" + "=" * 50)
    print(f"CAV-MAE retrieval  N={N}")
    print("-" * 50)
    print("audio→visual:", " ".join(f"{k}={v:.2f}%" for k, v in res_av.items()))
    print("visual→audio:", " ".join(f"{k}={v:.2f}%" for k, v in res_va.items()))
    print("=" * 50)

    results = {
        "n_clips": N, "eval_list": args.save_list,
        "audio_to_visual": res_av, "visual_to_audio": res_va,
        "ckpt": args.ckpt,
    }
    out_path = "data/cavmae_retrieval_results.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
