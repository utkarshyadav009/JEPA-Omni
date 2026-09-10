"""scripts/cavmae_sync_retrieval.py — CAV-MAE Sync audio<->visual retrieval baseline.

Runs edsonroteia/cav-mae-sync (CVPR 2025, model_3388_25 / "sync_pretrain_registers_cls_4s")
on our fixed 1545-clip VGGSound eval gallery (data/vggsound_eval_1545.txt) and computes
audio->visual and visual->audio R@1/R@5/R@10.

Preprocessing (matched to the model's own src/dataloader_sync.py and src/retrieval.py,
NOT our own pipeline):
  Audio:
    - torchaudio-style mono waveform (mean-subtracted), kaldi-compliance fbank:
      htk_compat=True, num_mel_bins=128, dither=0.0, frame_shift=10ms, window=hanning
      (src/dataloader_sync.py:_wav2fbank), padded/cut to a 1024-frame base spectrogram.
    - For each of 16 sync frame indices, a target_length=416-frame (i.e. the "4s" variant)
      segment is cropped from the base spectrogram via the model's own
      map_frame_to_spectrogram(frame_idx, num_frames=16, spectrogram_length, target_length=416)
      mapping (centers the audio segment on the video frame's temporal position).
    - Normalized with the model's own dataset stats: mean=-5.081, std=4.4849
      (src/retrieval.py audio_conf, NOT vanilla CAV-MAE's -4.2677/4.5689).
  Video:
    - 16 frames sampled at evenly-spaced bin centers across the clip duration
      (t_i = (i+0.5)/16 * duration), matching the model's total_frame=16 convention
      (exact extraction timestamps are not specified in the released data-prep script,
      which expects pre-extracted frame_i/ folders from the original CAV-MAE ffmpeg
      script; evenly-spaced bin-center sampling is the documented CAV-MAE convention).
    - Resize(224, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize(mean=[.4850,.4560,.4060],
      std=[.2290,.2240,.2250]) (src/dataloader_sync.py AudiosetDataset.preprocess, eval branch).
  Model config (read off the checkpoint's own state_dict shapes, confirmed against
  src/retrieval.py's hardcoded 'model_3388_25' entry): CAVMAESync(audio_length=416,
  modality_specific_depth=11, num_register_tokens=8, cls_token=True, total_frame=16,
  contrastive_heads=False). Embedding = CLS token (cls_a / cls_v from forward_feat),
  L2-normalized, per sync-frame (16 embeddings per clip per modality).

Retrieval: 'diagonal_mean' aggregation strategy (src/retrieval.py get_agg_sim_mat,
the strategy suggested in the repo's own README example command) — for clip pair (i,j),
similarity = mean over k in [0,16) of cos(audio_cls[i,k], video_cls[j,k]). Computed here as
a vectorized batched matmul (mathematically identical to the repo's nested-loop
implementation) instead of the repo's O(N^2 * 16) python double loop, which the README
says takes up to 40 minutes unparallelized.

Usage:
    conda run -n jepa-omni python scripts/cavmae_sync_retrieval.py
"""

from __future__ import annotations
import argparse, os, sys, json, hashlib, time
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torchcodec.decoders import VideoDecoder, AudioDecoder

REPO_DIR = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/cavmae_sync/repo"
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# ── timm compatibility shim ──────────────────────────────────────────────
# timm>=1.0 (installed in this env) removed the `qk_scale` kwarg from
# vision_transformer.Attention; the repo's src/models/cav_mae_sync.py hardcodes
# passing it (classic MAE-era timm<=0.4.x signature). Same fix already applied
# successfully in scripts/equiav_retrieval.py — standard scaled-dot-product ViT
# self-attention, numerically identical to what old timm did. Must patch before
# `import models` below, since cav_mae_sync.py does `from timm...import Attention`.
import torch.nn as nn
import timm.models.vision_transformer as _tvt

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
        x = self.proj_drop(self.proj(x))
        return x

_tvt.Attention = _CompatAttention

import models  # noqa: E402  (repo's src/models/__init__.py -> CAVMAESync = CAVMAE)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORM_MEAN, NORM_STD = -5.081, 4.4849
MELBINS = 128
BASE_TARGET_LEN = 1024     # full-spectrogram pad/cut length in _wav2fbank
SEG_TARGET_LEN = 416       # per-frame segment length (the "4s" variant)
TOTAL_FRAME = 16
IM_RES = 224

IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD = [0.2290, 0.2240, 0.2250]

img_preprocess = T.Compose([
    T.Resize(IM_RES, interpolation=Image.BICUBIC),
    T.CenterCrop(IM_RES),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def map_frame_to_spectrogram(frame_index, num_frames, spectrogram_length, target_length):
    frame_position = int(round(frame_index * spectrogram_length / num_frames))
    start = max(0, frame_position - target_length // 2)
    end = start + target_length
    if end > spectrogram_length:
        end = spectrogram_length
        start = max(0, end - target_length)
    return start, end


def wav2fbank(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    # waveform: (1, n_samples) mono, mean-subtracted, matching repo's _wav2fbank
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=MELBINS, dither=0.0, frame_shift=10,
    )
    n_frames = fbank.shape[0]
    p = BASE_TARGET_LEN - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:BASE_TARGET_LEN, :]
    return fbank


def load_clip(video_path: str):
    """Returns (fbanks[16,416,128], images[16,3,224,224]) or raises."""
    vd = VideoDecoder(video_path)
    duration = vd.metadata.duration_seconds
    n_frames_avail = len(vd)
    fps = vd.metadata.average_fps

    ad = AudioDecoder(video_path)
    samples = ad.get_all_samples()
    wav = samples.data  # (channels, n_samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sr = samples.sample_rate
    base_fbank = wav2fbank(wav, sr)  # (1024, 128)
    spec_len = base_fbank.shape[0]

    fbanks, images = [], []
    for i in range(TOTAL_FRAME):
        t = (i + 0.5) / TOTAL_FRAME * duration
        frame_idx = min(int(round(t * fps)), n_frames_avail - 1)
        frame = vd[frame_idx]  # (3,H,W) uint8
        img = Image.fromarray(frame.permute(1, 2, 0).numpy())
        images.append(img_preprocess(img))

        start, end = map_frame_to_spectrogram(i, TOTAL_FRAME, spec_len, SEG_TARGET_LEN)
        seg = base_fbank[start:end, :]
        if seg.shape[0] < SEG_TARGET_LEN:
            seg = torch.nn.functional.pad(seg, (0, 0, 0, SEG_TARGET_LEN - seg.shape[0]))
        seg = (seg - NORM_MEAN) / NORM_STD
        fbanks.append(seg)

    return torch.stack(fbanks), torch.stack(images)


def build_model(ckpt_path: str, device: str):
    model = models.CAVMAESync(
        audio_length=SEG_TARGET_LEN, modality_specific_depth=11,
        num_register_tokens=8, cls_token=True, total_frame=TOTAL_FRAME,
        contrastive_heads=False,
    )
    model = torch.nn.DataParallel(model)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(sd, strict=False)
    print("load_state_dict:", msg)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def embed_clip(model, fbanks, images, device):
    a = fbanks.to(device)         # (16,416,128)
    v = images.to(device)         # (16,3,224,224)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, _, cls_a, cls_v = model.module.forward_feat(a, v)
    cls_a = F.normalize(cls_a.float(), dim=-1)
    cls_v = F.normalize(cls_v.float(), dim=-1)
    return cls_a.cpu(), cls_v.cpu()  # (16,768) each


def diagonal_mean_sim(A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # A, V: (N, 16, 768), L2-normalized. sim[i,j] = mean_k A[i,k].V[j,k]
    N, K, D = A.shape
    sim = torch.zeros(N, N)
    for k in range(K):
        sim += A[:, k, :] @ V[:, k, :].T
    return sim / K


def compute_metrics(sim: torch.Tensor):
    N = sim.shape[0]
    ranks = torch.argsort(sim, dim=1, descending=True)
    gt = torch.arange(N).unsqueeze(1)
    hit_rank = (ranks == gt).float().argmax(dim=1)
    out = {}
    for k in (1, 5, 10):
        out[f"R@{k}"] = round((hit_rank < k).float().mean().item() * 100, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/cavmae_sync/cav_mae_sync.pth")
    parser.add_argument("--video-dir", default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    parser.add_argument("--eval-list", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "cavmae_sync_retrieval_results.json"))
    args = parser.parse_args()

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}")

    model = build_model(args.ckpt, args.device)

    all_a, all_v, ok_ids = [], [], []
    failed = []
    t0 = time.time()
    for i, cid in enumerate(clip_ids):
        vpath = os.path.join(args.video_dir, cid + ".mp4")
        try:
            fbanks, images = load_clip(vpath)
            cls_a, cls_v = embed_clip(model, fbanks, images, args.device)
            all_a.append(cls_a)
            all_v.append(cls_v)
            ok_ids.append(cid)
        except Exception as e:
            failed.append((cid, str(e)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(clip_ids)} processed, {len(failed)} failed, {time.time()-t0:.1f}s elapsed", flush=True)

    N = len(ok_ids)
    print(f"Evaluating {N} clips on {args.device}", flush=True)
    if failed:
        print(f"FAILED to process {len(failed)} clips (decode/inference errors):")
        for cid, err in failed[:20]:
            print(f"  {cid}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")
    if N == 1545:
        print("clips_seen=1545 (full-gallery OK)")
    else:
        print(f"clips_seen={N} != 1545 requested -- evaluating on the intersection, count stated above")

    A = torch.stack(all_a)  # (N,16,768)
    V = torch.stack(all_v)

    sim_a2v = diagonal_mean_sim(A, V)          # audio->visual: rows=audio query, cols=visual gallery
    sim_v2a = diagonal_mean_sim(V, A)          # visual->audio

    r_a2v = compute_metrics(sim_a2v)
    r_v2a = compute_metrics(sim_v2a)
    print("audio→visual:", r_a2v)
    print("visual→audio:", r_v2a)

    sha256 = hashlib.sha256()
    with open(args.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    ckpt_sha256 = sha256.hexdigest()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    params_m = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6

    out = {
        "n_clips": N,
        "clips_requested": 1545,
        "eval_list": "data/vggsound_eval_1545.txt",
        "failed_clips": len(failed),
        "audio_to_visual": r_a2v,
        "visual_to_audio": r_v2a,
        "ckpt": args.ckpt,
        "ckpt_sha256": ckpt_sha256,
        "repo": "https://github.com/edsonroteia/cav-mae-sync",
        "params_millions": round(params_m, 2),
        "pretrain_corpus": "AudioSet-2M + VGGSound (fine-grained sync fine-tuning of a CAV-MAE-initialized model; data-prep instructions explicitly reuse the original CAV-MAE AudioSet+VGGSound pipeline)",
        "contamination_flag": "IN-DISTRIBUTION",
        "preprocessing": (
            "Audio: kaldi-compliance fbank (htk_compat=True, 128 mel bins, dither=0, "
            "10ms frame shift, hanning window) over the full waveform, padded/cut to 1024 "
            "frames, then a 416-frame segment cropped per sync-frame via the model's own "
            "map_frame_to_spectrogram mapping, normalized with mean=-5.081/std=4.4849. "
            "Video: 16 frames at evenly-spaced bin centers across clip duration, "
            "Resize(224,bicubic)->CenterCrop(224)->ToTensor->ImageNet normalize. "
            "Model: CAVMAESync(audio_length=416, modality_specific_depth=11, "
            "num_register_tokens=8, cls_token=True, total_frame=16), CLS-token embedding, "
            "L2-normalized per sync-frame, diagonal_mean aggregation across the 16 frames."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda run -n jepa-omni python scripts/cavmae_sync_retrieval.py",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
