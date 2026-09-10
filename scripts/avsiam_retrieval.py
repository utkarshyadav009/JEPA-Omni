"""scripts/avsiam_retrieval.py — AVSiam audio<->visual retrieval baseline.

Runs AVSiam (Lin & Bertasius, "Siamese Vision Transformers are Scalable
Audio-visual Learners", arXiv:2403.19638) on the fixed 1545-clip VGGSound
eval gallery (data/vggsound_eval_1545.txt) and computes:
    audio->visual  R@1, R@5, R@10
    visual->audio  R@1, R@5, R@10

Model: imports the ACTUAL CAVMAE_BASE class from the official repo
(https://github.com/GenjiB/AVSiam, cloned to
/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/avsiam/repo) rather than
reimplementing it, since it has a nontrivial Siamese weight-sharing structure
(shared vit_base backbone, separate patch_embed_a / patch_embed for the two
modalities, an `ast_base` deep-copy of the visual tower used for the audio
forward pass). Two repo-quality issues required workarounds (see STUBS below):
the repo's own __pycache__ ships compiled .pyc files for `yb_tome.py`,
`cav_mae.py`, `cav_mae_large.py`, `cav_mae_huge.py`, `cav_mae_base_clip.py`
and `cav_mae_base_dino.py` with NO corresponding .py source anywhere in the
repo (never committed) and NOT loadable under our Python 3.11 (pyc magic
number mismatch vs. the cpython-38/39 bytecode). `cav_mae_base.py` (which DOES
have source, and is what the "Base" checkpoint uses) imports `yb_tome` and the
external `tome` (Token Merging) package at module level purely for an
optional ToMe fast-inference path that `forward_encoder` (what we call) never
touches -- so both are stubbed out rather than genuinely reimplemented.
`cav_mae_base.py` also unconditionally torch.load()s a hardcoded path
(`/mnt/opr/yblin/cav-pt/src/adapt_weights/jx_vit_base_patch16_224_in21k-*.pth`,
the authors' own cluster) inside __init__ as a (README: "not necessary") init
tweak; we intercept that one call and return an empty state dict, which is a
no-op because we immediately overwrite ALL weights with the real downloaded
checkpoint right after construction.

Audio preprocessing (from the repo's own src/dataloader.py):
  - torchaudio.compliance.kaldi.fbank(htk_compat=True, sample_frequency=16000,
    use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
    frame_shift=10)  [dataloader.py:328]
  - waveform mean-subtracted before fbank (`waveform - waveform.mean()`) [line 288]
  - padded/truncated to target_length=1024 frames [lines 333-343]
  - normalised (fbank - (-5.081)) / 4.4849  [src/retrieval.py audio_conf, line 506
    of dataloader.py applies it]
  - NOTE: the repo's own loader passes `sample_frequency=sr` using whatever
    rate `torchaudio.load()` returns from their pre-extracted .wav files, i.e.
    it does not resample itself -- their offline extraction pipeline is not
    public, so we resample to 16kHz ourselves before calling kaldi.fbank
    (standard AudioSet/VGGSound convention, also what CAV-MAE and AST assume).

Video preprocessing (from src/dataloader.py, eval branch of `randselect_img`,
lines 347-362, and the eval transform, lines 144-149):
  - eval mode uses a SINGLE FIXED frame at index frame_use=5 out of 10 frames
    pre-extracted evenly across the clip (audio_conf['frame_use']=5 in
    src/retrieval.py) -- i.e. a frame near the temporal midpoint. We decode
    the raw mp4 and take the frame at index round(0.5 * n_frames) to match.
  - torchvision Resize(224, BICUBIC) [shorter side] -> CenterCrop(224)
  - Normalize(mean=[0.4850,0.4560,0.4060], std=[0.2290,0.2240,0.2250])

Retrieval head: mean-pool all patch tokens (dim=1) then L2-normalize, exactly
as the repo's own src/retrieval.py:get_retrieval_result does (mean pool ->
F.normalize -> cosine similarity), applied here over the FULL 1545x1545
gallery in both directions rather than the repo's per-batch-of-100 loop.

Usage:
    conda run -n jepa-omni python scripts/avsiam_retrieval.py --variant base
    conda run -n jepa-omni python scripts/avsiam_retrieval.py --variant base_plus
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import types
import hashlib
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/avsiam/repo"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD  = [0.2290, 0.2240, 0.2250]
AUDIO_NORM_MEAN = -5.081
AUDIO_NORM_STD  = 4.4849
AUDIO_MELBINS   = 128
AUDIO_TARGET_LEN = 1024
VIDEO_SIZE = 224


# ── STUB missing/optional modules (see module docstring) ───────────────────
def _install_stubs():
    yb_tome = types.ModuleType("yb_tome")
    yb_tome.yb_bipartite_soft_matching = lambda *a, **k: (lambda x: x, lambda x: x)
    sys.modules["yb_tome"] = yb_tome
    sys.modules["models.yb_tome"] = yb_tome  # relative import inside models/cav_mae_base.py

    tome_pkg = types.ModuleType("tome")
    tome_merge = types.ModuleType("tome.merge")
    tome_merge.bipartite_soft_matching = lambda *a, **k: (lambda x: x, lambda x: x)
    tome_merge.merge_source = lambda *a, **k: None
    tome_merge.merge_wavg = lambda *a, **k: (None, None)
    tome_pkg.merge = tome_merge
    sys.modules["tome"] = tome_pkg
    sys.modules["tome.merge"] = tome_merge

    if "ipdb" not in sys.modules:
        try:
            import ipdb  # noqa: F401
        except ImportError:
            ipdb_stub = types.ModuleType("ipdb")
            ipdb_stub.set_trace = lambda *a, **k: None
            sys.modules["ipdb"] = ipdb_stub

    # Intercept the hardcoded adapt_weights torch.load — harmless no-op,
    # everything gets overwritten by the real checkpoint right after init.
    _orig_load = torch.load
    def _patched_load(f, *a, **kw):
        if isinstance(f, str) and "adapt_weights" in f:
            print(f"  [stub] skipping hardcoded adapt_weights load: {f}")
            return {}
        return _orig_load(f, *a, **kw)
    torch.load = _patched_load


_install_stubs()

import models as avsiam_models  # noqa: E402  (from REPO_ROOT/src)


# ── Preprocessing ────────────────────────────────────────────────────────
def waveform_to_fbank(wav: Tensor, sr: int = 16000) -> Tensor:
    """(n_samples,) float32 @ 16kHz -> (AUDIO_TARGET_LEN, AUDIO_MELBINS) normalised fbank."""
    import torchaudio
    wav = wav.float().unsqueeze(0)
    wav = wav - wav.mean()
    try:
        fbank = torchaudio.compliance.kaldi.fbank(
            wav, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type="hanning", num_mel_bins=AUDIO_MELBINS, dither=0.0,
            frame_shift=10,
        )
    except Exception:
        fbank = torch.zeros(AUDIO_TARGET_LEN, AUDIO_MELBINS) + 0.01
    n = fbank.shape[0]
    p = AUDIO_TARGET_LEN - n
    if p > 0:
        fbank = F.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[:AUDIO_TARGET_LEN, :]
    fbank = (fbank - AUDIO_NORM_MEAN) / AUDIO_NORM_STD
    return fbank


def decode_mid_frame(video_path: str, size: int = VIDEO_SIZE) -> Tensor:
    """Frame at ~50% (matches eval-mode frame_use=5-of-10), resized+cropped+normalised."""
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(video_path, device="cpu")
    n = int(getattr(decoder.metadata, "num_frames", None) or len(decoder))
    idx = n // 2
    frame = decoder.get_frames_at(indices=[idx]).data[0].float() / 255.0  # (3,H,W)
    c, h, w = frame.shape
    short = min(h, w)
    scale = size / short
    new_h, new_w = round(h * scale), round(w * scale)
    frame = F.interpolate(frame.unsqueeze(0), (new_h, new_w), mode="bicubic",
                           align_corners=False, antialias=True)[0]
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    frame = frame[:, top:top + size, left:left + size]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (frame - mean) / std


# ── Retrieval metrics (same convention as cavmae_retrieval.py) ─────────────
def recall_at_k(sim: Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: str, device: str) -> nn.Module:
    model = avsiam_models.CAVMAE_BASE(modality_specific_depth=11)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_clean = {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()}
    own_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd_clean.items() if k in own_keys}
    result = model.load_state_dict(filtered, strict=False)
    print(f"AVSiam CAVMAE_BASE: loaded {len(filtered)}/{len(own_keys)} keys, "
          f"missing={len(result.missing_keys)}, unexpected_in_ckpt="
          f"{len(sd_clean) - len(filtered)}")
    if result.missing_keys:
        print("  missing (first 10):", result.missing_keys[:10])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, filtered, own_keys


@torch.no_grad()
def encode_pair(model: nn.Module, spec: Tensor, frame: Tensor) -> Tuple[Tensor, Tensor]:
    """spec: (B, T, mel)  frame: (B, 3, H, W)  -> (audio_emb, video_emb) mean-pooled, L2-normed.

    NOTE: deliberately bypasses model.forward_encoder(). That method (also what
    CAVMAE_BASE.forward() calls, i.e. nominally "the" training path in this
    repo's *current* source) routes audio through `self.ast_base` — a
    module-level `copy.deepcopy(self.vit_base)` taken at __init__ time, i.e.
    an independent, never-checkpoint-updated copy still sitting at its
    ImageNet-pretrained init. We verified this directly: the released
    as2m_pretrained.20.pth / as+vgg+acav_pretrained.pth checkpoints contain
    ZERO "ast_base.*" keys (confirmed by inspecting the raw state dict), while
    every audio-relevant vit_base.* key (patch_embed_a, pos_embed_a, norm_a,
    and each block's norm1_a/norm2_a alongside the shared attn/mlp) IS
    present. Using forward_encoder as literally written would silently run
    the audio branch through 293 untrained parameters. The only reading
    consistent with 100% of the checkpoint's (non-decoder) weights actually
    being exercised is the paper's own stated design -- ONE shared ViT stack
    (`vit_base.blocks`) for both modalities, differentiated only by the
    modality-conditioned norm1_a/norm1_v + norm2_a/norm2_v pairs each Block
    already carries (see Block.forward(x, modality=...) in cav_mae_base.py) --
    exactly mirroring how the video branch is (uncontroversially) computed.
    We reimplement that directly here. (The `norm_pre`/`norm_pre_a` addition
    forward_encoder also does is skipped as a no-op: norm_pre is nn.Identity
    for this timm backbone, and a global positive scalar on the block input
    is absorbed exactly by the very next operation, a LayerNorm.)
    """
    a = spec.unsqueeze(1).transpose(2, 3)      # (B,1,mel,T)
    v = frame                                   # (B,3,H,W)
    a = model.vit_base.patch_embed_a(a) + model.vit_base.pos_embed_a
    v = model.vit_base.patch_embed(v) + model.vit_base.pos_embed[:, 1:]
    for blk in model.vit_base.blocks:
        a = blk(a, "a")
        v = blk(v, "v")
    ca = model.vit_base.norm_a(a)
    cv = model.vit_base.norm(v)
    a_emb = F.normalize(ca.mean(dim=1), dim=-1)
    v_emb = F.normalize(cv.mean(dim=1), dim=-1)
    return a_emb, v_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "base_plus"], default="base")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--video-dir",
                     default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    ap.add_argument("--eval-list", default="data/vggsound_eval_1545.txt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_map = {
        "base": "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/avsiam/as2m_pretrained.20.pth",
        "base_plus": "/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/avsiam/as+vgg+acav_pretrained.pth",
    }
    ckpt_path = args.ckpt or ckpt_map[args.variant]

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}")
    N_requested = len(clip_ids)

    print(f"Loading AVSiam ({args.variant}) from {ckpt_path}")
    model, filtered, own_keys = load_model(ckpt_path, args.device)
    n_params_m = sum(v.numel() for v in model.state_dict().values()) / 1e6
    print(f"Total params in model.state_dict(): {n_params_m:.1f}M")

    print(f"Evaluating {N_requested} clips on {args.device}", flush=True)

    valid_ids = []
    audio_embs, video_embs = [], []
    fail_count = 0
    for i in range(0, N_requested, args.batch_size):
        batch_ids = clip_ids[i:i + args.batch_size]
        specs, frames, kept = [], [], []
        for vid in batch_ids:
            vpath = os.path.join(args.video_dir, vid + ".mp4")
            try:
                from torchcodec.decoders import AudioDecoder
                dec = AudioDecoder(vpath, sample_rate=16000)
                w = dec.get_all_samples().data
                w = w.mean(0) if w.shape[0] > 1 else w[0]
                spec = waveform_to_fbank(w)
                frame = decode_mid_frame(vpath)
            except Exception as e:
                fail_count += 1
                print(f"  FAILED {vid}: {e}")
                continue
            specs.append(spec)
            frames.append(frame)
            kept.append(vid)
        if not kept:
            continue
        spec_batch = torch.stack(specs).to(args.device)
        frame_batch = torch.stack(frames).to(args.device)
        a_emb, v_emb = encode_pair(model, spec_batch, frame_batch)
        audio_embs.append(a_emb.cpu())
        video_embs.append(v_emb.cpu())
        valid_ids.extend(kept)
        if (i // args.batch_size) % 5 == 0:
            print(f"  {i + len(batch_ids)}/{N_requested}", flush=True)

    audio_embs = torch.cat(audio_embs, dim=0)
    video_embs = torch.cat(video_embs, dim=0)
    N = audio_embs.shape[0]

    if N == N_requested:
        print(f"clips_seen={N}  (full-gallery OK, matches {N_requested})")
    else:
        print(f"WARNING: clips_seen={N} != clips_requested={N_requested} "
              f"({fail_count} clips failed to decode) — evaluating on the intersection.")

    sim_av = audio_embs @ video_embs.T
    sim_va = video_embs @ audio_embs.T
    res_av = recall_at_k(sim_av)
    res_va = recall_at_k(sim_va)

    print("\n" + "=" * 50)
    print(f"AVSiam ({args.variant}) retrieval  N={N}")
    print("-" * 50)
    print("audio_to_visual:", " ".join(f"{k}={v:.2f}%" for k, v in res_av.items()))
    print("visual_to_audio:", " ".join(f"{k}={v:.2f}%" for k, v in res_va.items()))
    print("=" * 50)

    preprocessing_note = (
        "Audio: torchaudio.compliance.kaldi.fbank(htk_compat=True, sample_frequency=16000 "
        "[resampled by us; repo's own loader trusts pre-extracted wav rate], use_energy=False, "
        "window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10), waveform "
        "mean-subtracted first, padded/truncated to 1024 frames, normalised "
        "(fbank-(-5.081))/4.4849 [src/dataloader.py:288,328,333-343,506; audio_conf in "
        "src/retrieval.py]. Video: single frame at ~50% of clip duration (matches eval-mode "
        "frame_use=5-of-10 pre-extracted frames), Resize(224,BICUBIC,antialias) shorter side, "
        "CenterCrop(224), Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) "
        "[src/dataloader.py:144-149,347-362]. Embedding: mean-pool all patch tokens then "
        "L2-normalize, matching src/retrieval.py:get_retrieval_result."
    )

    contamination = {
        "base": "HELD-OUT",       # AudioSet-2M only
        "base_plus": "IN-DISTRIBUTION",  # AudioSet-2M + VGGSound + ACAV2.4M
    }[args.variant]
    corpus = {
        "base": "AudioSet-2M",
        "base_plus": "AudioSet-2M + VGGSound + ACAV2.4M",
    }[args.variant]

    results = {
        "n_clips": N,
        "clips_requested": N_requested,
        "eval_list": args.eval_list,
        "audio_to_visual": res_av,
        "visual_to_audio": res_va,
        "ckpt": ckpt_path,
        "ckpt_sha256": sha256_of(ckpt_path),
        "repo": "https://github.com/GenjiB/AVSiam",
        "variant": "Base" if args.variant == "base" else "Base+",
        "params_millions": round(n_params_m, 1),
        "pretrain_corpus": corpus,
        "contamination_flag": contamination,
        "preprocessing": preprocessing_note,
        "date_evaluated": "2026-09-02",
        "eval_command": f"conda run -n jepa-omni python scripts/avsiam_retrieval.py --variant {args.variant}",
    }
    out_path = f"data/avsiam_{args.variant}_retrieval_results.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
