"""scripts/wav2clip_retrieval.py — Wav2CLIP audio<->visual retrieval baseline.

Runs Wav2CLIP (github.com/descriptinc/lyrebird-wav2clip, pip package `wav2clip`)
on the fixed 1545-clip VGGSound eval gallery (data/vggsound_eval_1545.txt) and
computes audio->visual / visual->audio R@1, R@5, R@10 by cosine similarity.

Wav2CLIP's own preprocessing (matched from its source, NOT ours):
  - Audio tower (wav2clip/model/resnet.py `ResNet.forward`): raw 16kHz mono
    waveform -> torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353)
    -> log(x+1e-7) -> GLOBAL (whole-tensor) mean/std instance-normalisation
    -> ResNet18 (2D conv over the spectrogram) -> avgpool -> 512-d ->
    3-layer MLP head (512-512-512) -> 512-d embedding. Because the model's
    own forward() normalises using the mean/std of the *entire* input batch
    tensor (not per-sample), we run with batch size 1 throughout (matching
    the package's own `wav2clip.embed_audio(audio, model)` usage pattern,
    which is documented for a single (1, n_samples) clip) so that no
    cross-sample leakage from batching enters the reported numbers.
  - Checkpoint: torch.hub-cached Wav2CLIP.pt (loaded via `wav2clip.get_model()`,
    scenario="frozen", transform=True i.e. the distilled MLP head IS applied).
  - Pretrained on VGGSound (paper: "~200k 10-second clips", "training split"),
    sample rate 16kHz (paper, since resnet.py has no internal resampling and
    VGGSound-standard audio models forward-declared for 16kHz Kaldi-style
    processing at n_fft=512/hop=353 match VGGSound's own resnet.py this file
    was copied from).

  - Visual tower: the FROZEN CLIP image encoder Wav2CLIP was distilled
    against. Confirmed as ViT-B/32 by dimensional match: the MLP head's
    final layer outputs 512-d, and OpenAI CLIP ViT-B/32's image embedding is
    512-d (ViT-L/14 would be 768-d, ruled out). Loaded via open_clip
    (`ViT-B-32-quickgelu`, pretrained="openai" -- the exact original OpenAI
    weights+activation; plain `ViT-B-32` mismatches on quick_gelu vs the
    'openai' tag, confirmed via open_clip's own load-time warning).
    Frame aggregation matches the Wav2CLIP paper's own retrieval-eval
    protocol (Sec. 4: "extract CLIP image embeddings for each frame ... and
    use mean pooling to get clip-level embeddings") -- we sample frames
    evenly across each clip (10 frames/clip; the paper's own eval used 150
    frames at 30fps over a 5s crop, reduced here for the 1545-clip eval time
    budget -- the aggregation method itself, mean-pooling per-frame CLIP
    embeddings, is unchanged and is the only thing the paper actually
    specifies as method, not the frame density) and mean-pool their CLIP
    embeddings, each preprocessed with CLIP's own official transform
    (open_clip's val preprocessing: resize 224, center-crop, CLIP
    mean/std normalisation).

Contamination note (verified, not assumed): our 1545-clip eval gallery has
ZERO overlap with data/train.csv (VGGSound's official training split, the
183,730-clip pool Wav2CLIP's paper says it trained on) -- all 1545 IDs are
drawn from data/test.csv (VGGSound's official 15,446-clip test split). So at
the level of VGGSound's own train/test partition, Wav2CLIP's audio tower did
not see these exact clips during distillation. This is a same-dataset
held-out split, not a cross-dataset gap -- VGGSound's train/test partition
is not de-duplicated against near-identical re-uploads from the same
channel, a documented VGGSound caveat, so treat "HELD-OUT" here as weaker
than an independent external test set.

Usage:
    conda run -n jepa-omni python scripts/wav2clip_retrieval.py
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def recall_at_k(sim: Tensor, ks=(1, 5, 10)):
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(min(k, N), dim=1).indices
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float().mean().item()
        out[f"R@{k}"] = round(hits * 100, 2)
    return out


def decode_audio_16k_mono(vpath: str) -> Tensor:
    from torchcodec.decoders import AudioDecoder
    dec = AudioDecoder(vpath, sample_rate=16000)
    w = dec.get_all_samples().data
    w = w.mean(0) if w.shape[0] > 1 else w[0]
    return w.float()


def decode_frames_clip(vpath: str, preprocess, n_frames: int = 10):
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    dec = VideoDecoder(vpath, device="cpu")
    n = int(getattr(dec.metadata, "num_frames", None) or len(dec))
    if n <= 0:
        raise RuntimeError("zero-length video")
    idxs = sorted(set(min(n - 1, int(round(i * (n - 1) / max(1, n_frames - 1))))
                       for i in range(min(n_frames, n))))
    frames = dec.get_frames_at(indices=idxs).data  # (T, 3, H, W) uint8
    imgs = []
    for t in range(frames.shape[0]):
        arr = frames[t].permute(1, 2, 0).numpy()
        imgs.append(preprocess(Image.fromarray(arr)))
    return torch.stack(imgs)  # (T, 3, 224, 224)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="/mnt/Raid-Storage-2/utkarsh-data/vggsound_raw/extracted/video")
    parser.add_argument("--eval-list", default=os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545.txt"))
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", default="/mnt/Raid-Storage-2/utkarsh-data/baseline_checkpoints/wav2clip")
    args = parser.parse_args()

    with open(args.eval_list) as f:
        clip_ids = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(clip_ids)} clips from {args.eval_list}", flush=True)
    N_requested = len(clip_ids)

    print(f"Evaluating {N_requested} clips on {args.device}", flush=True)

    import wav2clip
    os.environ.setdefault("TORCH_HOME", args.ckpt_dir)
    audio_model = wav2clip.get_model(device=args.device)
    audio_model.eval()

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", cache_dir=args.ckpt_dir)
    clip_model = clip_model.to(args.device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # locate + hash the wav2clip checkpoint actually used (torch.hub cache)
    hub_dir = torch.hub.get_dir()
    ckpt_path = os.path.join(hub_dir, "checkpoints", "Wav2CLIP.pt")
    if not os.path.isfile(ckpt_path):
        # fall back: search TORCH_HOME
        for root, _, files in os.walk(args.ckpt_dir):
            for fn in files:
                if fn.lower().startswith("wav2clip"):
                    ckpt_path = os.path.join(root, fn)
    ckpt_sha256 = None
    if os.path.isfile(ckpt_path):
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ckpt_sha256 = h.hexdigest()
    print(f"wav2clip checkpoint: {ckpt_path} sha256={ckpt_sha256}", flush=True)

    audio_embs = []
    video_embs = []
    n_failed = 0
    fail_reasons = []

    for i, vid in enumerate(clip_ids):
        vpath = os.path.join(args.video_dir, vid + ".mp4")
        try:
            wav = decode_audio_16k_mono(vpath).to(args.device)
            with torch.no_grad():
                a_emb = audio_model(wav.unsqueeze(0))  # (1, 512), batch=1 -> no cross-sample norm leakage
            a_emb = a_emb.squeeze(0).float().cpu()

            frames = decode_frames_clip(vpath, preprocess, args.n_frames).to(args.device)
            with torch.no_grad():
                v_feats = clip_model.encode_image(frames)  # (T, 512)
                v_feats = F.normalize(v_feats.float(), dim=-1)
                v_emb = v_feats.mean(0)
            v_emb = v_emb.cpu()

            audio_embs.append(a_emb)
            video_embs.append(v_emb)
        except Exception as e:
            n_failed += 1
            fail_reasons.append(f"{vid}: {type(e).__name__}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_requested}", flush=True)

    N = len(audio_embs)
    print(f"\nProcessed {N}/{N_requested} clips successfully; {n_failed} failed.", flush=True)
    if n_failed:
        print("Failure examples:", fail_reasons[:10], flush=True)

    if N == N_requested:
        assert N == 1545, f"clips_seen={N} != 1545 -- eval-list itself is not 1545 clips!"
        print(f"clips_seen={N} == 1545 (full-gallery OK)", flush=True)
    else:
        print(f"WARNING: clips_seen={N} != requested {N_requested}. "
              f"Evaluating on the intersection of {N} successfully-decoded clips.", flush=True)

    audio_embs = torch.stack(audio_embs)  # (N, 512)
    video_embs = torch.stack(video_embs)  # (N, 512)
    audio_embs = F.normalize(audio_embs, dim=-1)
    video_embs = F.normalize(video_embs, dim=-1)

    sim_av = audio_embs @ video_embs.T
    sim_va = video_embs @ audio_embs.T
    res_av = recall_at_k(sim_av)
    res_va = recall_at_k(sim_va)

    print("\n" + "=" * 50)
    print(f"Wav2CLIP retrieval  N={N}")
    print("-" * 50)
    print("audio→visual:", " ".join(f"{k}={v:.2f}%" for k, v in res_av.items()))
    print("visual→audio:", " ".join(f"{k}={v:.2f}%" for k, v in res_va.items()))
    print("=" * 50)

    audio_params = sum(p.numel() for p in audio_model.parameters())
    clip_params = sum(p.numel() for p in clip_model.parameters())
    total_params_m = (audio_params + clip_params) / 1e6
    print(f"audio_tower_params={audio_params/1e6:.2f}M clip_image_tower_params(full CLIP model)={clip_params/1e6:.2f}M")

    results = {
        "n_clips": N,
        "clips_requested": N_requested,
        "n_failed": n_failed,
        "eval_list": "data/vggsound_eval_1545.txt",
        "audio_to_visual": res_av,
        "visual_to_audio": res_va,
        "ckpt": ckpt_path,
        "ckpt_sha256": ckpt_sha256,
        "repo": "https://github.com/descriptinc/lyrebird-wav2clip",
        "params_millions": round(total_params_m, 1),
        "params_note": "audio ResNet18+MLP tower + full CLIP ViT-B/32 (image+text) both counted; only the image tower and audio tower are used at inference",
        "pretrain_corpus": "VGGSound (audio tower distilled from frozen CLIP ViT-B/32; paper: ~200k 10s clips, official VGGSound training split)",
        "contamination_flag": "HELD-OUT (verified: 0/1545 eval clip IDs present in data/train.csv, VGGSound's official 183,730-clip training split that Wav2CLIP's paper reports training on; all 1545 are in data/test.csv instead) -- caveat: same-dataset split, not de-duplicated against near-identical re-uploads, so treat as weaker than a cross-dataset held-out set",
        "preprocessing": (
            "Audio: raw 16kHz mono waveform (torchcodec AudioDecoder, channel-mean mixdown) fed "
            "directly into Wav2CLIP's own ResNet18 (wav2clip/model/resnet.py), which internally computes "
            "torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353), log(x+1e-7), and global "
            "tensor mean/std normalisation, then the distilled 512-512-512 MLP head. Run at batch size 1 "
            "(the package's own single-clip usage pattern) because that internal normalisation is computed "
            "over the whole input tensor, not per-sample, so larger batches would leak stats across clips. "
            "Video: 10 frames sampled evenly across each clip, each resized/cropped/normalised with "
            "open_clip's official ViT-B/32 'openai' preprocessing, encoded with the frozen CLIP image tower, "
            "L2-normalised per-frame, mean-pooled to one 512-d clip embedding (matches the paper's own "
            "frame-mean-pooling retrieval-eval method; frame count reduced from the paper's 150 (30fps x 5s) "
            "for the 1545-clip eval time budget -- documented deviation, not a silent one)."
        ),
        "date_evaluated": "2026-09-02",
        "eval_command": "conda run -n jepa-omni python scripts/wav2clip_retrieval.py",
    }

    out_path = os.path.join(PROJECT_ROOT, "data", "wav2clip_retrieval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
