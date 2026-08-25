"""scripts/jepa_memory_phase2_av_extract.py — joint AUDIO-VISUAL identity features.

Phase 2a trained the identity head on voice alone, because that was where clean
data existed. That is the wrong shape for the north star: people are recognised by
FACE at least as much as by voice. This script supplies the vision half, paired with
audio for the SAME identity, so the head can be trained on both.

CORPUS: `blueskyheaven/voxceleb2-mp4-binary` -- the only mirror found that carries
VoxCeleb2 VIDEO. Each row is (mp4 bytes, speaker_id, file_name), and the mp4 is
already a tracked FACE CROP with its own audio, so no detector or bounding boxes
are needed and the vision and audio halves are guaranteed to be the same person at
the same moment.

THE ONE COMPROMISE, STATED UP FRONT: this mirror flattened VoxCeleb2's original
`speaker/youtube_video/segment` path down to a per-speaker running index, so the
source-video grouping is not directly available.
  * The PRIMARY protocol is unaffected: train/test is **speaker-disjoint** (the head
    never sees a test identity), which is the open-set task that matters and needs
    no video grouping at all.
  * For enrol/query WITHIN a test speaker, we split on the file_name INDEX RANGE
    (enrol on the low half, query on the high half). Verified directly that the
    index is assigned in the original traversal order -- a speaker's clips arrive in
    contiguous numeric clusters ({1-7}, {11-17}, {26-30}, ...), i.e. video by video.
    A range split is therefore an APPROXIMATE video split, contaminated only by the
    single source video that straddles the midpoint. It is labelled that way
    everywhere and must never be quoted as a clean cross-session number -- the clean
    cross-session voice number comes from the wds mirror (Phase 1B/2a), which does
    preserve the real paths.
  * An attempt to RECOVER exact grouping by joining to the wds mirror was made and
    is NOT yet resolved: the test compared against an incomplete wds shard set, so
    it was invalid rather than negative. Revisit once all 779 wds shards are local.

Both modalities use STATISTICS pooling (mean+std) for interface parity with
models/jepa_identity_head.py, even though Phase 2a measured that stats pooling
bought nothing over mean for audio -- keeping both halves lets the eval re-check
that on vision instead of assuming it transfers.

Usage:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/jepa_memory_phase2_av_extract.py \
        --shard-idx $i --num-shards 4 &
    done; wait
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models.jepa_identity_head import stats_pool  # noqa: E402

N_FRAMES = 16          # production StreamingConfig.n_vision_frames
CROP_RES = 256         # V-JEPA2 ViT-L native
AUDIO_SR = 16000
MAX_SEC = 10.0
MIN_SEC = 2.0


def decode_mp4(raw: bytes):
    """-> (frames (T,3,256,256) uint8, wav (n,) float32 @16k) or (None, None)."""
    from torchcodec.decoders import VideoDecoder, AudioDecoder
    try:
        vd = VideoDecoder(raw, device="cpu", num_ffmpeg_threads=4)
        n = vd.metadata.num_frames
        if n < 4:
            return None, None
        idx = torch.linspace(0, n - 1, N_FRAMES).long().tolist()
        fr = vd.get_frames_at(indices=idx).data.float()          # (T,3,H,W)
        fr = F.interpolate(fr, size=(CROP_RES, CROP_RES), mode="bilinear", align_corners=False)
        frames = fr.clamp(0, 255).to(torch.uint8)
    except Exception:
        return None, None
    try:
        w = AudioDecoder(raw, sample_rate=AUDIO_SR).get_all_samples().data
        w = w.mean(0) if w.dim() > 1 else w
        w = w[: int(MAX_SEC * AUDIO_SR)].float().numpy()
        if len(w) < MIN_SEC * AUDIO_SR:
            return None, None
    except Exception:
        return None, None
    return frames, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="/dev/shm/voxceleb2/data")
    ap.add_argument("--out-dir", default="/dev/shm/jepa_mem_p2av")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-per-speaker", type=int, default=24)
    ap.add_argument("--cpu-threads", type=int, default=24)
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    files = files[args.shard_idx::args.num_shards]
    print(f"[p2av:{args.shard_idx}] {len(files)} parquet shards", flush=True)

    from models.vision_encoder import VisionEncoder
    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    vis_enc = VisionEncoder(device=str(device))
    base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    print(f"[p2av:{args.shard_idx}] encoders loaded", flush=True)

    rows: List[Dict] = []
    V: List[torch.Tensor] = []
    A: List[torch.Tensor] = []
    per_spk: Dict[str, int] = defaultdict(int)
    n_ok = n_bad = 0
    t0 = time.time()

    for fi, path in enumerate(files):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            d = batch.to_pydict()
            for vid, spk, fn in zip(d["video"], d["speaker_id"], d["file_name"]):
                if per_spk[spk] >= args.max_per_speaker:
                    continue
                raw = vid["bytes"] if isinstance(vid, dict) else vid
                frames, wav = decode_mp4(raw)
                if frames is None:
                    n_bad += 1; continue
                try:
                    with torch.no_grad():
                        vt = vis_enc.encode(frames.unsqueeze(0).to(device))[0]     # (N,1024)
                        v_stats = stats_pool(vt.float().unsqueeze(0), dim=1)[0]    # (2048,)
                        wt = torch.from_numpy(wav)
                        b = base.encode(wt.view(1, 1, -1).to(device))[0]
                        nn_ = nat.encode(wt.view(1, 1, -1).expand(1, 2, -1).to(device))[0]
                        amb = (b.float() + nn_.float()) * 0.5 if b.shape[0] == nn_.shape[0] else b.float()
                        a_stats = stats_pool(amb.unsqueeze(0), dim=1)[0]           # (1536,)
                except Exception as e:
                    print(f"[p2av:{args.shard_idx}] embed fail {spk}/{fn}: {e!r}", flush=True)
                    n_bad += 1; continue
                V.append(v_stats.cpu()); A.append(a_stats.cpu())
                rows.append({"speaker": spk, "file_name": fn, "idx": int(fn)})
                per_spk[spk] += 1; n_ok += 1
                if n_ok % 250 == 0:
                    print(f"[p2av:{args.shard_idx}] ok={n_ok} bad={n_bad} "
                          f"speakers={len(per_spk)} {time.time()-t0:.0f}s", flush=True)
        print(f"[p2av:{args.shard_idx}] finished parquet {fi+1}/{len(files)} "
              f"ok={n_ok} speakers={len(per_spk)}", flush=True)

    out = os.path.join(args.out_dir, f"shard{args.shard_idx}.pt")
    torch.save({"rows": rows, "vision": torch.stack(V, 0), "audio": torch.stack(A, 0)}, out)
    print(f"[p2av:{args.shard_idx}] DONE ok={n_ok} bad={n_bad} speakers={len(per_spk)} "
          f"-> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
