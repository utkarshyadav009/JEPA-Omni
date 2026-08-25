"""scripts/jepa_memory_phase2_voice_extract.py — Phase 2 voice-identity feature dump.

Same frozen WavJEPA encoder and same cross-video VoxCeleb2 protocol as Phase 1B, with
two changes:
  * scale: enough SPEAKERS to train a head and still hold out unseen ones (Phase 1B's
    250 speakers were sized for a probe, not for training).
  * pooling: saves STATISTICS pooling (concat of mean and std over the token axis)
    instead of mean-only. Phase 1B used mean-only, so keeping both halves lets the
    eval report the pooling gain separately from the head's own contribution rather
    than conflating them.

Sharded across GPUs; each shard writes its own .pt.

Usage:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python scripts/jepa_memory_phase2_voice_extract.py \
        --shard-idx $i --num-shards 4 --n-speakers 1400 &
    done; wait
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from scripts.jepa_memory_phase1b_voice import build_index, decode_m4a, AUDIO_SR, MAX_SEC, MIN_SEC  # noqa: E402
from models.jepa_identity_head import stats_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/dev/shm/vox2wds")
    ap.add_argument("--out-dir", default="/dev/shm/jepa_mem_p2voice")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-speakers", type=int, default=1400)
    ap.add_argument("--videos-per-speaker", type=int, default=6)
    ap.add_argument("--segs-per-video", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)          # SAME seed on every shard ->
                                                    # identical speaker selection, so
                                                    # shards partition one consistent pool
    idx = build_index(args.shard_dir)
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for (spk, vid) in idx:
        by_spk[spk].append(vid)
    ok = sorted([s for s, v in by_spk.items() if len(v) >= args.videos_per_speaker])
    speakers = sorted(rng.choice(ok, size=min(args.n_speakers, len(ok)), replace=False).tolist())
    # partition SPEAKERS across shards (not clips) so no speaker is split across files
    mine = set(speakers[args.shard_idx::args.num_shards])
    print(f"[p2v:{args.shard_idx}] {len(ok)} eligible speakers, pool={len(speakers)}, "
          f"this shard={len(mine)}", flush=True)

    want: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    n_target = 0
    for spk in speakers:                             # iterate the FULL pool so the rng
        vids = sorted(by_spk[spk])                   # draw sequence matches every shard
        vids = [vids[i] for i in rng.permutation(len(vids))[: args.videos_per_speaker]]
        for vid in vids:
            members = idx[(spk, vid)]
            sel = [members[i] for i in rng.permutation(len(members))[: args.segs_per_video]]
            if spk not in mine:
                continue
            for tp, mn in sel:
                want[tp].append((mn, spk, vid)); n_target += 1
    print(f"[p2v:{args.shard_idx}] target {n_target} clips", flush=True)

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    print(f"[p2v:{args.shard_idx}] encoders loaded", flush=True)

    rows: List[Dict] = []
    stats: List[torch.Tensor] = []
    t0 = time.time(); n_ok = n_bad = 0
    for ti, (tp, items) in enumerate(sorted(want.items())):
        with tarfile.open(tp) as t:
            for mn, spk, vid in items:
                try:
                    raw = t.extractfile(mn).read()
                except Exception:
                    n_bad += 1; continue
                w = decode_m4a(raw)
                if w is None or len(w) < MIN_SEC * AUDIO_SR:
                    n_bad += 1; continue
                w = w[: int(MAX_SEC * AUDIO_SR)]
                wt = torch.from_numpy(w)
                try:
                    with torch.no_grad():
                        b = base.encode(wt.view(1, 1, -1).to(device))[0]
                        n = nat.encode(wt.view(1, 1, -1).expand(1, 2, -1).to(device))[0]
                        amb = (b.float() + n.float()) * 0.5 if b.shape[0] == n.shape[0] else b.float()
                        sp = stats_pool(amb.unsqueeze(0), dim=1)[0]      # (1536,)
                except Exception as e:
                    print(f"[p2v:{args.shard_idx}] fail {mn}: {e!r}", flush=True)
                    n_bad += 1; continue
                stats.append(sp.cpu())
                rows.append({"speaker": spk, "video": vid, "member": mn})
                n_ok += 1
        if (ti + 1) % 25 == 0:
            print(f"[p2v:{args.shard_idx}] tar {ti+1}/{len(want)} ok={n_ok} bad={n_bad} "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = os.path.join(args.out_dir, f"shard{args.shard_idx}.pt")
    torch.save({"rows": rows, "stats": torch.stack(stats, 0)}, out)
    print(f"[p2v:{args.shard_idx}] DONE ok={n_ok} bad={n_bad} -> {out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
