"""scripts/jepa_memory_phase1b_voice.py — Phase 1B (voice half) of the JEPA-memory
track: does the DEPLOYED audio path carry speaker identity ACROSS SESSIONS?

Why this exists: Phase 1A could not answer it. EasyCom's only cross-session identity
(p1) has no close microphone, so every audio-bearing stream there was confounded by
microphone channel (guests = close mic, p1 = glasses array). This script removes that
confound entirely by using VoxCeleb2, where "different session" is unambiguous.

DATA + PROTOCOL
  `gaunernst/voxceleb2-dev-wds` — 5,994 speakers, 779 shards, audio only, and
  critically it PRESERVES the original `speaker_id/youtube_video_id/segment` key
  (verified directly: 'id02139/yCPbcLeT5SI/00147.m4a'). The middle field is a distinct
  source YouTube video = a genuinely different recording session: different day,
  different room, different microphone, different channel. Enrolling on one set of
  source videos and testing on HELD-OUT source videos is therefore a real
  cross-session test, not a within-recording split.

  (The other mirror, `blueskyheaven/voxceleb2-mp4-binary`, DOES have video but was
  checked and FLATTENED the path to a bare segment index — the source-video grouping
  is gone, so a split on it would silently leak same-recording segments. It is
  deliberately not used here. That is why this script covers the voice half only;
  the vision half still needs a source-grouped face-video corpus.)

STREAMS (both frozen, both deployment-relevant — this is a real "which encoder should
the memory use" decision, not just a probe):
  wavjepa    (768) WavJEPA base+nat mean == exactly what M2 consumes as `ambient`
  moonshine  (416) MoonshineSpeechEncoder states — ALREADY resident on the Jetson at
                   37ms/turn for STT, so using it for voice identity costs no extra
                   on-device latency

Usage:
    python scripts/jepa_memory_phase1b_voice.py --shard-dir /dev/shm/vox2wds \
        --n-speakers 200 --videos-per-speaker 4 --segs-per-video 4
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

AUDIO_SR = 16000
MAX_SEC = 10.0          # matches extract_features_av.CLIP_DURATION_S / M2's window
MIN_SEC = 3.0


def build_index(shard_dir: str) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """(speaker, source_video) -> [(tar_path, member_name), ...]"""
    idx: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for tp in sorted(glob.glob(os.path.join(shard_dir, "*.tar"))):
        with tarfile.open(tp) as t:
            for m in t.getmembers():
                if not m.name.endswith(".m4a"):
                    continue
                parts = m.name.split("/")
                if len(parts) != 3:
                    continue
                idx[(parts[0], parts[1])].append((tp, m.name))
    return idx


def decode_m4a(raw: bytes) -> np.ndarray | None:
    from torchcodec.decoders import AudioDecoder
    try:
        d = AudioDecoder(raw, sample_rate=AUDIO_SR)
        w = d.get_all_samples().data          # (ch, n)
        w = w.mean(0) if w.dim() > 1 else w
        return w.float().numpy()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/dev/shm/vox2wds")
    ap.add_argument("--n-speakers", type=int, default=200)
    ap.add_argument("--videos-per-speaker", type=int, default=4)
    ap.add_argument("--segs-per-video", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/JEPA_MEMORY_PHASE1B_VOICE_RESULTS.json")
    ap.add_argument("--emb-out", default="/dev/shm/jepa_mem_p1b/voice_emb.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("[p1b] indexing shards ...", flush=True)
    idx = build_index(args.shard_dir)
    by_spk: Dict[str, List[str]] = defaultdict(list)
    for (spk, vid) in idx:
        by_spk[spk].append(vid)
    # need enough DISTINCT source videos to make a cross-video split meaningful
    ok = sorted([s for s, v in by_spk.items() if len(v) >= args.videos_per_speaker])
    print(f"[p1b] {len(idx)} (speaker,video) groups, {len(by_spk)} speakers, "
          f"{len(ok)} with >={args.videos_per_speaker} distinct source videos", flush=True)
    speakers = sorted(rng.choice(ok, size=min(args.n_speakers, len(ok)), replace=False).tolist())

    # collect the clips we actually want, grouped per tar so each tar opens once
    want: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)   # tar -> [(member, spk, vid)]
    n_target = 0
    for spk in speakers:
        vids = sorted(by_spk[spk])
        vids = [vids[i] for i in rng.permutation(len(vids))[: args.videos_per_speaker]]
        for vid in vids:
            members = idx[(spk, vid)]
            sel = [members[i] for i in rng.permutation(len(members))[: args.segs_per_video]]
            for tp, mn in sel:
                want[tp].append((mn, spk, vid)); n_target += 1
    print(f"[p1b] {len(speakers)} speakers, target {n_target} clips", flush=True)

    from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO, WAVJEPA_NAT_REPO
    from models.m4_speech import MoonshineSpeechEncoder
    base = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))
    nat = AudioEncoder(WAVJEPA_NAT_REPO, n_channels=2, device=str(device))
    moon = MoonshineSpeechEncoder().to(device).eval()
    print("[p1b] encoders loaded", flush=True)

    rows: List[Dict] = []
    embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
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
                dur = len(w) / AUDIO_SR
                wt = torch.from_numpy(w)
                try:
                    with torch.no_grad():
                        b = base.encode(wt.view(1, 1, -1).to(device))[0]
                        n = nat.encode(wt.view(1, 1, -1).expand(1, 2, -1).to(device))[0]
                        amb = (b.float() + n.float()) * 0.5 if b.shape[0] == n.shape[0] else b.float()
                        hidden, _ = moon([w], [dur], device)
                    embs["wavjepa"].append(amb.mean(0).cpu())
                    embs["moonshine"].append(hidden.float().mean(1)[0].cpu())
                except Exception as e:
                    print(f"[p1b] embed fail {mn}: {e!r}", flush=True); n_bad += 1; continue
                rows.append({"speaker": spk, "video": vid, "member": mn, "dur": dur})
                n_ok += 1
        if (ti + 1) % 20 == 0:
            print(f"[p1b] tar {ti+1}/{len(want)}  ok={n_ok} bad={n_bad}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"[p1b] embedded {n_ok} clips ({n_bad} failed) in {time.time()-t0:.0f}s", flush=True)
    E = {k: torch.stack(v, 0) for k, v in embs.items()}
    os.makedirs(os.path.dirname(args.emb_out), exist_ok=True)
    torch.save({"rows": rows, "emb": E}, args.emb_out)

    # ── eval: enroll on some source videos, test on HELD-OUT source videos ──
    res: Dict[str, Dict] = {}
    spk_list = sorted({r["speaker"] for r in rows})
    sidx = {s: i for i, s in enumerate(spk_list)}
    by_sv: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_sv[(r["speaker"], r["video"])].append(i)

    enroll_idx: Dict[str, List[int]] = defaultdict(list)
    query_idx: List[int] = []
    for s in spk_list:
        vids = sorted({v for (sp, v) in by_sv if sp == s})
        if len(vids) < 2:
            continue
        n_en = max(1, len(vids) - max(1, len(vids) // 2))
        for v in vids[:n_en]:
            enroll_idx[s].extend(by_sv[(s, v)])
        for v in vids[n_en:]:
            query_idx.extend(by_sv[(s, v)])

    print(f"\n[p1b] CROSS-VIDEO split: {len(spk_list)} speakers, "
          f"{sum(len(v) for v in enroll_idx.values())} enroll / {len(query_idx)} query clips")
    print(f"\n{'stream':<12} {'top1':>7} {'ci95':>16} {'chance':>8} {'shuf':>7} "
          f"{'AUC':>7} {'TAR@FAR1%':>10}")
    for k, Z in E.items():
        z = F.normalize(Z.float(), dim=-1)
        ids = [s for s in spk_list if enroll_idx.get(s)]
        cent = torch.stack([F.normalize(z[enroll_idx[s]].mean(0, keepdim=True), dim=-1)[0] for s in ids])
        q = [i for i in query_idx if rows[i]["speaker"] in ids]
        sims = z[q] @ cent.T
        pred = sims.argmax(1).numpy()
        true = np.array([ids.index(rows[i]["speaker"]) for i in q])
        corr = (pred == true).astype(float)
        boot = [corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000)]
        shuf = (rng.permutation(len(ids))[pred] == true).astype(float).mean()

        # verification: genuine = correct speaker (cross-video by construction)
        gen = sims[np.arange(len(q)), true].numpy()
        mask = np.ones(sims.shape, bool); mask[np.arange(len(q)), true] = False
        imp = sims.numpy()[mask]
        allv = np.concatenate([gen, imp]); order = allv.argsort()
        ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
        auc = (ranks[: len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp))
        tar = float((gen >= np.quantile(imp, 0.99)).mean())

        res[k] = {"top1": float(corr.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                  float(np.percentile(boot, 97.5))], "chance": 1.0 / len(ids),
                  "shuffled": float(shuf), "auc": float(auc), "tar_at_far1pct": tar,
                  "n_query": len(q), "n_speakers": len(ids)}
        r = res[k]
        print(f"{k:<12} {r['top1']:7.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
              f"{r['chance']:8.4f} {r['shuffled']:7.3f} {r['auc']:7.3f} {r['tar_at_far1pct']:10.3f}")

    out = {"protocol": "cross-source-video (different YouTube recording = different session)",
           "n_clips": len(rows), "n_speakers": len(spk_list), "results": res}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p1b] wrote {args.out}")

    best = max(res, key=lambda k: res[k]["top1"])
    b = res[best]
    print("\n== GATE ==")
    print(f"  cross-session voice top-1 >= 3x chance: {b['top1']:.3f} vs chance "
          f"{b['chance']:.4f} ({best})  -> {'PASS' if b['top1'] >= 3*b['chance'] else 'FAIL'}")
    print(f"  shuffle control at chance             : {b['shuffled']:.4f}")


if __name__ == "__main__":
    main()
