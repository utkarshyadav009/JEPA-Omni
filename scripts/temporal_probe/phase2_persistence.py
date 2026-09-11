"""scripts/temporal_probe/phase2_persistence.py — Phase 2, persistence curve.

Question: how long does information survive in the fused latent, and does the
fusion add any persistence BEYOND the window its input already spans?

DELTA IS CAPPED AT >= 10 s AND CANNOT BE FINER. The Ego4D feature cache was
extracted at a 10 s NON-OVERLAPPING stride (every start_sec in the held-out
manifest is a multiple of 10; window index x 10 = start second), and the raw
video no longer exists on this machine -- `find` returns 0 mp4 files across both
Ego4D trees. So Delta in {1,2,5} s is unobtainable without re-acquiring the
corpus. See docs/ERRATA_PROPOSED.md E-8/E-9 and docs/CORPUS_OPTIONS.md.

Window ordering is recovered from the cache ids, which encode it:
`ego4d_<file-uuid>_w<index>`. Only CONSECUTIVE index runs are used, so a
"Delta = 20 s" pair is genuinely 20 s apart in the source file and not two
windows that happen to sit either side of a gap.

Four curves on the SAME window pairs:
  world_state : m2.encode_world_state              (the object under test)
  vision      : mean-pooled raw V-JEPA2 tokens     (reference a -- its own input)
  ambient     : mean-pooled raw WavJEPA tokens     (reference b -- its own input)
  floor       : random pairs from DIFFERENT files  (reference c)

Batch size 1 throughout: a single clip cannot be padded, so the padding leak
(E-1) cannot touch these numbers by construction.

Usage:
    python scripts/temporal_probe/phase2_persistence.py --max-files 50 \
        --out docs/artifacts/temporal_probe/phase2_persistence.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, re, subprocess, sys, time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import _ts_to_tdm_bins

EXPECTED_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"
STRIDE_S = 10.0
MAX_TDM = 512
VIS_SPAT = 16


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def load_window(cache: str, wid: str, device):
    d = torch.load(os.path.join(cache, wid[:2], wid + ".pt"), map_location="cpu", weights_only=True)
    dur = float(d.get("clip_duration_s", 10.0))
    vis = d["vision"]                                  # (32,16,1024)
    T_v, S_v, D_v = vis.shape
    vis_flat = vis.reshape(T_v * S_v, D_v).float()
    vts = d["vision_ts"].unsqueeze(1).expand(T_v, S_v, 2).reshape(T_v * S_v, 2)
    vb = _ts_to_tdm_bins(vts, dur, MAX_TDM)
    base, nat = d["ambient_base"], d["ambient_nat"]
    aud = ((base.float() + nat.float()) * 0.5) if base.shape[0] == nat.shape[0] else base.float()
    ab = _ts_to_tdm_bins(d["ambient_base_ts"], dur, MAX_TDM)
    feats = {"vision": vis_flat.unsqueeze(0).to(device), "ambient": aud.unsqueeze(0).to(device)}
    tbins = {"vision": vb.unsqueeze(0).to(device), "ambient": ab.unsqueeze(0).to(device)}
    return feats, tbins, vis_flat.mean(0), aud.mean(0)


def halflife(curve: Dict[int, float], floor: float) -> float:
    """Delta at which cosine falls to the midpoint between its Delta=0 value and the floor.
    Linear interpolation between the bracketing measured deltas; None if never reached."""
    ds = sorted(curve)
    c0 = curve[ds[0]]
    mid = (c0 + floor) / 2.0
    for a, b in zip(ds, ds[1:]):
        if curve[a] >= mid >= curve[b]:
            if curve[a] == curve[b]:
                return float(a)
            return a + (curve[a] - mid) / (curve[a] - curve[b]) * (b - a)
    return float("nan")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--cache", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_ego4d_train_v1")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--min-run", type=int, default=10)
    ap.add_argument("--deltas", default="0,1,2,3,6", help="index offsets; x10 s")
    ap.add_argument("--floor-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sha = sha256_file(a.ckpt)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"ABORT: ckpt sha256 {sha} != {EXPECTED_SHA}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                       max_tdm_bins=MAX_TDM, dropout=0.0)).to(device)
    m2.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=False)["model"])
    m2.eval()

    pat = re.compile(r"^ego4d_(.+)_w(\d+)$")
    byfile = defaultdict(list)
    for sh in os.listdir(a.cache):
        p = os.path.join(a.cache, sh)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.endswith(".pt"):
                m = pat.match(f[:-3])
                if m:
                    byfile[m.group(1)].append(int(m.group(2)))

    # longest CONSECUTIVE run per file; longest runs first so big deltas are covered
    runs = []
    for fid, ws in byfile.items():
        ws = sorted(ws); best = cur = [ws[0]]
        for x, y in zip(ws, ws[1:]):
            cur = cur + [y] if y - x == 1 else [y]
            if len(cur) > len(best):
                best = cur
        if len(best) >= a.min_run:
            runs.append((fid, best))
    runs.sort(key=lambda t: -len(t[1]))
    runs = runs[: a.max_files]
    print(f"[phase2] {len(byfile)} files; {len(runs)} used "
          f"(run length {len(runs[-1][1])}-{len(runs[0][1])} windows)", flush=True)

    deltas = [int(x) for x in a.deltas.split(",")]
    sums = {k: defaultdict(list) for k in ("world_state", "vision", "ambient")}
    sums_n = {k: defaultdict(list) for k in ("world_state", "vision", "ambient")}
    per_file: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    t0 = time.time()
    for i, (fid, ws) in enumerate(runs):
        W, V, A = [], [], []
        for w in ws:
            feats, tbins, vmean, amean = load_window(a.cache, f"ego4d_{fid}_w{w:04d}", device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                W.append(m2.encode_world_state(feats, tbins).float()[0].cpu())
            V.append(vmean); A.append(amean)
        per_file[fid] = {"world_state": W, "vision": V, "ambient": A}
        for name, S in (("world_state", W), ("vision", V), ("ambient", A)):
            for d in deltas:
                for j in range(len(S) - d):
                    x, y = S[j], S[j + d]
                    sums[name][d].append(F.cosine_similarity(x, y, dim=0).item())
                    sums_n[name][d].append(
                        F.cosine_similarity(F.normalize(x, dim=0), F.normalize(y, dim=0), dim=0).item())
        if (i + 1) % 10 == 0:
            print(f"[phase2] {i+1}/{len(runs)} files  {time.time()-t0:.0f}s", flush=True)

    # FLOOR (reference c): random pairs drawn from DIFFERENT files, one per stream,
    # using the SAME pairing for all three so the floors are directly comparable.
    rng = random.Random(a.seed)
    fids = list(per_file)
    floor_raw = {k: [] for k in ("world_state", "vision", "ambient")}
    floor_nrm = {k: [] for k in ("world_state", "vision", "ambient")}
    for _ in range(a.floor_pairs):
        f1, f2 = rng.sample(fids, 2)
        i1 = rng.randrange(len(per_file[f1]["world_state"]))
        i2 = rng.randrange(len(per_file[f2]["world_state"]))
        for k in floor_raw:
            x, y = per_file[f1][k][i1], per_file[f2][k][i2]
            floor_raw[k].append(F.cosine_similarity(x, y, dim=0).item())
            floor_nrm[k].append(F.cosine_similarity(F.normalize(x, dim=0),
                                                    F.normalize(y, dim=0), dim=0).item())

    res = {"deltas_seconds": [d * STRIDE_S for d in deltas], "curves": {}, "floor": {}}
    mean = lambda v: sum(v) / len(v)
    for k in floor_raw:
        res["floor"][k] = {"raw_cosine": round(mean(floor_raw[k]), 6),
                           "l2norm_cosine": round(mean(floor_nrm[k]), 6),
                           "n_pairs": len(floor_raw[k])}
    for name in sums:
        raw = {d * STRIDE_S: round(mean(sums[name][d]), 6) for d in deltas}
        nrm = {d * STRIDE_S: round(mean(sums_n[name][d]), 6) for d in deltas}
        res["curves"][name] = {"raw_cosine": raw, "l2norm_cosine": nrm,
                               "n_pairs": {d * STRIDE_S: len(sums[name][d]) for d in deltas}}
    # half-life PER STREAM, each against its OWN different-file floor
    for name in sums:
        hl_raw = halflife(res["curves"][name]["raw_cosine"], res["floor"][name]["raw_cosine"])
        hl_nrm = halflife(res["curves"][name]["l2norm_cosine"], res["floor"][name]["l2norm_cosine"])
        res["curves"][name]["half_life_s_raw"] = None if hl_raw != hl_raw else round(hl_raw, 2)
        res["curves"][name]["half_life_s_l2norm"] = None if hl_nrm != hl_nrm else round(hl_nrm, 2)
        res["curves"][name]["half_life_note"] = (
            "None = cosine never fell to the midpoint between Delta=0 and the floor within "
            "the measured range (max Delta = %ds), so the half-life is BEYOND the measurable "
            "window, not absent." % int(max(res["deltas_seconds"])))

    res.update({"n_files": len(runs), "min_run": a.min_run, "stride_s": STRIDE_S,
                "file_ids": [f for f, _ in runs], "ckpt": a.ckpt, "ckpt_sha256": sha,
                "batch_size": 1,
                "note": "Delta >= 10 s is a HARD FLOOR: the cache was extracted at a 10 s "
                        "non-overlapping stride and the Ego4D raw video is gone."})
    print(json.dumps({k: res[k] for k in ("deltas_seconds", "curves", "floor")}, indent=2)[:2500], flush=True)
    json.dump({"script": "scripts/temporal_probe/phase2_persistence.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "result": res}, open(a.out, "w"), indent=2)
    print(f"[phase2] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
