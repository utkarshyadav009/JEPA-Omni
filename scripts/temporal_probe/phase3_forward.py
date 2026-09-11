"""scripts/temporal_probe/phase3_forward.py — Phase 3, forward-prediction probe.

Question: does W(t) contain information about W(t+Delta) BEYOND the fact that the
world changes slowly?

WHAT THIS TESTS, STATED UP FRONT: whether the SPACE is predictable, not whether
the MODEL predicts. RUN-2 was trained with lam_pred = 0.0 and has no term of any
kind referencing a future window. A positive result here would say the latent
space admits a learnable forward map, NOT that the architecture performs
prediction. Do not let any report imply otherwise.

Delta >= 10 s is a hard floor (10 s non-overlapping extraction stride; Ego4D raw
video is gone). See docs/ERRATA_PROPOSED.md E-9.

Baselines reported beside every learned result, per instruction:
  IDENTITY   : predict W(t+Delta) = W(t) unchanged
  CORPUS MEAN: predict the training-split mean
  RESCALED   : per-dimension affine rescaling of W(t) (a*W+b fitted per dim)

Split is FILE-DISJOINT. R@1 retrieves the true future from ALL candidate futures
in the held-out files; that gallery size is reported with the number.

Usage:
    python scripts/temporal_probe/phase3_forward.py \
        --out docs/artifacts/temporal_probe/phase3_forward.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import _ts_to_tdm_bins

EXPECTED_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"
MAX_TDM, VIS_SPAT = 512, 16


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def load_ws(cache, wid, m2, device):
    d = torch.load(os.path.join(cache, wid[:2], wid + ".pt"), map_location="cpu", weights_only=True)
    dur = float(d.get("clip_duration_s", 10.0))
    vis = d["vision"]; T_v, S_v, D_v = vis.shape
    vts = d["vision_ts"].unsqueeze(1).expand(T_v, S_v, 2).reshape(T_v * S_v, 2)
    base, nat = d["ambient_base"], d["ambient_nat"]
    aud = ((base.float() + nat.float()) * 0.5) if base.shape[0] == nat.shape[0] else base.float()
    feats = {"vision": vis.reshape(T_v * S_v, D_v).float().unsqueeze(0).to(device),
             "ambient": aud.unsqueeze(0).to(device)}
    tbins = {"vision": _ts_to_tdm_bins(vts, dur, MAX_TDM).unsqueeze(0).to(device),
             "ambient": _ts_to_tdm_bins(d["ambient_base_ts"], dur, MAX_TDM).unsqueeze(0).to(device)}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                          enabled=(device.type == "cuda")):
        return m2.encode_world_state(feats, tbins).float()[0].cpu()


def metrics(pred, true, tag):
    cos = F.cosine_similarity(pred, true, dim=-1)
    sim = F.normalize(pred, dim=-1) @ F.normalize(true, dim=-1).T
    gt = torch.arange(true.shape[0])
    r1 = (sim.argmax(1) == gt).float().mean().item() * 100
    r5 = (sim.topk(min(5, sim.shape[1]), dim=1).indices == gt[:, None]).any(1).float().mean().item() * 100
    return {"method": tag, "cosine_mean": round(cos.mean().item(), 4),
            "cosine_std": round(cos.std().item(), 4),
            "R@1": round(r1, 2), "R@5": round(r5, 2), "gallery_size": int(true.shape[0])}


@torch.no_grad()
def fit_ridge(X, Y, lam=1.0):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1])
    return torch.linalg.solve(A, Xb.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--cache", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_ego4d_train_v1")
    ap.add_argument("--max-files", type=int, default=120)
    ap.add_argument("--min-run", type=int, default=10)
    ap.add_argument("--deltas", default="1,2,3", help="index offsets; x10 s")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sha = sha256_file(a.ckpt)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"ABORT: sha {sha}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                       max_tdm_bins=MAX_TDM, dropout=0.0)).to(device)
    m2.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=False)["model"])
    m2.eval()

    pat = re.compile(r"^ego4d_(.+)_w(\d+)$"); byf = defaultdict(list)
    for sh in os.listdir(a.cache):
        p = os.path.join(a.cache, sh)
        if os.path.isdir(p):
            for f in os.listdir(p):
                m = pat.match(f[:-3]) if f.endswith(".pt") else None
                if m:
                    byf[m.group(1)].append(int(m.group(2)))
    runs = []
    for fid, ws in byf.items():
        ws = sorted(ws); best = cur = [ws[0]]
        for x, y in zip(ws, ws[1:]):
            cur = cur + [y] if y - x == 1 else [y]
            if len(cur) > len(best):
                best = cur
        if len(best) >= a.min_run:
            runs.append((fid, best))
    runs.sort(key=lambda t: -len(t[1]))
    runs = runs[: a.max_files]

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(runs), generator=g).tolist()
    n_test = int(len(runs) * a.test_frac)
    test_ids = {runs[i][0] for i in perm[:n_test]}
    print(f"[phase3] files: {len(runs)} total -> {len(runs)-n_test} train / {n_test} test "
          f"(FILE-DISJOINT)", flush=True)

    W = {}
    t0 = time.time()
    for i, (fid, ws) in enumerate(runs):
        W[fid] = [load_ws(a.cache, f"ego4d_{fid}_w{w:04d}", m2, device) for w in ws]
        if (i + 1) % 20 == 0:
            print(f"[phase3] encoded {i+1}/{len(runs)} files {time.time()-t0:.0f}s", flush=True)

    out = {}
    for d in [int(x) for x in a.deltas.split(",")]:
        tr_x, tr_y, te_x, te_y = [], [], [], []
        for fid, ws in runs:
            S = W[fid]
            for j in range(len(S) - d):
                (te_x if fid in test_ids else tr_x).append(S[j])
                (te_y if fid in test_ids else tr_y).append(S[j + d])
        Xtr, Ytr = torch.stack(tr_x), torch.stack(tr_y)
        Xte, Yte = torch.stack(te_x), torch.stack(te_y)
        rows = [metrics(Xte, Yte, "IDENTITY (copy W(t))"),
                metrics(Ytr.mean(0, keepdim=True).expand_as(Yte), Yte, "corpus mean")]
        # per-dimension affine rescale fitted on train
        mx, sx = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        my, sy = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
        rows.append(metrics((Xte - mx) / sx * sy + my, Yte, "per-dim rescaled copy"))
        Wr = fit_ridge(Xtr, Ytr)
        rows.append(metrics(torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1) @ Wr, Yte, "ridge"))
        mlp = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024))
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xn, Xten = (Xtr - mu) / sd, (Xte - mu) / sd
        for ep in range(200):
            opt.zero_grad()
            loss = F.mse_loss(mlp(Xn), Ytr)
            loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            rows.append(metrics(mlp(Xten), Yte, "2-layer MLP"))
        out[f"{d*10}s"] = {"n_train_pairs": Xtr.shape[0], "n_test_pairs": Xte.shape[0],
                           "gallery_size": Xte.shape[0], "results": rows}
        print(f"\n[phase3] Delta = {d*10}s   test pairs = {Xte.shape[0]} (= R@1 gallery size)", flush=True)
        for r in rows:
            print(f"    {r['method']:<24} cos={r['cosine_mean']:.4f}  R@1={r['R@1']:6.2f}  R@5={r['R@5']:6.2f}", flush=True)

    json.dump({"script": "scripts/temporal_probe/phase3_forward.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ckpt": a.ckpt, "ckpt_sha256": sha,
               "n_files": len(runs), "n_test_files": n_test, "split": "FILE-DISJOINT",
               "seed": a.seed, "stride_s": 10.0,
               "caveat": "Tests whether the SPACE is predictable, not whether the MODEL "
                         "predicts: RUN-2 trained with lam_pred=0.0 and has no term "
                         "referencing any future window.",
               "by_delta": out}, open(a.out, "w"), indent=2)
    print(f"[phase3] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
