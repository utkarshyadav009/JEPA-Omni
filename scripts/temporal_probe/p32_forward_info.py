"""P3.2 — forward-information probe. Does W(t) carry information about the FUTURE beyond
symmetric temporal persistence and scene identity?

RECEPTIVE FIELD, MEASURED NOT ASSUMED: V-JEPA2 (vjepa2-vitl-fpc64-256) samples 64 frames
uniformly across the FULL 10 s window and WavJEPA runs 100 Hz over the same 10 s, so ONE
window's receptive field is the whole 10 s. Any target within +-10 s of W(t) therefore shares
input content with it. Consequences, both forced by that number rather than chosen:
  * Delta < 10 s is NOT a prediction horizon at any stride -- it is partly retrieval of content
    already inside the input. Not reported.
  * The within-file gallery excludes everything within +-10 s of the query.

The Ego4D cache is already in the valid regime: 10 s NON-OVERLAPPING stride == Delta=10 s with
zero input overlap, 77,831 consecutive pairs on disk.

Stages (run in this order; e is first on purpose):
  e  falsifier      -- mismatched-file pairs must land at chance, else nothing is interpretable
  a  persistence    -- cos(W(t),W(t+d)) vs a different-file floor, rescaled per model
  b  within-file    -- retrieve the true future from the SAME recording (primary)
  c  backward       -- same everything, predicting W(t-d). forward-minus-backward is the headline
  d  residual       -- g: W(t) -> W(t+d) - W(t)
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import _ts_to_tdm_bins

CACHE = "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_ego4d_train_v1"
STRIDE_S, WINDOW_S, MAX_TDM = 10.0, 10.0, 512
CKPTS = {"RUN-2 step19000": "checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt",
         "RUN-4 step18000": "checkpoints/m2_run4_padfix_ta896/step18000.pt"}
CAP = {"RUN-2 step19000": None, "RUN-4 step18000": 896}     # each model at its OWN training length


def runs_by_file(min_run):
    pat = re.compile(r"^ego4d_(.+)_w(\d+)$"); byf = defaultdict(list)
    for sh in os.listdir(CACHE):
        p = os.path.join(CACHE, sh)
        if os.path.isdir(p):
            for f in os.listdir(p):
                m = pat.match(f[:-3]) if f.endswith(".pt") else None
                if m: byf[m.group(1)].append(int(m.group(2)))
    out = []
    for fid, ws in byf.items():
        ws = sorted(ws); best = cur = [ws[0]]
        for x, y in zip(ws, ws[1:]):
            cur = cur + [y] if y - x == 1 else [y]
            if len(cur) > len(best): best = cur
        if len(best) >= min_run: out.append((fid, best))
    out.sort(key=lambda t: -len(t[1]))
    return out


@torch.no_grad()
def encode_file(m2, fid, wins, dev, cap):
    W = []
    for w in wins:
        wid = f"ego4d_{fid}_w{w:04d}"
        d = torch.load(os.path.join(CACHE, wid[:2], wid + ".pt"), map_location="cpu", weights_only=True)
        dur = float(d.get("clip_duration_s", 10.0))
        vis = d["vision"]; T, S, D = vis.shape
        vts = d["vision_ts"].unsqueeze(1).expand(T, S, 2).reshape(T * S, 2)
        base, nat = d["ambient_base"], d["ambient_nat"]
        aud = ((base.float() + nat.float()) * 0.5) if base.shape[0] == nat.shape[0] else base.float()
        ab = _ts_to_tdm_bins(d["ambient_base_ts"], dur, MAX_TDM)
        if cap is not None and aud.shape[0] > cap:
            aud, ab = aud[:cap], ab[:cap]
        feats = {"vision": vis.reshape(T * S, D).float().unsqueeze(0).to(dev),
                 "ambient": aud.unsqueeze(0).to(dev)}
        tb = {"vision": _ts_to_tdm_bins(vts, dur, MAX_TDM).unsqueeze(0).to(dev),
              "ambient": ab.unsqueeze(0).to(dev)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            W.append(m2.encode_world_state(feats, tb).float()[0].cpu())
    return torch.stack(W)


def load_m2(path, dev):
    m = AVJepaPredictor(AVJepaConfig()).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"]); m.eval()
    return m


def fit_ridge(X, Y, lam=1.0):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    return torch.linalg.solve(Xb.T @ Xb + lam * torch.eye(Xb.shape[1]), Xb.T @ Y)


def apply_ridge(Wr, X):
    return torch.cat([X, torch.ones(X.shape[0], 1)], 1) @ Wr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="e", choices=["encode", "e"])
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--max-files", type=int, default=300)
    ap.add_argument("--wcache", default="/home/utkarsh/JEPA-Omni/data/p32_wcache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(a.wcache, exist_ok=True)
    files = runs_by_file(a.min_run)[: a.max_files]
    print(f"[p32] {len(files)} files, run lengths {len(files[-1][1])}-{len(files[0][1])}", flush=True)

    # ---- encode W(t) for every selected window, per checkpoint (cached) ----
    for name, cp in CKPTS.items():
        tag = name.split()[0].lower() + "_" + os.path.basename(cp).replace(".pt", "")
        out_p = os.path.join(a.wcache, f"{tag}.pt")
        if os.path.exists(out_p):
            print(f"[p32] {name}: cached", flush=True); continue
        m2 = load_m2(cp, dev); store = {}
        t0 = time.time()
        for i, (fid, wins) in enumerate(files):
            store[fid] = {"wins": wins, "W": encode_file(m2, fid, wins, dev, CAP[name]).half()}
            if (i + 1) % 25 == 0:
                print(f"[p32] {name}: {i+1}/{len(files)} files {time.time()-t0:.0f}s", flush=True)
        torch.save(store, out_p); del m2; torch.cuda.empty_cache()
        print(f"[p32] {name}: wrote {out_p}", flush=True)

    if a.stage == "encode":
        sys.exit(0)

    # ---- P3.2e FALSIFIER: mismatched-file pairs must land at chance ----
    print("\n[p32e] FALSIFIER — training on MISMATCHED-file pairs", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    report = {}
    for name in CKPTS:
        tag = name.split()[0].lower() + "_" + os.path.basename(CKPTS[name]).replace(".pt", "")
        store = torch.load(os.path.join(a.wcache, f"{tag}.pt"), map_location="cpu", weights_only=False)
        fids = sorted(store)
        ntest = max(1, len(fids) // 3)
        perm = torch.randperm(len(fids), generator=g).tolist()
        test = {fids[i] for i in perm[:ntest]}
        for d in (1, 2, 3, 6):
            def pairs(split_test):
                X, Y = [], []
                fl = [f for f in fids if (f in test) == split_test]
                for f in fl:
                    W = store[f]["W"].float()
                    if W.shape[0] <= d: continue
                    # MISMATCH: target comes from a DIFFERENT randomly chosen file
                    for j in range(W.shape[0] - d):
                        o = fl[torch.randint(len(fl), (1,), generator=g).item()]
                        Wo = store[o]["W"].float()
                        if Wo.shape[0] <= d or o == f: continue
                        k = torch.randint(Wo.shape[0] - d, (1,), generator=g).item()
                        X.append(W[j]); Y.append(Wo[k + d])
                return (torch.stack(X), torch.stack(Y)) if X else (None, None)
            Xtr, Ytr = pairs(False); Xte, Yte = pairs(True)
            if Xtr is None or Xte is None: continue
            Wr = fit_ridge(Xtr, Ytr)
            P = apply_ridge(Wr, Xte)
            sim = F.normalize(P, dim=-1) @ F.normalize(Yte, dim=-1).T
            gt = torch.arange(Yte.shape[0])
            r1 = (sim.argmax(1) == gt).float().mean().item() * 100
            chance = 100.0 / Yte.shape[0]
            report.setdefault(name, {})[f"{int(d*STRIDE_S)}s"] = {
                "R@1": round(r1, 3), "chance": round(chance, 3),
                "ratio_to_chance": round(r1 / chance, 2), "gallery": int(Yte.shape[0])}
            print(f"[p32e] {name:<18} d={int(d*STRIDE_S):>3}s  R@1={r1:6.3f}%  "
                  f"chance={chance:.3f}%  ratio={r1/chance:5.2f}x  n={Yte.shape[0]}", flush=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"\n[p32e] wrote {a.out}", flush=True)
