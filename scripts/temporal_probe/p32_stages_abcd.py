"""P3.2 a-d: persistence, within-file retrieval, backward control, residual.
Falsifier (p32e) already PASSED at chance for both models at every Delta.

EXCLUSION BAND, forced by the measured receptive field: each encoder spans the full 10 s
window, so a gallery entry shares input content with the query only if their 10 s spans
overlap. At Ego4D's 10 s NON-overlapping stride that is offset 0 alone -- the query itself.
Offset +-1 is exactly adjacent with zero overlap, and excluding it would delete the
Delta=10 s TARGET. So "+-10 s exclusion" == "drop the query's own vector", which is also
precisely the artifact that gave IDENTITY its spurious R@5 advantage in Phase 3.
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")

STRIDE_S = 10.0
WC = "/home/utkarsh/JEPA-Omni/data/p32_wcache"
TAGS = {"RUN-2 step19000": "run-2_step19000", "RUN-4 step18000": "run-4_step18000"}


def fit_ridge(X, Y, lam=1.0):
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    return torch.linalg.solve(Xb.T @ Xb + lam * torch.eye(Xb.shape[1]), Xb.T @ Y)
def app_ridge(Wr, X):
    return torch.cat([X, torch.ones(X.shape[0], 1)], 1) @ Wr
def fit_mlp(X, Y, epochs=200, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    m = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Linear(1024, Y.shape[1]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xn = (X - mu) / sd
    for _ in range(epochs):
        opt.zero_grad(); F.mse_loss(m(Xn), Y).backward(); opt.step()
    m.eval()
    return lambda Z: m((Z - mu) / sd).detach()


def within_file_eval(store, fids, predict, d, backward=False, shuffle=False, seed=0):
    """Per-file gallery = every window in the recording EXCEPT the query itself (offset 0).
    Returns macro (per-file mean) and micro (pooled) R@1/R@5 plus cosine."""
    g = torch.Generator().manual_seed(seed)
    per_file, pooled_hit1, pooled_hit5, pooled_cos, pooled_n = [], 0, 0, [], 0
    for fid in fids:
        W = store[fid]["W"].float()
        if shuffle:
            W = W[torch.randperm(W.shape[0], generator=g)]
        N = W.shape[0]
        if N <= d + 1: continue
        qi = list(range(d, N)) if backward else list(range(0, N - d))
        ti = [i - d for i in qi] if backward else [i + d for i in qi]
        Q, T = W[qi], W[ti]
        P = predict(Q)
        Gal = W                                    # gallery = all windows in this recording
        sim = F.normalize(P, dim=-1) @ F.normalize(Gal, dim=-1).T      # (nq, N)
        for k, q in enumerate(qi):
            sim[k, q] = -1e9                        # +-10 s exclusion == drop the query itself
        gt = torch.tensor(ti)
        rk = sim.argsort(1, descending=True)
        h1 = (rk[:, 0] == gt).float(); h5 = (rk[:, :5] == gt[:, None]).any(1).float()
        cos = F.cosine_similarity(P, T, dim=-1)
        per_file.append({"fid": fid, "n_q": len(qi), "gallery": N - 1,
                         "R@1": h1.mean().item() * 100, "R@5": h5.mean().item() * 100,
                         "cos": cos.mean().item()})
        pooled_hit1 += h1.sum().item(); pooled_hit5 += h5.sum().item()
        pooled_cos.append(cos.sum().item()); pooled_n += len(qi)
    if not per_file: return None
    r1 = [p["R@1"] for p in per_file]; gal = [p["gallery"] for p in per_file]
    return {"macro_R@1": round(st.mean(r1), 3),
            "macro_R@5": round(st.mean([p["R@5"] for p in per_file]), 3),
            "micro_R@1": round(100 * pooled_hit1 / pooled_n, 3),
            "micro_R@5": round(100 * pooled_hit5 / pooled_n, 3),
            "cos": round(sum(pooled_cos) / pooled_n, 4),
            "median_file_R@1": round(st.median(r1), 3),
            "iqr_file_R@1": [round(sorted(r1)[len(r1)//4], 2), round(sorted(r1)[3*len(r1)//4], 2)],
            "mean_gallery": round(st.mean(gal), 1), "n_files": len(per_file), "n_queries": pooled_n,
            "chance_R@1": round(100 * st.mean([1.0 / gg for gg in gal]), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="1,2,3,6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = [int(x) for x in a.deltas.split(",")]
    SEEDS = [int(x) for x in a.seeds.split(",")]
    report = {}

    for name, tag in TAGS.items():
        store = torch.load(f"{WC}/{tag}.pt", map_location="cpu", weights_only=False)
        fids = sorted(store)
        report[name] = {}
        # ---------- P3.2a persistence, rescaled against own Delta=0 and own floor ----------
        floor = []
        gg = torch.Generator().manual_seed(0)
        for _ in range(3000):
            f1, f2 = [fids[i] for i in torch.randint(len(fids), (2,), generator=gg).tolist()]
            if f1 == f2: continue
            W1, W2 = store[f1]["W"].float(), store[f2]["W"].float()
            i1 = torch.randint(W1.shape[0], (1,), generator=gg).item()
            i2 = torch.randint(W2.shape[0], (1,), generator=gg).item()
            floor.append(F.cosine_similarity(W1[i1], W2[i2], dim=0).item())
        fl = st.mean(floor)
        pers = {"floor": round(fl, 4)}
        for d in D:
            cs = []
            for fid in fids:
                W = store[fid]["W"].float()
                if W.shape[0] <= d: continue
                cs += F.cosine_similarity(W[:-d], W[d:], dim=-1).tolist()
            raw = st.mean(cs)
            pers[f"{int(d*STRIDE_S)}s"] = {"raw_cos": round(raw, 4),
                                           "rescaled": round((raw - fl) / (1.0 - fl), 4)}
        report[name]["a_persistence"] = pers

        # ---------- splits (file-disjoint) ----------
        for seed in SEEDS:
            gs = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(fids), generator=gs).tolist()
            nte = max(1, len(fids) // 3)
            te = [fids[i] for i in perm[:nte]]; tr = [fids[i] for i in perm[nte:]]
            for d in D:
                dl = f"{int(d*STRIDE_S)}s"
                def build(fl_, back):
                    X, Y = [], []
                    for f in fl_:
                        W = store[f]["W"].float()
                        if W.shape[0] <= d: continue
                        if back: X.append(W[d:]); Y.append(W[:-d])
                        else:    X.append(W[:-d]); Y.append(W[d:])
                    return torch.cat(X), torch.cat(Y)
                for back, dirn in ((False, "forward"), (True, "backward")):
                    Xtr, Ytr = build(tr, back)
                    Wr = fit_ridge(Xtr, Ytr); mlp = fit_mlp(Xtr, Ytr, seed=seed)
                    mean_y = Ytr.mean(0, keepdim=True)
                    mx, sx = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
                    my, sy = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
                    methods = {
                        "IDENTITY":   lambda Q: Q,
                        "corpus_mean":lambda Q: mean_y.expand(Q.shape[0], -1),
                        "rescaled":   lambda Q: (Q - mx) / sx * sy + my,
                        "ridge":      lambda Q: app_ridge(Wr, Q),
                        "mlp":        mlp,
                    }
                    for mn, fn in methods.items():
                        r = within_file_eval(store, te, fn, d, backward=back, seed=seed)
                        if r: report[name].setdefault(f"b_{dirn}", {}).setdefault(dl, {}).setdefault(mn, []).append(r)
                    # temporal-shuffle control (ridge only; the strongest simple learner)
                    rs = within_file_eval(store, te, lambda Q: app_ridge(Wr, Q), d,
                                          backward=back, shuffle=True, seed=seed)
                    if rs: report[name].setdefault(f"b_{dirn}_shuffled", {}).setdefault(dl, {}).setdefault("ridge", []).append(rs)
                # ---------- P3.2d residual ----------
                Xtr, Ytr = build(tr, False)
                Wr_res = fit_ridge(Xtr, Ytr - Xtr)
                r = within_file_eval(store, te, lambda Q: Q + app_ridge(Wr_res, Q), d, seed=seed)
                if r: report[name].setdefault("d_residual", {}).setdefault(dl, {}).setdefault("ridge_residual", []).append(r)
        print(f"[p32abcd] {name} done", flush=True)

    json.dump(report, open(a.out, "w"), indent=2)
    print(f"[p32abcd] wrote {a.out}", flush=True)
