"""P6 -- INFORMATION-EXISTENCE PROBES. Does W_t contain ANY measurable information about t+D?

The RUN-5 pilot showed the tested post-hoc PREDICTORS do not beat persistence. That is a weak
test of INFORMATION CONTENT, because a predictor fighting `future = current + change` looks bad
whether (a) there is no change information, or (b) the change information exists but direct
future-state prediction cannot extract it against a strong persistence baseline.

These probes separate those. They are linear/ridge ONLY -- an MLP is run only where a linear
probe already finds signal. No new architectures, no new losses, no new data.

=====================  METRICS PREDECLARED BEFORE ANY RUN  =====================
R2 on HELD-OUT videos = 1 - SS_res / SS_tot, with SS_tot taken about the TRAIN mean.
  R2 = 0  means "no better than predicting the training mean".
  R2 < 0  means "worse than the training mean".

For every target we report, in this order:
  R2_forward      ridge W_t -> target(t+D)
  R2_backward     ridge W_t -> target(t-D)
  R2_shuffled     ridge fitted on temporally shuffled pairs (the null)
  R2_persistence  use t's OWN value of the property as the prediction (where defined)

SIGNAL CRITERION, fixed now:
  a target carries forward information iff
      R2_forward - R2_shuffled >= 0.05   AND   R2_forward >= 3 x bootstrap SE
  and it carries information BEYOND PERSISTENCE iff additionally
      R2_forward - R2_persistence >= 0.05

THE THREE-WAY DIAGNOSTIC (section 5 of the brief). Two INPUTS, same probe, same targets:
  W    : W_t                              -- the fused representation (1024)
  VA   : [vision_mean_t ; ambient_mean_t] -- pre-fusion frozen encoder summaries (1024+768)
  VA predicts but W does not  -> fusion into W DISCARDS predictive information
  neither predicts            -> little predictable signal at this horizon
  W predicts but InfoNCE failed -> the prediction OBJECTIVE was the bottleneck

CAVEAT, stated up front: vision_mean / ambient_mean are MEAN-POOLED over the window, not the
full token sequences. They are a cheap faithful proxy for "pre-fusion features", not the full
token probe -- that would need the 610 GB feature cache and is not cheap.

Temporal rules are unchanged: D = 10 s, window 10 s, so pairs share ZERO input. Evaluation uses
the 10 s-spaced within-file grid. No same-window leakage anywhere.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch

sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
WS_DIR = "/home/utkarsh/JEPA-Omni/data/epic_kitchens_ws"
M2_SHA = "27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4"


def load(ws_dir):
    out = {}
    for fn in sorted(f for f in os.listdir(ws_dir) if f.endswith(".pt")):
        d = torch.load(os.path.join(ws_dir, fn), map_location="cpu", weights_only=True)
        assert d["m2_ckpt_sha256"] == M2_SHA
        out[fn[:-3]] = d
    return out


def split_participants(vids, every=5):
    ps = sorted({v.split("_")[0] for v in vids}); hold = set(ps[::every])
    return ([v for v in vids if v.split("_")[0] not in hold],
            [v for v in vids if v.split("_")[0] in hold], sorted(hold))


def pair_index(d, delta, grid=None):
    """(i, j) with start[j] == start[i] + delta; grid restricts i to a lattice."""
    st = d["start_s"].tolist(); pos = {round(x, 3): k for k, x in enumerate(st)}
    out = []
    for k, x in enumerate(st):
        if grid is not None and abs((x / grid) - round(x / grid)) > 1e-6: continue
        j = pos.get(round(x + delta, 3))
        if j is not None: out.append((k, j))
    return out


def gather(data, vids, delta, grid, shuffle_seed=None):
    """Returns W_i, VA_i, W_j, VIS_j, AUD_j  (float32, CPU).

    shuffle_seed permutes the INPUT rows within each file, never the target index. Permuting the
    target was the first implementation and it is INVALID for difference targets: dW = W_j - W_i
    under a shuffled j has much larger variance, inflating SS_tot and therefore R2. It made the
    null outscore the real pairing (dW_full: null 0.337 vs real 0.227), which is a broken null
    rather than an absent signal. Permuting inputs leaves the target distribution untouched.
    """
    Wi, Vi, Wj, VJ, AJ, DW = [], [], [], [], [], []
    g = torch.Generator().manual_seed(shuffle_seed or 0)
    for v in vids:
        d = data[v]; idx = pair_index(d, delta, grid)
        if not idx: continue
        W = d["world_state"].float(); VM = d["vision_mean"].float(); AM = d["ambient_mean"].float()
        i = torch.tensor([a for a, _ in idx]); j = torch.tensor([b for _, b in idx])
        wi, vi = W[i], torch.cat([VM[i], AM[i]], 1)
        wj, vj, aj = W[j], VM[j], AM[j]
        dW = wj - wi                                  # computed from the TRUE pairing
        if shuffle_seed is not None:
            perm = torch.randperm(len(i), generator=g)
            wi, vi = wi[perm], vi[perm]               # inputs only
        Wi.append(wi); Vi.append(vi); Wj.append(wj); VJ.append(vj); AJ.append(aj)
        DW.append(dW)
    return (torch.cat(Wi), torch.cat(Vi), torch.cat(Wj), torch.cat(VJ), torch.cat(AJ),
            torch.cat(DW))


def ridge_fit(X, Y, dev, lams=(1e1, 1e2, 1e3, 1e4), val_frac=0.15):
    """Closed form, lambda picked on a held-out slice of TRAIN (never on eval)."""
    n = X.shape[0]; nv = int(n * val_frac)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, va = perm[nv:], perm[:nv]
    Xt, Yt = X[tr].to(dev), Y[tr].to(dev); Xv, Yv = X[va].to(dev), Y[va].to(dev)
    mx, my = Xt.mean(0, keepdim=True), Yt.mean(0, keepdim=True)
    Xc, Yc = Xt - mx, Yt - my
    A = Xc.T @ Xc; B = Xc.T @ Yc; I = torch.eye(A.shape[0], device=dev)
    best, bw = None, None
    for lam in lams:
        Wm = torch.linalg.solve(A + lam * I, B)
        r = ((Xv - mx) @ Wm + my - Yv).pow(2).sum().item()
        if best is None or r < best: best, bw = r, (Wm, mx, my)
    return bw


def r2(pred, Y, train_mean):
    ss_res = (pred - Y).pow(2).sum().item()
    ss_tot = (Y - train_mean).pow(2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def boot_se(pred, Y, train_mean, n=200, seed=0):
    g = torch.Generator(device=Y.device).manual_seed(seed)
    N = Y.shape[0]; vals = []
    for _ in range(n):
        idx = torch.randint(N, (N,), generator=g, device=Y.device)
        vals.append(r2(pred[idx], Y[idx], train_mean))
    t = torch.tensor(vals)
    return float(t.std().item())


TARGETS = ["W_full", "W_norm", "W_mean", "W_std", "W_pca8",
           "dW_full", "dW_norm", "dW_pca8", "vis_full", "vis_norm", "aud_full", "aud_norm"]


def make_target(name, Wi, Wj, VJ, AJ, pca_W=None, pca_dW=None, DW=None):
    if name == "W_full":  return Wj
    if name == "W_norm":  return Wj.norm(dim=1, keepdim=True)
    if name == "W_mean":  return Wj.mean(1, keepdim=True)
    if name == "W_std":   return Wj.std(1, keepdim=True)
    if name == "W_pca8":  return Wj @ pca_W
    if name == "dW_full": return DW if DW is not None else (Wj - Wi)
    if name == "dW_norm": return (DW if DW is not None else (Wj - Wi)).norm(dim=1, keepdim=True)
    if name == "dW_pca8": return (DW if DW is not None else (Wj - Wi)) @ pca_dW
    if name == "vis_full": return VJ
    if name == "vis_norm": return VJ.norm(dim=1, keepdim=True)
    if name == "aud_full": return AJ
    if name == "aud_norm": return AJ.norm(dim=1, keepdim=True)
    raise ValueError(name)


def persistence_pred(name, Wi, Vi, VJ, AJ):
    """t's OWN value of the property -- the persistence baseline. None where undefined."""
    VM, AM = Vi[:, :1024], Vi[:, 1024:]
    if name == "W_full":  return Wi
    if name == "W_norm":  return Wi.norm(dim=1, keepdim=True)
    if name == "W_mean":  return Wi.mean(1, keepdim=True)
    if name == "W_std":   return Wi.std(1, keepdim=True)
    if name == "dW_full": return torch.zeros_like(Wi)          # "nothing changes"
    if name == "dW_norm": return None
    if name == "vis_full": return VM
    if name == "vis_norm": return VM.norm(dim=1, keepdim=True)
    if name == "aud_full": return AM
    if name == "aud_norm": return AM.norm(dim=1, keepdim=True)
    return None


class MLPProbe(torch.nn.Module):
    def __init__(self, din, dout, h=2048):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(din), torch.nn.Linear(din, h),
                                       torch.nn.GELU(), torch.nn.Linear(h, dout))
    def forward(self, x): return self.net(x)


def mlp_fit(X, Y, dev, steps=15000, bs=4096, lr=1e-4, seed=0, val_frac=0.15, patience=10):
    """Small MLP, MSE, with EARLY STOPPING on a held-out slice of TRAIN.

    The first version had no early stopping and OVERFIT badly: held-out R2 got WORSE with more
    training (-0.026 at 3k steps, -0.270 at 15k), while ridge -- which picks lambda on a
    train-internal split -- reached +0.227 on the same data. Comparing an unregularised MLP
    against a regularised ridge is rigged, and the resulting "no non-linear signal" reading would
    have been an artifact of that. The MLP now gets the SAME discipline: a train-internal
    validation split, with the best checkpoint restored.

    Run ONLY where the linear probe already found signal (predeclared rule).
    """
    torch.manual_seed(seed)
    n = X.shape[0]; nv = int(n * val_frac)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr_i, va_i = perm[nv:], perm[:nv]
    Xt, Yt = X[tr_i], Y[tr_i]
    Xv, Yv = X[va_i].to(dev), Y[va_i].to(dev)
    mx, my = Xt.mean(0, keepdim=True).to(dev), Yt.mean(0, keepdim=True).to(dev)
    sy = torch.ones_like(my)   # centre only -- see note above; scaling misaligns the objective
    m = MLPProbe(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed); N = Xt.shape[0]
    best, best_state, bad, check = float("inf"), None, 0, max(200, steps // 50)
    for step in range(steps):
        idx = torch.randint(N, (min(bs, N),), generator=g)
        loss = torch.nn.functional.mse_loss(m((Xt[idx].to(dev) - mx)), (Yt[idx].to(dev) - my) / sy)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % check == 0:
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(m(Xv - mx), (Yv - my) / sy).item()
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: m.load_state_dict(best_state)
    m.eval()
    return lambda Xe: (m((Xe.to(dev) - mx)) * sy + my)


def pca_reduce(train_M, dims, dev):
    """Fit PCA on TRAIN, return a projection to `dims`. Capacity control: VA has 1792 dims vs
    W's 1024, so part of VA's advantage could be CAPACITY rather than CONTENT."""
    mu = train_M.mean(0, keepdim=True)
    Mc = (train_M - mu).to(dev)
    V = torch.linalg.svd(Mc, full_matrices=False)[2][:dims].T.cpu()
    return mu, V


def run(data, tr_v, ev_v, delta, dev, grid_train=None, grid_eval=10.0,
        va_dims=None, probe="ridge", mlp_targets=None):
    tr = gather(data, tr_v, delta, grid_train)
    ev = gather(data, ev_v, delta, grid_eval)
    ev_bwd = gather(data, ev_v, -delta, grid_eval)
    tr_shuf = gather(data, tr_v, delta, grid_train, shuffle_seed=1)
    ev_shuf = gather(data, ev_v, delta, grid_eval, shuffle_seed=2)
    print(f"[p6] train pairs {tr[0].shape[0]:,}  eval pairs {ev[0].shape[0]:,}  "
          f"eval-bwd {ev_bwd[0].shape[0]:,}", flush=True)

    # PCA bases fitted on TRAIN only
    def pca8(M):
        Mc = (M - M.mean(0, keepdim=True)).to(dev)
        return torch.linalg.svd(Mc, full_matrices=False)[2][:8].T.cpu()
    pca_W = pca8(tr[2]); pca_dW = pca8(tr[5])

    # CAPACITY CONTROL: reduce VA to va_dims so it is matched to W's 1024.
    va_proj = None
    if va_dims is not None:
        mu, V = pca_reduce(tr[1], va_dims, dev)
        va_proj = (mu, V)
        print(f"[p6] CAPACITY CONTROL: VA {tr[1].shape[1]} -> {va_dims} dims via train-fitted PCA",
              flush=True)

    def maybe_reduce(name, M):
        if name == "VA" and va_proj is not None:
            mu, V = va_proj
            return (M - mu) @ V
        return M

    out = {}
    tgt_list = TARGETS if mlp_targets is None else mlp_targets
    for inp_name, ti, ei, ebi, tsi, esi in (("W", 0, 0, 0, 0, 0), ("VA", 1, 1, 1, 1, 1)):
        for tgt in tgt_list:
            Ytr = make_target(tgt, tr[0], tr[2], tr[3], tr[4], pca_W, pca_dW, tr[5])
            Yev = make_target(tgt, ev[0], ev[2], ev[3], ev[4], pca_W, pca_dW, ev[5])
            Ybw = make_target(tgt, ev_bwd[0], ev_bwd[2], ev_bwd[3], ev_bwd[4], pca_W, pca_dW, ev_bwd[5])
            Yts = make_target(tgt, tr_shuf[0], tr_shuf[2], tr_shuf[3], tr_shuf[4], pca_W, pca_dW, tr_shuf[5])
            Yes = make_target(tgt, ev_shuf[0], ev_shuf[2], ev_shuf[3], ev_shuf[4], pca_W, pca_dW, ev_shuf[5])
            Xtr = maybe_reduce(inp_name, tr[ti]); Xev = maybe_reduce(inp_name, ev[ei])
            Xbw = maybe_reduce(inp_name, ev_bwd[ebi]); Xts = maybe_reduce(inp_name, tr_shuf[tsi])
            Xes = maybe_reduce(inp_name, ev_shuf[esi])
            tmean = Ytr.mean(0, keepdim=True).to(dev)
            Y = Yev.to(dev); Yb = Ybw.to(dev); Ys = Yes.to(dev)

            if probe == "ridge":
                Wm, mx, my = ridge_fit(Xtr, Ytr, dev)
                P = (Xev.to(dev) - mx) @ Wm + my
                Pb = (Xbw.to(dev) - mx) @ Wm + my
                Wm2, mx2, my2 = ridge_fit(Xts, Yts, dev)
                Ps = (Xes.to(dev) - mx2) @ Wm2 + my2
            else:
                f = mlp_fit(Xtr, Ytr, dev); P = f(Xev); Pb = f(Xbw)
                fs = mlp_fit(Xts, Yts, dev); Ps = fs(Xes)
            r_f = r2(P, Y, tmean); se_f = boot_se(P, Y, tmean)
            r_b = r2(Pb, Yb, tmean)
            r_s = r2(Ps, Ys, Yts.mean(0, keepdim=True).to(dev))

            pp = persistence_pred(tgt, ev[0], ev[1], ev[3], ev[4])
            r_p = r2(pp.to(dev), Y, tmean) if pp is not None else None

            sig = (r_f - r_s) >= 0.05 and r_f >= 3 * se_f
            beyond = (r_p is not None) and sig and (r_f - r_p) >= 0.05
            out[f"{inp_name}:{tgt}"] = {"R2_fwd": round(r_f, 4), "se": round(se_f, 4),
                                        "R2_bwd": round(r_b, 4), "R2_shuf": round(r_s, 4),
                                        "R2_persist": (round(r_p, 4) if r_p is not None else None),
                                        "signal": bool(sig), "beyond_persistence": bool(beyond),
                                        "dim": Yev.shape[1]}
            ps = f"{r_p:+.4f}" if r_p is not None else "   n/a "
            print(f"[p6] {inp_name:>2}:{tgt:<9} R2_fwd {r_f:+.4f}±{se_f:.4f}  bwd {r_b:+.4f}  "
                  f"shuf {r_s:+.4f}  persist {ps}  "
                  f"{'SIGNAL' if sig else '  --  '}{'  BEYOND-PERSISTENCE' if beyond else ''}",
                  flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--grid-train", type=float, default=2.0,
                    help="train-pair lattice in s; 2.0 keeps ~150k pairs without 90%% overlap")
    ap.add_argument("--va-dims", type=int, default=0, help="PCA-reduce VA to this many dims (capacity control)")
    ap.add_argument("--probe", default="ridge", choices=["ridge", "mlp"])
    ap.add_argument("--targets", default="", help="comma list; default all")
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p6_information_probes.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    print("[p6] loading...", flush=True)
    data = load(WS_DIR); vids = sorted(data)
    tr_v, ev_v, held = split_participants(vids)
    print(f"[p6] participant split: train {len(tr_v)} eval {len(ev_v)}, held-out {held}", flush=True)
    print(f"[p6] SIGNAL CRITERION (predeclared): R2_fwd - R2_shuf >= 0.05 AND R2_fwd >= 3*SE;"
          f"  BEYOND-PERSISTENCE additionally R2_fwd - R2_persist >= 0.05", flush=True)
    res = {"delta": a.delta, "grid_train": a.grid_train, "held_out": held,
           "va_dims": a.va_dims or None, "probe": a.probe,
           "probes": run(data, tr_v, ev_v, a.delta, dev, a.grid_train, 10.0,
                         va_dims=(a.va_dims or None), probe=a.probe,
                         mlp_targets=([t for t in a.targets.split(",") if t] or None))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p6] wrote {a.out}", flush=True)
