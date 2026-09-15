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


def run(data, tr_v, ev_v, delta, dev, grid_train=None, grid_eval=10.0):
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

    out = {}
    for inp_name, ti, ei, ebi, tsi, esi in (("W", 0, 0, 0, 0, 0), ("VA", 1, 1, 1, 1, 1)):
        for tgt in TARGETS:
            Ytr = make_target(tgt, tr[0], tr[2], tr[3], tr[4], pca_W, pca_dW, tr[5])
            Yev = make_target(tgt, ev[0], ev[2], ev[3], ev[4], pca_W, pca_dW, ev[5])
            Ybw = make_target(tgt, ev_bwd[0], ev_bwd[2], ev_bwd[3], ev_bwd[4], pca_W, pca_dW, ev_bwd[5])
            Yts = make_target(tgt, tr_shuf[0], tr_shuf[2], tr_shuf[3], tr_shuf[4], pca_W, pca_dW, tr_shuf[5])
            Yes = make_target(tgt, ev_shuf[0], ev_shuf[2], ev_shuf[3], ev_shuf[4], pca_W, pca_dW, ev_shuf[5])
            Xtr, Xev, Xbw, Xts, Xes = tr[ti], ev[ei], ev_bwd[ebi], tr_shuf[tsi], ev_shuf[esi]

            Wm, mx, my = ridge_fit(Xtr, Ytr, dev)
            tmean = Ytr.mean(0, keepdim=True).to(dev)
            P = (Xev.to(dev) - mx) @ Wm + my; Y = Yev.to(dev)
            r_f = r2(P, Y, tmean); se_f = boot_se(P, Y, tmean)

            Pb = (Xbw.to(dev) - mx) @ Wm + my; Yb = Ybw.to(dev)
            r_b = r2(Pb, Yb, tmean)

            Wm2, mx2, my2 = ridge_fit(Xts, Yts, dev)
            Ps = (Xes.to(dev) - mx2) @ Wm2 + my2; Ys = Yes.to(dev)
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
           "probes": run(data, tr_v, ev_v, a.delta, dev, a.grid_train, 10.0)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p6] wrote {a.out}", flush=True)
