"""P7 -- FUSION BOTTLENECK DIAGNOSIS. What does fusion preserve, and what does it discard?

The question, stated as the comparison that matters:

    V-JEPA2 ---+                        V-JEPA2 + WavJEPA
               +--> future                      |
    WavJEPA ---+                              FUSION
                                                 |
                              versus             v
                                                 W
                                                 |
                                                 v
                                              future

Four inputs, ALL PCA-reduced to a COMMON dimension so capacity cannot explain any difference:
    V   vision_mean(t)                    (1024 raw)
    A   ambient_mean(t)                   ( 768 raw)   <- the natural minimum
    VA  [vision_mean ; ambient_mean](t)   (1792 raw)
    W   world_state(t)                    (1024 raw)

Targets (change first -- persistence cannot help there, so it is the cleanest directional test):
    dW, dV, dA      X(t+10) - X(t)
    W_next, V_next, A_next

METRICS ARE THE PREDECLARED ONES FROM P6, UNCHANGED: R2 on held-out participants about the TRAIN
mean; forward, backward, input-permuted null, persistence. Ridge only -- the MLP probe is
inconclusive (p6 3.4) and is NOT being tuned further.

NOT A SWEEP. One probe, one regularisation rule (lambda on a train-internal split), one split.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import importlib.util
_sp = importlib.util.spec_from_file_location("p6", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p6_information_probes.py")
p6 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(p6)

INPUTS = ["V", "A", "VA", "W"]
TGTS = ["dW", "dV", "dA", "W_next", "V_next", "A_next"]


def slice_input(name, Wi, Vi):
    VM, AM = Vi[:, :1024], Vi[:, 1024:]
    return {"V": VM, "A": AM, "VA": Vi, "W": Wi}[name]


def target(name, Wi, Vi, Wj, VJ, AJ, DW):
    VM, AM = Vi[:, :1024], Vi[:, 1024:]
    return {"dW": DW, "dV": VJ - VM, "dA": AJ - AM,
            "W_next": Wj, "V_next": VJ, "A_next": AJ}[name]


def persist(name, Wi, Vi):
    VM, AM = Vi[:, :1024], Vi[:, 1024:]
    z = lambda t: torch.zeros_like(t)
    return {"dW": z(Wi), "dV": z(VM), "dA": z(AM),
            "W_next": Wi, "V_next": VM, "A_next": AM}[name]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=768, help="common capacity for every input")
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--grid-train", type=float, default=2.0)
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p7_fusion_diagnosis.json")
    a = ap.parse_args()
    dev = torch.device("cuda")

    data = p6.load(p6.WS_DIR); vids = sorted(data)
    tr_v, ev_v, held = p6.split_participants(vids)
    tr = p6.gather(data, tr_v, a.delta, a.grid_train)
    ev = p6.gather(data, ev_v, a.delta, 10.0)
    eb = p6.gather(data, ev_v, -a.delta, 10.0)
    ts = p6.gather(data, tr_v, a.delta, a.grid_train, shuffle_seed=1)
    es = p6.gather(data, ev_v, a.delta, 10.0, shuffle_seed=2)
    print(f"[p7] train {tr[0].shape[0]:,}  eval {ev[0].shape[0]:,}  held-out kitchens {held}", flush=True)
    print(f"[p7] CAPACITY MATCH: every input PCA-reduced to {a.dims} dims (fitted on train)", flush=True)

    projs = {}
    for nm in INPUTS:
        M = slice_input(nm, tr[0], tr[1])
        projs[nm] = p6.pca_reduce(M, a.dims, dev) if M.shape[1] > a.dims else None

    def X(nm, pack):
        M = slice_input(nm, pack[0], pack[1])
        if projs[nm] is None: return M
        mu, V = projs[nm]; return (M - mu) @ V

    res = {}
    print(f"\n{'target':<8} {'input':<4} {'R2_fwd':>9} {'R2_bwd':>9} {'fwd/bwd':>8} {'null':>8} {'persist':>8}")
    for tgt in TGTS:
        for nm in INPUTS:
            Ytr = target(tgt, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])
            Yev = target(tgt, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5])
            Ybw = target(tgt, eb[0], eb[1], eb[2], eb[3], eb[4], eb[5])
            # NULL: permuted INPUTS, TRUE targets. The targets must come from the UNSHUFFLED
            # pack -- dV = V(t+10) - V(t) derives from the input V(t), so computing it from the
            # shuffled pack changes the TARGET distribution (larger variance -> inflated SS_tot
            # -> inflated R2). That is the same defect already fixed once for dW, and here it
            # made the null OUTSCORE the real pairing (dV:V null 0.346 vs real 0.276).
            # Rows align: gather() permutes only inputs, leaving row order and j-side intact.
            Yts = target(tgt, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])
            Yes = target(tgt, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5])
            Wm, mx, my = p6.ridge_fit(X(nm, tr), Ytr, dev)
            tmean = Ytr.mean(0, keepdim=True).to(dev)
            P = (X(nm, ev).to(dev) - mx) @ Wm + my; Y = Yev.to(dev)
            r_f = p6.r2(P, Y, tmean); se = p6.boot_se(P, Y, tmean)
            Pb = (X(nm, eb).to(dev) - mx) @ Wm + my
            r_b = p6.r2(Pb, Ybw.to(dev), tmean)
            Wm2, mx2, my2 = p6.ridge_fit(X(nm, ts), Yts, dev)
            Ps = (X(nm, es).to(dev) - mx2) @ Wm2 + my2
            r_s = p6.r2(Ps, Yes.to(dev), Yts.mean(0, keepdim=True).to(dev))
            pp = persist(tgt, ev[0], ev[1])
            r_p = p6.r2(pp.to(dev), Y, tmean)
            ratio = r_f / max(r_b, 1e-6) if r_b > 0 else float("inf")
            res[f"{tgt}:{nm}"] = {"R2_fwd": round(r_f, 4), "se": round(se, 4),
                                  "R2_bwd": round(r_b, 4), "fwd_bwd_ratio": round(min(ratio, 999), 2),
                                  "null": round(r_s, 4), "persist": round(r_p, 4)}
            print(f"{tgt:<8} {nm:<4} {r_f:>+9.4f} {r_b:>+9.4f} {min(ratio,999):>8.2f} "
                  f"{r_s:>+8.4f} {r_p:>+8.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"dims": a.dims, "delta": a.delta, "held_out": held, "probes": res},
              open(a.out, "w"), indent=1)
    print(f"\n[p7] wrote {a.out}", flush=True)
