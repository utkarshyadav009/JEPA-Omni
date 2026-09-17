"""Checkpoint decay analysis: does the mechanism effect rise then fall within ONE run?

RUN-5's full run retained 9% of the 3,000-step pilot's effect. Two explanations fit that:
  (a) TRANSIENT -> CONVERGED DECAY. The predictive gradient moves a plastic fusion early; later
      the 4.2M head satisfies the loss on its own and the AV term reasserts itself.
  (b) NOISE. The pilot and the full run differ in AV batch (24 vs 50) and are separate runs, so
      the "decay" could be run-to-run variation dressed up as a trajectory.

This distinguishes them by scoring M1-M3 at six checkpoints of the SAME run, so batch size,
seed, data order and config are all held constant and only training time varies.

  a clean rise-then-fall  -> (a) is supported, and the head-absorption story is tellable
  a noisy or flat curve   -> (a) is NOT supported, and that story must NOT be told

The pilot's step3000 (batch 24, separate run) is scored too, as a cross-check on whether the
batch difference matters at all.

RUN-4 is encoded ONCE and reused as the baseline for every point -- p8 re-encodes it per call,
which would be six redundant passes over ~150k windows.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import importlib.util
_sp = importlib.util.spec_from_file_location("p8", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p8_mechanism_eval.py")
p8 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(p8)
p6 = p8.p6


def score(ckpt, tr_v, ev_v, dev):
    Wt, VMt, AMt, ot, pt = p8.encode(ckpt, tr_v, dev)
    We, VMe, AMe, oe, pe = p8.encode(ckpt, ev_v, dev)
    itr, jtr = p8.pairs(ot, pt); iev, jev = p8.pairs(oe, pe)
    ibw, jbw = p8.pairs(oe, pe, back=True)
    f_ = p8.probe(Wt[itr], Wt[jtr]-Wt[itr], We[iev], We[jev]-We[iev], dev)
    b_ = p8.probe(Wt[itr], Wt[jtr]-Wt[itr], We[ibw], We[jbw]-We[ibw], dev)
    return {"M1_dV":     p8.probe(Wt[itr], VMt[jtr]-VMt[itr], We[iev], VMe[jev]-VMe[iev], dev),
            "M3_V_next": p8.probe(Wt[itr], VMt[jtr],          We[iev], VMe[jev],          dev),
            "M4_dA":     p8.probe(Wt[itr], AMt[jtr]-AMt[itr], We[iev], AMe[jev]-AMe[iev], dev),
            "M2_dW_fwd": f_, "M2_dW_bwd": b_, "M2_ratio": f_ / max(b_, 1e-6)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1000,3000,6000,10000,15000,20000")
    ap.add_argument("--run5-dir", default="checkpoints/RUN-5/full")
    ap.add_argument("--pilot-ckpt", default="checkpoints/RUN-5/mechanism_pilot/step3000.pt")
    ap.add_argument("--run4-ckpt", default="checkpoints/m2_run4_padfix_ta896/step18000.pt")
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p10_decay.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    allv = sorted(f[:-3] for f in os.listdir(p8.WS_DIR) if f.endswith(".pt"))
    tr_v, ev_v, held = p6.split_participants(allv)
    print(f"[p10] train {len(tr_v)} eval {len(ev_v)}; held-out {held}", flush=True)

    res = {}
    print("[p10] baseline RUN-4 (encoded once)...", flush=True)
    res["RUN-4"] = score(a.run4_ckpt, tr_v, ev_v, dev)
    base = res["RUN-4"]
    print(f"[p10] RUN-4  dV {base['M1_dV']:.4f}  V_next {base['M3_V_next']:.4f}  "
          f"ratio {base['M2_ratio']:.2f}x", flush=True)

    for st in [int(x) for x in a.steps.split(",")]:
        ck = os.path.join(a.run5_dir, f"step{st}.pt")
        if not os.path.exists(ck):
            print(f"[p10] MISSING {ck}"); continue
        r = score(ck, tr_v, ev_v, dev); res[f"step{st}"] = r
        print(f"[p10] step{st:<6} dV {r['M1_dV']:.4f} ({r['M1_dV']-base['M1_dV']:+.4f})  "
              f"V_next {r['M3_V_next']:.4f} ({r['M3_V_next']-base['M3_V_next']:+.4f})  "
              f"ratio {r['M2_ratio']:.2f}x  dA {r['M4_dA']:.4f}", flush=True)

    if os.path.exists(a.pilot_ckpt):
        r = score(a.pilot_ckpt, tr_v, ev_v, dev); res["pilot_step3000_batch24"] = r
        print(f"[p10] pilot3k(bs24) dV {r['M1_dV']:.4f} ({r['M1_dV']-base['M1_dV']:+.4f})  "
              f"ratio {r['M2_ratio']:.2f}x   <- cross-check on the batch-size difference", flush=True)

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p10] wrote {a.out}", flush=True)
