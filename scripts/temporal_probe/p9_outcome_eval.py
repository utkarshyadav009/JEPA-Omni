"""RUN-5 v2 OUTCOME evaluation (docs/RUN5_SPEC_V2.md section 6.2).

The mechanism tier asks whether the fusion RETAINS more. This tier asks whether that CONVERTS.
They are separate on purpose: FUSION_BOTTLENECK.md section 5 warns the directional signal is
small in absolute terms (R2 0.045 from vision), so retaining it need not move R@1 against a
persistence baseline that is very strong at R@1. Mechanism-passes-outcome-fails is a RESULT,
to be reported, not tuned away.

  O1  beat PERSISTENCE (IDENTITY)               >= +2.0 and >= 3x SE
  O2  forward - backward, SEPARATELY TRAINED    >= +2.0 and >= 3x SE
      backward predictor
  O3  shuffle control                           at chance (< 1.5x)
  O4  mismatched-file falsifier                 at chance (< 1.5x); different PARTICIPANT,
                                                random target index
  O5  VGGSound n=1,545 R@1 within 2.0 of        read from the training log, not recomputed here
      41.77 (v->a) / 41.88 (a->v)

O2 MUST use a separately trained backward predictor. In the v1 pilot the naive version -- the
forward model run backwards -- gave +3.082 +- 0.154, a clean PASS at 20x SE, while the correct
control gave +1.261, a FAIL. That artifact was large and highly significant, not marginal.

The retrieval protocol, gallery construction, chance calculation and controls are imported from
p5_pilot.py unchanged, so RUN-5 is scored on exactly the harness RUN-4 and the v1 pilot were.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import importlib.util

def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

p5 = _load("p5", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p5_pilot.py")
p6 = _load("p6", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p6_information_probes.py")
p8 = _load("p8", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p8_mechanism_eval.py")

WS_DIR = p8.WS_DIR
GRID_S = 10.0


def build_data(ckpt, video_ids, dev):
    """{video: (W, start_s)} in p5_pilot's expected shape, from THIS checkpoint's fusion."""
    W, VM, AM, order, pos = p8.encode(ckpt, video_ids, dev)
    out = {}
    for k, (vid, w) in enumerate(order):
        out.setdefault(vid, [[], []])
        out[vid][0].append(W[k]); out[vid][1].append(float(w))
    return {v: (torch.stack(a).half(), torch.tensor(b)) for v, (a, b) in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="RUN-5")
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p9_outcome.json")
    a = ap.parse_args()
    dev = torch.device("cuda")

    allv = sorted(f[:-3] for f in os.listdir(WS_DIR) if f.endswith(".pt"))
    tr_v, ev_v, held = p6.split_participants(allv)
    print(f"[p9] {a.label}: train {len(tr_v)} eval {len(ev_v)} videos; held-out {held}", flush=True)

    print("[p9] encoding train split...", flush=True)
    data_tr = build_data(a.ckpt, tr_v, dev)
    print("[p9] encoding eval split...", flush=True)
    data_ev = build_data(a.ckpt, ev_v, dev)
    data = {**data_tr, **data_ev}
    tr_ids = sorted(data_tr); ev_ids = sorted(data_ev)

    pairs = p5.build_pairs(data, tr_ids, a.delta, stride=2.0)
    bwd_pairs = [(vi, j, i) for (vi, i, j) in pairs]
    print(f"[p9] training pairs {len(pairs):,}", flush=True)

    res = {"label": a.label, "ckpt": a.ckpt, "held_out": held, "arms": {}}
    idf = lambda x: x
    A = res["arms"]
    A["falsifier_identity"] = p5.evaluate(data, ev_ids, a.delta, idf, dev, mismatch=True, seed=0)
    A["IDENTITY_forward"]  = p5.evaluate(data, ev_ids, a.delta, idf, dev, seed=0)
    A["IDENTITY_backward"] = p5.evaluate(data, ev_ids, a.delta, idf, dev, backward=True, seed=0)
    print(f"[p9] IDENTITY (persistence) fwd {A['IDENTITY_forward']['micro_R@1']}  "
          f"bwd {A['IDENTITY_backward']['micro_R@1']}  "
          f"gallery {A['IDENTITY_forward']['mean_gallery']} chance {A['IDENTITY_forward']['chance_R@1']}",
          flush=True)

    fs, bs_ = [], []
    for sd in [int(x) for x in a.seeds.split(",")]:
        m = p5.train_predictor(data, tr_ids, pairs, dev, a.steps, 256, 3e-4, 0.05, sd)
        m.eval(); fn = lambda x: m(x)
        mb = p5.train_predictor(data, tr_ids, bwd_pairs, dev, a.steps, 256, 3e-4, 0.05, sd)
        mb.eval(); fnb = lambda x: mb(x)
        with torch.no_grad():
            A[f"fwd_s{sd}"]  = p5.evaluate(data, ev_ids, a.delta, fn,  dev, seed=sd)
            A[f"bwd_s{sd}"]  = p5.evaluate(data, ev_ids, a.delta, fnb, dev, backward=True, seed=sd)
            A[f"shuf_s{sd}"] = p5.evaluate(data, ev_ids, a.delta, fn,  dev, shuffle=True, seed=sd)
            A[f"fals_s{sd}"] = p5.evaluate(data, ev_ids, a.delta, fn,  dev, mismatch=True, seed=sd)
            A[f"collapse_s{sd}"] = p5.collapse_metrics(data, ev_ids, dev, fn)
        fs.append(A[f"fwd_s{sd}"]["micro_R@1"]); bs_.append(A[f"bwd_s{sd}"]["micro_R@1"])
        print(f"[p9] seed {sd}: fwd {fs[-1]}  bwd(sep-trained) {bs_[-1]}  "
              f"shuf {A[f'shuf_s{sd}']['micro_R@1']}  fals {A[f'fals_s{sd}']['micro_R@1']}", flush=True)

    import statistics as st
    se = lambda v: st.stdev(v)/len(v)**0.5 if len(v) > 1 else 0.0
    fm, fse = st.mean(fs), se(fs)
    gaps = [x-y for x, y in zip(fs, bs_)]; gm, gse = st.mean(gaps), se(gaps)
    idf1 = A["IDENTITY_forward"]["micro_R@1"]
    ch = A["IDENTITY_forward"]["chance_R@1"]; chf = A["falsifier_identity"]["chance_R@1"]
    shm = st.mean([A[f"shuf_s{s}"]["micro_R@1"] for s in [int(x) for x in a.seeds.split(",")]])
    fam = st.mean([A[f"fals_s{s}"]["micro_R@1"] for s in [int(x) for x in a.seeds.split(",")]])
    v = {"O1": (fm-idf1) >= 2.0 and (fm-idf1) >= 3*fse,
         "O2": gm >= 2.0 and gm >= 3*gse,
         "O3": (shm/ch) < 1.5, "O4": (fam/chf) < 1.5}
    print(f"\n[p9] O1 beat persistence  {fm:.3f} vs {idf1:.3f} = {fm-idf1:+.3f} +-{fse:.3f}  "
          f"{'PASS' if v['O1'] else 'FAIL'}")
    print(f"[p9] O2 fwd-bwd           {gm:+.3f} +-{gse:.3f}   {'PASS' if v['O2'] else 'FAIL'}")
    print(f"[p9] O3 shuffle           {shm:.3f} / {ch:.3f} = {shm/ch:.2f}x  {'PASS' if v['O3'] else 'FAIL'}")
    print(f"[p9] O4 falsifier         {fam:.3f} / {chf:.3f} = {fam/chf:.2f}x  {'PASS' if v['O4'] else 'FAIL'}")
    print(f"[p9] O5 VGGSound non-regression: read from the training log (not computed here)")
    print(f"\n[p9] OUTCOME GATE (O1-O4): {'PASS' if all(v.values()) else 'FAIL'}")
    res["verdict"] = v
    res["summary"] = {"fwd": fm, "fwd_se": fse, "identity": idf1, "gap": gm, "gap_se": gse,
                      "shuffle_ratio": shm/ch, "falsifier_ratio": fam/chf}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p9] wrote {a.out}", flush=True)
