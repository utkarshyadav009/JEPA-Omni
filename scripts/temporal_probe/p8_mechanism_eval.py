"""RUN-5 v2 MECHANISM evaluation (docs/RUN5_SPEC_V2.md section 6.1).

Did the predictive-fusion objective change what the fusion RETAINS?

  M1  W -> dV      forward R2          required: +0.05 over RUN-4
  M2  W -> dW      forward/backward    required: >= 2.00x (RUN-4 1.14x)
  M3  W -> V_next  forward R2          required: +0.05 over RUN-4
  M4  the dA gain must NOT exceed the dV gain, else it is not the visual mechanism claimed

RUN-4'S BASELINE IS RE-MEASURED HERE, NOT QUOTED. The spec's numbers came from the 1 s-stride
world-state files; RUN-5's W must be recomputed from TOKENS, and the token cache is 2 s-stride.
Scoring RUN-5 on one grid against a RUN-4 number from another is exactly the mismatch that
produced the 41.68/41.35-vs-41.77/41.88 confusion (CANONICAL_NUMBERS 1.2). Both checkpoints go
through the identical path; thresholds are therefore GAINS over the re-measured row.

Probe, split, capacity matching and metrics are unchanged from p7_fusion_diagnosis.py.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
import torch
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
import importlib.util
_sp = importlib.util.spec_from_file_location("p6", "/home/utkarsh/JEPA-Omni/scripts/temporal_probe/p6_information_probes.py")
p6 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(p6)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import _cap_ambient_len, flat_pad

WS_DIR = "/home/utkarsh/JEPA-Omni/data/epic_kitchens_ws"
FEAT_DIRS = ["/home/utkarsh/JEPA-Omni/data/feature_cache_epic_kitchens",
             "/mnt/Raid-Storage-2/utkarsh-data/feature_cache_epic_kitchens"]
STRIDE_S, DELTA_S = 2.0, 10.0
OFFSET = int(DELTA_S / STRIDE_S)          # 5 windows


class _WinDS(torch.utils.data.Dataset):
    """Every window of the given videos, in (video, window) order -- not just paired ones,
    because dW needs W at BOTH ends of the pair under the SAME checkpoint."""
    def __init__(self, paths, max_tdm_bins=512, audio_mode="mean"):
        self.paths = paths
        self._tx = AVCachedDataset.__new__(AVCachedDataset)
        self._tx.max_tdm_bins = max_tdm_bins; self._tx.audio_mode = audio_mode
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        d = torch.load(self.paths[i], map_location="cpu", weights_only=True)
        return self._tx.transform(d, os.path.basename(self.paths[i])[:-3])


def index_windows(video_ids):
    by = defaultdict(dict)
    for root in FEAT_DIRS:
        if not os.path.isdir(root): continue
        for shard in os.listdir(root):
            sd = os.path.join(root, shard)
            if not os.path.isdir(sd): continue
            for fn in os.listdir(sd):
                if not (fn.startswith("ek_") and fn.endswith(".pt") and "_w" in fn): continue
                vid, w = fn[3:-3].rsplit("_w", 1)
                if vid in video_ids: by[vid][int(w)] = os.path.join(sd, fn)
    return by


@torch.no_grad()
def encode(ckpt, video_ids, dev, cap_t=896, bs=24):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = AVJepaPredictor(AVJepaConfig()).to(dev); m.load_state_dict(ck["model"]); m.eval()
    by = index_windows(set(video_ids))
    order, paths = [], []
    for vid in sorted(by):
        for w in sorted(by[vid]):
            order.append((vid, w)); paths.append(by[vid][w])
    dl = torch.utils.data.DataLoader(_WinDS(paths), batch_size=bs, shuffle=False,
                                     num_workers=6, collate_fn=av_collate_fn)
    W, VM, AM = [], [], []
    for b in dl:
        _cap_ambient_len(b["feats"], b["tbins"], b["padding_mask"], max_t=cap_t)
        f = {k: v.to(dev) for k, v in b["feats"].items()}
        t = {k: v.to(dev) for k, v in b["tbins"].items()}
        pd = {k: v.to(dev) for k, v in b["padding_mask"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ws = m.encode_world_state(f, t, key_padding_mask=flat_pad(pd, f))
        W.append(ws.float().cpu()); VM.append(f["vision"].float().mean(1).cpu())
        AM.append(f["ambient"].float().mean(1).cpu())
    del m; torch.cuda.empty_cache()
    pos = {k: i for i, k in enumerate(order)}
    return torch.cat(W), torch.cat(VM), torch.cat(AM), order, pos


def pairs(order, pos, back=False):
    i_idx, j_idx = [], []
    for (vid, w) in order:
        k = pos.get((vid, w - OFFSET * int(STRIDE_S)) if back else (vid, w + OFFSET * int(STRIDE_S)))
        if k is not None:
            i_idx.append(pos[(vid, w)]); j_idx.append(k)
    return torch.tensor(i_idx), torch.tensor(j_idx)


def probe(Xtr, Ytr, Xev, Yev, dev):
    Wm, mx, my = p6.ridge_fit(Xtr, Ytr, dev)
    tmean = Ytr.mean(0, keepdim=True).to(dev)
    return p6.r2((Xev.to(dev) - mx) @ Wm + my, Yev.to(dev), tmean)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run5-ckpt", required=True)
    ap.add_argument("--run4-ckpt", default="checkpoints/m2_run4_padfix_ta896/step18000.pt")
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p8_mechanism.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    allv = sorted(f[:-3] for f in os.listdir(WS_DIR) if f.endswith(".pt"))
    tr_v, ev_v, held = p6.split_participants(allv)
    print(f"[p8] train {len(tr_v)} eval {len(ev_v)} videos; held-out {held}", flush=True)

    res = {}
    for name, ck in (("RUN-4", a.run4_ckpt), ("RUN-5", a.run5_ckpt)):
        print(f"[p8] encoding {name}: {ck}", flush=True)
        Wt, VMt, AMt, ot, pt = encode(ck, tr_v, dev)
        We, VMe, AMe, oe, pe = encode(ck, ev_v, dev)
        itr, jtr = pairs(ot, pt); iev, jev = pairs(oe, pe)
        ibw, jbw = pairs(oe, pe, back=True)
        r = {}
        r["M1_dV"] = probe(Wt[itr], VMt[jtr] - VMt[itr], We[iev], VMe[jev] - VMe[iev], dev)
        r["M3_V_next"] = probe(Wt[itr], VMt[jtr], We[iev], VMe[jev], dev)
        r["M4_dA"] = probe(Wt[itr], AMt[jtr] - AMt[itr], We[iev], AMe[jev] - AMe[iev], dev)
        f_ = probe(Wt[itr], Wt[jtr] - Wt[itr], We[iev], We[jev] - We[iev], dev)
        b_ = probe(Wt[itr], Wt[jtr] - Wt[itr], We[ibw], We[jbw] - We[ibw], dev)
        r["M2_dW_fwd"], r["M2_dW_bwd"] = f_, b_
        r["M2_ratio"] = f_ / max(b_, 1e-6)
        r["n_train_pairs"], r["n_eval_pairs"] = int(len(itr)), int(len(iev))
        res[name] = r
        print(f"[p8] {name}: dV {r['M1_dV']:.4f}  V_next {r['M3_V_next']:.4f}  dA {r['M4_dA']:.4f}  "
              f"dW fwd {f_:.4f} bwd {b_:.4f} ratio {r['M2_ratio']:.2f}x", flush=True)

    R4, R5 = res["RUN-4"], res["RUN-5"]
    g1, g3, g4 = R5["M1_dV"]-R4["M1_dV"], R5["M3_V_next"]-R4["M3_V_next"], R5["M4_dA"]-R4["M4_dA"]
    v = {"M1": (g1 >= 0.05), "M2": (R5["M2_ratio"] >= 2.00),
         "M3": (g3 >= 0.05), "M4": (g4 <= g1)}
    print(f"\n[p8] M1 dV gain      {g1:+.4f}  (>= +0.050)   {'PASS' if v['M1'] else 'FAIL'}")
    print(f"[p8] M2 dW ratio     {R5['M2_ratio']:.2f}x   (>= 2.00x)    {'PASS' if v['M2'] else 'FAIL'}")
    print(f"[p8] M3 V_next gain  {g3:+.4f}  (>= +0.050)   {'PASS' if v['M3'] else 'FAIL'}")
    print(f"[p8] M4 dA gain      {g4:+.4f}  (<= dV gain {g1:+.4f})  {'PASS' if v['M4'] else 'FAIL'}")
    print(f"\n[p8] MECHANISM GATE: {'PASS' if all(v.values()) else 'FAIL'}")
    res["gains"] = {"M1": g1, "M3": g3, "M4": g4}; res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p8] wrote {a.out}", flush=True)
