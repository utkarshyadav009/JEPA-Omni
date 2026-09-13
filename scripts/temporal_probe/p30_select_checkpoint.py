"""P3.0 — post-hoc checkpoint selection for RUN-4 by held-out R@1 (E-14 option (a)).

`best.pt` is selected on training loss_ema and is IGNORED here; commit 0eb3337 never fixed
that (ERRATA E-14). Every tagged stepN000.pt is scored on the held-out gallery at the
model's own training length (T_a=896) through the corrected harness.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor, effective_rank
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from train_m2 import pool_and_project, _cap_ambient_len

def score(ck_path, dev, cap_t, seed):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    m = AVJepaPredictor(AVJepaConfig()).to(dev); m.load_state_dict(ck["model"]); m.eval()
    vp = nn.Linear(1024, 256).to(dev); vp.load_state_dict(ck["vision_proj"]); vp.eval()
    ap = nn.Linear(1024, 256).to(dev); ap.load_state_dict(ck["ambient_proj"]); ap.eval()
    ids = [l.strip() for l in open("data/vggsound_eval_1545.txt") if l.strip()]
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        ids = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
    ds = AVCachedDataset("/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k", ids, 512, "mean")
    dl = torch.utils.data.DataLoader(ds, batch_size=48, shuffle=False, num_workers=6,
                                     collate_fn=av_collate_fn)
    ZV, ZA, WS = [], [], []
    with torch.no_grad():
        for b in dl:
            _cap_ambient_len(b["feats"], b["tbins"], b["padding_mask"], max_t=cap_t)
            f = {k: v.to(dev) for k, v in b["feats"].items()}
            t = {k: v.to(dev) for k, v in b["tbins"].items()}
            pd = {k: v.to(dev) for k, v in b["padding_mask"].items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zv, za = pool_and_project(m, vp, ap, f, t, pad=pd)
                ws = m.encode_world_state(f, t, key_padding_mask=torch.cat([pd["vision"], pd["ambient"]], 1))
            ZV.append(zv.float().cpu()); ZA.append(za.float().cpu()); WS.append(ws.float().cpu())
    zv = torch.cat(ZV); za = torch.cat(ZA); W = torch.cat(WS)
    N = zv.shape[0]; gt = torch.arange(N); sim = zv @ za.T
    r = lambda rk, k: round((rk[:, :k] == gt[:, None]).any(1).float().mean().item() * 100, 2)
    del m, vp, ap; torch.cuda.empty_cache()
    return {"v2a_R@1": r((-sim).argsort(1), 1), "a2v_R@1": r((-sim.T).argsort(1), 1),
            "v2a_R@5": r((-sim).argsort(1), 5), "a2v_R@5": r((-sim.T).argsort(1), 5),
            "matched_cos": round(sim.diagonal().mean().item(), 4),
            "eff_rank": round(effective_rank(W), 2), "n": N}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=896); ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for cp in a.ckpts.split(","):
        step = int(os.path.basename(cp).replace("step", "").replace(".pt", ""))
        runs = [score(cp, dev, a.cap, s) for s in [int(x) for x in a.seeds.split(",")]]
        agg = {k: round(sum(r[k] for r in runs) / len(runs), 2) for k in runs[0] if k != "n"}
        agg["range_v2a_R@1"] = round(max(r["v2a_R@1"] for r in runs) - min(r["v2a_R@1"] for r in runs), 2)
        res[step] = agg
        print(f"[p30] step {step:>5}  v→a {agg['v2a_R@1']:>6.2f}  a→v {agg['a2v_R@1']:>6.2f}  "
              f"eff_rank {agg['eff_rank']:>6.2f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
