"""RUN-5 PILOT (P4.11) -- can a predictor trained on a FROZEN RUN-4 representation extract
forward information that RUN-4 does not expose?

This is the gate. If a frozen-representation predictor cannot beat PERSISTENCE, joint training
does not run. See docs/RUN5_SPEC.md section 2.1 for the pre-registered criteria, fixed before
any number here existed.

WHAT IS AND IS NOT TRAINED
  trained : P_phi only -- LayerNorm -> 1024->2048 GELU -> 2048->1024. ~4.2M params.
  frozen  : everything. W_t and W_{t+D} are precomputed by RUN-4 step18000
            (sha256 27b33c8c...). Stop-gradient is VACUOUS here and that is deliberate:
            the LeJEPA-vs-V-JEPA2 disagreement about stop-grad does not need resolving
            to run this gate (docs/PREDICTION_LITERATURE.md section 2).

WHY InfoNCE AND NOT L2
  Under L2/cosine the IDENTITY map is already near-optimal -- P3.2 measured IDENTITY beating
  ridge by 4.5 R@1 -- so a regression loss is minimised by learning to copy and would report
  success for a model that learned nothing. Negatives make copying insufficient.

EVALUATION GRID -- stated because it is a real choice
  Training uses every 1 s-strided pair. EVALUATION uses a 10 s-spaced grid within each file, so
  no two gallery entries share input. At 1 s stride the gallery would otherwise be packed with
  90%-overlapping near-duplicates of the target, which measures window overlap rather than
  prediction. The 10 s grid also matches P3.2's Ego4D protocol, so the numbers are comparable.

BASELINES ARE RECOMPUTED ON THIS CORPUS. P3.2's IDENTITY = 14.81 is an EGO4D number and is NOT
the bar; the bar is IDENTITY measured on this same Epic-Kitchens eval split.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

WS_DIR = "/home/utkarsh/JEPA-Omni/data/epic_kitchens_ws"
M2_SHA = "27b33c8cebe656f26e51987cc49a5b8bf4452845f14d9e8a41eb9d1a6c1848a4"
GRID_S = 10.0          # eval grid spacing == window length == zero input overlap


class Predictor(nn.Module):
    def __init__(self, d=1024, h=2048):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x): return self.net(x)


def load_corpus(ws_dir):
    files = sorted(f for f in os.listdir(ws_dir) if f.endswith(".pt"))
    data = {}
    for fn in files:
        d = torch.load(os.path.join(ws_dir, fn), map_location="cpu", weights_only=True)
        assert d["m2_ckpt_sha256"] == M2_SHA, f"{fn}: wrong checkpoint"
        data[fn[:-3]] = (d["world_state"], d["start_s"])
    return data


def split_participants(vids, every=5):
    ps = sorted({v.split("_")[0] for v in vids})
    hold = set(ps[::every])
    tr = sorted(v for v in vids if v.split("_")[0] not in hold)
    ev = sorted(v for v in vids if v.split("_")[0] in hold)
    return tr, ev, sorted(hold)


def build_pairs(data, vids, delta, stride=1.0):
    """(video_idx, i, j) with start_s[j] == start_s[i] + delta. Training uses the full 1s grid."""
    out = []
    for vi, v in enumerate(vids):
        st = data[v][1]
        pos = {round(float(x), 3): k for k, x in enumerate(st.tolist())}
        for k, x in enumerate(st.tolist()):
            j = pos.get(round(float(x) + delta, 3))
            if j is not None: out.append((vi, k, j))
    return out


def grid_indices(st, grid=GRID_S):
    """indices of windows on a `grid`-spaced lattice -> gallery entries share no input."""
    want, out, s = 0.0, [], st.tolist()
    pos = {round(float(x), 3): k for k, x in enumerate(s)}
    while want <= s[-1] + 1e-6:
        k = pos.get(round(want, 3))
        if k is not None: out.append(k)
        want += grid
    return out


@torch.no_grad()
def evaluate(data, vids, delta, fn_pred, dev, backward=False, shuffle=False, mismatch=False,
             seed=0, min_gallery=8):
    """Within-file retrieval. Returns macro/micro R@1, R@5, mean gallery size, chance."""
    g = torch.Generator().manual_seed(seed)
    per_file, hits1, hits5, tot, galls = [], 0, 0, 0, []
    # CHANCE must be the QUERY-WEIGHTED mean of 1/G, not 1/mean(G). Gallery sizes are very
    # skewed here (files run 1 to 370 grid windows), and by Jensen E[1/G] >> 1/E[G] -- the
    # first version of this reported a falsifier at "2x chance" that was in fact exactly at
    # chance, because the denominator was wrong.
    chance_num, chance5_num = 0.0, 0.0
    for v in vids:
        W, st = data[v]
        gi = grid_indices(st)
        if len(gi) < min_gallery + 1: continue
        Wg = W[gi].float().to(dev)                       # (G,1024) gallery, non-overlapping
        if shuffle:                                       # destroy temporal order, keep the set
            Wg = Wg[torch.randperm(Wg.shape[0], generator=g)]
        G = Wg.shape[0]
        step = int(round(delta / GRID_S))
        if backward: qidx = list(range(step, G)); tidx = [i - step for i in qidx]
        else:        qidx = list(range(G - step)); tidx = [i + step for i in qidx]
        if not qidx: continue
        Q = Wg[qidx]
        P = fn_pred(Q)                                    # (n,1024) prediction
        if mismatch:
            # FALSIFIER. Two properties this MUST have, both learned the hard way:
            #  (a) the partner file must come from a DIFFERENT PARTICIPANT. Videos are named
            #      P<kitchen>_<n> and sorted, so "the next file" is the same kitchen 95% of the
            #      time -- the first version of this scored 4.78 against 1.78 chance purely
            #      because the "mismatched" file was the same kitchen.
            #  (b) the target index must be RANDOM, not the query's own position carried over.
            #      Position within a recording correlates with content, and clamping indices
            #      into a shorter file piles targets onto the last window.
            cands = [o for o in vids if o.split("_")[0] != v.split("_")[0]]
            if not cands: continue
            other = cands[int(torch.randint(len(cands), (1,), generator=g).item())]
            Wo, sto = data[other]; go = grid_indices(sto)
            if len(go) < min_gallery + 1: continue
            Wg = Wo[go].float().to(dev); G = Wg.shape[0]
            tidx = torch.randint(G, (len(qidx),), generator=g).tolist()
        Pn, Gn = F.normalize(P, dim=1), F.normalize(Wg, dim=1)
        sim = Pn @ Gn.T                                   # (n,G)
        for r, q in enumerate(qidx):                      # exclude the query's OWN window
            if not mismatch: sim[r, q] = -1e4
        rk = (-sim).argsort(1)
        gt = torch.tensor(tidx, device=dev)
        h1 = (rk[:, :1] == gt[:, None]).any(1).float()
        h5 = (rk[:, :5] == gt[:, None]).any(1).float()
        per_file.append(h1.mean().item() * 100)
        hits1 += h1.sum().item(); hits5 += h5.sum().item(); tot += len(qidx); galls.append(G)
        chance_num += len(qidx) * (1.0 / G); chance5_num += len(qidx) * (min(5, G) / G)
    if tot == 0: return None
    return {"macro_R@1": round(sum(per_file)/len(per_file), 3),
            "micro_R@1": round(100*hits1/tot, 3), "micro_R@5": round(100*hits5/tot, 3),
            "n_files": len(per_file), "n_queries": tot,
            "mean_gallery": round(sum(galls)/len(galls), 1),
            "chance_R@1": round(100 * chance_num / tot, 3),
            "chance_R@5": round(100 * chance5_num / tot, 3)}


def fit_ridge(data, vids, pairs, dev, lam=1e2, cap=200000):
    X, Y = [], []
    for (vi, i, j) in pairs[:cap]:
        W = data[vids[vi]][0]; X.append(W[i]); Y.append(W[j])
    X = torch.stack(X).float().to(dev); Y = torch.stack(Y).float().to(dev)
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=dev)
    return torch.linalg.solve(A, X.T @ Y)                 # (d,d)


def collapse_metrics(data, vids, dev, fn=None):
    Ws = []
    for v in vids[:80]:
        W, st = data[v]; gi = grid_indices(st)
        if gi: Ws.append(W[gi].float())
    W = torch.cat(Ws).to(dev)
    if fn is not None: W = fn(W)
    sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
    from models.av_jepa_predictor import effective_rank
    eff = effective_rank(W)      # participation ratio -- the SAME definition as RUN-4's 73.53
    Wn = F.normalize(W, dim=1)
    idx = torch.randperm(Wn.shape[0], device=dev)[:2000]
    cross = (Wn[idx] @ Wn[idx].T)
    cross = cross[~torch.eye(cross.shape[0], dtype=torch.bool, device=dev)].mean().item()
    return {"eff_rank": round(eff, 2), "mean_cross_cos": round(cross, 4),
            "min_dim_std": round(float(W.std(0).min().item()), 6)}


def train_predictor(data, vids, pairs, dev, steps, bs, lr, temp, seed, log_every=500):
    """InfoNCE: P_phi(W_t) must retrieve stopgrad(W_{t+D}) among OTHER VIDEOS' futures.

    Negatives are drawn from different videos by construction -- one pair per video per batch.
    A same-video negative at a nearby offset is a near-duplicate of the positive (persistence
    cosine 0.78 at 10 s) and would make the loss dominated by an impossible discrimination.
    """
    torch.manual_seed(seed)
    model = Predictor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    by_vid = {}
    for k, (vi, i, j) in enumerate(pairs): by_vid.setdefault(vi, []).append(k)
    vid_list = sorted(by_vid)
    g = torch.Generator().manual_seed(seed)
    t0, losses = time.time(), []
    for step in range(steps):
        take = min(bs, len(vid_list))
        vs = [vid_list[x] for x in torch.randperm(len(vid_list), generator=g)[:take].tolist()]
        idx = [by_vid[v][torch.randint(len(by_vid[v]), (1,), generator=g).item()] for v in vs]
        A = torch.stack([data[vids[pairs[k][0]]][0][pairs[k][1]] for k in idx]).float().to(dev)
        B = torch.stack([data[vids[pairs[k][0]]][0][pairs[k][2]] for k in idx]).float().to(dev)
        P = model(A)
        logits = F.normalize(P, dim=1) @ F.normalize(B, dim=1).T / temp   # B is already detached
        loss = F.cross_entropy(logits, torch.arange(logits.shape[0], device=dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            acc = (logits.argmax(1) == torch.arange(logits.shape[0], device=dev)).float().mean()
            print(f"[pilot] step {step+1}/{steps} loss {sum(losses[-log_every:])/log_every:.4f} "
                  f"in-batch acc {100*acc:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="participant", choices=["participant", "video"])
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p5_pilot.json")
    a = ap.parse_args()
    dev = torch.device("cuda")

    print("[pilot] loading corpus...", flush=True)
    data = load_corpus(WS_DIR)
    vids = sorted(data)
    if a.split == "participant":
        tr_v, ev_v, held = split_participants(vids)
    else:
        held = None; tr_v = [v for k, v in enumerate(vids) if k % 5]; ev_v = vids[::5]
    print(f"[pilot] split={a.split}  train {len(tr_v)} videos  eval {len(ev_v)} videos"
          + (f"  held-out kitchens {held}" if held else ""), flush=True)

    pairs = build_pairs(data, tr_v, a.delta)
    print(f"[pilot] training pairs at D={a.delta}s: {len(pairs):,}", flush=True)

    res = {"config": vars(a), "held_out": held, "n_train_videos": len(tr_v),
           "n_eval_videos": len(ev_v), "n_train_pairs": len(pairs), "arms": {}}

    # ---- falsifier FIRST: if mismatched pairs are not at chance nothing below is readable ----
    idf = lambda x: x
    f = evaluate(data, ev_v, a.delta, idf, dev, mismatch=True, seed=0)
    res["arms"]["falsifier_identity"] = f
    print(f"[pilot] FALSIFIER (mismatched files): R@1 {f['micro_R@1']}  chance {f['chance_R@1']}", flush=True)

    # ---- baselines on THIS corpus (P3.2's 14.81 was Ego4D and is NOT the bar) ----
    for name, fwd in (("IDENTITY", False),):
        res["arms"]["IDENTITY_forward"] = evaluate(data, ev_v, a.delta, idf, dev, seed=0)
        res["arms"]["IDENTITY_backward"] = evaluate(data, ev_v, a.delta, idf, dev, backward=True, seed=0)
    Wr = fit_ridge(data, tr_v, pairs, dev)
    ridge = lambda x: x @ Wr
    res["arms"]["ridge_forward"] = evaluate(data, ev_v, a.delta, ridge, dev, seed=0)
    res["arms"]["ridge_backward"] = evaluate(data, ev_v, a.delta, ridge, dev, backward=True, seed=0)
    print(f"[pilot] IDENTITY fwd micro R@1 {res['arms']['IDENTITY_forward']['micro_R@1']}  "
          f"bwd {res['arms']['IDENTITY_backward']['micro_R@1']}  "
          f"(gallery {res['arms']['IDENTITY_forward']['mean_gallery']}, "
          f"chance {res['arms']['IDENTITY_forward']['chance_R@1']})", flush=True)
    print(f"[pilot] ridge    fwd micro R@1 {res['arms']['ridge_forward']['micro_R@1']}  "
          f"bwd {res['arms']['ridge_backward']['micro_R@1']}", flush=True)

    # ---- the pilot predictor, over seeds ----
    # TWO predictors per seed. A forward-trained predictor evaluated backward is NOT the
    # backward control: it would be worse backward by construction, simply because it was
    # trained forward, and the gap would measure our training direction rather than the data.
    # The spec's "identical pipeline" means a SEPARATELY TRAINED backward predictor, so the
    # headline gap compares forward-trained-on-forward against backward-trained-on-backward
    # -- i.e. "is the future more predictable than the past?", which is the actual question.
    bwd_pairs = [(vi, j, i) for (vi, i, j) in pairs]      # same pairs, roles swapped
    for sd in [int(x) for x in a.seeds.split(",")]:
        m = train_predictor(data, tr_v, pairs, dev, a.steps, a.bs, a.lr, a.temp, sd)
        m.eval(); fn = lambda x: m(x)
        mb = train_predictor(data, tr_v, bwd_pairs, dev, a.steps, a.bs, a.lr, a.temp, sd)
        mb.eval(); fnb = lambda x: mb(x)
        with torch.no_grad():
            res["arms"][f"pilot_forward_s{sd}"] = evaluate(data, ev_v, a.delta, fn, dev, seed=sd)
            res["arms"][f"pilot_bwdmodel_backward_s{sd}"] = evaluate(data, ev_v, a.delta, fnb, dev, backward=True, seed=sd)
            res["arms"][f"pilot_fwdmodel_backward_s{sd}"] = evaluate(data, ev_v, a.delta, fn, dev, backward=True, seed=sd)
            res["arms"][f"pilot_shuffled_s{sd}"] = evaluate(data, ev_v, a.delta, fn, dev, shuffle=True, seed=sd)
            res["arms"][f"pilot_falsifier_s{sd}"] = evaluate(data, ev_v, a.delta, fn, dev, mismatch=True, seed=sd)
            res["arms"][f"collapse_s{sd}"] = collapse_metrics(data, ev_v, dev, fn)
        f_ = res["arms"][f"pilot_forward_s{sd}"]["micro_R@1"]
        b_ = res["arms"][f"pilot_bwdmodel_backward_s{sd}"]["micro_R@1"]
        print(f"[pilot] seed {sd}: FWD-model-fwd {f_}  BWD-model-bwd {b_}  GAP {f_-b_:+.3f}  "
              f"(fwd-model-bwd {res['arms'][f'pilot_fwdmodel_backward_s{sd}']['micro_R@1']}, "
              f"not the control)  shuf {res['arms'][f'pilot_shuffled_s{sd}']['micro_R@1']}  "
              f"fals {res['arms'][f'pilot_falsifier_s{sd}']['micro_R@1']}  "
              f"collapse {res['arms'][f'collapse_s{sd}']}", flush=True)
        torch.save(m.state_dict(), f"checkpoints/p5_pilot_d{int(a.delta)}_s{sd}.pt")
        torch.save(mb.state_dict(), f"checkpoints/p5_pilot_bwd_d{int(a.delta)}_s{sd}.pt")
    res["arms"]["collapse_baseline"] = collapse_metrics(data, ev_v, dev, None)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[pilot] wrote {a.out}", flush=True)
