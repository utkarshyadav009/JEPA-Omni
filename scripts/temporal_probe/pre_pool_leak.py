"""scripts/temporal_probe/pre_pool_leak.py — P0.3d. READ-ONLY. Fixes nothing.

Quantifies the residual leak in the TRAINING-side path used by the query predictor
and the M3/M4 connectors.

The defect: train_query_predictor.build_sources (and train_m3.py:342 / train_m4.py:153)
correctly build a padding mask and hand it to the DOWNSTREAM module -- but the M2 call
that produces their input,

    src["m2"] = m2.encode_pre_pool_tokens(feats, tbins)

runs AVJepaPredictor._backbone with key_padding_mask=None. So the REAL tokens have
already attended to pad tokens across all 8 layers before the downstream mask is
applied. The downstream mask removes the pad POSITIONS; it cannot undo their influence
on the surviving positions.

This script measures how large that influence is, on the real eval distribution:

  A. contamination  = per-token relative L2 change between the real tokens produced
                      WITH padding present and the same tokens produced with the
                      padding masked out. This is the size of the defect.
  B. pad-slot norm  = how large the pad tokens' own post-backbone activations are
                      relative to real ones (they are not zero: after in_proj a zero
                      feature still picks up the layer bias, the modality embedding
                      and temporal_emb[0]).
  C. batch-1 control= the same tokens computed alone, where no padding can exist.
                      Distance to this is the absolute error of the batched path.

Usage:
    python scripts/temporal_probe/pre_pool_leak.py \
        --out docs/artifacts/temporal_probe/p03d_pre_pool_leak.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from data.av_cached_dataset import AVCachedDataset, av_collate_fn
from scripts.temporal_probe.padded_eval import flat_padding_mask

EXPECTED_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--cache-dir", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k")
    ap.add_argument("--eval-subset", default="data/vggsound_eval_1545.txt")
    ap.add_argument("--batch-size", type=int, default=64, help="the QP trains at 64 micro-batch")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sha = sha256_file(args.ckpt)
    if sha != EXPECTED_SHA:
        raise SystemExit(f"ABORT: ckpt sha256 {sha} != {EXPECTED_SHA}")

    device = torch.device("cuda")
    m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                       max_tdm_bins=512, dropout=0.0)).to(device)
    m2.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    m2.eval()

    ids = [l.strip() for l in open(args.eval_subset) if l.strip()]
    ds = AVCachedDataset(cache_dir=args.cache_dir, clip_ids=ids, max_tdm_bins=512, audio_mode="mean")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=6, collate_fn=av_collate_fn)

    tot = {"contam_rel_l2": [], "contam_cos": [], "pad_norm_ratio": [],
           "bs1_rel_l2": [], "pad_frac": []}
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        flat = flat_padding_mask(pad, feats)

        H_leak = m2.encode_pre_pool_tokens(feats, tbins).float()                       # as used in training
        H_mask = m2.encode_pre_pool_tokens(feats, tbins, key_padding_mask=flat).float()  # corrected

        real = ~flat
        a, b = H_leak[real], H_mask[real]
        tot["contam_rel_l2"].append(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean().item())
        tot["contam_cos"].append(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())
        if flat.any():
            tot["pad_norm_ratio"].append(
                (H_leak[flat].norm(dim=-1).mean() / H_leak[real].norm(dim=-1).mean()).item())
        tot["pad_frac"].append(flat.float().mean().item())

        # batch-1 control on the first 4 clips of this batch: no padding can exist
        if bi == 0:
            errs = []
            for i in range(min(4, feats["vision"].shape[0])):
                na = int((~pad["ambient"][i]).sum())
                f1 = {"vision": feats["vision"][i:i+1], "ambient": feats["ambient"][i:i+1, :na]}
                t1 = {"vision": tbins["vision"][i:i+1], "ambient": tbins["ambient"][i:i+1, :na]}
                h1 = m2.encode_pre_pool_tokens(f1, t1).float()[0]
                nv = feats["vision"].shape[1]
                hb = torch.cat([H_leak[i, :nv], H_leak[i, nv:nv+na]], 0)
                errs.append(((hb - h1).norm(dim=-1) / h1.norm(dim=-1).clamp_min(1e-6)).mean().item())
            tot["bs1_rel_l2"] = errs

    res = {k: (round(sum(v) / len(v), 6) if v else None) for k, v in tot.items() if k != "bs1_rel_l2"}
    res["bs1_rel_l2_mean"] = round(sum(tot["bs1_rel_l2"]) / len(tot["bs1_rel_l2"]), 6)
    res["bs1_rel_l2_per_clip"] = [round(x, 6) for x in tot["bs1_rel_l2"]]
    res.update({"batch_size": args.batch_size, "n_batches": args.n_batches,
                "gallery": args.eval_subset, "gallery_size": len(ds)})
    print(json.dumps(res, indent=2), flush=True)

    json.dump({"script": "scripts/temporal_probe/pre_pool_leak.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "ckpt": args.ckpt, "ckpt_sha256": sha, "result": res},
              open(args.out, "w"), indent=2)
    print(f"[pre_pool_leak] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
