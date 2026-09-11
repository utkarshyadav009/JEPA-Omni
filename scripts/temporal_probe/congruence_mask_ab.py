"""scripts/temporal_probe/congruence_mask_ab.py — P0.4 item 3.

The ears-following result (0.650 -> 0.070, checkpoints/AV_CONGRUENCE_EVAL.json)
CANNOT be re-run at batch size 1: scripts/eval_av_congruence.py swaps audio WITHIN
a batch (`perm = torch.roll(torch.arange(B), 1)`) and skips any batch with B < 2.
Batch size is therefore not a free variable in that harness -- changing it changes
which clip's audio is swapped in as well as the padding, confounding the test.

So this runs the correct single-variable A/B instead. The ONLY unmasked thing in
the query-predictor path is the M2 backbone call inside build_sources:

    src["m2"] = m2.encode_pre_pool_tokens(feats, tbins)      # key_padding_mask=None

Everything downstream (QueryPredictor's cross-attention) already receives and applies
the mask. This script monkeypatches build_sources to pass the mask into that one call
and re-runs the identical harness on the identical batches with the identical swap
pairing. Nothing on disk is modified; train_query_predictor.py and eval_av_congruence.py
are imported, not edited.

Usage:
    python scripts/temporal_probe/congruence_mask_ab.py \
        --arms abl_A_m2_vision_ambient:mean --n-clips 320 \
        --out docs/artifacts/temporal_probe/p04_congruence_mask_ab.json
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import train_m3
train_m3.CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")

import train_query_predictor as TQP
import scripts.eval_av_congruence as EAC
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.text_target import build_text_target

_ORIG_BUILD_SOURCES = TQP.build_sources


def build_sources_masked(batch, m2, device, names):
    """Identical to train_query_predictor.build_sources EXCEPT that the m2 stream's
    backbone pass receives the padding mask. Token order inside the m2 stream is
    [vision; ambient] (AVJepaPredictor._embed concatenates over feats.items()), which
    is exactly how build_sources already builds msk["m2"] -- so the same concatenation
    is reused rather than re-derived."""
    feats = {k: v.to(device, non_blocking=True).float() for k, v in batch["feats"].items()}
    tbins = {k: v.to(device, non_blocking=True) for k, v in batch["tbins"].items()}
    pads = {k: v.to(device, non_blocking=True) for k, v in batch["pad"].items()}
    src, msk = {}, {}
    if "m2" in names:
        flat = torch.cat([pads["vision"], pads["ambient"]], 1)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            src["m2"] = m2.encode_pre_pool_tokens(feats, tbins, key_padding_mask=flat).float()
        msk["m2"] = flat
    if "vision" in names:
        src["vision"] = feats["vision"]; msk["vision"] = pads["vision"]
    if "ambient" in names:
        src["ambient"] = feats["ambient"]; msk["ambient"] = pads["ambient"]
    if "scene" in names:
        sc = batch.get("scene")
        if sc is None:
            raise KeyError("'scene' stream requested but this batch has no SigLIP2 features")
        src["scene"] = sc.to(device, non_blocking=True).float()
    return src, msk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--cache-dir", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k")
    ap.add_argument("--scene-file", default="")
    ap.add_argument("--arms", default="abl_A_m2_vision_ambient:mean")
    ap.add_argument("--n-clips", type=int, default=320)
    ap.add_argument("--text-backbone", default="embeddinggemma")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    EAC.CACHE_DIR = args.cache_dir
    TQP.CACHE_DIR = args.cache_dir
    device = torch.device("cuda")

    m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                       max_tdm_bins=512, dropout=0.0)).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device,
                                   weights_only=False)["model"], strict=True)
    m2.eval()
    tt = build_text_target(args.text_backbone, shared_dim=1536, device=str(device))

    _, te_pairs = TQP.build_splits(EAC.VGGSOUND_FIELDS)
    te = TQP.group_by_clip(te_pairs, EAC.VGGSOUND_FIELDS)
    scene = (torch.load(args.scene_file, map_location="cpu", weights_only=False)
             if args.scene_file else {})
    if scene:
        te = {k: v for k, v in te.items() if k in scene}
    ids = sorted(te.keys())
    print(f"[cong_ab] {len(ids)} eval clips, using first {args.n_clips}", flush=True)

    runs = {}
    for variant, fn in (("leaked_m2_unmasked", _ORIG_BUILD_SOURCES),
                        ("fixed_m2_masked", build_sources_masked)):
        TQP.build_sources = fn
        EAC.build_sources = fn
        for spec in [a for a in args.arms.split(",") if a.strip()]:
            name, amode = spec.split(":")
            t0 = time.time()
            r = EAC.eval_arm(f"checkpoints/{name}/best.pt", m2, tt, te, ids, scene,
                             device, args.n_clips, amode)
            r["wall_s"] = round(time.time() - t0, 1)
            runs.setdefault(name, {})[variant] = r
            print(f"[cong_ab] {name:<28} {variant:<20} {json.dumps(r)}", flush=True)
    TQP.build_sources = _ORIG_BUILD_SOURCES
    EAC.build_sources = _ORIG_BUILD_SOURCES

    for name, v in runs.items():
        if len(v) == 2:
            a, b = v["leaked_m2_unmasked"], v["fixed_m2_masked"]
            d = {k: round(b[k] - a[k], 4) for k in a
                 if isinstance(a.get(k), (int, float)) and k != "wall_s"}
            v["delta_fixed_minus_leaked"] = d
            print(f"[cong_ab] DELTA {name}: {json.dumps(d)}", flush=True)

    json.dump({"script": "scripts/temporal_probe/congruence_mask_ab.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "note": "batch size is NOT a free variable here: eval_av_congruence swaps "
                       "audio within a batch and skips B<2, so batch-size-1 is impossible. "
                       "The single variable tested is the m2 backbone's padding mask.",
               "n_clips": args.n_clips, "runs": runs},
              open(args.out, "w"), indent=2)
    print(f"[cong_ab] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
