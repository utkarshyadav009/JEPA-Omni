"""scripts/temporal_probe/qp_rescore_bs1.py — P0.4 item 2.

Re-score a trained query-predictor arm at BATCH SIZE 1 and compare against the
original batch-48 protocol on the IDENTICAL clip set.

Why batch size 1 is the decisive test: train_query_predictor.collate pads vision
and ambient to the longest clip IN THE BATCH. At batch size 1 no padding can
exist, so any movement between the two is attributable to padding. The clip set,
the clip ORDER, the query phrasings (same seed -> same rng draws) and the
checkpoint are held identical; only the batch size differs.

What the static audit says to expect (see docs/ERRATA_PROPOSED.md E-6):
QueryPredictor.forward DOES receive and apply the padding mask, and it pools over
its 8 fixed LATENTS rather than over tokens -- so neither of the two leaks that
corrupted the gallery eval is present in the same form. The one residual is that
AVJepaPredictor.encode_pre_pool_tokens (which builds the 'm2' stream) runs the M2
backbone UNMASKED, so m2's real tokens attend to pad tokens before the QP's own
mask is applied. Also, unlike the gallery eval, one side of this retrieval is TEXT,
which has no access to a clip's pad count -- so the shared-nuisance mechanism that
inflated VGGSound R@1 cannot operate here. This script measures rather than assumes.

Usage:
    python scripts/temporal_probe/qp_rescore_bs1.py \
        --qp-ckpt checkpoints/sig_runA_matched3stream/best.pt \
        --out docs/artifacts/temporal_probe/p04_sig_runA.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import train_m3
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.query_predictor import QueryPredictor, QueryPredictorConfig
from models.text_target import build_text_target

M2_SHA = "e1a8231ec9fbae6cf2b8288c89612ee1facd2dc6938781c22cf06295b31bedf8"
SRC_DIMS = {"m2": 1024, "vision": 1024, "ambient": 768, "scene": 768}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qp-ckpt", required=True)
    p.add_argument("--cache-dir", default="/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k")
    p.add_argument("--captions", default=os.path.join(PROJECT_ROOT, "scripts",
                                                       "qwen_omni_full_captions_v2.jsonl"))
    p.add_argument("--max-clips", type=int, default=624,
                   help="held EQUAL across both batch sizes so the retrieval gallery "
                        "size -- which cross_clip_r1 depends on -- is identical")
    p.add_argument("--batch-sizes", default="48,1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene-file", default="",
                   help="re-extracted SigLIP2 scene dict {clip_id:(8,768)} for the 4-stream "
                        "arms, from scripts/temporal_probe/extract_scene_subset.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    train_m3.CAPTIONS_PATH = args.captions
    from train_query_predictor import (QueryClipDataset, collate, group_by_clip,
                                       build_splits, evaluate)
    VGGSOUND_FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief",
                       "gpt_summary_detailed", "gpt_sound_acoustic",
                       "gpt_sound_acoustic_v1_original"]

    ck = torch.load(args.qp_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    names = [x.strip() for x in cfg["token_sources"].split(",") if x.strip()]
    scene = None
    if "scene" in names:
        if not args.scene_file:
            raise SystemExit(
                "ABORT: this arm uses the 'scene' stream. Its SigLIP2 features lived in "
                "/dev/shm/scene_all, which is empty (the box rebooted). Re-extract with "
                "scripts/temporal_probe/extract_scene_subset.py and pass --scene-file.")
        scene = torch.load(args.scene_file, map_location="cpu", weights_only=False)
        shapes = {tuple(v.shape) for v in scene.values()}
        print(f"[qp_rescore] scene: {len(scene)} clips, shapes={shapes}", flush=True)

    m2_sha = sha256_file(cfg["m2_ckpt"])
    if m2_sha != M2_SHA:
        raise SystemExit(f"ABORT: M2 ckpt sha256 {m2_sha} != {M2_SHA}")

    device = torch.device("cuda")
    # infer shared_dim from the checkpoint rather than trusting a default
    shared_dim = ck["query_predictor"]["head.weight"].shape[0]
    qpcfg = QueryPredictorConfig(source_dims={k: SRC_DIMS[k] for k in names},
                                 shared_dim=shared_dim)
    qp = QueryPredictor(qpcfg).to(device)
    qp.load_state_dict(ck["query_predictor"], strict=True)
    qp.eval()

    m2 = AVJepaPredictor(AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0,
                                       max_tdm_bins=512, dropout=0.0)).to(device)
    m2.load_state_dict(torch.load(cfg["m2_ckpt"], map_location=device,
                                   weights_only=False)["model"], strict=True)
    m2.eval()

    # An EMPTY saved text_target_proj means the target projection was nn.Identity,
    # i.e. the frozen-SigLIP2 configuration (siglip_shared_dim=None). Passing a dim
    # there instead builds an UNTRAINED Linear and scores at chance -- so the flag is
    # derived from the checkpoint, never assumed. (models/text_target.py:148-158
    # records that the Identity arm measures within-clip 0.654 / R@1 0.489, which is
    # sig_runA's logged result, confirming this mapping.)
    has_proj = bool(ck.get("text_target_proj"))
    tt = build_text_target(cfg["text_backbone"], shared_dim=1536, device=str(device),
                           siglip_shared_dim=(qpcfg.shared_dim if has_proj else None))
    if has_proj:
        tt.proj.load_state_dict(ck["text_target_proj"])
    print(f"[qp_rescore] text target: {cfg['text_backbone']} "
          f"proj={'trained Linear' if has_proj else 'Identity (frozen)'}", flush=True)

    _, vte_pairs = build_splits(VGGSOUND_FIELDS)
    vte = group_by_clip(vte_pairs, VGGSOUND_FIELDS)
    if scene is not None:
        # reproduce restrict_to_scene=True against the RE-EXTRACTED set. The original
        # /dev/shm set covered 13,579/13,679 (99.3%) and its exact membership is
        # unrecoverable, so the absolute value is a near-match, not a reproduction.
        # The bs48-vs-bs1 DELTA is unaffected: both sides use this identical subset.
        n0 = len(vte)
        vte = {k: v for k, v in vte.items() if k in scene}
        print(f"[qp_rescore] restrict_to_scene: {n0} -> {len(vte)} clips "
              f"(re-extracted subset; NOT the original membership)", flush=True)
    print(f"[qp_rescore] eval clips available={len(vte)}  streams={names}  "
          f"audio_mode={cfg['audio_mode']}", flush=True)

    runs = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        ds = QueryClipDataset(vte, args.cache_dir, VGGSOUND_FIELDS,
                              scene_feats=scene, audio_mode=cfg["audio_mode"])
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6,
                            collate_fn=collate)
        # fresh rng with the SAME seed => identical held-out query phrasings both runs
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        m = evaluate(qp, tt, m2, loader, VGGSOUND_FIELDS, device, rng, names,
                     max_clips=args.max_clips)
        m.update({"batch_size": bs, "wall_s": round(time.time() - t0, 1)})
        runs.append(m)
        print(f"[qp_rescore] bs={bs:<3} {json.dumps(m)}", flush=True)

    base = runs[0]
    for r in runs[1:]:
        r["delta_vs_bs%d" % base["batch_size"]] = {
            k: round(r[k] - base[k], 4)
            for k in ("within_clip_acc", "swapped_query_acc", "cross_clip_r1")}
        print(f"[qp_rescore] DELTA bs{r['batch_size']} vs bs{base['batch_size']}: "
              f"{json.dumps(r['delta_vs_bs%d' % base['batch_size']])}", flush=True)

    json.dump({"script": "scripts/temporal_probe/qp_rescore_bs1.py",
               "command": " ".join([sys.executable] + sys.argv),
               "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True).stdout.strip() or "MISSING",
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "qp_ckpt": args.qp_ckpt, "qp_ckpt_step": ck["step"],
               "qp_cfg": {k: str(v) for k, v in cfg.items()},
               "m2_ckpt": cfg["m2_ckpt"], "m2_ckpt_sha256": m2_sha,
               "cache_dir": args.cache_dir, "streams": names,
               "eval_clips_available": len(vte),
               "scene_file": args.scene_file or None,
               "scene_subset_is_original_membership": False if scene is not None else None, "max_clips": args.max_clips,
               "seed": args.seed, "logged_reference": ck.get("metrics", {}),
               "runs": runs},
              open(args.out, "w"), indent=2)
    print(f"[qp_rescore] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
