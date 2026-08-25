"""scripts/eval_candidate_sets.py — captions or tags? predictor or zero-shot?

Four combinations are now buildable, and they are NOT interchangeable. This measures them
instead of assuming, because each one trades something real:

  1. CAPTIONS via PREDICTOR   -- in-geometry. The predictor was optimised to land on caption
                                 embeddings, and this is the ONLY path that uses audio
                                 (m2/ambient streams). 177 MiB of candidates.
  2. TAGS via PREDICTOR       -- extrapolation. Same model, but tag points were never a
                                 training target. 2 MiB of candidates.
  3. TAGS via ZERO-SHOT       -- SigLIP2 image embedding scored directly against tags. No
                                 predictor, no audio, no query-conditioning -- but nothing
                                 learned in between, so it generalises off-corpus. 2 MiB.
  4. CAPTIONS via ZERO-SHOT   -- control for 3, isolating candidate granularity from path.

METRICS, and why these ones:
  * `caption_r1` (paths 1 and 4): correct clip's caption among all eval clips at a fixed
    field. The standard number, comparable to every recorded run.
  * `tag_precision@k` (paths 2 and 3): a retrieved tag COUNTS as correct if it appears as a
    content word in that clip's own ground-truth captions. There is no per-clip tag label in
    this corpus, so this is a proxy, not a ground truth -- it rewards tags the caption
    actually mentions and punishes hallucinated ones. Reported with a shuffled-clip control
    (`tag_precision_shuffled`) so the number is only meaningful by its GAP to chance: a
    vocabulary of common words scores non-zero against any caption.
  * the ROOM frame, which is the actual deployment condition and the reason for all of this.
    No corpus metric can settle it, so it is printed for inspection rather than scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
train_m3.CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl")
from train_m3 import build_splits, CACHE_DIR
from train_query_predictor import QueryClipDataset, collate, group_by_clip, build_sources
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.query_predictor import (QueryPredictor, QueryPredictorConfig, QUERY_BANK,
                                    VGGSOUND_FIELDS)

STOP = set("""a an the of in on at to for with and or is are was were be been being this that
these those it its as by from into over under near very there here they he she his her their
someone something people person while during after before then than which who whom whose
video shows depicts featuring showing appears seen visible camera scene clip footage""".split())


def load_scene(scene_dir: str):
    sc = {}
    for p in sorted(glob.glob(os.path.join(scene_dir, "*.pt"))):
        sc.update(torch.load(p, map_location="cpu", weights_only=False))
    return sc


def content_words(texts) -> set:
    out = set()
    for t in texts:
        for w in re.findall(r"[a-z]+", t.lower()):
            if len(w) >= 4 and w not in STOP:
                out.add(w)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sig_armC_baseonly/best.pt")
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--bank", default="checkpoints/bank_siglip2_fp16.pt")
    ap.add_argument("--tags", default="checkpoints/candidates_siglip2.pt")
    ap.add_argument("--scene-dir", default="/dev/shm/scene_all")
    ap.add_argument("--n-clips", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--audio-mode", default="base")
    ap.add_argument("--frame", default="jetson_artifacts/benchmarks/home/cam_gst.jpg")
    ap.add_argument("--out", default="checkpoints/CANDIDATE_SET_EVAL.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]
    sd = ck["query_predictor"]
    shared_dim, query_dim = int(sd["head.weight"].shape[0]), int(sd["q_proj.weight"].shape[1])
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768, "scene": 768}
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=query_dim, shared_dim=shared_dim)).to(device)
    qp.load_state_dict(sd); qp.eval()
    print(f"[cand] {args.ckpt} streams={names} shared={shared_dim} query={query_dim}", flush=True)

    from models.text_target import SigLIP2TextTarget
    # The two retrieval paths live in DIFFERENT spaces and must not be mixed:
    #   predictor path -> the LEARNED space (raw text through the trained proj)
    #   zero-shot path -> the RAW pretrained space, because that is the one SigLIP2's IMAGE
    #                     tower shares. Scoring an image against projected text would be
    #                     comparing vectors from two different geometries.
    tp = ck.get("text_target_proj") or {}
    tt = SigLIP2TextTarget(device=str(device), shared_dim=(shared_dim if tp else None))
    if tp:
        tt.proj.load_state_dict(tp)
        print(f"[cand] loaded trained proj -> learned space is {shared_dim}-d", flush=True)
    else:
        print("[cand] checkpoint has NO proj (Identity/frozen run)", flush=True)

    def load_raw(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return F.normalize(d["emb"].float(), dim=-1).to(device), d["text"]

    B_raw, B_txt = load_raw(args.bank)      # captions, RAW SigLIP2 space
    T_raw, T_txt = load_raw(args.tags)      # tags,     RAW SigLIP2 space
    proj = (lambda z: F.normalize(tt.proj(z), dim=-1)) if tp else (lambda z: z)
    with torch.no_grad():
        B_prj, T_prj = proj(B_raw), proj(T_raw)
    print(f"[cand] bank {tuple(B_raw.shape)} | tags {tuple(T_raw.shape)}", flush=True)

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"],
                       strict=True)
    m2.eval()

    scene = load_scene(args.scene_dir)
    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    ids = [c for c in sorted(te) if c in scene][: args.n_clips]
    sub = {k: te[k] for k in ids}
    dl = DataLoader(QueryClipDataset(sub, CACHE_DIR, VGGSOUND_FIELDS, scene_feats=scene,
                                     audio_mode=args.audio_mode),
                    batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    FIELD = "gpt_summary_detailed"
    fi = VGGSOUND_FIELDS.index(FIELD)
    q = QUERY_BANK[FIELD][-1]

    zq_all, zscene_all, gt_all = [], [], []
    for b in dl:
        Bn = len(b["clip_id"])
        src, msk = build_sources(b, m2, device, names)
        qe = tt.encode_text_frozen_raw([q] * Bn).to(device)
        zq_all.append(F.normalize(qp(src, qe, msk).float(), dim=-1))
        # zero-shot path: mean-pool the clip's SigLIP2 frames into one image embedding
        zscene_all.append(F.normalize(b["scene"].to(device).float().mean(1), dim=-1))
        gt_all += [b["caps"][i] for i in range(Bn)]
    z_q = torch.cat(zq_all, 0); z_s = torch.cat(zscene_all, 0)
    N = z_q.shape[0]
    print(f"[cand] {N} held-out clips scored", flush=True)

    # ---- caption R@1: correct clip's own caption among all N clips' captions at FIELD ----
    own_raw = F.normalize(tt.encode_text_frozen_raw([g[fi] for g in gt_all]).float(), dim=-1)
    with torch.no_grad():
        own_prj = proj(own_raw)
    res = {}
    for tag, z, own in (("captions_via_predictor", z_q, own_prj),
                        ("captions_via_zeroshot", z_s, own_raw)):
        r1 = float(((z @ own.T).argmax(1) == torch.arange(N, device=device)).float().mean())
        res[tag] = {"caption_r1": r1, "n": N}
        print(f"[cand] {tag:26s} caption_r1={r1:.3f}", flush=True)

    # ---- tag precision@k against the clip's own caption content words ----
    perm = torch.randperm(N).tolist()
    for tag, z, T_emb in (("tags_via_predictor", z_q, T_prj),
                          ("tags_via_zeroshot", z_s, T_raw)):
        top = (z @ T_emb.T).topk(args.topk, dim=1).indices.cpu()
        hit = shuf = tot = 0
        for i in range(N):
            cw = content_words(gt_all[i])
            cw_s = content_words(gt_all[perm[i]])
            for j in top[i].tolist():
                w = set(re.findall(r"[a-z]+", T_txt[j].lower()))
                hit += int(bool(w & cw)); shuf += int(bool(w & cw_s)); tot += 1
        res[tag] = {f"tag_precision@{args.topk}": hit / tot,
                    f"tag_precision_shuffled@{args.topk}": shuf / tot,
                    "gap": (hit - shuf) / tot, "n": N}
        print(f"[cand] {tag:26s} p@{args.topk}={hit/tot:.3f} "
              f"shuffled={shuf/tot:.3f} gap={(hit-shuf)/tot:+.3f}", flush=True)

    # ---- the room frame: the actual deployment condition ----
    if os.path.exists(args.frame):
        from scripts.siglip2_room_test import load_frame
        img = load_frame(args.frame, 90, 1.8)
        px = tt.processor(images=[img], return_tensors="pt").to(device)
        px = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in px.items()}
        o = tt.base.get_image_features(**px)
        o = o.pooler_output if hasattr(o, "pooler_output") else o
        zi = F.normalize(o.float(), dim=-1)
        room = {}
        # the room frame is scored ZERO-SHOT, so RAW space on both sides
        for nm, E, TX in (("tags", T_raw, T_txt), ("captions", B_raw, B_txt)):
            s = (zi @ E.T)[0]; tk = s.topk(8)
            room[nm] = [{"score": float(v), "text": TX[i]}
                        for v, i in zip(tk.values.tolist(), tk.indices.tolist())]
            print(f"\n== ROOM frame, zero-shot vs {nm} ==")
            for r in room[nm]:
                print(f"  [{r['score']:+.4f}] {r['text'][:110]}")
        res["room_frame"] = room

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[cand] wrote {args.out}")


if __name__ == "__main__":
    main()
