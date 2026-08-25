"""scripts/eval_av_congruence.py — does the model actually LISTEN, or is it guessing sound
from pictures?

WHY THIS EXISTS. The 4-way stream ablation found that dropping audio and M2 entirely
(arm D: scene+vision) scored 0.546 against 0.566 for the full stack — only ~0.02 apart. Read
naively that says ears are nearly worthless. But that benchmark is caption retrieval over
VGGSound/Action100M, which is overwhelmingly VISUAL: 4 of the 6 VGGSound question types are
about what is seen, and even the sound questions are usually guessable from the picture (you
can see the guitar). **A model that ignored audio completely would still score well.** So that
number cannot settle whether M2's audio-visual congruence earns its place.

THE TEST. Give the model MISMATCHED audio and ask what it hears.

    features = vision from clip i  +  ambient from clip j        (i != j)
    question = "What do you hear?"          (a SOUND question)
    candidates = clip i's sound caption  vs  clip j's sound caption

  * A model that genuinely uses audio retrieves **clip j's** caption -> follows the EARS.
  * A model that infers sound from pictures retrieves **clip i's** caption -> follows the EYES.

`audio_following_rate` is therefore a direct measure of whether the ears are load-bearing,
and it is immune to the visual shortcut that inflates ordinary caption retrieval. Chance is
0.5 (two candidates). A vision-only arm should sit near 0.0 (always the eyes), not 0.5 —
because it is not guessing, it is systematically answering from the picture.

CONTROL: the same clips scored with MATCHED audio. A model must stay accurate there too;
following the ears is only meaningful if it does not wreck normal performance.

Run over every trained ablation arm so M2's contribution is measured on its own job.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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
from models.text_target import TextTarget

SOUND_FIELDS = ["gpt_sound_acoustic", "gpt_sound_acoustic_v1_original"]


def load_scene(scene_dir: str):
    sc = {}
    for p in sorted(glob.glob(os.path.join(scene_dir, "*.pt"))):
        sc.update(torch.load(p, map_location="cpu", weights_only=False))
    return sc


@torch.no_grad()
def eval_arm(ckpt_path: str, m2, tt, te, ids, scene, device, n_clips: int,
             audio_mode: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = [s.strip() for s in ck["cfg"]["token_sources"].split(",")]
    SRC = {"m2": 1024, "vision": 1024, "ambient": 768, "scene": 768}
    # geometry off the checkpoint, not hardcoded: two target spaces now exist
    sd = ck["query_predictor"]
    shared_dim = int(sd["head.weight"].shape[0])
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=tt.native_dim,
                                             shared_dim=shared_dim)).to(device)
    qp.load_state_dict(sd); qp.eval()
    tp = ck.get("text_target_proj") or {}
    if tp:
        tt.proj.load_state_dict(tp)

    sub = {k: te[k] for k in ids[:n_clips]}
    dl = DataLoader(QueryClipDataset(sub, CACHE_DIR, VGGSOUND_FIELDS, scene_feats=scene,
                                     audio_mode=audio_mode),
                    batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

    has_audio = ("ambient" in names) or ("m2" in names)
    ear = eye = n = 0
    matched_ok = 0
    for batch in dl:
        B = len(batch["clip_id"])
        if B < 2:
            continue
        perm = torch.roll(torch.arange(B), 1)          # clip i gets clip i-1's audio

        # ---- MATCHED control ----
        src, msk = build_sources(batch, m2, device, names)
        for f in SOUND_FIELDS:
            fi = VGGSOUND_FIELDS.index(f)
            q = QUERY_BANK[f][-1]
            qe = tt.encode_text_frozen_raw([q] * B).to(device)
            z = qp(src, qe, msk)
            own = F.normalize(tt.encode_text([batch["caps"][b][fi] for b in range(B)]).float(), dim=-1)
            oth = own[perm]
            matched_ok += int(((z * own).sum(-1) > (z * oth).sum(-1)).sum())

        # ---- MISMATCHED: swap the AUDIO between clips, keep vision/scene ----
        sw = dict(batch)
        sw["feats"] = dict(batch["feats"]); sw["tbins"] = dict(batch["tbins"]); sw["pad"] = dict(batch["pad"])
        sw["feats"]["ambient"] = batch["feats"]["ambient"][perm]
        sw["tbins"]["ambient"] = batch["tbins"]["ambient"][perm]
        sw["pad"]["ambient"] = batch["pad"]["ambient"][perm]
        src_s, msk_s = build_sources(sw, m2, device, names)
        for f in SOUND_FIELDS:
            fi = VGGSOUND_FIELDS.index(f)
            q = QUERY_BANK[f][-1]
            qe = tt.encode_text_frozen_raw([q] * B).to(device)
            z = qp(src_s, qe, msk_s)
            eyes = F.normalize(tt.encode_text([batch["caps"][b][fi] for b in range(B)]).float(), dim=-1)
            ears = eyes[perm]                          # caption of the clip the AUDIO came from
            s_ear = (z * ears).sum(-1)
            s_eye = (z * eyes).sum(-1)
            ear += int((s_ear > s_eye).sum()); eye += int((s_eye >= s_ear).sum()); n += B
    del qp
    torch.cuda.empty_cache()
    return {"streams": names, "audio_mode": audio_mode, "has_audio": has_audio,
            "n_trials": n, "audio_following_rate": ear / max(1, n),
            "vision_following_rate": eye / max(1, n),
            "matched_control_acc": matched_ok / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    ap.add_argument("--scene-dir", default="/dev/shm/scene_all")
    ap.add_argument("--n-clips", type=int, default=320)
    ap.add_argument("--text-backbone", default="embeddinggemma")
    ap.add_argument("--arms", default="abl_A_m2_vision_ambient:mean,abl_B_plus_scene:mean,"
                                      "abl_C_scene_baseonly:base,abl_D_scene_vision_only:mean",
                    help="comma list of ckptdir:audio_mode")
    ap.add_argument("--out", default="checkpoints/AV_CONGRUENCE_EVAL.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    te = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    scene = load_scene(args.scene_dir)
    ids = [c for c in sorted(te) if c in scene]
    print(f"[cong] {len(ids)} scene-covered held-out clips; using {args.n_clips}", flush=True)

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()
    if args.text_backbone.startswith("siglip"):
        from models.text_target import SigLIP2TextTarget
        probe = torch.load(f"checkpoints/{args.arms.split(',')[0].split(':')[0]}/best.pt",
                           map_location="cpu", weights_only=False)
        sdim = int(probe["query_predictor"]["head.weight"].shape[0]) \
            if (probe.get("text_target_proj") or {}) else None
        tt = SigLIP2TextTarget(device=str(device), shared_dim=sdim)
    else:
        tt = TextTarget(backbone="embeddinggemma", shared_dim=1536, unfreeze_base=False,
                        device=str(device))

    ARMS = [tuple(a.split(":")) for a in args.arms.split(",")]
    res = {}
    for name, amode in ARMS:
        p = f"checkpoints/{name}/best.pt"
        if not os.path.exists(p):
            print(f"[cong] skip {name} (no checkpoint)", flush=True); continue
        r = eval_arm(p, m2, tt, te, ids, scene, device, args.n_clips, amode)
        res[name] = r
        print(f"[cong] {name:24s} audio_following={r['audio_following_rate']:.3f} "
              f"matched_control={r['matched_control_acc']:.3f} n={r['n_trials']}", flush=True)

    print("\n== AV CONGRUENCE: swap the audio, ask what it HEARS ==")
    print(f"{'arm':<24} {'has audio':>10} {'follows EARS':>13} {'follows EYES':>13} {'matched ctrl':>13}")
    for k, r in res.items():
        print(f"{k:<24} {str(r['has_audio']):>10} {r['audio_following_rate']:13.3f} "
              f"{r['vision_following_rate']:13.3f} {r['matched_control_acc']:13.3f}")
    print("\nchance = 0.500. A vision-only arm should sit near 0.0 (systematically the eyes),")
    print("not 0.5 -- it is not guessing, it is answering from the picture.")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[cong] wrote {args.out}")


if __name__ == "__main__":
    main()
