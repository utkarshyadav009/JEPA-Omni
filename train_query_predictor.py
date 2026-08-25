"""train_query_predictor.py — trains models/query_predictor.py::QueryPredictor.

THE CLAIM BEING TRAINED AND THEN FALSIFIED: the thinker can ask perception a
question in natural language and get back an answer that is (a) grounded in THIS
scene and (b) actually responsive to THAT question.

Both halves need separate negatives, and this is the whole design of the loss:
  * cross-CLIP negatives  -> the answer must describe THIS scene, not another one.
    (the existing embed predictor already gets this right)
  * within-CLIP negatives -> for one clip, the 5 caption fields are all true
    statements about it, differing only in WHAT WAS ASKED. Putting a clip's OTHER
    field captions in the candidate set is what forces the query to carry
    information. Without them the model can ignore the query entirely and still
    score perfectly, which is exactly the degenerate solution to avoid.
Both appear in ONE similarity matrix: z_q (B, D) against all B*K in-batch
captions, correct index = (clip i, field asked of clip i).

DATA — both captioned corpora this project has built up, mixed per-batch. No new
extraction: both feature caches and both caption files already exist.
  * VGGSound (scripts/qwen_omni_full_captions_v2.jsonl, 188,657 clips over a
    199,007-clip cache) is really a **3 aspects x 2 granularities grid**, which is
    what makes it the right corpus here -- a query has to resolve BOTH axes:
        action  brief 4.8w  / detailed 37.0w
        summary brief 8.9w  / detailed 58.1w
        sound   brief 12.2w / detailed 18.0w   (the v1_original field is the long one)
    Field coverage measured at 99-100%, so requiring all 6 costs almost nothing.
  * Action100M (scripts/action100m_captions.jsonl, 399,934 clips + cache) supplies
    only the action row (3.2w / 27.1w) but ~2x the clips, and that row IS the
    brief<->detailed axis the "describe it in more detail" use case lives on.
    Its known 3.4% literal "N/A" placeholder captions are filtered (the repo
    already found and fixed this once -- see load_action100m_splits).
Ego4D's 134k cached clips are deliberately NOT used: they carry no captions in
this project, so there is no query/answer supervision to draw from them.

K differs per corpus (6 vs 2), so batches are drawn from ONE corpus at a time
(--p-vggsound) -- a mixed batch would make the candidate matrix ragged and the
chance level vary per example.

Frozen: V-JEPA2 + WavJEPA (already cached), the locked M2 predictor, and
TextTarget's EmbeddingGemma base. Trainable: QueryPredictor + TextTarget.proj
(the same split train_m2_embed_predictor.py uses, so results are comparable).

Usage:
    python train_query_predictor.py --steps 3000 --batch-clips 96
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
from train_m3 import build_splits, CACHE_DIR, _cap_ambient_len
from train_m2_embed_predictor import load_action100m_splits, ACTION100M_CACHE_DIR
from data.av_cached_dataset import AVCachedDataset
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.query_predictor import (QueryPredictor, QueryPredictorConfig,
                                    QUERY_BANK, get_query,
                                    VGGSOUND_FIELDS, ACTION100M_FIELDS)
from models.text_target import TextTarget

PLACEHOLDERS = ("N/A", "NA", "NONE", "NULL")


def group_by_clip(pairs: List[Tuple[str, str, str]], fields: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for cid, field, text in pairs:
        if text and text.strip() and text.strip().upper() not in PLACEHOLDERS:
            out[cid][field] = text.strip()
    # keep only clips carrying EVERY field: within-clip negatives are the whole
    # point, and a ragged candidate set would make chance vary per example
    return {c: d for c, d in out.items() if len(d) == len(fields)}


class QueryClipDataset(Dataset):
    """One item per CLIP, carrying all of that corpus's caption fields, so the
    collate can build within-clip negatives.

    `scene_feats` optionally attaches the SigLIP2 scene stream ({clip_id: (K,768)}). It is a
    SEPARATE dict rather than part of the AV cache because the JEPA trunk's cache is frozen
    and shared by every other trainer -- adding a stream must not disturb it."""

    def __init__(self, clips: Dict[str, Dict[str, str]], cache_dir: str,
                 fields: List[str], max_tdm_bins: int = 512, scene_feats=None,
                 audio_mode: str = "mean"):
        self.ids = sorted(clips)
        self.caps = clips
        self.fields = fields
        self.scene_feats = scene_feats
        # audio_mode: "mean" = WavJEPA base+nat averaged (what M2 was TRAINED on, deployed
        # default) | "base" = base only. The cache stores base and nat separately, so this
        # is a load-time switch, not a re-extraction. Worth testing because the user's own
        # M2 ablation measured base-only (ambient->vision R@1 37.99%) slightly BEATING the
        # mean (37.15%), while nat costs 469 ms on the Jetson -- the single biggest
        # perception component -- and is fed duplicated mono, outside its binaural
        # training distribution.
        self.av = AVCachedDataset(cache_dir=cache_dir, clip_ids=self.ids,
                                  max_tdm_bins=max_tdm_bins, audio_mode=audio_mode)

    def __len__(self) -> int:
        return len(self.av)

    def __getitem__(self, i: int) -> Dict:
        item = self.av[i]
        cid = self.ids[i]
        item["clip_id"] = cid
        item["fields"] = self.fields
        item["caps"] = [self.caps[cid][f] for f in self.fields]
        if self.scene_feats is not None:
            item["scene"] = self.scene_feats.get(cid)
        return item


def collate(batch: List[Dict]) -> Dict:
    B = len(batch)
    mv = max(b["feats"]["vision"].shape[0] for b in batch)
    ma = max(b["feats"]["ambient"].shape[0] for b in batch)
    vis = torch.zeros(B, mv, batch[0]["feats"]["vision"].shape[-1], dtype=batch[0]["feats"]["vision"].dtype)
    aud = torch.zeros(B, ma, batch[0]["feats"]["ambient"].shape[-1], dtype=batch[0]["feats"]["ambient"].dtype)
    vb = torch.zeros(B, mv, dtype=torch.long); ab = torch.zeros(B, ma, dtype=torch.long)
    vp = torch.ones(B, mv, dtype=torch.bool); apad = torch.ones(B, ma, dtype=torch.bool)
    for i, b in enumerate(batch):
        nv, na = b["feats"]["vision"].shape[0], b["feats"]["ambient"].shape[0]
        vis[i, :nv] = b["feats"]["vision"]; aud[i, :na] = b["feats"]["ambient"]
        vb[i, :nv] = b["tbins"]["vision"];  ab[i, :na] = b["tbins"]["ambient"]
        vp[i, :nv] = False; apad[i, :na] = False
    out_scene = None
    if batch[0].get("scene") is not None:
        ms = max(b["scene"].shape[0] for b in batch)
        out_scene = torch.zeros(B, ms, batch[0]["scene"].shape[-1], dtype=torch.float32)
        for i, b in enumerate(batch):
            out_scene[i, :b["scene"].shape[0]] = b["scene"].float()
    return {"feats": {"vision": vis, "ambient": aud},
            "scene": out_scene,
            "tbins": {"vision": vb, "ambient": ab},
            "pad": {"vision": vp, "ambient": apad},
            "caps": [b["caps"] for b in batch],
            "fields": batch[0]["fields"],
            "clip_id": [b["clip_id"] for b in batch]}


def build_sources(batch, m2, device, names: List[str]):
    """Assemble the requested token streams. All are free from the existing AV cache.
    m2      = the cross-modal FUSED view (audio-visual congruence -- M2's trained job)
    vision  = M2's own INPUT read directly (the detail M2 was measured to discard)
    ambient = the WavJEPA tokens read directly
    The m2 stream's token order is [vision; ambient] (see AVJepaPredictor._embed),
    so its padding mask is the concatenation of the two input masks."""
    feats = {k: v.to(device, non_blocking=True).float() for k, v in batch["feats"].items()}
    tbins = {k: v.to(device, non_blocking=True) for k, v in batch["tbins"].items()}
    pads = {k: v.to(device, non_blocking=True) for k, v in batch["pad"].items()}
    src, msk = {}, {}
    if "m2" in names:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            src["m2"] = m2.encode_pre_pool_tokens(feats, tbins).float()
        msk["m2"] = torch.cat([pads["vision"], pads["ambient"]], 1)
    if "vision" in names:
        src["vision"] = feats["vision"]; msk["vision"] = pads["vision"]
    if "ambient" in names:
        src["ambient"] = feats["ambient"]; msk["ambient"] = pads["ambient"]
    if "scene" in names:
        sc = batch.get("scene")
        if sc is None:
            raise KeyError("'scene' stream requested but this batch has no SigLIP2 features "
                           "-- pass scene_feats to QueryClipDataset")
        src["scene"] = sc.to(device, non_blocking=True).float()
    return src, msk


@torch.no_grad()
def evaluate(qp, tt, m2, loader, fields, device, rng, names, max_clips: int = 600) -> Dict[str, float]:
    """Three falsifiers in one pass, on ONE corpus's field set:

      within_clip_acc : pick the correct FIELD among the SAME clip's K captions
                        (chance 1/K). This is THE metric -- it can only be beaten
                        by using the query, since all K captions describe the same
                        clip and differ only in what was asked.
      swapped_query   : identical, but asked with a DIFFERENT field's question.
                        Must collapse toward chance; if it does not, the model is
                        ignoring the query and within_clip_acc is meaningless.
      cross_clip_r1   : correct CLIP among all eval clips at a fixed field. Guards
                        against buying query-sensitivity by wrecking scene grounding.

    Every query uses the HELD-OUT phrasing, never seen in training, so this also
    tests generalisation to wording the thinker was never trained on."""
    K = len(fields)
    qp.eval(); tt.base.eval()
    Zq_blocks, Zs_blocks, Zt_blocks, sizes = [], [], [], []
    n = 0
    for batch in loader:
        if n >= max_clips:
            break
        src, msk = build_sources(batch, m2, device, names)
        B = next(iter(src.values())).shape[0]
        qb, sb = [], []
        for fi, f in enumerate(fields):
            qe = tt.encode_text_frozen_raw([get_query(f, rng, train=False)] * B).to(device)
            qb.append(qp(src, qe, msk).cpu())
            wrong = fields[(fi + K // 2) % K]                # a genuinely different intent
            se = tt.encode_text_frozen_raw([get_query(wrong, rng, train=False)] * B).to(device)
            sb.append(qp(src, se, msk).cpu())
        Zq_blocks.append(torch.stack(qb, 1))                 # (B, K, D)
        Zs_blocks.append(torch.stack(sb, 1))
        Zt_blocks.append(tt.encode_text([c for caps in batch["caps"] for c in caps]).cpu())
        sizes.append(B); n += B

    Zq = torch.cat(Zq_blocks, 0)                             # (N, K, D)
    Zs = torch.cat(Zs_blocks, 0)
    Zt = torch.cat(Zt_blocks, 0).view(-1, K, Zq.shape[-1])   # (N, K, D)
    N = Zq.shape[0]

    hit = (torch.einsum("nkd,njd->nkj", Zq, Zt).argmax(-1) == torch.arange(K)).float().mean()
    hit_sw = (torch.einsum("nkd,njd->nkj", Zs, Zt).argmax(-1) == torch.arange(K)).float().mean()

    fi = fields.index("gpt_action_detailed")
    sim = Zq[:, fi] @ Zt[:, fi].T                            # (N, N)
    r1 = float((sim.argmax(1) == torch.arange(N)).float().mean())

    qp.train(); tt.base.train(tt.unfreeze_base)
    return {"within_clip_acc": float(hit), "swapped_query_acc": float(hit_sw),
            "within_clip_chance": 1.0 / K, "cross_clip_r1": r1, "n_clips": float(N)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--out-dir", default="checkpoints/query_predictor_v1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-clips", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--text-backbone", default="embeddinggemma")
    p.add_argument("--p-vggsound", type=float, default=0.5,
                   help="probability a step draws from VGGSound (6 fields) vs Action100M (2)")
    p.add_argument("--max-clips-per-corpus", type=int, default=0)
    p.add_argument("--captions-path", default=os.path.join(PROJECT_ROOT, "scripts",
                   "qwen_omni_full_captions_v2.jsonl"),
                   help="v2 = the CORRECTED rich captions (what the locked M3 used) AND the "
                        "only file carrying gpt_sound_acoustic_v1_original, i.e. the 6th field "
                        "that completes the 3x2 aspect-x-granularity grid. v1 has 5 fields only.")
    p.add_argument("--token-sources", default="m2,vision",
                   help="comma-separated streams the query cross-attends over. "
                        "'m2' keeps the AV-congruence fused view (M2 stays in the "
                        "pipeline); 'vision' adds a direct path to the raw pooled "
                        "ViT-L tokens M2 discards; 'ambient' adds raw WavJEPA.")
    p.add_argument("--lambda-within", type=float, default=0.0,
                   help="weight on a separate within-clip (K-way) cross-entropy term; "
                        "0.0 reproduces the original single-softmax loss")
    p.add_argument("--scene-dir", default="",
                   help="dir of SigLIP2 scene shards ({clip_id: (K,768)}). Required when "
                        "'scene' is in --token-sources. Action100M only -- raw VGGSound "
                        "video no longer exists on this machine, so VGGSound has no scene "
                        "stream and is skipped when 'scene' is requested.")
    p.add_argument("--audio-mode", default="mean", choices=["mean", "base", "nat"],
                   help="WavJEPA combination. 'mean'=base+nat (M2's training default); "
                        "'base'=drop nat, which is a measured 469 ms Jetson win.")
    p.add_argument("--restrict-to-scene", action="store_true",
                   help="train/eval only on clips that HAVE a SigLIP2 scene vector. Set this "
                        "for EVERY arm of an ablation (including no-scene controls) so the "
                        "arms differ in architecture, not in which data they saw.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    names = [x.strip() for x in args.token_sources.split(",") if x.strip()]

    # build_splits reads train_m3's module-global at call time
    train_m3.CAPTIONS_PATH = args.captions_path

    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── VGGSound: the full 3 aspects x 2 granularities grid ──
    vtr_pairs, vte_pairs = build_splits(VGGSOUND_FIELDS)
    vtr = group_by_clip(vtr_pairs, VGGSOUND_FIELDS)
    vte = group_by_clip(vte_pairs, VGGSOUND_FIELDS)

    # ── Action100M: action row only, but ~400k clips of brief<->detailed ──
    atr_all, ate_all = defaultdict(list), defaultdict(list)
    for f in ACTION100M_FIELDS:
        tr_p, te_p = load_action100m_splits(f)
        for cid, fld, txt in tr_p:
            atr_all[cid].append((cid, fld, txt))
        for cid, fld, txt in te_p:
            ate_all[cid].append((cid, fld, txt))
    atr = group_by_clip([t for v in atr_all.values() for t in v], ACTION100M_FIELDS)
    ate = group_by_clip([t for v in ate_all.values() for t in v], ACTION100M_FIELDS)

    if args.max_clips_per_corpus:
        vtr = {k: vtr[k] for k in sorted(vtr)[: args.max_clips_per_corpus]}
        atr = {k: atr[k] for k in sorted(atr)[: args.max_clips_per_corpus]}
    print(f"[qp] VGGSound  clips with all {len(VGGSOUND_FIELDS)} fields: "
          f"train={len(vtr)} test={len(vte)}", flush=True)
    print(f"[qp] Action100M clips with all {len(ACTION100M_FIELDS)} fields: "
          f"train={len(atr)} test={len(ate)}", flush=True)

    scene = None
    if "scene" in names:
        import glob as _g
        scene = {}
        for sp in sorted(_g.glob(os.path.join(args.scene_dir, "*.pt"))):
            scene.update(torch.load(sp, map_location="cpu", weights_only=False))
        print(f"[qp] scene stream: {len(scene)} clips with SigLIP2 features", flush=True)

    if args.restrict_to_scene:
        if scene is None:
            import glob as _g
            scene = {}
            for sp in sorted(_g.glob(os.path.join(args.scene_dir, "*.pt"))):
                scene.update(torch.load(sp, map_location="cpu", weights_only=False))
        n0 = (len(vtr), len(atr))
        vtr = {k: v for k, v in vtr.items() if k in scene}
        vte = {k: v for k, v in vte.items() if k in scene}
        atr = {k: v for k, v in atr.items() if k in scene}
        ate = {k: v for k, v in ate.items() if k in scene}
        print(f"[qp] restricted to scene-covered clips: VGGSound {n0[0]}->{len(vtr)} "
              f"Action100M {n0[1]}->{len(atr)} (same pool for EVERY ablation arm)", flush=True)

    mk = lambda c, cd, fl, bs, sh, nw: DataLoader(
        QueryClipDataset(c, cd, fl, scene_feats=scene, audio_mode=args.audio_mode),
        batch_size=bs, shuffle=sh,
        num_workers=nw, collate_fn=collate, drop_last=sh, pin_memory=True,
        persistent_workers=nw > 0)
    dl_v = mk(vtr, CACHE_DIR, VGGSOUND_FIELDS, args.batch_clips, True, 8) if vtr else None
    dl_a = mk(atr, ACTION100M_CACHE_DIR, ACTION100M_FIELDS, args.batch_clips, True, 8)
    ev_v = mk(vte, CACHE_DIR, VGGSOUND_FIELDS, 48, False, 4) if vte else None
    ev_a = mk(ate, ACTION100M_CACHE_DIR, ACTION100M_FIELDS, 48, False, 4)

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()
    for prm in m2.parameters():
        prm.requires_grad_(False)

    tt = TextTarget(backbone=args.text_backbone, shared_dim=1536, unfreeze_base=False, device=str(device))
    SRC_DIMS = {"m2": m2cfg.d_model, "vision": 1024, "ambient": 768, "scene": 768}
    qp = QueryPredictor(QueryPredictorConfig(
        source_dims={s_: SRC_DIMS[s_] for s_ in names},
        query_dim=tt.native_dim, shared_dim=1536)).to(device)
    print(f"[qp] token sources = {names}", flush=True)
    print(f"[qp] QueryPredictor trainable = "
          f"{sum(x.numel() for x in qp.parameters())/1e6:.1f}M "
          f"(+ TextTarget.proj {sum(x.numel() for x in tt.proj.parameters())/1e6:.1f}M)", flush=True)

    opt = torch.optim.AdamW(list(qp.parameters()) + list(tt.proj.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def infinite(dl):
        while True:
            for b in dl:
                yield b

    it_v = infinite(dl_v) if dl_v is not None else None
    it_a = infinite(dl_a)
    best = -1.0; log: List[Dict] = []; t0 = time.time()

    for step in range(args.steps):
        use_v = (it_v is not None) and (rng.random() < args.p_vggsound)
        batch = next(it_v if use_v else it_a)
        fields = batch["fields"]; K = len(fields)

        src, msk = build_sources(batch, m2, device, names)
        B = next(iter(src.values())).shape[0]

        ask = [int(rng.integers(0, K)) for _ in range(B)]
        with torch.no_grad():
            qe = tt.encode_text_frozen_raw(
                [get_query(fields[f], rng, train=True) for f in ask]).to(device)

        z_q = qp(src, qe, msk)                                   # (B, 1536)
        z_t = tt.encode_text([c for caps in batch["caps"] for c in caps])   # (B*K, 1536)
        logits = (z_q @ z_t.T) / args.temp                       # (B, B*K)
        target = torch.tensor([i * K + ask[i] for i in range(B)], device=device)
        loss = F.cross_entropy(logits, target)
        if args.lambda_within > 0.0:
            # separate K-way term over ONLY this clip's own captions, so
            # query-sensitivity cannot be diluted as the batch grows --
            # see train_query_predictor_ddp.gathered_query_loss for the mechanism
            logits_w = torch.einsum("bd,bkd->bk", z_q, z_t.view(B, K, -1)) / args.temp
            loss = loss + args.lambda_within * F.cross_entropy(
                logits_w, torch.tensor(ask, device=device))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(qp.parameters()) + list(tt.proj.parameters()), 1.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            acc = float((logits.argmax(1) == target).float().mean())
            print(f"[qp] step {step:5d}/{args.steps} [{'vgg' if use_v else 'a100m'}] "
                  f"loss={loss.item():.4f} acc={acc:.3f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            mv = evaluate(qp, tt, m2, ev_v, VGGSOUND_FIELDS, device, rng, names) if ev_v else {
                "within_clip_acc": float("nan"), "within_clip_chance": float("nan"),
                "swapped_query_acc": float("nan"), "cross_clip_r1": float("nan")}
            ma = evaluate(qp, tt, m2, ev_a, ACTION100M_FIELDS, device, rng, names)
            m = {"step": step, "vggsound": mv, "action100m": ma}
            log.append(m)
            print(f"[qp] EVAL {step}: VGG within={mv['within_clip_acc']:.3f} "
                  f"(chance {mv['within_clip_chance']:.3f} swap {mv['swapped_query_acc']:.3f}) "
                  f"R@1={mv['cross_clip_r1']:.3f} | A100M within={ma['within_clip_acc']:.3f} "
                  f"(chance {ma['within_clip_chance']:.3f} swap {ma['swapped_query_acc']:.3f}) "
                  f"R@1={ma['cross_clip_r1']:.3f}", flush=True)
            score = (ma["within_clip_acc"] + ma["cross_clip_r1"]) if ev_v is None else \
                    (mv["within_clip_acc"] + mv["cross_clip_r1"])
            if score > best:
                best = score
                torch.save({"step": step, "query_predictor": qp.state_dict(),
                            "text_target_proj": tt.proj.state_dict(),
                            "cfg": vars(args), "metrics": m},
                           os.path.join(args.out_dir, "best.pt"))

    torch.save({"step": args.steps, "query_predictor": qp.state_dict(),
                "text_target_proj": tt.proj.state_dict(), "cfg": vars(args), "log": log},
               os.path.join(args.out_dir, "last.pt"))
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[qp] DONE best_score={best:.4f} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
