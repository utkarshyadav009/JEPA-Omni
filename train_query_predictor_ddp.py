"""train_query_predictor_ddp.py — DDP + GradCache for the query predictor.

WHY: cross-clip R@1 is the only query-predictor metric still short, and it is
NEGATIVE-COUNT bound, not step bound. Measured directly: the single-GPU unified run
plateaued at R@1 ~0.46 from step ~2000 (0.462/0.449/0.458/0.466/0.458 over the last
five evals), so more steps are exhausted; and batch 288 OOMed on a 95GiB card
(needed 11.5GiB on top of 90.2GiB), so a bigger single batch is capped between 192
and 288. DDP+GradCache is the only remaining lever.

The other metric, within-clip query-sensitivity, is deliberately NOT expected to move:
its candidate set is fixed at K captions per clip regardless of batch, and it is already
0.897 against a 0.167 chance level with the swapped-query control at 0.004.

MECHANISM (3-phase GradCache, adapted from train_m2_embed_predictor.py's proven
implementation rather than reinvented):
  Phase 1 (no grad): forward each micro-batch, cache z_q and z_t.
  Phase 2: concat into leaf tensors, all_gather z_t across ranks with the
           GRAD-PRESERVING torch.distributed.nn.functional.all_gather, compute the
           true global loss, backward once to get d(loss)/d(local embeddings).
  Phase 3 (with grad): replay each micro-batch and backward a surrogate dot-product
           against its slice of that cached gradient. Routes exactly the gradient a
           single giant all-ranks backward would, while never holding more than one
           micro-batch of activations.

Effective global batch = n_micro x micro_batch x world_size.

TWO CORRECTNESS REQUIREMENTS specific to this trainer:
  * **All ranks must draw from the SAME corpus on a given step.** VGGSound has K=6
    fields and Action100M K=2; if ranks disagreed the gathered candidate matrix would
    be ragged and the target indices wrong. The corpus choice is therefore derived
    from the STEP NUMBER via a shared seed, not from per-rank randomness.
  * **The asked field and query phrasing must be identical between Phase 1 and
    Phase 3**, or the replayed forward would not match the cached gradient. Both are
    sampled ONCE per micro-batch before Phase 1 and reused.

Also broadcasts qp + text_target.proj from rank 0 at startup. This guards a real bug
this project already hit once (commit 0efdbf5 in train_m2.py): sync_grads averages
GRADIENTS every step but does NOT fix divergent random INITIALISATION, so without an
explicit broadcast each rank would train its own uncoordinated copy forever.

Usage:
    torchrun --nproc_per_node=4 train_query_predictor_ddp.py \
        --token-sources m2,vision,ambient --micro-batch 64 --n-micro 4 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import train_m3
from train_m3 import build_splits, CACHE_DIR
from train_m2_embed_predictor import load_action100m_splits, ACTION100M_CACHE_DIR
from train_query_predictor import (QueryClipDataset, collate, group_by_clip,
                                   build_sources, evaluate)
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.query_predictor import (QueryPredictor, QueryPredictorConfig, get_query,
                                    VGGSOUND_FIELDS, ACTION100M_FIELDS)
from models.text_target import TextTarget, build_text_target
from utils import is_distributed, get_rank, get_local_rank, get_world_size, is_main_process


def setup() -> torch.device:
    if is_distributed():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
            return torch.device("cuda", get_local_rank())
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_grads(modules: List[nn.Module]) -> None:
    if not (is_distributed() and get_world_size() > 1):
        return
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = _flatten_dense_tensors([p.grad for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= get_world_size()
    for p, s in zip(params, _unflatten_dense_tensors(flat, [p.grad for p in params])):
        p.grad.copy_(s)


def broadcast_module(m: nn.Module) -> None:
    if not (is_distributed() and get_world_size() > 1):
        return
    for p in m.parameters():
        dist.broadcast(p.data, src=0)
    for b in m.buffers():
        dist.broadcast(b.data, src=0)


def gathered_query_loss(z_q_local, z_t_local, K: int, ask: List[int], temperature: float,
                        lambda_within: float = 0.0):
    """z_q_local (B,D); z_t_local (B*K,D). Candidates are gathered across ranks so the
    cross-clip negative pool is the GLOBAL batch. Global candidate index of this rank's
    clip i, field f = rank*(B*K) + i*K + f.

    TWO-TERM LOSS (lambda_within > 0) -- fixes a measured regression, not a guess.
    Scaling the batch 96 -> 1024 moved cross-clip R@1 0.458 -> 0.715 but cost within-clip
    query-sensitivity 0.897 -> 0.857. The mechanism: within-clip negatives (the anchor
    clip's OTHER K-1 captions) are the only pressure forcing the query to matter, and they
    are a vanishing fraction of the candidate pool as batch grows --
        B=96,   K=6: 5/576  = 0.87% of candidates
        B=1024, K=6: 5/6144 = 0.08% of candidates   (~10x dilution)
    so one softmax lets the cross-clip objective swamp them.

    The fix is to stop making a single softmax arbitrate between two different jobs:
      L = L_global  (over ALL gathered candidates -> scene grounding)
        + lambda_within * L_within  (over ONLY this clip's K captions -> query-sensitivity)
    The within term is K-way regardless of batch size, so its gradient share cannot be
    diluted by scaling. It is also purely LOCAL (no gather), which is what keeps it
    compatible with GradCache: it is a function of the same leaf tensors, so Phase 3's
    replay needs no change at all.

    lambda_within=0.0 reproduces the original single-softmax loss exactly, so every
    number already recorded stays reproducible."""
    B = z_q_local.shape[0]
    if is_distributed() and get_world_size() > 1:
        import torch.distributed.nn as dist_nn
        z_t_global = torch.cat(dist_nn.functional.all_gather(z_t_local), dim=0)
        rank = get_rank()
        assert z_t_global.shape[0] == B * K * get_world_size(), (
            f"ragged gather: {z_t_global.shape[0]} != {B*K}*{get_world_size()} -- "
            "ranks disagreed on batch size or corpus (K)"
        )
    else:
        z_t_global, rank = z_t_local, 0
    logits = (z_q_local @ z_t_global.t()) / temperature
    target = torch.tensor([rank * (B * K) + i * K + ask[i] for i in range(B)],
                          device=z_q_local.device)
    loss_global = F.cross_entropy(logits, target)
    loss = loss_global
    metrics = {"global_cands": int(z_t_global.shape[0])}

    if lambda_within > 0.0:
        # (B,K) logits against ONLY this clip's own captions -- local, no gather
        z_t_own = z_t_local.view(B, K, -1)
        logits_w = torch.einsum("bd,bkd->bk", z_q_local, z_t_own) / temperature
        target_w = torch.tensor(ask, device=z_q_local.device)
        loss_within = F.cross_entropy(logits_w, target_w)
        loss = loss_global + lambda_within * loss_within
        with torch.no_grad():
            metrics["loss_within"] = float(loss_within)
            metrics["acc_within"] = (logits_w.argmax(1) == target_w).float().mean().item()

    with torch.no_grad():
        metrics["acc"] = (logits.argmax(1) == target).float().mean().item()
        metrics["loss_global"] = float(loss_global)
    return loss, metrics


def gradcache_step(qp, tt, m2, micros, names, K, temperature, device, rng_shared,
                   lambda_within: float = 0.0):
    """3-phase GradCache. `micros` is a list of collated batches for THIS rank."""
    # sample asks/queries ONCE -- phase 3 must replay phase 1 exactly
    plans = []
    for b in micros:
        Bm = len(b["clip_id"])
        ask = [int(rng_shared.integers(0, K)) for _ in range(Bm)]
        qtext = [get_query(b["fields"][f], rng_shared, train=True) for f in ask]
        plans.append((ask, qtext))

    zq_c, zt_c = [], []
    with torch.no_grad():
        for b, (ask, qtext) in zip(micros, plans):
            src, msk = build_sources(b, m2, device, names)
            qe = tt.encode_text_frozen_raw(qtext).to(device)
            zq_c.append(qp(src, qe, msk))
            zt_c.append(tt.encode_text([c for caps in b["caps"] for c in caps]))

    z_q = torch.cat(zq_c, 0).detach().requires_grad_(True)
    z_t = torch.cat(zt_c, 0).detach().requires_grad_(True)
    ask_all = [a for ask, _ in plans for a in ask]
    loss, metrics = gathered_query_loss(z_q, z_t, K, ask_all, temperature, lambda_within)
    loss.backward()
    g_q, g_t = z_q.grad.detach(), z_t.grad.detach()

    off_q = off_t = 0
    for b, (ask, qtext) in zip(micros, plans):
        Bm = len(b["clip_id"])
        src, msk = build_sources(b, m2, device, names)
        qe = tt.encode_text_frozen_raw(qtext).to(device)
        zq_i = qp(src, qe, msk)
        zt_i = tt.encode_text([c for caps in b["caps"] for c in caps])
        surrogate = (zq_i.float() * g_q[off_q:off_q + Bm]).sum() \
                  + (zt_i.float() * g_t[off_t:off_t + Bm * K]).sum()
        surrogate.backward()
        off_q += Bm; off_t += Bm * K
    return float(loss.detach()), metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--out-dir", default="checkpoints/query_predictor_ddp")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--n-micro", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--text-backbone", default="embeddinggemma")
    p.add_argument("--token-sources", default="m2,vision,ambient")
    p.add_argument("--p-vggsound", type=float, default=0.5)
    p.add_argument("--captions-path", default=os.path.join(PROJECT_ROOT, "scripts",
                   "qwen_omni_full_captions_v2.jsonl"))
    p.add_argument("--lambda-within", type=float, default=0.0,
                   help="weight on the separate within-clip (K-way) cross-entropy term. "
                        "0.0 = original single-softmax loss (reproduces all recorded runs). "
                        "Try 0.3-1.0 at batch 1024 to recover the query-sensitivity that "
                        "batch scaling diluted; see gathered_query_loss's docstring.")
    p.add_argument("--seed", type=int, default=0)
    # ── scene stream (SigLIP2), ported from train_query_predictor.py ──
    p.add_argument("--scene-dir", default="/dev/shm/scene_all",
                   help="dir of {clip_id: (K,768)} SigLIP2 shards for the 'scene' stream")
    p.add_argument("--restrict-to-scene", action="store_true",
                   help="keep only scene-covered clips, so every arm trains on ONE pool")
    p.add_argument("--audio-mode", default="mean", choices=["mean", "base"],
                   help="'mean' = WavJEPA base+nat (what M2 was trained on) | 'base' = base "
                        "only, which drops the 469 ms nat forward on the Jetson")
    # ── frozen SigLIP2 text space (see models/text_target.py::SigLIP2TextTarget) ──
    p.add_argument("--siglip-shared-dim", type=int, default=0,
                   help="dim of the TRAINABLE projection on top of the frozen SigLIP2 text "
                        "tower. 0 = Identity, which was trained head-to-head and LOST "
                        "(within-clip 0.654 vs 0.883); see SigLIP2TextTarget's docstring. "
                        "The projection can be applied offline at pre-encode time, so it "
                        "costs no on-device memory.")
    p.add_argument("--text-cache-dir", default="",
                   help="dir of pre-encoded {text, emb} shards from "
                        "scripts/encode_captions_siglip2.py. Only valid with a siglip "
                        "backbone (the space must match). Skips the text tower entirely.")
    args = p.parse_args()
    names = [x.strip() for x in args.token_sources.split(",") if x.strip()]
    train_m3.CAPTIONS_PATH = args.captions_path

    device = setup()
    torch.manual_seed(args.seed); random.seed(args.seed)
    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)

    vtr = group_by_clip(build_splits(VGGSOUND_FIELDS)[0], VGGSOUND_FIELDS)
    vte = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    atr_p, ate_p = [], []
    for f in ACTION100M_FIELDS:
        tr_p, te_p = load_action100m_splits(f)
        atr_p += tr_p; ate_p += te_p
    atr = group_by_clip(atr_p, ACTION100M_FIELDS)
    ate = group_by_clip(ate_p, ACTION100M_FIELDS)
    if is_main_process():
        print(f"[ddp] VGGSound train={len(vtr)} test={len(vte)} | "
              f"Action100M train={len(atr)} test={len(ate)}", flush=True)

    scene = None
    if "scene" in names or args.restrict_to_scene:
        import glob as _g
        scene = {}
        for sp in sorted(_g.glob(os.path.join(args.scene_dir, "*.pt"))):
            scene.update(torch.load(sp, map_location="cpu", weights_only=False))
        if is_main_process():
            print(f"[ddp] scene stream: {len(scene)} clips with SigLIP2 features "
                  f"from {args.scene_dir}", flush=True)
        # A previous run silently trained on ZERO scene clips because the loader globbed
        # 'shard*.pt' while the VGGSound shards were named 'vgg_shard*.pt'. Fail loudly.
        if not scene:
            raise RuntimeError(f"no scene features found under {args.scene_dir}")

    if args.restrict_to_scene:
        n0 = (len(vtr), len(atr))
        vtr = {k: v for k, v in vtr.items() if k in scene}
        vte = {k: v for k, v in vte.items() if k in scene}
        atr = {k: v for k, v in atr.items() if k in scene}
        ate = {k: v for k, v in ate.items() if k in scene}
        if is_main_process():
            print(f"[ddp] restricted to scene-covered clips: VGGSound {n0[0]}->{len(vtr)} "
                  f"Action100M {n0[1]}->{len(atr)}", flush=True)
        if not vtr or not atr:
            raise RuntimeError("restrict-to-scene emptied a corpus -- check scene coverage")

    def mk(clips, cd, fl, shuffle):
        ds = QueryClipDataset(clips, cd, fl, scene_feats=scene, audio_mode=args.audio_mode)
        smp = DistributedSampler(ds, shuffle=shuffle, drop_last=True) if is_distributed() else None
        return DataLoader(ds, batch_size=args.micro_batch, sampler=smp,
                          shuffle=(smp is None and shuffle), num_workers=6,
                          collate_fn=collate, drop_last=True, pin_memory=True,
                          persistent_workers=True), smp

    dl_v, smp_v = mk(vtr, CACHE_DIR, VGGSOUND_FIELDS, True)
    dl_a, smp_a = mk(atr, ACTION100M_CACHE_DIR, ACTION100M_FIELDS, True)
    ev_v = DataLoader(QueryClipDataset(vte, CACHE_DIR, VGGSOUND_FIELDS, scene_feats=scene,
                                       audio_mode=args.audio_mode), batch_size=48,
                      shuffle=False, num_workers=4, collate_fn=collate)
    ev_a = DataLoader(QueryClipDataset(ate, ACTION100M_CACHE_DIR, ACTION100M_FIELDS,
                                       scene_feats=scene, audio_mode=args.audio_mode),
                      batch_size=48, shuffle=False, num_workers=4, collate_fn=collate)

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()
    for prm in m2.parameters():
        prm.requires_grad_(False)

    tt = build_text_target(args.text_backbone, shared_dim=1536, device=str(device),
                           siglip_shared_dim=(args.siglip_shared_dim or None))
    is_siglip = args.text_backbone.startswith("siglip")
    if args.text_cache_dir:
        if not is_siglip:
            raise ValueError("--text-cache-dir holds SigLIP2-space vectors; it is only valid "
                             "with a siglip backbone")
        import glob as _g
        ctexts, cvecs = [], []
        for sp in sorted(_g.glob(os.path.join(args.text_cache_dir, "text_shard*.pt"))):
            d = torch.load(sp, map_location="cpu", weights_only=False)
            if d.get("siglip") and d["siglip"] != tt.repo:
                raise RuntimeError(f"text cache built with {d['siglip']} but target is {tt.repo}")
            ctexts += d["text"]; cvecs.append(d["emb"])
        if not ctexts:
            raise RuntimeError(f"no text shards under {args.text_cache_dir}")
        tt.load_cache(ctexts, torch.cat(cvecs, 0))
        if is_main_process():
            print(f"[ddp] text cache: {len(ctexts)} pre-encoded captions "
                  f"({torch.cat(cvecs,0).numel()*2/2**20:.0f} MiB) -- text tower idle", flush=True)

    # shared_dim now comes from the TARGET, not a constant: SigLIP2's frozen joint space is
    # 768-d and cannot be re-projected without leaving the image tower's space.
    shared = tt.shared_dim
    SRC = {"m2": m2cfg.d_model, "vision": 1024, "ambient": 768, "scene": 768}
    qp = QueryPredictor(QueryPredictorConfig(source_dims={s: SRC[s] for s in names},
                                             query_dim=tt.native_dim, shared_dim=shared)).to(device)
    broadcast_module(qp); broadcast_module(tt.proj)      # see module docstring
    if is_main_process():
        print(f"[ddp] target space: {args.text_backbone} dim={shared} "
              f"trainable_text_params={sum(p.numel() for p in tt.proj.parameters())}", flush=True)
    if is_main_process():
        eff = args.micro_batch * args.n_micro * max(1, get_world_size())
        print(f"[ddp] sources={names} world={get_world_size()} "
              f"EFFECTIVE GLOBAL BATCH={eff} clips", flush=True)

    opt = torch.optim.AdamW(list(qp.parameters()) + list(tt.proj.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def inf(dl, smp):
        ep = 0
        while True:
            if smp is not None:
                smp.set_epoch(ep)
            for b in dl:
                yield b
            ep += 1

    it_v, it_a = inf(dl_v, smp_v), inf(dl_a, smp_a)
    best = -1.0; log = []; t0 = time.time()
    rng_eval = np.random.default_rng(args.seed)

    for step in range(args.steps):
        # corpus choice derived from the STEP so every rank agrees on K
        step_rng = np.random.default_rng(args.seed * 100003 + step)
        use_v = step_rng.random() < args.p_vggsound
        it = it_v if use_v else it_a
        fields = VGGSOUND_FIELDS if use_v else ACTION100M_FIELDS
        micros = [next(it) for _ in range(args.n_micro)]

        opt.zero_grad(set_to_none=True)
        loss, metrics = gradcache_step(qp, tt, m2, micros, names, len(fields),
                                       args.temp, device, step_rng, args.lambda_within)
        sync_grads([qp, tt.proj])
        torch.nn.utils.clip_grad_norm_(list(qp.parameters()) + list(tt.proj.parameters()), 1.0)
        opt.step(); sched.step()

        if step % 50 == 0 and is_main_process():
            extra = (f" within={metrics['acc_within']:.3f}" if "acc_within" in metrics else "")
            print(f"[ddp] step {step:5d}/{args.steps} [{'vgg' if use_v else 'a100m'}] "
                  f"loss={loss:.4f} acc={metrics['acc']:.3f}{extra} "
                  f"cands={metrics['global_cands']} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)

        if ((step + 1) % args.eval_every == 0 or step + 1 == args.steps) and is_main_process():
            mv = evaluate(qp, tt, m2, ev_v, VGGSOUND_FIELDS, device, rng_eval, names)
            ma = evaluate(qp, tt, m2, ev_a, ACTION100M_FIELDS, device, rng_eval, names)
            log.append({"step": step, "vggsound": mv, "action100m": ma})
            print(f"[ddp] EVAL {step}: VGG within={mv['within_clip_acc']:.3f} "
                  f"(chance {mv['within_clip_chance']:.3f} swap {mv['swapped_query_acc']:.3f}) "
                  f"R@1={mv['cross_clip_r1']:.3f} | A100M within={ma['within_clip_acc']:.3f} "
                  f"R@1={ma['cross_clip_r1']:.3f}", flush=True)
            sc = mv["within_clip_acc"] + mv["cross_clip_r1"]
            if sc > best:
                best = sc
                torch.save({"step": step, "query_predictor": qp.state_dict(),
                            "text_target_proj": tt.proj.state_dict(),
                            "cfg": vars(args), "metrics": log[-1]},
                           os.path.join(args.out_dir, "best.pt"))
        if is_distributed():
            dist.barrier()

    if is_main_process():
        with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"[ddp] DONE best={best:.4f} -> {args.out_dir}", flush=True)
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
