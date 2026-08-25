"""train_m2_embed_predictor.py — Phase 1 of the VL-JEPA-style embedding
predictor plan (see /home/utkarsh/.claude/plans/serene-soaring-abelson.md).

Adopts VL-JEPA's (arXiv 2512.10942) training OBJECTIVE and inference PATTERN
-- predict a continuous text embedding via InfoNCE instead of autoregressive
token generation -- WITHOUT assuming its heavy predictor architecture wins
here. This project already ran that exact comparison once, at the M1 stage:
models/predictor.py's "llama_last8" mode (last 8 layers of an LLM, the
literal VL-JEPA recipe) LOST to the simple "mlp" mode (m1_experiment_results.md:
R@1 13.9-16.2% llama_last8 vs 16.8% mlp at matched scale; the eventual M1
winner at 22.5% R@1 was mlp scaled up, not llama_last8). Real VL-JEPA needed
3.3B training samples at batch 24k to hit its reported numbers -- nowhere
near what's feasible here. So: run BOTH modes head-to-head on real held-out
data before picking one, per direct instruction, rather than assuming either
wins.

Frozen: V-JEPA2 ViT-L + WavJEPA-base/nat (via the locked M2 AVJepaPredictor's
encode_pre_pool_tokens -- the SAME frozen pre-pool token sequence
M3Connector already consumes, no new fusion code).
Trainable: a NEW models/predictor.py Predictor instance (mode=mlp or
llama_last8) consuming those pre-pool tokens, and models/text_target.py's
TextTarget (Y-Encoder, EmbeddingGemma-300M preset, matching VL-JEPA's
Y-Encoder exactly) -- trained jointly with bidirectional InfoNCE
(models/losses.py's info_nce, already used by M1).

Data: VGGSound only for this Phase-1 comparison (existing cache + captions,
no extraction blocker) -- reuses train_m3.py's build_splits/GRANULARITY_TAGS/
CACHE_DIR exactly. Ego4D has no caption/text supervision in this project (used
only for M2's separate cross-modal objective) so it is not part of this
text-InfoNCE training track. Action100M is Phase 2 (separate extraction
script), added once available.

Usage:
    python train_m2_embed_predictor.py --predictor-mode mlp --n-clips 8000 --steps 3000
    python train_m2_embed_predictor.py --predictor-mode llama_last8 --n-clips 8000 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import hashlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, GRANULARITY_TAGS, CACHE_DIR, _cap_ambient_len
from data.av_cached_dataset import AVCachedDataset
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.predictor import Predictor
from models.text_target import TextTarget
from models.losses import info_nce
from utils import is_distributed, get_rank, get_local_rank, get_world_size, is_main_process

ACTION100M_CACHE_DIR = "/home/utkarsh/raid2-data/feature_cache_action100m"
ACTION100M_CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "scripts", "action100m_captions.jsonl")


# ── Distributed + GradCache (2026-08-02): scale effective batch to 1000+ ────
# Directly requested per the batch-size finding (M1's own history: batch
# 8->256 was the single biggest lever, 16.8%->22.5% R@1; independently
# confirmed via literature -- DPR ablation batch 16->128 = +3pp top-20 acc,
# modern embedders train at 8k-32k batch specifically because gains don't
# plateau). This project ALREADY has a proven GradCache+DDP implementation
# for exactly this problem (train_m2.py's gradcache_contrastive_step +
# gathered_info_nce) -- adapted here for (predictor, text_target) instead of
# (vision_proj, ambient_proj). Memory-tested empirically (2026-08-02): batch
# 256/384 fit in one forward on this GPU, 512+ OOM -- GradCache lets us use
# a much larger LOGICAL batch by chunking into micro-batches that never
# exceed the safe single-forward size, while DDP across 4 GPUs multiplies
# the effective global batch further (4 ranks x logical-per-rank batch).

def setup_distributed() -> torch.device:
    use_cuda = torch.cuda.is_available()
    if is_distributed():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(get_local_rank())
            return torch.device("cuda", get_local_rank())
        return torch.device("cpu")
    return torch.device("cuda" if use_cuda else "cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_grads(module: nn.Module) -> None:
    """Manual grad all-reduce (matches train_m2.py's sync_grads exactly) --
    module.parameters() must already be identical across ranks (broadcast
    from rank 0 at construction, via each rank loading the SAME checkpoint/
    initial random state) for this to converge correctly."""
    if not (is_distributed() and get_world_size() > 1):
        return
    params = list(module.parameters())
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = _flatten_dense_tensors([p.grad for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= get_world_size()
    for p, synced in zip(params, _unflatten_dense_tensors(flat, [p.grad for p in params])):
        p.grad.copy_(synced)


def _soft_targets_from_text_sim(sim_anchor: torch.Tensor, sim_candidates: torch.Tensor,
                                 self_offset: int, soft_temp: float) -> torch.Tensor:
    """Builds soft cross-entropy targets from caption-to-caption cosine similarity
    (inputs must already be L2-normalized, so the dot product below IS cosine
    similarity -- no extra normalize() needed).

    FOUND (2026-08-03, per direct instruction after researching Soft-InfoNCE):
    hard one-hot InfoNCE targets treat every non-matching caption in the batch
    as an equally-wrong negative. This project already found, independently,
    that many captions are near-duplicates across DIFFERENT clips (the
    gpt_action_brief-field finding that motivated the switch to
    gpt_action_detailed: many different video segments could plausibly share
    the same generic short caption) -- those near-duplicates get punished as
    false negatives under a hard target. Using a similarity source to build a
    soft label directly measures "how similar is this candidate's real caption
    to the anchor's real caption" and lets genuinely-similar-but-different
    captions receive partial credit instead of being penalised as if wrong.
    The true anchor position is self-similarity = 1.0, always the maximum, so
    softmax naturally keeps it dominant -- this softens false-negative
    pressure, it does not replace the correct-answer signal.

    CORRECTION (2026-08-04): the original version of this docstring called the
    caller's embeddings "frozen, stable" -- that was WRONG when sim_* comes
    from TextTarget.encode_text(), whose output passes through the trainable
    .proj head (one of this project's own trained modules). See
    TextTarget.encode_text_frozen_raw() for the actually-frozen variant, added
    after checking the reference Soft-InfoNCE implementation
    (github.com/Alex-HaochenLi/Soft-InfoNCE, EMNLP 2023) which deliberately
    uses an EXTERNAL frozen source (BM25/SimCSE/fixed checkpoint) specifically
    to avoid the circularity of a model influencing its own soft targets.
    Both variants are wired through gathered_info_nce_embed's
    soft_infonce_frozen_sim flag -- this function itself is agnostic to which
    embedding source it's given.

    soft_temp is a SEPARATE, typically much colder temperature than the main
    InfoNCE temperature -- it controls how much mass leaks to near-duplicates.
    Too warm (large) blurs the whole target toward uniform; too cold
    (small) collapses back to near-one-hot. Default chosen colder than the
    main temperature (0.05 vs 0.07) so only genuinely close captions get
    meaningful mass.
    """
    with torch.no_grad():
        text_sim = sim_anchor @ sim_candidates.t()             # (B_local, global_B), true cosine sim
        soft = torch.softmax(text_sim / soft_temp, dim=-1)
    return soft


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """KL-style soft-label cross-entropy: -sum_j target[i,j] * log_softmax(logits)[i,j],
    averaged over the batch. Reduces to standard F.cross_entropy when
    soft_targets is one-hot."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def gathered_info_nce_embed(z_p_local: torch.Tensor, z_t_local: torch.Tensor, temperature: float,
                             soft_infonce: bool = False, soft_temp: float = 0.05,
                             sim_local: torch.Tensor = None):
    """Same as train_m2.py's gathered_info_nce, adapted for (predicted
    embedding, text embedding) instead of (vision, ambient) -- structurally
    identical: two aligned embedding streams from the same batch. Negatives
    gathered across ALL DDP ranks via the grad-preserving
    torch.distributed.nn.functional.all_gather. Falls back to plain in-batch
    info_nce when not running distributed and soft_infonce is off.

    soft_infonce=True replaces the hard one-hot cross-entropy target with a
    soft target built from caption-to-caption similarity (see
    _soft_targets_from_text_sim) -- see that function's docstring for why.

    sim_local: optional (B_local, D_sim) tensor to use as the similarity
    source instead of z_t_local -- pass TextTarget.encode_text_frozen_raw()'s
    output here for the genuinely-frozen variant (see that method's
    docstring for why this differs from just reusing z_t_local, which passes
    through the trainable .proj head). If None, falls back to z_t_local
    (the original, simpler, NOT-fully-frozen variant)."""
    if not (is_distributed() and get_world_size() > 1):
        if not soft_infonce:
            return info_nce(z_p_local, z_t_local, temperature=temperature)
        z_p_global, z_t_global = z_p_local, z_t_local
        sim_global = sim_local if sim_local is not None else z_t_local
        rank, B_local = 0, z_p_local.shape[0]
    else:
        import torch.distributed.nn as dist_nn
        rank = get_rank()
        B_local = z_p_local.shape[0]

        z_p_global = torch.cat(dist_nn.functional.all_gather(z_p_local), dim=0)
        z_t_global = torch.cat(dist_nn.functional.all_gather(z_t_local), dim=0)
        global_B = z_p_global.shape[0]

        expected_global_B = B_local * get_world_size()
        assert global_B == expected_global_B, (
            f"ragged gather: global_B={global_B} != {B_local}*{get_world_size()}="
            f"{expected_global_B} -- a rank dropped or sent a different local batch size"
        )

        if soft_infonce:
            if sim_local is not None:
                sim_global = torch.cat(dist_nn.functional.all_gather(sim_local), dim=0)
            else:
                sim_global = z_t_global

    labels = torch.arange(B_local, device=z_p_local.device) + rank * B_local
    logits_p2t = (z_p_local @ z_t_global.t()) / temperature
    logits_t2p = (z_t_local @ z_p_global.t()) / temperature

    if soft_infonce:
        # Soft targets built from caption-to-caption similarity -- symmetric
        # use: p2t's candidates are the global similarity pool, t2p's
        # candidates are the same pool (comparing text-to-text both
        # directions is the correct symmetric relation; there is no separate
        # "predicted-embedding" similarity structure to lean on here, by
        # design -- see _soft_targets_from_text_sim's docstring).
        sim_anchor_local = sim_local if sim_local is not None else z_t_local
        soft_p2t = _soft_targets_from_text_sim(sim_anchor_local, sim_global, rank * B_local, soft_temp)
        soft_t2p = _soft_targets_from_text_sim(sim_anchor_local, sim_global, rank * B_local, soft_temp)
        loss_p2t = _soft_cross_entropy(logits_p2t, soft_p2t)
        loss_t2p = _soft_cross_entropy(logits_t2p, soft_t2p)
    else:
        loss_p2t = F.cross_entropy(logits_p2t, labels)
        loss_t2p = F.cross_entropy(logits_t2p, labels)
    loss = 0.5 * (loss_p2t + loss_t2p)

    with torch.no_grad():
        acc_v2t = (logits_p2t.argmax(dim=1) == labels).float().mean().item()
        acc_t2v = (logits_t2p.argmax(dim=1) == labels).float().mean().item()

    return loss, {"acc_v2t": acc_v2t, "acc_t2v": acc_t2v, "global_B": B_local * max(1, get_world_size() if is_distributed() else 1)}


def gradcache_step(predictor, text_target, m2, micro_batches, temperature: float, device,
                    soft_infonce: bool = False, soft_temp: float = 0.05,
                    soft_infonce_frozen_sim: bool = False):
    """GradCache composed with gathered_info_nce_embed -- lets the EFFECTIVE
    per-rank batch (and thus, with DDP, the global batch) exceed what fits
    in one forward pass, by chunking into micro_batches.

    Phase 1 (no grad): forward each microbatch through frozen M2 + trainable
    predictor + trainable text_target, cache embeddings.
    Phase 2: concat this rank's cached embeddings into ONE leaf tensor, run
    gathered_info_nce_embed (differentiable all_gather -> true global loss),
    backward ONCE to get the target gradient w.r.t. this rank's local embeddings.
    Phase 3 (with grad): replay each microbatch's forward again, backward a
    surrogate dot-product against its slice of the target gradient -- routes
    the exact same gradient signal into predictor/text_target's parameters
    as one giant all-ranks backward would, without ever holding more than
    one microbatch's activations at a time.

    soft_infonce_frozen_sim=True computes TextTarget.encode_text_frozen_raw()
    once during Phase 1 and reuses it for the soft-target similarity source
    (never needs Phase-3 replay -- it's always no_grad and doesn't depend on
    any trainable parameter, unlike z_t which passes through the trainable
    .proj head and DOES need the full 3-phase treatment).

    Does NOT call sync_grads() -- caller must do that exactly once after this returns.
    """
    zp_chunks, zt_chunks, zt_frozen_chunks = [], [], []
    with torch.no_grad():
        for batch in micro_batches:
            feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
            tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pre_pool = m2.encode_pre_pool_tokens(feats, tbins)
            z_p = predictor(pre_pool)
            z_t = text_target.encode_text(batch["caption_text"])
            zp_chunks.append(z_p)
            zt_chunks.append(z_t)
            if soft_infonce and soft_infonce_frozen_sim:
                zt_frozen_chunks.append(text_target.encode_text_frozen_raw(batch["caption_text"]))

    z_p_local = torch.cat(zp_chunks, 0).detach().requires_grad_(True)
    z_t_local = torch.cat(zt_chunks, 0).detach().requires_grad_(True)
    sim_local = torch.cat(zt_frozen_chunks, 0) if zt_frozen_chunks else None

    loss, metrics = gathered_info_nce_embed(z_p_local, z_t_local, temperature=temperature,
                                             soft_infonce=soft_infonce, soft_temp=soft_temp,
                                             sim_local=sim_local)
    loss.backward()
    target_grad_p = z_p_local.grad.detach()
    target_grad_t = z_t_local.grad.detach()

    offset = 0
    for batch in micro_batches:
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        b = feats["vision"].shape[0]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pre_pool = m2.encode_pre_pool_tokens(feats, tbins)
        z_p_i = predictor(pre_pool)
        z_t_i = text_target.encode_text(batch["caption_text"])
        surrogate = (z_p_i.float() * target_grad_p[offset:offset + b]).sum() \
                  + (z_t_i.float() * target_grad_t[offset:offset + b]).sum()
        surrogate.backward()
        offset += b

    return float(loss.detach()), metrics


def load_action100m_splits(field: str, test_frac: float = 0.1) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Returns (train_pairs, test_pairs) of (clip_id, field, caption_text)
    from scripts/action100m_captions.jsonl (written by
    scripts/extract_features_action100m.py). No pre-existing train/test CSV
    convention for Action100M (unlike VGGSound's data/train.csv/test.csv),
    so split deterministically by hashing video_uid (NOT clip_id) -- keeps
    every segment from the same video on the same side of the split, since
    multiple segments from one video (even non-overlapping ones) share
    visual/audio style and would leak information across train/test if
    split independently."""
    train_pairs, test_pairs = [], []
    if not os.path.exists(ACTION100M_CAPTIONS_PATH):
        return train_pairs, test_pairs
    with open(ACTION100M_CAPTIONS_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["clip_id"]
            uid = r["video_uid"]
            text = r.get(field)
            # FOUND (2026-08-02): 3.4% of gpt_action_brief captions are literal
            # placeholder strings ("N/A"/"NA"/etc), not null -- `if not text`
            # doesn't catch these since a non-empty string is truthy. Pure noise:
            # teaches the model nothing and pollutes the embedding space.
            if not text or text.strip().upper() in ("N/A", "NA", "NONE", "NULL"):
                continue
            h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100
            bucket = test_pairs if h < int(test_frac * 100) else train_pairs
            bucket.append((cid, field, text))
    return train_pairs, test_pairs


class M2EmbedDataset(Dataset):
    """(clip, field, caption) triples -> AV feats/tbins + caption text.
    Mirrors train_m3.py's M3CaptionDataset but without the tokenized
    prefix/caption_ids fields -- this training track has no autoregressive
    decoder, TextTarget does its own tokenization internally."""

    def __init__(self, pairs: List[Tuple[str, str, str]], cache_dir: str, max_tdm_bins: int = 512):
        self.pairs = pairs
        clip_ids = [cid for cid, _, _ in pairs]
        self.av_ds = AVCachedDataset(cache_dir=cache_dir, clip_ids=clip_ids,
                                      max_tdm_bins=max_tdm_bins, audio_mode="mean")
        self.captions = [text for _, _, text in pairs]

    def __len__(self) -> int:
        return len(self.av_ds)

    def __getitem__(self, idx: int) -> Dict:
        item = self.av_ds[idx]
        item["caption_text"] = self.captions[idx]
        return item


def collate_fn(batch: List[Dict], pad_val: int = 0) -> Dict:
    B = len(batch)
    max_v = max(b["feats"]["vision"].shape[0] for b in batch)
    max_a = max(b["feats"]["ambient"].shape[0] for b in batch)
    vis = torch.zeros(B, max_v, batch[0]["feats"]["vision"].shape[-1], dtype=batch[0]["feats"]["vision"].dtype)
    aud = torch.zeros(B, max_a, batch[0]["feats"]["ambient"].shape[-1], dtype=batch[0]["feats"]["ambient"].dtype)
    vis_bins = torch.zeros(B, max_v, dtype=torch.long)
    aud_bins = torch.zeros(B, max_a, dtype=torch.long)
    vis_pad = torch.ones(B, max_v, dtype=torch.bool)
    aud_pad = torch.ones(B, max_a, dtype=torch.bool)
    for i, b in enumerate(batch):
        nv = b["feats"]["vision"].shape[0]
        na = b["feats"]["ambient"].shape[0]
        vis[i, :nv] = b["feats"]["vision"]
        aud[i, :na] = b["feats"]["ambient"]
        vis_bins[i, :nv] = b["tbins"]["vision"]
        aud_bins[i, :na] = b["tbins"]["ambient"]
        vis_pad[i, :nv] = False
        aud_pad[i, :na] = False
    return {
        "feats": {"vision": vis, "ambient": aud},
        "tbins": {"vision": vis_bins, "ambient": aud_bins},
        "padding_mask": {"vision": vis_pad, "ambient": aud_pad},
        "caption_text": [b["caption_text"] for b in batch],
    }


@torch.no_grad()
def retrieval_eval(predictor, text_target, m2, device, loader, max_clips=1000) -> Dict[str, float]:
    predictor.eval(); text_target.base.eval()
    zp_all, zt_all = [], []
    n = 0
    for batch in loader:
        if n >= max_clips:
            break
        feats = {k: v.to(device).float() for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pre_pool = m2.encode_pre_pool_tokens(feats, tbins)
        z_p = predictor(pre_pool)
        z_t = text_target.encode_text(batch["caption_text"])
        zp_all.append(z_p.cpu()); zt_all.append(z_t.cpu())
        n += z_p.shape[0]
    z_p = torch.cat(zp_all, 0)[:max_clips]
    z_t = torch.cat(zt_all, 0)[:max_clips]
    N = z_p.shape[0]
    gt = torch.arange(N)
    sim = z_p @ z_t.T
    results = {}
    for name, ranked in [("pred→text", (-sim).argsort(1)), ("text→pred", (-sim.T).argsort(1))]:
        for k in (1, 5, 10):
            hits = (ranked[:, :k] == gt.unsqueeze(1)).any(1).float().mean().item()
            results[f"{name}_R@{k}"] = round(hits * 100, 2)
    results["n_clips"] = float(N)
    predictor.train(); text_target.base.train(text_target.unfreeze_base)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--predictor-mode", required=True, choices=["mlp", "llama_last8"])
    p.add_argument("--text-backbone", default="embeddinggemma", choices=["embeddinggemma", "minilm"])
    p.add_argument("--field", default="gpt_action_brief", choices=list(GRANULARITY_TAGS.keys()))
    p.add_argument("--n-clips", type=int, default=8000, help="cap on training pairs, for a fast Phase-1 comparison")
    p.add_argument("--batch-size", type=int, default=64,
                   help="MICRO-batch size -- the safe single-forward chunk size (empirically "
                        "tested 2026-08-02: 256/384 fit on one GPU, 512+ OOM). GradCache chains "
                        "--n-microbatches of these together into a larger logical per-rank batch "
                        "without ever holding more than one micro-batch's activations at once.")
    p.add_argument("--n-microbatches", type=int, default=1,
                   help="GradCache chunks: logical per-rank batch = batch_size * n_microbatches. "
                        "With DDP (torchrun --nproc_per_node=N), global batch = that * N ranks. "
                        "1 = plain single-forward step (GradCache/gather still used if distributed).")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--soft-infonce", action="store_true",
                   help="replace the hard one-hot InfoNCE target with a soft target built "
                        "from TRUE caption-to-caption cosine similarity (see "
                        "_soft_targets_from_text_sim docstring) -- mitigates false negatives "
                        "from near-duplicate captions across different clips (the same class "
                        "of issue that motivated switching gpt_action_brief->gpt_action_detailed).")
    p.add_argument("--soft-temp", type=float, default=0.05,
                   help="temperature for the soft-InfoNCE target distribution (separate from "
                        "--temperature, the main InfoNCE logit temperature). Colder = less mass "
                        "leaks to near-duplicate captions.")
    p.add_argument("--soft-infonce-frozen-sim", action="store_true",
                   help="use TextTarget.encode_text_frozen_raw() (pre-proj, genuinely frozen "
                        "EmbeddingGemma embeddings) as the soft-InfoNCE similarity source instead "
                        "of encode_text()'s output (which passes through the trainable .proj head "
                        "and so is NOT fully frozen -- see encode_text_frozen_raw's docstring). "
                        "Matches the anti-circularity design of the reference Soft-InfoNCE "
                        "implementation (github.com/Alex-HaochenLi/Soft-InfoNCE, EMNLP 2023), which "
                        "deliberately uses an external frozen similarity source. No-op if "
                        "--soft-infonce is not also set.")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--use-action100m", action="store_true",
                   help="combine VGGSound with scripts/action100m_captions.jsonl "
                        "(from scripts/extract_features_action100m.py) for training. "
                        "Reports held-out R@1 on VGGSound and Action100M SEPARATELY, "
                        "not just combined -- so we can see whether Action100M actually "
                        "helps or just adds noise, not assume it from more data alone.")
    p.add_argument("--action100m-n-clips", type=int, default=None,
                   help="cap on Action100M training pairs (default: use all available)")
    p.add_argument("--skip-vggsound", action="store_true",
                   help="Action100M-only run (isolates whether Action100M alone can learn the "
                        "task, vs the earlier combined run's confound of batch dilution from "
                        "mixing two very differently-sized datasets). Requires --use-action100m.")
    args = p.parse_args()
    if args.skip_vggsound:
        assert args.use_action100m, "--skip-vggsound requires --use-action100m"

    out_dir = args.out_dir or f"checkpoints/m2_embed_predictor_{args.predictor_mode}"
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)

    device = setup_distributed()
    if is_main_process():
        print(f"[train] device={device} world_size={get_world_size()} "
              f"mode={args.predictor_mode} field={args.field} "
              f"micro_batch={args.batch_size} n_microbatches={args.n_microbatches} "
              f"logical_per_rank_batch={args.batch_size * args.n_microbatches} "
              f"global_batch={args.batch_size * args.n_microbatches * get_world_size()}", flush=True)

    print("[train] loading locked M2 predictor (frozen)...", flush=True)
    m2_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    m2.load_state_dict(m2ckpt["model"], strict=True)
    m2.eval()
    for prm in m2.parameters():
        prm.requires_grad_(False)

    print(f"[train] building trainable Predictor(mode={args.predictor_mode})...", flush=True)
    predictor = Predictor(in_dim=m2_cfg.d_model, shared_dim=1536, mode=args.predictor_mode).to(device)

    print(f"[train] building TextTarget(backbone={args.text_backbone})...", flush=True)
    text_target = TextTarget(backbone=args.text_backbone, shared_dim=1536, unfreeze_base=False, device=str(device))

    if is_distributed() and get_world_size() > 1:
        # FOUND (2026-08-02, via git history -- train_m2.py commit 0efdbf5
        # "Fix GradCache DDP head synchronization"): sync_grads() only
        # averages GRADIENTS every step; it does NOT correct different
        # random initializations across ranks. Each rank builds its own
        # Predictor/TextTarget.proj independently above, so WITHOUT this
        # broadcast, every rank starts from a different random init and
        # never converges -- averaging gradients from divergent weights is
        # not the same as training one shared model. Broadcast ALL
        # trainable modules from rank 0, not just the main predictor (that
        # exact omission -- broadcasting predictor but forgetting
        # vision_proj/ambient_proj -- was the bug in the M2 script).
        for module in (predictor, text_target.proj):
            for t in module.state_dict().values():
                dist.broadcast(t, src=0)
        print(f"[train] rank {get_rank()}: broadcast predictor + text_target.proj weights from rank 0", flush=True)

    rng = random.Random(args.seed)
    train_datasets = []
    vgg_eval_loader = None
    action100m_eval_loader = None

    if not args.skip_vggsound:
        print(f"[train] building VGGSound splits, field={args.field!r}, n_clips cap={args.n_clips}...", flush=True)
        vgg_train_pairs, vgg_test_pairs = build_splits(args.field)
        rng.shuffle(vgg_train_pairs)
        vgg_train_pairs = vgg_train_pairs[: args.n_clips]
        rng.shuffle(vgg_test_pairs)
        vgg_test_pairs = vgg_test_pairs[:1000]
        train_datasets.append(M2EmbedDataset(vgg_train_pairs, CACHE_DIR, max_tdm_bins=m2_cfg.max_tdm_bins))
        vgg_eval_loader = DataLoader(
            M2EmbedDataset(vgg_test_pairs, CACHE_DIR, max_tdm_bins=m2_cfg.max_tdm_bins),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        print("[train] --skip-vggsound: Action100M-ISOLATED run, no VGGSound in training or eval", flush=True)

    if args.use_action100m:
        a100_train_pairs, a100_test_pairs = load_action100m_splits(args.field)
        rng.shuffle(a100_train_pairs)
        if args.action100m_n_clips is not None:
            a100_train_pairs = a100_train_pairs[: args.action100m_n_clips]
        rng.shuffle(a100_test_pairs)
        a100_test_pairs = a100_test_pairs[:1000]
        print(f"[train] Action100M: {len(a100_train_pairs)} train pairs, "
              f"{len(a100_test_pairs)} held-out pairs (video-level split, "
              f"scripts/action100m_captions.jsonl)", flush=True)
        if len(a100_train_pairs) == 0:
            print("[train]   WARNING: 0 Action100M train pairs found -- check "
                  f"{ACTION100M_CAPTIONS_PATH} exists and has field {args.field!r}", flush=True)
        else:
            train_datasets.append(M2EmbedDataset(a100_train_pairs, ACTION100M_CACHE_DIR,
                                                  max_tdm_bins=m2_cfg.max_tdm_bins))
        if len(a100_test_pairs) > 0:
            action100m_eval_loader = DataLoader(
                M2EmbedDataset(a100_test_pairs, ACTION100M_CACHE_DIR, max_tdm_bins=m2_cfg.max_tdm_bins),
                batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    logical_batch = args.batch_size * args.n_microbatches
    if is_main_process():
        print(f"[train] combined training set: {len(train_ds)} pairs "
              f"({' + '.join(str(len(d)) for d in train_datasets)})", flush=True)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if is_distributed() and get_world_size() > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                               sampler=sampler, collate_fn=collate_fn, num_workers=4, drop_last=True)

    def infinite(loader):
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b in loader:
                yield b
            epoch += 1
    batches = infinite(train_loader)

    trainable_params = list(predictor.parameters()) + list(text_target.proj.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    n_trainable = sum(t.numel() for t in trainable_params if t.requires_grad)
    if is_main_process():
        print(f"[train] trainable_params={n_trainable/1e6:.1f}M total_steps={args.steps} "
              f"logical_per_rank_batch={logical_batch} "
              f"global_batch={logical_batch * get_world_size()}", flush=True)

    loss_ema = None
    results_log = []
    best_score = -1.0
    t_start = time.time()
    for step in range(args.steps):
        micro_batches = [next(batches) for _ in range(args.n_microbatches)]

        optimizer.zero_grad()
        loss_val, metrics = gradcache_step(predictor, text_target, m2, micro_batches,
                                            temperature=args.temperature, device=device,
                                            soft_infonce=args.soft_infonce, soft_temp=args.soft_temp,
                                            soft_infonce_frozen_sim=args.soft_infonce_frozen_sim)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        sync_grads(predictor)
        sync_grads(text_target.proj)
        optimizer.step()

        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        if step % 50 == 0 and is_main_process():
            elapsed = time.time() - t_start
            print(f"[train] step {step}/{args.steps} loss={loss_val:.4f} "
                  f"loss_ema={loss_ema:.4f} acc_v2t={metrics.get('acc_v2t', 0):.3f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            if is_main_process():
                r = {"step": step}
                if vgg_eval_loader is not None:
                    r_vgg = retrieval_eval(predictor, text_target, m2, device, vgg_eval_loader)
                    print(f"[eval] step {step} VGGSound: {json.dumps(r_vgg)}", flush=True)
                    r["vggsound"] = r_vgg
                if action100m_eval_loader is not None:
                    r_a100 = retrieval_eval(predictor, text_target, m2, device, action100m_eval_loader)
                    print(f"[eval] step {step} Action100M: {json.dumps(r_a100)}", flush=True)
                    r["action100m"] = r_a100
                results_log.append(r)
                ckpt = {
                    "step": step, "predictor_mode": args.predictor_mode, "field": args.field,
                    "use_action100m": args.use_action100m,
                    "predictor": predictor.state_dict(),
                    "text_target_proj": text_target.proj.state_dict(),
                    "n_trainable_params": n_trainable,
                    "results_log": results_log,
                }
                torch.save(ckpt, os.path.join(out_dir, "last.pt"))
                # FOUND (2026-08-02, bs2048 run): held-out R@1 peaked around
                # step 300-350 then mildly declined by the final step (train
                # acc kept climbing to 75% while held-out R@1 flattened/dipped
                # -- classic overfitting) but last.pt only ever held the FINAL
                # checkpoint, overwritten every eval, so the actual best
                # checkpoint was never saved anywhere. Track best-by-held-out-
                # R@1 (averaged across whichever eval sets are active) and
                # save it separately.
                score_parts = []
                if "vggsound" in r:
                    score_parts.append(r["vggsound"]["pred→text_R@1"] + r["vggsound"]["text→pred_R@1"])
                if "action100m" in r:
                    score_parts.append(r["action100m"]["pred→text_R@1"] + r["action100m"]["text→pred_R@1"])
                score = sum(score_parts) / len(score_parts) if score_parts else 0.0
                if score > best_score:
                    best_score = score
                    torch.save(ckpt, os.path.join(out_dir, "best.pt"))
                    print(f"[train] step {step}: new best (score={score:.2f}), saved best.pt", flush=True)
            if is_distributed() and get_world_size() > 1:
                dist.barrier()

    if is_main_process():
        print(f"[train] DONE. wrote {out_dir}/last.pt", flush=True)
        print(f"[train] FINAL RESULTS: {json.dumps(results_log[-1] if results_log else {}, indent=2)}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
