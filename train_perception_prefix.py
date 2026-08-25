"""train_perception_prefix.py — the mercury capacity probe: can a small projector make a
frozen BMO LLM describe a scene in its OWN words, well enough to replace bank retrieval?

THE QUESTION THIS ANSWERS (and the reason it runs on mercury before any Jetson work):
the Ultravox track closed with "the 350M decoder is a hard wall" for ASR. Captioning is a
different task — semantic gist rather than exact token fidelity — so that verdict should not
be assumed to transfer, but it is the top risk. This trains the projector and scores it
against a pre-existing, directly comparable baseline: **M3's word-overlap F1 = 0.317** on the
same VGGSound rich captions.

MODEL: the BMO-finetuned **Qwen3-0.6B thinker v3** (`checkpoints/bmo_thinker_qwen3_v3_merged`)
— per direct instruction, the thinker rather than the fast tier, because deliberate scene
description is exactly the thinker's job and the fast tier should stay free for instant
replies. It is also the SAME weights deployed as `bmo_thinker_qwen3_v3_Q8_0.gguf`, so the
projector learns to speak into the brain that is actually on the device.

DATA: the whole captioned corpus, per instruction — VGGSound (6 granularities x 172,593
clips with all fields) + Action100M (2 granularities x 345,754 clips). One training example
= (clip, field): the prefix encodes the scene, the field's question is TEXT in the prompt,
the field's caption is the target. So one clip teaches 6 different questions.

FROZEN: V-JEPA2 / WavJEPA (cached), M2, and the entire LLM. TRAINABLE: the projector only.

Usage:
    torchrun --nproc_per_node=4 train_perception_prefix.py --steps 6000
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
from train_query_predictor import QueryClipDataset, collate, group_by_clip, build_sources
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.perception_prefix import PerceptionPrefix, PerceptionPrefixConfig
from models.query_predictor import QUERY_BANK, VGGSOUND_FIELDS, ACTION100M_FIELDS
from utils import is_distributed, get_rank, get_local_rank, get_world_size, is_main_process

MAX_ANS_TOK = 96


def setup() -> torch.device:
    if is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_local_rank())
        return torch.device("cuda", get_local_rank())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_grads(mods: List[nn.Module]) -> None:
    if not (is_distributed() and get_world_size() > 1):
        return
    ps = [p for m in mods for p in m.parameters() if p.requires_grad]
    for p in ps:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = _flatten_dense_tensors([p.grad for p in ps])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM); flat /= get_world_size()
    for p, s in zip(ps, _unflatten_dense_tensors(flat, [p.grad for p in ps])):
        p.grad.copy_(s)


def build_batch(tok, emb_layer, prefix, questions, answers, device):
    """[prefix soft tokens][ "Task: q\\n" ][ answer ] with the loss on the ANSWER only.

    The question is masked out of the loss for the same reason train_m3.py masks its
    granularity tag: the model must not be trained to predict its own instruction, only the
    description that follows it."""
    B, K, H = prefix.shape
    ids_q = [tok(f"Task: {q}\n", add_special_tokens=False)["input_ids"] for q in questions]
    ids_a = [tok(a, add_special_tokens=False)["input_ids"][:MAX_ANS_TOK] + [tok.eos_token_id]
             for a in answers]
    L = K + max(len(q) + len(a) for q, a in zip(ids_q, ids_a))
    inp = torch.zeros(B, L, H, device=device, dtype=prefix.dtype)
    lab = torch.full((B, L), -100, dtype=torch.long, device=device)
    att = torch.zeros(B, L, dtype=torch.long, device=device)
    for i, (q, a) in enumerate(zip(ids_q, ids_a)):
        n = K + len(q) + len(a)
        txt = torch.tensor(q + a, device=device)
        inp[i, :K] = prefix[i]
        inp[i, K:K + len(q) + len(a)] = emb_layer(txt).to(prefix.dtype)
        lab[i, K + len(q):n] = torch.tensor(a, device=device)   # answer only
        att[i, :n] = 1
    return inp, lab, att


@torch.no_grad()
def evaluate(model, tok, emb_layer, proj, m2, loader, fields, device, names,
             max_clips=200, gen_tokens=48) -> Dict[str, float]:
    """Word-overlap F1 against the reference caption -- the SAME metric M3 reported (0.317),
    so the comparison is like-for-like. Also reports a query-sensitivity check: does asking a
    different question change the generated text?"""
    proj.eval()
    f1s, changed, n, first_texts = [], 0, 0, None
    for batch in loader:
        if n >= max_clips:
            break
        src, msk = build_sources(batch, m2, device, names)
        pfx = proj(src, msk).to(torch.bfloat16)
        B = pfx.shape[0]
        for fi, f in enumerate(fields):
            q = QUERY_BANK[f][-1]                              # held-out phrasing
            ids_q = tok(f"Task: {q}\n", add_special_tokens=False)["input_ids"]
            qe = emb_layer(torch.tensor(ids_q, device=device)).to(pfx.dtype)
            inputs = torch.cat([pfx, qe.unsqueeze(0).expand(B, -1, -1)], 1)
            out = model.generate(inputs_embeds=inputs,
                                 attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=device),
                                 max_new_tokens=gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(out, skip_special_tokens=True)
            for bi in range(B):
                ref = set(batch["caps"][bi][fi].lower().split())
                hyp = set(texts[bi].lower().split())
                if ref and hyp:
                    inter = len(ref & hyp)
                    p, r = inter / len(hyp), inter / len(ref)
                    f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
            if fi == 0:
                first_texts = texts
            if fi == len(fields) - 1 and first_texts is not None:
                changed += sum(1 for a, b in zip(first_texts, texts) if a.strip() != b.strip())
        n += B
    proj.train()
    return {"word_overlap_f1": float(np.mean(f1s)) if f1s else 0.0,
            "n_captions": len(f1s), "n_clips": n,
            "query_changes_output_frac": changed / max(1, n)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="checkpoints/bmo_thinker_qwen3_v3_merged")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_run2_vggsound197k_ego4d134k_neg200/step19000.pt")
    p.add_argument("--out-dir", default="checkpoints/perception_prefix_thinker")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-prefix", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--p-vggsound", type=float, default=0.6)
    p.add_argument("--token-sources", default="m2,vision,ambient")
    p.add_argument("--captions-path", default=os.path.join(PROJECT_ROOT, "scripts",
                   "qwen_omni_full_captions_v2.jsonl"))
    p.add_argument("--max-clips-per-corpus", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    names = [x.strip() for x in args.token_sources.split(",") if x.strip()]
    train_m3.CAPTIONS_PATH = args.captions_path

    device = setup()
    torch.manual_seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device).eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)
    emb_layer = llm.get_input_embeddings()
    H = llm.config.hidden_size
    if is_main_process():
        print(f"[pp] LLM={args.llm} hidden={H} params={sum(x.numel() for x in llm.parameters())/1e6:.0f}M "
              f"(FROZEN)", flush=True)

    vtr = group_by_clip(build_splits(VGGSOUND_FIELDS)[0], VGGSOUND_FIELDS)
    vte = group_by_clip(build_splits(VGGSOUND_FIELDS)[1], VGGSOUND_FIELDS)
    atr_p, ate_p = [], []
    for f in ACTION100M_FIELDS:
        a, b = load_action100m_splits(f); atr_p += a; ate_p += b
    atr = group_by_clip(atr_p, ACTION100M_FIELDS)
    ate = group_by_clip(ate_p, ACTION100M_FIELDS)
    if args.max_clips_per_corpus:
        vtr = {k: vtr[k] for k in sorted(vtr)[: args.max_clips_per_corpus]}
        atr = {k: atr[k] for k in sorted(atr)[: args.max_clips_per_corpus]}
    if is_main_process():
        print(f"[pp] VGGSound train={len(vtr)} test={len(vte)} | Action100M train={len(atr)} "
              f"test={len(ate)}", flush=True)

    def mk(clips, cd, fl, shuffle, bs):
        ds = QueryClipDataset(clips, cd, fl)
        smp = DistributedSampler(ds, shuffle=shuffle, drop_last=True) if is_distributed() else None
        return DataLoader(ds, batch_size=bs, sampler=smp, shuffle=(smp is None and shuffle),
                          num_workers=6, collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=True), smp

    dl_v, sv = mk(vtr, CACHE_DIR, VGGSOUND_FIELDS, True, args.batch)
    dl_a, sa = mk(atr, ACTION100M_CACHE_DIR, ACTION100M_FIELDS, True, args.batch)
    ev_v, _ = mk(vte, CACHE_DIR, VGGSOUND_FIELDS, False, 8)

    m2cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    m2 = AVJepaPredictor(m2cfg).to(device)
    m2.load_state_dict(torch.load(args.m2_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    m2.eval()
    for prm in m2.parameters():
        prm.requires_grad_(False)

    SRC = {"m2": 1024, "vision": 1024, "ambient": 768}
    proj = PerceptionPrefix(PerceptionPrefixConfig(
        source_dims={s: SRC[s] for s in names}, llm_hidden=H, n_prefix=args.n_prefix)).to(device)
    if is_distributed() and get_world_size() > 1:
        for prm in proj.parameters():
            dist.broadcast(prm.data, src=0)
    if is_main_process():
        print(f"[pp] projector trainable = {sum(x.numel() for x in proj.parameters())/1e6:.1f}M "
              f"-> {args.n_prefix} soft tokens x {H}", flush=True)

    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def inf(dl, smp):
        e = 0
        while True:
            if smp is not None:
                smp.set_epoch(e)
            for b in dl:
                yield b
            e += 1

    it_v, it_a = inf(dl_v, sv), inf(dl_a, sa)
    log, best, t0 = [], -1.0, time.time()

    for step in range(args.steps):
        srng = np.random.default_rng(args.seed * 7919 + step)
        use_v = srng.random() < args.p_vggsound
        batch = next(it_v if use_v else it_a)
        fields = batch["fields"]
        src, msk = build_sources(batch, m2, device, names)
        B = next(iter(src.values())).shape[0]
        ask = [int(srng.integers(0, len(fields))) for _ in range(B)]
        qs = [QUERY_BANK[fields[f]][int(srng.integers(0, len(QUERY_BANK[fields[f]]) - 1))] for f in ask]
        ans = [batch["caps"][i][ask[i]] for i in range(B)]

        pfx = proj(src, msk).to(torch.bfloat16)   # match the frozen LLM dtype
        inp, lab, att = build_batch(tok, emb_layer, pfx, qs, ans, device)
        out = llm(inputs_embeds=inp, attention_mask=att)
        logits = out.logits[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads([proj])
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 and is_main_process():
            print(f"[pp] step {step:5d}/{args.steps} [{'vgg' if use_v else 'a100m'}] "
                  f"loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)

        if ((step + 1) % args.eval_every == 0 or step + 1 == args.steps) and is_main_process():
            m = evaluate(llm, tok, emb_layer, proj, m2, ev_v, VGGSOUND_FIELDS, device, names)
            m["step"] = step; log.append(m)
            print(f"[pp] EVAL {step}: word_overlap_F1={m['word_overlap_f1']:.4f} "
                  f"(M3 baseline 0.317)  query_changes_output={m['query_changes_output_frac']:.2f} "
                  f"n={m['n_captions']}", flush=True)
            if m["word_overlap_f1"] > best:
                best = m["word_overlap_f1"]
                torch.save({"step": step, "projector": proj.state_dict(),
                            "cfg": vars(args), "metrics": m},
                           os.path.join(args.out_dir, "best.pt"))
        if is_distributed():
            dist.barrier()

    if is_main_process():
        with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"[pp] DONE best_F1={best:.4f} -> {args.out_dir}", flush=True)
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
