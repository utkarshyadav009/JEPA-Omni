"""scripts/train_glr_thinker.py — train the GLR transition head on BMO's thinker.

Implements the training half of arXiv:2606.02248 against `checkpoints/bmo_thinker_qwen3_v5_merged`
(Qwen3-0.6B, the paper's own primary model). Only `models/glr_transition_head.py::TransitionHead`
is trainable -- ~1M params on top of a fully frozen backbone.

WHY THE DATA ALREADY FITS. GLR anchors latent trajectories on real textual CoT traces, so it
needs `(question, chain-of-thought, answer)` triples. `data/bmo_thinker_corpus_v6c.jsonl` is
exactly that shape and exactly our domain -- 1,489 rows of `prompt` / `reasoning` / `answer` /
`tools` across companion 360, perception_social 357, reasoning 330, orchestration 263,
grounded 179. The paper trains on 10K math examples; we have 1.5K in-domain ones, which is the
right trade for a 1M-param head whose job is a *local* geometric correction rather than new
reasoning ability. If it underfits, `--corpus` takes the larger v7 when it lands.

THE TWO CONSTRAINTS THAT ARE EASY TO BREAK AND FATAL IF BROKEN:

  * **No cross-entropy on the latent replacement positions.** The paper is explicit
    (`ce_latent_tokens=False`): supervising them drags each latent state toward an immediate
    vocabulary prediction, undoing the geometric relaxation that produces the token savings.
    Here, CE is masked to answer tokens only, and there is an assert that no thought position
    carries a CE label.
  * **The input embedding matrix stays frozen.** `Δe_i = E_in(t_i) - E_in(t_{i-1})` are the
    regression targets; training `E_in` makes them non-stationary and the head chases a moving
    objective. Everything is frozen here anyway, but the assert states the reason.

TWO FORWARD PASSES PER BATCH, which is why this is slower per step than plain SFT:
  1. teacher pass over the true discrete sequence, collecting hidden states at thought positions;
  2. student pass with thought-token embeddings replaced by the rolled-out latents,
     `ê_i = ê_{i-1} + g_φ(h_{i-1})`, scoring CE on the answer span only.

DEPLOYMENT IS ALREADY PROVEN (2026-08-16), which is why this is worth training rather than
prototyping: `scripts/probe_llamacpp_hidden_states.py` passes on the Jetson -- per-step hidden
states come back at dim=1024 via a raw `llama_get_embeddings_ith` pointer read (the wrapper's
own accessor raises the same PEP-3118 buffer error already documented for `get_logits_ith`), and
`llama_batch.embd` feeds embeddings in byte-identically. Both halves of the rollout work.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/train_glr_thinker.py --epochs 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.glr_transition_head import (GLRConfig, TransitionHead, build_delta_targets,
                                        transition_loss)

IGNORE = -100


class ThinkerCoTDataset(Dataset):
    """Builds `q <think> t_{1:m} </think> a_{1:l}` and records the three spans.

    The thought span is tracked explicitly rather than re-found by searching for `<think>` at
    train time: tokenizers merge across boundaries, and a span located by string search can be
    off by a token in a way that silently corrupts both the CE mask and the Δe targets."""

    def __init__(self, path: str, tok, max_len: int = 1024, min_thought: int = 8):
        self.rows, self.tok, self.max_len = [], tok, max_len
        skipped = 0
        for line in Path(path).open():
            if not line.strip():
                continue
            r = json.loads(line)
            reasoning = (r.get("reasoning") or "").strip()
            answer = (r.get("answer") or "").strip()
            prompt = (r.get("prompt") or "").strip()
            if not (reasoning and answer and prompt):
                skipped += 1
                continue
            # transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=True),
            # not a plain list, so template to TEXT and encode -- one unambiguous code path
            # across versions, and it keeps `head_ids` a list we can concatenate.
            head_txt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, enable_thinking=True, tokenize=False)
            head_ids = tok.encode(head_txt, add_special_tokens=False)
            open_ids = tok.encode("<think>\n", add_special_tokens=False)
            th_ids = tok.encode(reasoning, add_special_tokens=False)
            close_ids = tok.encode("\n</think>\n\n", add_special_tokens=False)
            ans_ids = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
            if len(th_ids) < min_thought:
                skipped += 1
                continue
            ids = head_ids + open_ids + th_ids + close_ids + ans_ids
            if len(ids) > max_len:
                skipped += 1
                continue
            t0 = len(head_ids) + len(open_ids)
            self.rows.append({
                "ids": ids,
                "t0": t0, "t1": t0 + len(th_ids),                      # thought span [t0, t1)
                "a0": t0 + len(th_ids) + len(close_ids),               # answer span start
            })
        print(f"[glr-data] {len(self.rows)} usable rows ({skipped} skipped: missing fields, "
              f"thought < {min_thought} tokens, or > {max_len} tokens)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    n = max(len(b["ids"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, n), pad_id, dtype=torch.long)
    attn = torch.zeros((B, n), dtype=torch.long)
    tmask = torch.zeros((B, n), dtype=torch.bool)          # thought positions
    labels = torch.full((B, n), IGNORE, dtype=torch.long)   # CE on ANSWER ONLY
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"])
        attn[i, :L] = 1
        tmask[i, b["t0"]:b["t1"]] = True
        labels[i, b["a0"]:L] = ids[i, b["a0"]:L]
    return ids, attn, tmask, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/bmo_thinker_qwen3_v5_merged")
    ap.add_argument("--corpus", default="data/bmo_thinker_corpus_v6c.jsonl")
    ap.add_argument("--out", default="checkpoints/glr_thinker_v1")
    ap.add_argument("--epochs", type=int, default=5)          # paper: 5
    ap.add_argument("--lr", type=float, default=4e-5)         # paper: 4e-5 cosine, 5% warmup
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)           # paper effective batch 16
    ap.add_argument("--gamma", type=float, default=0.999)
    # L_Delta is now NORMALISED by the mean squared target norm (see transition_loss), so it is
    # ~1.0 for a head that predicts nothing and sits naturally alongside CE ~2. lambda=1 is
    # therefore meaningful rather than arbitrary.
    #
    # Both unnormalised choices were tried and both failed, which is why the normalisation is
    # there: lambda=1.0 on raw units gave L_Delta ~99.9% of the gradient (CE ignored);
    # lambda=1e-3 gave the reverse, and the K=10 rollout DIVERGED -- ||latent||=423, 330 tokens
    # against the K=0 baseline's 83, EOS rate 0.38, answer_f1 0.090 vs 0.234.
    ap.add_argument("--lambda-delta", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Stated as an assert, not a comment: the Delta targets are built FROM this matrix, so if it
    # ever becomes trainable the head is regressing against a moving target.
    emb = model.get_input_embeddings()
    assert not emb.weight.requires_grad, "input embeddings must stay frozen (non-stationary targets)"
    d_model = emb.weight.shape[1]

    cfg = GLRConfig(d_model=d_model, gamma=args.gamma, lambda_delta=args.lambda_delta)
    head = TransitionHead(cfg).to(dev).to(torch.float32)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[glr] d_model={d_model}  head params={n_head/1e6:.2f}M  "
          f"backbone frozen ({sum(p.numel() for p in model.parameters())/1e6:.0f}M)")

    ds = ThinkerCoTDataset(args.corpus, tok, max_len=args.max_len)
    n_val = max(1, int(len(ds) * args.val_frac))
    g = torch.Generator().manual_seed(0)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, pad))
    dlv = DataLoader(va, batch_size=args.batch, shuffle=False,
                     collate_fn=lambda b: collate(b, pad))

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    steps = max(1, (len(dl) // args.accum) * args.epochs)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    def run_batch(ids, attn, tmask, labels, train: bool):
        ids, attn = ids.to(dev), attn.to(dev)
        tmask, labels = tmask.to(dev), labels.to(dev)

        # ---- pass 1: teacher, over the true discrete sequence ----
        with torch.no_grad():
            e_in = emb(ids)                                              # (B,T,d)
            h = model(inputs_embeds=e_in, attention_mask=attn,
                      output_hidden_states=True).hidden_states[-1]       # (B,T,d)

        # displacements predicted from h_{i-1} -> shift hidden states right by one
        h_prev = torch.zeros_like(h)
        h_prev[:, 1:] = h[:, :-1]
        pred_delta = head(h_prev.float())                                # (B,T,d)
        tgt_delta = build_delta_targets(emb.weight.float(), ids, tmask)  # (B,T,d)
        l_delta = transition_loss(pred_delta, tgt_delta, tmask, cfg.gamma)

        # ---- pass 2: student, thought embeddings replaced by the rollout ----
        # e_hat_i = e_hat_{i-1} + g(h_{i-1}); a cumulative sum over the thought span gives the
        # same result in one shot, starting from the embedding just before the span.
        steps_in_span = (pred_delta * tmask.unsqueeze(-1)).cumsum(dim=1)
        start = torch.zeros_like(e_in)
        idx = tmask.float().argmax(dim=1)                                # first thought position
        for b in range(ids.shape[0]):
            if tmask[b].any():
                start[b] = e_in[b, max(idx[b].item() - 1, 0)]
        e_lat = (start + steps_in_span).to(e_in.dtype)
        e_stu = torch.where(tmask.unsqueeze(-1), e_lat, e_in)

        out = model(inputs_embeds=e_stu, attention_mask=attn)
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        # CE ON ANSWER TOKENS ONLY -- asserted, because getting this wrong silently defeats
        # the whole method rather than producing an obvious error.
        assert not (tgt != IGNORE).logical_and(tmask[:, 1:]).any(), \
            "CE label found on a latent/thought position (ce_on_latent must be False)"
        l_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               tgt.reshape(-1), ignore_index=IGNORE)
        loss = l_ce + cfg.lambda_delta * l_delta
        return loss, l_ce.detach(), l_delta.detach()

    best = float("inf")
    os.makedirs(args.out, exist_ok=True)
    gstep = 0
    for ep in range(args.epochs):
        head.train()
        t0 = time.time()
        for i, (ids, attn, tmask, labels) in enumerate(dl):
            loss, l_ce, l_d = run_batch(ids, attn, tmask, labels, True)
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 10 == 0:
                    print(f"[glr] ep{ep} step{gstep}/{steps} loss={loss.item():.4f} "
                          f"ce={l_ce.item():.4f} delta={l_d.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        head.eval()
        vl = vce = vd = 0.0
        with torch.no_grad():
            for ids, attn, tmask, labels in dlv:
                loss, l_ce, l_d = run_batch(ids, attn, tmask, labels, False)
                vl += loss.item(); vce += l_ce.item(); vd += l_d.item()
        n = max(1, len(dlv))
        vl, vce, vd = vl / n, vce / n, vd / n
        print(f"[glr] epoch {ep}: val_loss={vl:.4f} val_ce={vce:.4f} val_delta={vd:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            torch.save({"head": head.state_dict(), "cfg": cfg.__dict__,
                        "d_model": d_model, "base": args.model, "epoch": ep,
                        "val_loss": vl, "val_ce": vce, "val_delta": vd},
                       f"{args.out}/best.pt")
            print(f"[glr]   saved {args.out}/best.pt", flush=True)

    print(f"[glr] DONE best val_loss={best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
