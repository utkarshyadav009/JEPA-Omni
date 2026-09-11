"""scripts/train_speaker_v16.py -- train a BMO speaker candidate on the v16 pooled corpus.

Handles BOTH candidates through one code path so the A/B differs only in base
model, not in data handling:

  --candidate gemma : google/gemma-3-270m-it, FULL fine-tune
  --candidate lfm   : LFM2.5-350M, LoRA (same base + deploy path as production v10)

Data rendering lives in scripts/bmo_v16_data.py -- see that file for why the
multi-turn rows need a dedicated path and how the prompt shape is kept
byte-identical to models/m4_cognitive_core.py's inference path.

max_len=320 is MEASURED, not guessed: the longest rendered conversation in the
corpus is 275 tokens (LFM) / 277 (Gemma), p99=188/187. 320 truncates 0/11586
rows. The predecessor script sat at 96 for weeks and silently sliced the target
off all 783 directive rows, contributing exactly zero loss; the truncation count
is asserted to be 0 here and printed every run.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bmo_v16_data import (load_rows, rows_to_conversations, ConvDataset,
                                   derive_turn_end, collate, state_prefix)

CANDIDATES = {
    "gemma": dict(
        path="/home/utkarsh/hf_models/gemma-3-270m-it",
        template_kwargs={},
        mode="full",
        lr=5e-5,
        epochs=3,
        attn="eager",          # HF's own recommendation for Gemma3 training
        lora_targets=None,
    ),
    "lfm": dict(
        path="/home/utkarsh/hf_models/LFM2.5-350M",
        template_kwargs={"enable_thinking": False},
        mode="lora",
        lr=2e-4,
        epochs=4,
        attn=None,
        # LFM2 leaf-module names, NOT MiniCPM/Llama names. LFM2's hybrid conv+GQA
        # stack has no o_proj/gate_proj/up_proj/down_proj at all; using the Llama
        # list here would silently adapt ~zero real layers.
        lora_targets="q_proj,k_proj,v_proj,out_proj,w1,w2,w3",
    ),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    p.add_argument("--corpus", default="data/bmo_companion_corpus_v16_pooled.jsonl")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = CANDIDATES[args.candidate]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    out_dir = args.out_dir or f"checkpoints/bmo_speaker_v16_{args.candidate}"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    print(f"[v16:{args.candidate}] base={cfg['path']} mode={cfg['mode']} "
          f"lr={lr} epochs={epochs} bs={args.batch_size} max_len={args.max_len}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    turn_end = derive_turn_end(tok, cfg["template_kwargs"])
    print(f"[v16:{args.candidate}] turn_end derived from chat template = {turn_end!r} "
          f"(id={tok.convert_tokens_to_ids(turn_end)}); tokenizer.eos_token={tok.eos_token!r} "
          f"(id={tok.eos_token_id})", flush=True)

    rows = load_rows(args.corpus)
    convs, stats = rows_to_conversations(rows, seed=args.seed)
    print(f"[v16:{args.candidate}] CORPUS {json.dumps(stats)}", flush=True)
    n_multi_raw = sum(1 for r in rows if "messages" in r)
    assert stats["n_multi"] == n_multi_raw, (
        f"multi-turn rows lost! corpus has {n_multi_raw} `messages` rows, "
        f"only {stats['n_multi']} survived rendering")
    assert stats["n_skipped"] == 0, f"{stats['n_skipped']} rows skipped"

    idx = list(range(len(convs))); random.Random(args.seed).shuffle(idx)
    n_val = max(50, int(len(convs) * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_convs = [convs[i] for i in train_idx]
    val_convs = [convs[i] for i in val_idx]
    n_multi_train = sum(1 for c in train_convs if len(c) > 2)
    n_multi_val = sum(1 for c in val_convs if len(c) > 2)
    print(f"[v16:{args.candidate}] train={len(train_convs)} (multi-turn {n_multi_train}) "
          f"val={len(val_convs)} (multi-turn {n_multi_val})", flush=True)

    train_ds = ConvDataset(train_convs, tok, args.max_len, turn_end, cfg["template_kwargs"])
    val_ds = ConvDataset(val_convs, tok, args.max_len, turn_end, cfg["template_kwargs"])

    # ---- HARD GATE: no target may be cut, and every row must carry loss ----
    n_trunc = n_zero = 0
    sup_tokens = 0
    for i in range(len(train_ds)):
        ids, lab, tr = train_ds.encode(train_convs[i])
        n_trunc += int(tr)
        n_sup = int((lab != -100).sum())
        sup_tokens += n_sup
        n_zero += int(n_sup == 0)
    print(f"[v16:{args.candidate}] TRUNCATION CHECK: {n_trunc} rows exceed max_len={args.max_len}; "
          f"{n_zero} rows carry zero loss; {sup_tokens} supervised target tokens total", flush=True)
    assert n_trunc == 0, "a target is being truncated -- raise --max-len"
    assert n_zero == 0, "a row contributes zero loss"

    # ---- print one rendered TRAINING sample beside one rendered INFERENCE prompt ----
    mi = next(i for i, c in enumerate(train_convs) if len(c) > 4)
    ids, lab, _ = train_ds.encode(train_convs[mi])
    print(f"\n[v16:{args.candidate}] ===== RENDERED TRAINING SAMPLE (multi-turn) =====", flush=True)
    print(repr(tok.decode(ids)), flush=True)
    print(f"[v16:{args.candidate}] supervised spans: " +
          repr([tok.decode([t for t, l in zip(ids.tolist(), lab.tolist()) if l != -100])]), flush=True)
    infer_ctx = tok.apply_chat_template(train_convs[mi][:-1], add_generation_prompt=True,
                                        tokenize=False, **cfg["template_kwargs"])
    print(f"\n[v16:{args.candidate}] ===== RENDERED INFERENCE-SHAPED PROMPT (same conversation) =====", flush=True)
    print(repr(infer_ctx), flush=True)
    inf_ids = tok(infer_ctx, add_special_tokens=False)["input_ids"]
    match = ids[:len(inf_ids)].tolist() == inf_ids and lab[len(inf_ids)] != -100
    print(f"[v16:{args.candidate}] SHAPE MATCH (inference context is a token-exact prefix of the "
          f"training sequence, and supervision begins at the first generated token): {match}\n", flush=True)
    assert match, "train/inference prompt shape mismatch"

    # ---- model ----
    kw = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if cfg["attn"]:
        kw["attn_implementation"] = cfg["attn"]
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kw).to(device)
    if cfg["mode"] == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        targets = cfg["lora_targets"].split(",")
        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        missing = [t for t in targets if t not in present]
        assert not missing, f"LoRA targets not in this model: {missing}"
        print(f"[v16:{args.candidate}] LoRA targets all present in model: {targets}", flush=True)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0.05, target_modules=targets))
        model.print_trainable_parameters()
    else:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[v16:{args.candidate}] FULL fine-tune: {n_tr:,} trainable params", flush=True)
    model.gradient_checkpointing_disable()

    coll = lambda b: collate(b, tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    n_steps = len(train_loader) * epochs
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                              weight_decay=0.01, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(optim, max(10, n_steps // 20), n_steps)
    print(f"[v16:{args.candidate}] {epochs} epochs x {len(train_loader)} steps = {n_steps} steps", flush=True)

    curve = []
    best = (float("inf"), -1)
    step = 0; t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            run += loss.item(); nb += 1; step += 1
            if step % 25 == 0:
                print(f"[v16:{args.candidate}] step {step}/{n_steps} ep={ep} "
                      f"loss={run/nb:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"t={time.perf_counter()-t0:.0f}s", flush=True)
        train_loss = run / max(1, nb)

        model.eval(); vl = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                vl.append(model(**batch).loss.item())
        val_loss = sum(vl) / max(1, len(vl))
        curve.append({"epoch": ep, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})
        is_best = val_loss < best[0]
        print(f"[v16:{args.candidate}] === epoch {ep}: train={train_loss:.4f} val={val_loss:.4f}"
              f"{'  (new best)' if is_best else ''} ===", flush=True)
        if is_best:
            best = (val_loss, ep)
            model.save_pretrained(os.path.join(out_dir, "best"))
            tok.save_pretrained(os.path.join(out_dir, "best"))

    with open(os.path.join(out_dir, "loss_curve.json"), "w") as f:
        json.dump({"candidate": args.candidate, "mode": cfg["mode"], "lr": lr,
                   "epochs": epochs, "batch_size": args.batch_size, "max_len": args.max_len,
                   "corpus_stats": stats, "n_train": len(train_convs), "n_val": len(val_convs),
                   "n_multi_train": n_multi_train, "supervised_tokens": sup_tokens,
                   "turn_end": turn_end, "curve": curve,
                   "best_val_loss": best[0], "best_epoch": best[1]}, f, indent=2)
    print(f"[v16:{args.candidate}] DONE best_epoch={best[1]} best_val={best[0]:.4f} -> {out_dir}/best", flush=True)

    # ---- merged HF checkpoint, for convert_hf_to_gguf.py ----
    merged = os.path.join(out_dir, "merged")
    if cfg["mode"] == "lora":
        from peft import PeftModel
        del model; torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(cfg["path"], **kw)
        m = PeftModel.from_pretrained(base, os.path.join(out_dir, "best")).merge_and_unload()
        m.save_pretrained(merged); tok.save_pretrained(merged)
    else:
        m = AutoModelForCausalLM.from_pretrained(os.path.join(out_dir, "best"), **kw)
        m.save_pretrained(merged); tok.save_pretrained(merged)
    print(f"[v16:{args.candidate}] merged HF checkpoint -> {merged}", flush=True)


if __name__ == "__main__":
    main()
