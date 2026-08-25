"""LoRA fine-tune Qwen3-0.6B into BMO's System-2 THINKER on the distilled
reasoning corpus (data/bmo_thinker_corpus_v1_DRAFT.jsonl: prompt/reasoning/
answer/tools). Trains the native Qwen3 think format:
    user: {prompt}
    assistant: <think>{reasoning}</think>{answer}{tool_calls}
so the thinker deliberates then answers (and can emit tool calls for the
dispatcher). Loss only on the assistant completion (prompt masked). Distinct
from the fast tier's conversational corpus -- this is the reasoning objective.
"""
import argparse, json, os, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

IGNORE = -100


class ThinkerDS(Dataset):
    def __init__(self, rows, tok, max_len=1024):
        self.ex = []
        for r in rows:
            reasoning = (r.get("reasoning") or "").strip()
            answer = (r.get("answer") or "").strip()
            tools = r.get("tools") or []
            if not answer:
                continue
            tool_str = (" " + " ".join(tools)) if tools else ""
            completion = f"<think>\n{reasoning}\n</think>\n\n{answer}{tool_str}"
            user = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           add_generation_prompt=True, tokenize=False)
            full = user + completion + tok.eos_token
            u_ids = tok(user, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = list(f_ids)
            for i in range(min(len(u_ids), len(labels))):
                labels[i] = IGNORE  # mask the prompt
            self.ex.append((f_ids, labels))

    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    ids, labs, att = [], [], []
    for f, l in batch:
        p = m - len(f)
        ids.append(f + [pad_id] * p); labs.append(l + [IGNORE] * p); att.append([1] * len(f) + [0] * p)
    return (torch.tensor(ids), torch.tensor(labs), torch.tensor(att))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/utkarsh/hf_models/Qwen3-0.6B")
    ap.add_argument("--corpus", default="data/bmo_thinker_corpus_v1_DRAFT.jsonl")
    ap.add_argument("--out-dir", default="checkpoints/bmo_thinker_qwen3_lora")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.corpus) if l.strip()]
    n_val = max(4, len(rows) // 10)
    tr, va = rows[n_val:], rows[:n_val]
    tr_ds, va_ds = ThinkerDS(tr, tok), ThinkerDS(va, tok)
    print(f"train={len(tr_ds)} val={len(va_ds)}", flush=True)
    coll = lambda b: collate(b, tok.pad_token_id)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, collate_fn=coll)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = len(tr_dl) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        for ids, labs, att in tr_dl:
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=labs.to(dev))
            out.loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for ids, labs, att in va_dl:
                vl += model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=labs.to(dev)).loss.item(); nb += 1
        vl /= max(1, nb)
        print(f"[thinker] epoch {ep} val_loss={vl:.4f}", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(os.path.join(args.out_dir, "best"))
            tok.save_pretrained(os.path.join(args.out_dir, "best"))
    print(f"DONE thinker best_val_loss={best:.4f} -> {args.out_dir}/best", flush=True)


if __name__ == "__main__":
    main()
