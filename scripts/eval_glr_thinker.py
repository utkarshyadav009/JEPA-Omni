"""scripts/eval_glr_thinker.py — does GLR actually buy us anything?

Trains nothing. Sweeps the latent budget K and reports the ONLY two numbers that decide whether
this ships: how many tokens the thinker generates, and whether the answer survives.

WHY THIS EVAL AND NOT val_loss. `checkpoints/glr_thinker_v1/best.pt` reached val_loss 3.5002
with val_ce flat at ~1.995 while val_delta fell 37%, which says the head learned the trajectory
without degrading teacher-forced likelihood. That is necessary and nowhere near sufficient:
teacher forcing never runs the rollout. The paper's own result is that accuracy DEGRADES at
large K from accumulated geometric drift (it reports a K sweep precisely because there is an
optimum, not a monotone win), so K must be chosen by measurement on our data.

THE BASELINE IS K=0, i.e. the deployed thinker generating its full `<think>` chain. That is the
thing we are trying to beat, and it is currently **1,749-3,509 ms on the Jetson** -- the
dominant leg of the whole BMO pipeline. Any K that does not cut generated tokens materially is
not worth the extra moving part.

METRICS, all automatic and all reported per K:

  gen_tokens   median tokens generated to completion. The point of the exercise.
                Latent steps are counted too (K + emitted tokens), which is CONSERVATIVE for
                GLR since a latent step skips the vocabulary projection and is cheaper than a
                decoded token -- the same accounting the paper uses.
  eos_rate     fraction that actually terminated instead of running into the cap. A "short"
                generation that never emits EOS is a hang, not a saving, and this is the first
                thing geometric drift breaks.
  answer_f1    token-level F1 against the corpus reference answer. Crude, but it is symmetric,
                needs no judge model, and is sensitive to exactly the failure we care about --
                the answer wandering off after the latent prefix. Compared RELATIVE to the K=0
                arm on the same prompts, never as an absolute quality claim.
  drift        mean L2 norm of the final latent vs the mean norm of real token embeddings. The
                paper attributes large-K degradation to drift; this makes it visible rather
                than inferred, so a bad K sweep can be diagnosed instead of just observed.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_glr_thinker.py --k 0,5,10,20,50
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.glr_transition_head import GLRConfig, TransitionHead
from scripts.train_glr_thinker import ThinkerCoTDataset


def token_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0)
    hit = 0
    rc = {}
    for w in r:
        rc[w] = rc.get(w, 0) + 1
    for w in p:
        if rc.get(w, 0) > 0:
            hit += 1
            rc[w] -= 1
    if hit == 0:
        return 0.0
    prec, rec = hit / len(p), hit / len(r)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate_glr(model, tok, head, prompt: str, k: int, max_new: int, dev):
    """K latent steps from E_in(<think>), then ordinary greedy decoding."""
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, enable_thinking=True,
                                  tokenize=False)
    ids = torch.tensor([tok.encode(txt, add_special_tokens=False)], device=dev)
    emb = model.get_input_embeddings()
    e = emb(ids)

    out = model(inputs_embeds=e, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    h = out.hidden_states[-1][:, -1:]
    cur = e[:, -1:]

    for _ in range(k):
        cur = cur + head(h.float()).to(cur.dtype)      # e_i = e_{i-1} + g(h_{i-1})
        out = model(inputs_embeds=cur, past_key_values=past,
                    use_cache=True, output_hidden_states=True)
        past, h = out.past_key_values, out.hidden_states[-1][:, -1:]

    drift = float(cur.float().norm()) if k else 0.0

    # hand back to normal token decoding
    gen, eos = [], False
    logits = model(inputs_embeds=cur, past_key_values=past, use_cache=True) if k else out
    nxt = int(logits.logits[0, -1].argmax()) if k else int(out.logits[0, -1].argmax())
    past = logits.past_key_values if k else out.past_key_values
    for _ in range(max_new):
        if nxt == tok.eos_token_id:
            eos = True
            break
        gen.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]], device=dev),
                  past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
    return tok.decode(gen, skip_special_tokens=True), len(gen), eos, drift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/glr_thinker_v1/best.pt")
    ap.add_argument("--corpus", default="data/bmo_thinker_corpus_v6c.jsonl")
    ap.add_argument("--k", default="0,5,10,20,50")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=320)   # production budget
    ap.add_argument("--out", default="checkpoints/glr_thinker_v1/eval.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = ck["base"]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(dev).eval()
    head = TransitionHead(GLRConfig(**{k: v for k, v in ck["cfg"].items()
                                       if k in GLRConfig.__dataclass_fields__}))
    head.load_state_dict(ck["head"]); head = head.to(dev).float().eval()
    print(f"[eval] {args.ckpt}  epoch={ck['epoch']} val_loss={ck['val_loss']:.4f}  base={base}")

    # SAME split and seed as training, so the eval prompts are genuinely held out.
    ds = ThinkerCoTDataset(args.corpus, tok, max_len=1024)
    n_val = max(1, int(len(ds) * 0.1))
    g = torch.Generator().manual_seed(0)
    _, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    rows = [json.loads(l) for l in Path(args.corpus).open() if l.strip()]
    refs = {}
    for r in rows:
        if r.get("prompt") and r.get("answer"):
            refs[r["prompt"].strip()] = r["answer"].strip()
    prompts = []
    for i in range(min(args.n, len(va))):
        ids = va[i]["ids"]
        txt = tok.decode(ids[:va[i]["t0"]], skip_special_tokens=True)
        for p, a in refs.items():
            if p[:40] in txt:
                prompts.append((p, a))
                break
    print(f"[eval] {len(prompts)} held-out prompts with references\n")

    table = {}
    for k in [int(x) for x in args.k.split(",")]:
        toks, f1s, eoss, drifts = [], [], [], []
        t0 = time.time()
        for p, ref in prompts:
            txt, n, eos, dr = generate_glr(model, tok, head, p, k, args.max_new, dev)
            ans = txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
            toks.append(n + k)          # conservative: latent steps counted as full tokens
            f1s.append(token_f1(ans, ref))
            eoss.append(eos)
            drifts.append(dr)
        table[k] = {
            "median_gen_tokens": statistics.median(toks),
            "mean_gen_tokens": round(statistics.mean(toks), 1),
            "eos_rate": round(sum(eoss) / max(len(eoss), 1), 3),
            "answer_f1": round(statistics.mean(f1s), 4),
            "mean_latent_norm": round(statistics.mean(drifts), 2),
            "sec": round(time.time() - t0, 1),
        }
        r = table[k]
        print(f"[eval] K={k:3d}  median_tokens={r['median_gen_tokens']:6.1f}  "
              f"eos={r['eos_rate']:.2f}  answer_f1={r['answer_f1']:.4f}  "
              f"|latent|={r['mean_latent_norm']:7.2f}  ({r['sec']}s)", flush=True)

    base_t = table.get(0, {}).get("median_gen_tokens") or 1
    base_f1 = table.get(0, {}).get("answer_f1") or 1e-9
    print("\n" + "=" * 74)
    print(f"{'K':>4}  {'median tok':>11}  {'vs K=0':>8}  {'eos':>5}  {'f1':>7}  {'f1 vs K=0':>10}")
    for k, r in table.items():
        print(f"{k:>4}  {r['median_gen_tokens']:>11.1f}  "
              f"{r['median_gen_tokens']/base_t:>7.2f}x  {r['eos_rate']:>5.2f}  "
              f"{r['answer_f1']:>7.4f}  {r['answer_f1']/base_f1:>9.2f}x")
    print("\nA K only wins if median tokens drop AND eos stays high AND f1 does not fall off.")
    json.dump(table, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
