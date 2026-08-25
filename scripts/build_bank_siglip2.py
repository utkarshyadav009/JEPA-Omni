"""scripts/build_bank_siglip2.py — build the deployable CAPTION bank in SigLIP2's frozen
text space.

WHAT CHANGED VERSUS THE OLD BANK. `perception_bank_max_fp16.pt` was 121,104 captions x 1536-d
encoded through a *trained* `text_target.proj`. That made it a function of one checkpoint:
the bank silently went out of geometry when paired with a different arm, which is exactly
what happened (it was built with `query_predictor_ddp_lw0.3`'s proj and then used with the
ablation arms). The new target space is SigLIP2's own frozen text space, so:

  * the bank is checkpoint-INDEPENDENT -- any arm trained against this space can use it;
  * it halves in size (768-d instead of 1536-d);
  * it needs no encoder at build time either, because
    `scripts/encode_captions_siglip2.py` already encoded every corpus caption once. This
    script is a SELECTION over that cache, not a re-encode.

`--n-captions` caps the bank. Bank size is a memory decision on a 7.6 GB device, not a
quality-free one, so it is explicit rather than "as many as fit".
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-cache-dir", default="/dev/shm/siglip2_text_cache")
    ap.add_argument("--captions-path",
                    default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_full_captions_v2.jsonl"))
    ap.add_argument("--n-captions", type=int, default=121104,
                    help="cap the bank; 0 = every cached caption")
    ap.add_argument("--min-chars", type=int, default=12,
                    help="drop stub captions that carry no retrievable content")
    ap.add_argument("--ckpt", default="",
                    help="query-predictor checkpoint whose TRAINED text_target_proj to "
                         "apply. Required unless the run used --siglip-shared-dim 0. The "
                         "cache holds RAW SigLIP2 vectors, so the projection is applied "
                         "here, OFFLINE -- which is why no text encoder ships to the "
                         "device even though the target space is learned.")
    ap.add_argument("--out", default="checkpoints/bank_siglip2_fp16.pt")
    args = ap.parse_args()

    texts, vecs, repo = [], [], None
    for sp in sorted(glob.glob(os.path.join(args.text_cache_dir, "text_shard*.pt"))):
        d = torch.load(sp, map_location="cpu", weights_only=False)
        repo = repo or d.get("siglip")
        texts += d["text"]; vecs.append(d["emb"])
    if not texts:
        raise SystemExit(f"no text shards under {args.text_cache_dir} -- run "
                         "scripts/encode_captions_siglip2.py first")
    emb = torch.cat(vecs, 0)
    print(f"[bank] cache: {len(texts)} captions, {tuple(emb.shape)} ({repo})", flush=True)

    keep = [i for i, t in enumerate(texts) if len(t) >= args.min_chars]
    print(f"[bank] {len(keep)} pass min_chars={args.min_chars}", flush=True)
    if args.n_captions and len(keep) > args.n_captions:
        # take the LONGEST captions: they carry the most retrievable detail, and the old
        # 121k bank was likewise dominated by the detailed fields
        keep.sort(key=lambda i: -len(texts[i]))
        keep = keep[: args.n_captions]
    keep.sort()

    sel = emb[torch.as_tensor(keep)].contiguous().float()
    if args.ckpt:
        import torch.nn as nn, torch.nn.functional as F
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tp = ck.get("text_target_proj") or {}
        if not tp:
            raise SystemExit(f"{args.ckpt} has no text_target_proj -- it was trained with "
                             "Identity, so build the bank without --ckpt")
        W, b = tp["weight"].float(), tp["bias"].float()
        sel = F.normalize(sel @ W.t() + b, dim=-1)
        print(f"[bank] applied trained proj {tuple(W.shape)} from {args.ckpt}", flush=True)
    else:
        import torch.nn.functional as F
        sel = F.normalize(sel, dim=-1)
    out_emb = sel.contiguous().to(torch.float16)
    out_txt = [texts[i] for i in keep]
    torch.save({"emb": out_emb, "text": out_txt, "siglip": repo, "space": "siglip2_frozen"},
               args.out)
    print(f"[bank] wrote {args.out}  {tuple(out_emb.shape)}  "
          f"{out_emb.numel()*2/2**20:.0f} MiB  (old bank was 355 MiB at 1536-d)", flush=True)


if __name__ == "__main__":
    main()
