"""scripts/m3_semantic_rescore.py — re-score action_detailed / summary_detailed
with a semantic (sentence-embedding cosine) metric instead of word-overlap F1.

Word-overlap F1 penalizes valid paraphrases in long free-text prose (many
ways to correctly describe the same detailed scene share few exact words),
which likely explains why the multi-granularity falsifier found the
smallest normal-vs-swapped gap on the two "detailed" fields. This rescores
normal/swapped/prior-baseline generations for those two fields using
sentence-transformers/all-MiniLM-L6-v2 (mean-pooled last hidden state,
L2-normalized, cosine similarity) -- used in its native embedding space
(NOT models/text_target.py's TextTarget, whose projection head into the
1536-d shared space is untrained/random for this purpose; raw MiniLM mean-
pooling IS what the model was actually trained for -- semantic textual
similarity).

Reads checkpoints/m3_multigran/multigran_falsifier_results.json's
"all_clips" field (per-clip ground_truth/normal/swapped text for all 200
held-out clips, written by the (rerun) m3_multigran_falsifier.py).

Usage:
    python scripts/m3_semantic_rescore.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, word_overlap_f1

DETAILED_FIELDS = ["gpt_action_detailed", "gpt_summary_detailed"]
RESULTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "m3_multigran", "multigran_falsifier_results.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "m3_multigran", "semantic_rescore_results.json")


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


@torch.no_grad()
def encode(texts, tokenizer, model, device, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        tok = tokenizer(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        hidden = model(**tok).last_hidden_state
        pooled = mean_pool(hidden, tok["attention_mask"])
        out.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[semrescore] loading sentence-transformers/all-MiniLM-L6-v2 (raw, no extra projection)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    all_clips = results["all_clips"]
    print(f"[semrescore] loaded {len(all_clips)} clips from {RESULTS_PATH}", flush=True)

    print("[semrescore] rebuilding train captions for prior baselines...", flush=True)
    train_pairs, _ = build_splits(DETAILED_FIELDS)
    from collections import Counter, defaultdict
    train_by_field = defaultdict(list)
    for cid, field, text in train_pairs:
        train_by_field[field].append(text)
    mode_caption = {f: Counter(train_by_field[f]).most_common(1)[0][0] for f in DETAILED_FIELDS}
    rng = random.Random(2)

    report = {}
    for field in DETAILED_FIELDS:
        gts, normals, swappeds = [], [], []
        for row in all_clips:
            fv = row["fields"][field]
            gts.append(fv["ground_truth"]); normals.append(fv["normal"]); swappeds.append(fv["swapped"])
        n = len(gts)
        random_train = [rng.choice(train_by_field[field]) for _ in range(n)]
        mode_train = [mode_caption[field]] * n

        print(f"[semrescore] encoding {field} ({n} clips x 5 text lists)...", flush=True)
        z_gt = encode(gts, tokenizer, model, device)
        z_normal = encode(normals, tokenizer, model, device)
        z_swapped = encode(swappeds, tokenizer, model, device)
        z_random = encode(random_train, tokenizer, model, device)
        z_mode = encode(mode_train, tokenizer, model, device)

        def mean_cos(z_a, z_b):
            return (z_a * z_b).sum(-1).mean().item()

        sem = {
            "n": n,
            "cos_normal": mean_cos(z_normal, z_gt),
            "cos_swapped": mean_cos(z_swapped, z_gt),
            "cos_mode_baseline": mean_cos(z_mode, z_gt),
            "cos_random_train_baseline": mean_cos(z_random, z_gt),
        }
        sem["semantic_gap_normal_minus_swapped"] = sem["cos_normal"] - sem["cos_swapped"]

        # word-overlap F1 recomputed here too, for a direct side-by-side of both metrics
        f1_normal = sum(word_overlap_f1(p, g) for p, g in zip(normals, gts)) / n
        f1_swapped = sum(word_overlap_f1(p, g) for p, g in zip(swappeds, gts)) / n
        sem["f1_normal"] = f1_normal
        sem["f1_swapped"] = f1_swapped
        sem["f1_gap_normal_minus_swapped"] = f1_normal - f1_swapped

        report[field] = sem
        print(f"[semrescore] {field}: "
              f"cos normal={sem['cos_normal']:.4f} swapped={sem['cos_swapped']:.4f} "
              f"mode={sem['cos_mode_baseline']:.4f} random={sem['cos_random_train_baseline']:.4f} "
              f"gap={sem['semantic_gap_normal_minus_swapped']:.4f}  ||  "
              f"F1 normal={f1_normal:.4f} swapped={f1_swapped:.4f} gap={sem['f1_gap_normal_minus_swapped']:.4f}",
              flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[semrescore] DONE. wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
