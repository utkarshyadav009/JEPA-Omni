"""scripts/step1_go_nogo.py — STEP 1 go/no-go controls.

Run AFTER 1500 steps of mask_mode=whole training to verify the model is
performing genuine cross-modal prediction (not shortcutting).

Controls (on a held-out batch from the training cache):

  (a) VISION-DROPOUT: zero the context modality's features.
      With mask_mode=whole, audio is context and vision is fully masked.
      If we zero audio, the model has ZERO useful context — predicts vision
      from nothing but positional/temporal embeddings.
      PASS = loss stays HIGH  (model needed audio to predict vision).
      FAIL = loss stays low   (model is shortcutting, e.g. predicts mean).

  (b) TEMPORAL-SHUFFLE: replace clip i's audio context with clip j≠i's audio.
      PASS = loss RISES sharply vs normal  (model learned audio-visual pairing).
      FAIL = loss unchanged               (model ignores audio content).

Usage:
    conda run -n jepa-omni python scripts/step1_go_nogo.py \
        --ckpt checkpoints/m2_step1/last.pt \
        --config configs/m2.yaml

Output: one table with three rows: normal / vision-dropout / temporal-shuffle.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.av_jepa_predictor import AVJepaPredictor, AVJepaConfig
from train_m2 import (
    build_dataloader,
    sample_cross_modal_mask,
    load_config,
    cfg_get,
    AttrDict,
)


def measure_loss(
    model: AVJepaPredictor,
    feats: dict,
    tbins: dict,
    mask: dict,
    device: torch.device,
) -> float:
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        loss, _ = model(feats, tbins, mask)
    return float(loss.detach())


def measure_cosine_distance(
    model: AVJepaPredictor,
    feats: dict,
    tbins: dict,
    mask: dict,
    device: torch.device,
) -> float:
    """Second, independent readout of the same forward pass: mean (1 - cosine)
    between predicted and (per-token-normalised) target latents at masked
    positions. Rebuilt from the model's own building-block methods (_embed,
    _backbone, out_head, mask_token) — forward() only returns a scalar loss,
    so this does not alter or duplicate forward()'s method body, it composes
    the same public/semi-public pieces from outside to get at the raw tensors.
    """
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        tokens, mod_ids, _ = model._embed(feats, tbins)
        B, S, d = tokens.shape
        flat_mask = torch.cat([mask[m] for m in feats], 1)
        q = model.mask_token.expand(B, S, d)
        tokens = torch.where(
            flat_mask.unsqueeze(-1),
            q + model._embed_pos_only(feats, tbins),
            tokens,
        )
        h = model._backbone(tokens)

        cos_dists = []
        offset = 0
        for m, x in feats.items():
            T_m = x.shape[1]
            seg = h[:, offset:offset + T_m]
            mm = mask[m]
            if mm.any():
                pred = model.out_head[m](seg[mm]).float()
                tgt = F.layer_norm(x[mm].detach().float(), (x.shape[-1],))
                cos = F.cosine_similarity(pred, tgt, dim=-1)
                cos_dists.append((1.0 - cos).mean())
            offset += T_m
    return float(torch.stack(cos_dists).mean().detach()) if cos_dists else float("nan")


def measure_floor(feats: dict, mask: dict) -> tuple:
    """Zero-predictor and mean-predictor smooth-L1 floor against the SAME
    per-token-normalised target used in training, at the SAME masked
    positions used for the go/no-go controls. No model involved -- a pure
    baseline, so we can confirm the trained model still beats it (i.e. it
    hasn't just gotten hard-but-trivial)."""
    floors_zero, floors_mean = [], []
    for m, x in feats.items():
        mm = mask[m]
        if mm.any():
            tgt = F.layer_norm(x[mm].detach().float(), (x.shape[-1],))
            zero_pred = torch.zeros_like(tgt)
            mean_pred = tgt.mean(0, keepdim=True).expand_as(tgt)
            floors_zero.append(F.smooth_l1_loss(zero_pred, tgt).item())
            floors_mean.append(F.smooth_l1_loss(mean_pred, tgt).item())
    zero_floor = sum(floors_zero) / len(floors_zero) if floors_zero else float("nan")
    mean_floor = sum(floors_mean) / len(floors_mean) if floors_mean else float("nan")
    return zero_floor, mean_floor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--n-batches", type=int, default=8,
                        help="Number of batches to average per condition.")
    parser.add_argument("--mask-mode", default="whole",
                        choices=["whole", "high_frac"],
                        help="Eval mask must mirror the checkpoint's training-time "
                             "masking for a fair control (default: whole).")
    parser.add_argument("--mask-frac", type=float, default=None,
                        help="Contiguous mask fraction, required for --mask-mode high_frac "
                             "(should match the fraction the checkpoint was trained with).")
    args = parser.parse_args()
    if args.mask_mode == "high_frac" and args.mask_frac is None:
        parser.error("--mask-mode high_frac requires --mask-frac")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── build & load model ─────────────────────────────────────────────────
    predictor_cfg = AVJepaConfig(
        d_model      = int(cfg_get(cfg, "model.d_model",       default=1024)),
        depth        = int(cfg_get(cfg, "model.depth",         default=8)),
        heads        = int(cfg_get(cfg, "model.heads",         default=8)),
        mlp_ratio    = float(cfg_get(cfg, "model.mlp_ratio",  default=4.0)),
        max_tdm_bins = int(cfg_get(cfg, "model.max_tdm_bins", default=512)),
        dropout      = float(cfg_get(cfg, "model.dropout",    default=0.0)),
    )
    model = AVJepaPredictor(predictor_cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"[go-nogo] Loaded checkpoint: step={ckpt.get('step', '?')}", flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    loader, _ = build_dataloader(cfg, limit=None)
    import random
    rng = random.Random(999)

    losses_normal  = []
    losses_dropout = []
    losses_shuffle = []
    cos_normal  = []
    cos_dropout = []
    cos_shuffle = []
    floor_zeros = []
    floor_means = []

    n_done = 0
    for batch in loader:
        if n_done >= args.n_batches:
            break
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}

        # Deterministically mask the second modality (vision) for consistent comparison
        mods = list(feats.keys())
        masked_mod = mods[1] if len(mods) > 1 else mods[0]
        ctx_mod    = [m for m in mods if m != masked_mod][0]

        mask = {m: torch.zeros_like(tbins[m], dtype=torch.bool) for m in mods}
        if args.mask_mode == "whole":
            mask[masked_mod] = torch.ones_like(tbins[masked_mod], dtype=torch.bool)
        else:
            # high_frac: contiguous block of args.mask_frac, mirrors training-time masking
            T = tbins[masked_mod].shape[1]
            Bsz = tbins[masked_mod].shape[0]
            n_mask = max(1, int(T * args.mask_frac))
            m_tensor = torch.zeros_like(tbins[masked_mod], dtype=torch.bool)
            for b in range(Bsz):
                start = rng.randint(0, max(0, T - n_mask))
                m_tensor[b, start:start + n_mask] = True
            mask[masked_mod] = m_tensor

        fz, fm = measure_floor(feats, mask)
        floor_zeros.append(fz)
        floor_means.append(fm)

        # ── (normal) cross-modal prediction ───────────────────────────────
        l_normal = measure_loss(model, feats, tbins, mask, device)
        c_normal = measure_cosine_distance(model, feats, tbins, mask, device)
        losses_normal.append(l_normal)
        cos_normal.append(c_normal)

        # ── (a) VISION-DROPOUT: zero context modality ──────────────────────
        feats_dropout = {m: v.clone() for m, v in feats.items()}
        feats_dropout[ctx_mod] = torch.zeros_like(feats_dropout[ctx_mod])
        l_dropout = measure_loss(model, feats_dropout, tbins, mask, device)
        c_dropout = measure_cosine_distance(model, feats_dropout, tbins, mask, device)
        losses_dropout.append(l_dropout)
        cos_dropout.append(c_dropout)

        # ── (b) TEMPORAL-SHUFFLE: mismatched audio context ─────────────────
        B = feats[ctx_mod].shape[0]
        perm = torch.randperm(B)
        # Ensure no identity mapping (roll if perm[i]==i for any i)
        for i in range(B):
            if perm[i] == i:
                perm[i], perm[(i + 1) % B] = perm[(i + 1) % B], perm[i]
        feats_shuffle = {m: v.clone() for m, v in feats.items()}
        feats_shuffle[ctx_mod] = feats[ctx_mod][perm]
        l_shuffle = measure_loss(model, feats_shuffle, tbins, mask, device)
        c_shuffle = measure_cosine_distance(model, feats_shuffle, tbins, mask, device)
        losses_shuffle.append(l_shuffle)
        cos_shuffle.append(c_shuffle)

        n_done += 1
        print(f"  batch {n_done}/{args.n_batches}  "
              f"normal={l_normal:.4f}/{c_normal:.4f}  "
              f"dropout={l_dropout:.4f}/{c_dropout:.4f}  "
              f"shuffle={l_shuffle:.4f}/{c_shuffle:.4f}  (loss/cos-dist)", flush=True)

    avg_n = sum(losses_normal)  / len(losses_normal)
    avg_d = sum(losses_dropout) / len(losses_dropout)
    avg_s = sum(losses_shuffle) / len(losses_shuffle)

    avg_cn = sum(cos_normal)  / len(cos_normal)
    avg_cd = sum(cos_dropout) / len(cos_dropout)
    avg_cs = sum(cos_shuffle) / len(cos_shuffle)

    dropout_delta = avg_d - avg_n
    shuffle_delta = avg_s - avg_n
    cos_dropout_delta = avg_cd - avg_cn
    cos_shuffle_delta = avg_cs - avg_cn

    avg_floor_zero = sum(floor_zeros) / len(floor_zeros)
    avg_floor_mean = sum(floor_means) / len(floor_means)
    beats_floor = avg_n < avg_floor_mean
    floor_margin_pct = (avg_floor_mean - avg_n) / (avg_floor_mean + 1e-8) * 100

    # ── Verdicts (loss-based, PRIMARY readout) ──────────────────────────────
    verdict_a = "PASS" if dropout_delta / (avg_n + 1e-8) > 0.20 else "FAIL"
    verdict_b = "PASS" if shuffle_delta / (avg_n + 1e-8) > 0.10 else "FAIL"

    # ── Verdicts (cosine-distance, SECOND/corroborating readout) ────────────
    # Same thresholds applied to 1-cos delta (relative to normal's 1-cos).
    verdict_a_cos = "PASS" if cos_dropout_delta / (avg_cn + 1e-8) > 0.20 else "FAIL"
    verdict_b_cos = "PASS" if cos_shuffle_delta / (avg_cn + 1e-8) > 0.10 else "FAIL"

    print()
    print("=" * 68)
    print("STEP 1 GO/NO-GO CONTROL TABLE — PRIMARY (training loss, smooth-L1 vs normalised target)")
    print(f"  masked_mod={masked_mod}  ctx_mod={ctx_mod}  "
          f"mask_mode={args.mask_mode}  mask_frac={args.mask_frac}  "
          f"n_batches={args.n_batches}")
    print("-" * 68)
    print(f"  Zero-predictor floor  = {avg_floor_zero:.4f}")
    print(f"  Mean-predictor floor  = {avg_floor_mean:.4f}")
    print(f"  Trained loss (normal) = {avg_n:.4f}  "
          f"({'beats' if beats_floor else 'DOES NOT beat'} mean floor by {floor_margin_pct:.1f}%)")
    print("-" * 68)
    print(f"  Condition                       Loss     Delta vs normal")
    print(f"  Normal (cross-modal)          {avg_n:7.4f}   —")
    print(f"  (a) Vision-dropout (zero ctx) {avg_d:7.4f}  +{dropout_delta:7.4f}  "
          f"→ {verdict_a}")
    print(f"  (b) Temporal-shuffle          {avg_s:7.4f}  +{shuffle_delta:7.4f}  "
          f"→ {verdict_b}")
    print("=" * 68)
    print("STEP 1 GO/NO-GO CONTROL TABLE — SECOND READOUT (1 - cosine, corroboration only)")
    print("-" * 68)
    print(f"  Condition                     1-cos     Delta vs normal")
    print(f"  Normal (cross-modal)          {avg_cn:7.4f}   —")
    print(f"  (a) Vision-dropout (zero ctx) {avg_cd:7.4f}  +{cos_dropout_delta:7.4f}  "
          f"→ {verdict_a_cos}")
    print(f"  (b) Temporal-shuffle          {avg_cs:7.4f}  +{cos_shuffle_delta:7.4f}  "
          f"→ {verdict_b_cos}")
    print("-" * 68)

    # ── Combined decision: BOTH readouts must PASS both controls ────────────
    all_pass = all(v == "PASS" for v in
                    (verdict_a, verdict_b, verdict_a_cos, verdict_b_cos))
    if all_pass:
        print("DECISION: GO — both controls PASS on BOTH loss and cosine readouts.")
    else:
        failed = [name for name, v in [
            ("(a) vision-dropout / loss", verdict_a),
            ("(b) temporal-shuffle / loss", verdict_b),
            ("(a) vision-dropout / cosine", verdict_a_cos),
            ("(b) temporal-shuffle / cosine", verdict_b_cos),
        ] if v == "FAIL"]
        print("DECISION: NO-GO — the following checks FAILED:")
        for f in failed:
            print(f"    - {f}")
        print("  Do NOT launch the week run.")
    print("=" * 68)


if __name__ == "__main__":
    main()
