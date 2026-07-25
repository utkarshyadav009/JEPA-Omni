"""scripts/m4c_gate_eval.py — M4c duplex-loop gate: turn-taking P/R
(EasyCom sessions = headline, VGGSound pseudo-timeline = mechanism check
only), interruption latency, end-to-end tick latency.

Usage:
    python scripts/m4c_gate_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train_m3 import build_splits, m3_collate_fn, _cap_ambient_len, CACHE_DIR, GRANULARITY_TAGS
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor
from models.m3_connector import M3Connector, M3ConnectorConfig
from models.m4_speech import WhisperSpeechEncoder, UltravoxProjector, UltravoxProjectorConfig
from models.m4_decision_head import SpeakSilenceHead, DecisionHeadConfig
from models.m4_duplex_loop import DuplexLoop
from data.av_cached_dataset import AVCachedDataset
from data.m4_pseudo_timeline import M4PseudoTimelineDataset, m4_collate_fn
from data.m4_easycom_turntaking import build_ticks, EasyComTurnTakingDataset

FIELD = "gpt_sound_acoustic"


def prf1(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return precision, recall, f1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--m2-ckpt", default="checkpoints/m2_fusion_20k_best/step19000_peak.pt")
    p.add_argument("--joint-ckpt", default="checkpoints/m4_joint/best.pt")
    p.add_argument("--whisper", default="openai/whisper-medium")
    p.add_argument("--decision-head-ckpt", default="checkpoints/m4_decision_head/best.pt")
    p.add_argument("--n-latency-samples", type=int, default=20)
    p.add_argument("--interrupt-at-step", type=int, default=5)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="checkpoints/m4_decision_head/m4c_gate_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"[m4c-gate] hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    print("[m4c-gate] loading frozen LLM (no LoRA)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(device)
    llm.eval()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print("[m4c-gate] loading frozen M2 predictor...", flush=True)
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8, mlp_ratio=4.0, max_tdm_bins=512, dropout=0.0)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    m2ckpt = torch.load(args.m2_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(m2ckpt["model"], strict=True)
    predictor.eval()

    print("[m4c-gate] loading M3 connector + M4b projector (post-joint, frozen)...", flush=True)
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device, weights_only=False)
    m3_cfg = M3ConnectorConfig(**joint_ckpt["m3_cfg"])
    m3_connector = M3Connector(m3_cfg).to(device)
    m3_connector.load_state_dict(joint_ckpt["m3_connector"])
    m3_connector.eval()
    m4b_cfg = UltravoxProjectorConfig(**joint_ckpt["m4b_cfg"])
    m4b_projector = UltravoxProjector(m4b_cfg).to(device)
    m4b_projector.load_state_dict(joint_ckpt["m4b_projector"])
    m4b_projector.eval()

    print("[m4c-gate] loading whisper encoder...", flush=True)
    whisper = WhisperSpeechEncoder(args.whisper, dtype=torch.bfloat16).to(device)

    print("[m4c-gate] loading trained decision head...", flush=True)
    dh_ckpt = torch.load(args.decision_head_ckpt, map_location=device, weights_only=False)
    dh_cfg = DecisionHeadConfig(**dh_ckpt["cfg"])
    decision_head = SpeakSilenceHead(dh_cfg).to(device)
    decision_head.load_state_dict(dh_ckpt["state_dict"])
    decision_head.eval()
    threshold = dh_ckpt["threshold"]

    loop = DuplexLoop(predictor, m3_connector, m4b_projector, whisper, decision_head,
                       llm, tokenizer, device, decision_threshold=threshold)

    results = {}

    # ============ EasyCom session-level turn-taking P/R (HEADLINE) ============
    print("\n[m4c-gate] === EasyCom held-out SESSION turn-taking P/R (headline) ===", flush=True)
    _, ec_test_ticks = build_ticks()
    ec_ds = EasyComTurnTakingDataset(ec_test_ticks)
    by_session = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(ec_ds)):
        item = ec_ds[i]
        sf, _ = loop.compute_speech_activity(item["waveform"], item["duration_sec"])
        pred_speak, prob = loop.decide(None, sf)
        gt_speak = item["is_speak"]
        sess = item["session"]
        by_session[sess]["n"] += 1
        if gt_speak and pred_speak: by_session[sess]["tp"] += 1; overall["tp"] += 1
        elif gt_speak and not pred_speak: by_session[sess]["fn"] += 1; overall["fn"] += 1
        elif not gt_speak and pred_speak: by_session[sess]["fp"] += 1; overall["fp"] += 1
        else: by_session[sess]["tn"] += 1; overall["tn"] += 1
        if (i + 1) % 500 == 0:
            print(f"[m4c-gate] EasyCom eval {i+1}/{len(ec_ds)}", flush=True)

    session_report = {}
    for sess, c in sorted(by_session.items()):
        precision, recall, f1 = prf1(c["tp"], c["fp"], c["fn"])
        session_report[sess] = {"n": c["n"], "speak_precision": precision, "speak_recall": recall, "speak_f1": f1,
                                 "silence_recall": c["tn"] / max(1, c["tn"] + c["fp"])}
        print(f"[m4c-gate]   session {sess}: n={c['n']}  speak_P={precision:.3f}  speak_R={recall:.3f}  "
              f"speak_F1={f1:.3f}  silence_R={session_report[sess]['silence_recall']:.3f}", flush=True)
    ov_p, ov_r, ov_f1 = prf1(overall["tp"], overall["fp"], overall["fn"])
    ov_silence_r = overall["tn"] / max(1, overall["tn"] + overall["fp"])
    results["easycom_session_turntaking"] = {"per_session": session_report,
                                              "overall": {"speak_precision": ov_p, "speak_recall": ov_r,
                                                          "speak_f1": ov_f1, "silence_recall": ov_silence_r,
                                                          "n": sum(c["n"] for c in by_session.values())}}
    print(f"[m4c-gate] OVERALL (held-out sessions 10/11/12): speak_P={ov_p:.3f} speak_R={ov_r:.3f} "
          f"speak_F1={ov_f1:.3f} silence_R={ov_silence_r:.3f}", flush=True)

    # ============ VGGSound pseudo-timeline (MECHANISM CHECK ONLY) ============
    print("\n[m4c-gate] === VGGSound pseudo-timeline P/R (MECHANISM CHECK, not a turn-taking result) ===", flush=True)
    _, vgg_test_pairs3 = build_splits(FIELD)
    vgg_test_pairs = [(c, t) for c, _, t in vgg_test_pairs3][:1000]
    vgg_ds = M4PseudoTimelineDataset(vgg_test_pairs, CACHE_DIR, tokenizer, silence_token_id=0)
    vc = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(vgg_ds)):
        item = vgg_ds[i]
        batch = m4_collate_fn([item], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        ws = loop.compute_world_state(feats, tbins)
        pred_speak, prob = loop.decide(ws, None)
        gt_speak = item["label"] == "speak"
        if gt_speak and pred_speak: vc["tp"] += 1
        elif gt_speak and not pred_speak: vc["fn"] += 1
        elif not gt_speak and pred_speak: vc["fp"] += 1
        else: vc["tn"] += 1
        if (i + 1) % 500 == 0:
            print(f"[m4c-gate] VGGSound eval {i+1}/{len(vgg_ds)}", flush=True)
    vp, vr, vf1 = prf1(vc["tp"], vc["fp"], vc["fn"])
    v_silence_r = vc["tn"] / max(1, vc["tn"] + vc["fp"])
    results["vggsound_pseudo_timeline_MECHANISM_CHECK"] = {"speak_precision": vp, "speak_recall": vr,
                                                             "speak_f1": vf1, "silence_recall": v_silence_r,
                                                             "n": sum(vc.values()), "label": "MECHANISM CHECK ONLY, NOT a turn-taking result"}
    print(f"[m4c-gate] VGGSound (mechanism check): speak_P={vp:.3f} speak_R={vr:.3f} speak_F1={vf1:.3f} "
          f"silence_R={v_silence_r:.3f}", flush=True)

    # ============ Interruption latency ============
    print("\n[m4c-gate] === interruption latency ===", flush=True)
    _, vgg_eval_pairs3 = build_splits(FIELD)
    rng.shuffle(vgg_eval_pairs3)
    sample_pairs = [(c, t) for c, _, t in vgg_eval_pairs3[:args.n_latency_samples]]
    ds = AVCachedDataset(cache_dir=CACHE_DIR, clip_ids=[c for c, _ in sample_pairs], max_tdm_bins=512, audio_mode="mean")
    all_token_latencies = []
    interrupt_halts = []
    for i in range(len(ds)):
        item = ds[i]
        batch = m3_collate_fn([{"feats": item["feats"], "tbins": item["tbins"], "clip_id": item["clip_id"],
                                 "prefix_ids": torch.zeros(0, dtype=torch.long), "caption_ids": torch.zeros(1, dtype=torch.long),
                                 "caption_text": "", "field": None}], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        pre_pool = loop.compute_pre_pool(feats, tbins)
        kpm = torch.cat([pad["vision"], pad["ambient"]], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            m3_lat = m3_connector(pre_pool.to(torch.bfloat16), kpm)
        prefix_ids = tokenizer(GRANULARITY_TAGS[FIELD], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        prefix = llm.get_input_embeddings()(prefix_ids)
        soft_prompt = torch.cat([m3_lat, prefix], dim=1)
        attn = torch.ones(soft_prompt.shape[:2], dtype=torch.long, device=device)
        result = loop.generate_interruptible(soft_prompt, attn, max_new_tokens=40,
                                              interrupt_at_step=args.interrupt_at_step)
        all_token_latencies.extend(result.per_token_latencies_ms)
        if result.interrupted:
            interrupt_halts.append(result.per_token_latencies_ms[-1] if result.per_token_latencies_ms else 0.0)

    import statistics
    token_lat_mean = statistics.mean(all_token_latencies)
    token_lat_p95 = statistics.quantiles(all_token_latencies, n=20)[18] if len(all_token_latencies) >= 20 else max(all_token_latencies)
    token_lat_max = max(all_token_latencies)
    results["interruption_latency_ms"] = {
        "per_token_forward_pass_mean_ms": token_lat_mean, "per_token_forward_pass_p95_ms": token_lat_p95,
        "per_token_forward_pass_max_ms": token_lat_max,
        "n_samples": len(ds), "interrupt_at_step": args.interrupt_at_step,
        "interpretation": "worst-case interruption latency = time for the in-flight token's forward pass to "
                           "complete once the halt signal is set, since we check between tokens not mid-token",
    }
    print(f"[m4c-gate] per-token forward pass: mean={token_lat_mean:.2f}ms  p95={token_lat_p95:.2f}ms  "
          f"max={token_lat_max:.2f}ms  (this IS the worst-case interruption latency)", flush=True)
    print(f"[m4c-gate] target sub-second: {'PASS' if token_lat_max < 1000 else 'FAIL'}", flush=True)

    # ============ End-to-end tick latency (perception -> decision -> first token) ============
    print("\n[m4c-gate] === end-to-end tick latency (perception -> decision -> first token) ===", flush=True)
    # Use the REAL tick-fraction-truncated pseudo-timeline items (not raw
    # full clips) -- a full/complete-scene World-State is, by the training
    # distribution's own design, a SILENCE tick ("already reported, nothing
    # new"); only the early/truncated fraction is labeled SPEAK. Sampling
    # from the tick dataset gives a realistic mix of both branches instead
    # of accidentally measuring only the cheap decision-only path.
    tick_sample_pairs = [(c, t) for c, _, t in vgg_eval_pairs3[:args.n_latency_samples]]
    tick_ds = M4PseudoTimelineDataset(tick_sample_pairs, CACHE_DIR, tokenizer, silence_token_id=0)
    tick_latencies = []
    for i in range(min(len(tick_ds), args.n_latency_samples * 4)):
        item = tick_ds[i]
        batch = m4_collate_fn([item], tokenizer.pad_token_id)
        feats = {k: v.to(device) for k, v in batch["feats"].items()}
        tbins = {k: v.to(device) for k, v in batch["tbins"].items()}
        pad = {k: v.to(device) for k, v in batch["padding_mask"].items()}
        _cap_ambient_len(feats, tbins, pad)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        ws = loop.compute_world_state(feats, tbins)
        pred_speak, prob = loop.decide(ws, None)
        t_decision = time.perf_counter()
        if pred_speak:
            pre_pool = loop.compute_pre_pool(feats, tbins)
            kpm = torch.cat([pad["vision"], pad["ambient"]], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                m3_lat = m3_connector(pre_pool.to(torch.bfloat16), kpm)
            prefix_ids = tokenizer(GRANULARITY_TAGS[FIELD], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            prefix = llm.get_input_embeddings()(prefix_ids)
            soft_prompt = torch.cat([m3_lat, prefix], dim=1)
            attn = torch.ones(soft_prompt.shape[:2], dtype=torch.long, device=device)
            with torch.no_grad():
                out = llm(inputs_embeds=soft_prompt, attention_mask=attn, use_cache=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        tick_latencies.append({"total_ms": (t1 - t0) * 1000.0, "decision_ms": (t_decision - t0) * 1000.0,
                                "generation_first_token_ms": (t1 - t_decision) * 1000.0, "decided_speak": pred_speak})

    total_ms = [t["total_ms"] for t in tick_latencies]
    speak_ms = [t["total_ms"] for t in tick_latencies if t["decided_speak"]]
    silence_ms = [t["total_ms"] for t in tick_latencies if not t["decided_speak"]]
    results["end_to_end_tick_latency_ms"] = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "note": "measured on the ACTUAL available hardware (see 'hardware' field) -- NOT a consumer RTX 4090, "
                "flagging so M5 capacity planning doesn't assume 4090-class throughput",
        "per_tick": tick_latencies,
        "mean_ms": statistics.mean(total_ms), "p95_ms": statistics.quantiles(total_ms, n=20)[18] if len(total_ms) >= 20 else max(total_ms),
        "max_ms": max(total_ms),
        "n_decided_speak": len(speak_ms), "n_decided_silence": len(silence_ms),
        "mean_ms_when_speak_branch": statistics.mean(speak_ms) if speak_ms else None,
        "mean_ms_when_silence_branch": statistics.mean(silence_ms) if silence_ms else None,
    }
    print(f"[m4c-gate] end-to-end tick latency: overall mean={statistics.mean(total_ms):.2f}ms  max={max(total_ms):.2f}ms  "
          f"| speak-branch(n={len(speak_ms)}) mean={statistics.mean(speak_ms) if speak_ms else float('nan'):.2f}ms  "
          f"| silence-branch(n={len(silence_ms)}) mean={statistics.mean(silence_ms) if silence_ms else float('nan'):.2f}ms  "
          f"(hardware: {results['end_to_end_tick_latency_ms']['hardware']})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[m4c-gate] DONE. wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
