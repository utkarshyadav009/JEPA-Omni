"""scripts/eval_m2_audioset20k.py — M2 Pipeline Evaluation on AudioSet-20K.

Evaluates trained M2 joint Audio-Visual predictor checkpoints on the AudioSet-20K
dataset (20,550 train audio clips / 18,886 test audio clips in WebDataset format).

Extraction:
  1. WavJEPA base encoder extracts frozen ambient features (B, 996, 768).
  2. Trained M2 Predictor (AVJepaPredictor) encodes World-State representations
     (B, 1024) via encode_world_state().

Evaluation Protocol (Standard AudioSet Linear Probing):
  1. Train a linear probe (nn.Linear(1024, 527)) on AudioSet-20K train set
     using BCEWithLogitsLoss for 30 epochs.
  2. Evaluate on AudioSet-20K eval test set.
  3. Compute mean Average Precision (mAP), Top-1 Accuracy, and Top-5 Accuracy.

Usage:
    /home/utkarsh/miniconda3/envs/jepa-omni/bin/python scripts/eval_m2_audioset20k.py \
        --ckpt checkpoints/m2_fusion_20k_best/step19000_peak.pt
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time
from typing import Dict, List, Tuple

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.audio_encoder import AudioEncoder, WAVJEPA_BASE_REPO
from models.av_jepa_predictor import AVJepaConfig, AVJepaPredictor

NUM_CLASSES = 527  # AudioSet 527 ontology classes
TARGET_AUDIO_LEN = 160000  # 10s @ 16kHz


class AudioSet20kDataset(Dataset):
    """Parses AudioSet-20K WebDataset tar files directly from disk."""

    def __init__(self, tar_paths: List[str], max_samples: int | None = None):
        super().__init__()
        self.items: List[Tuple[bytes, List[int]]] = []
        t0 = time.time()
        print(f"[AudioSet20k] Indexing {len(tar_paths)} tar files...", flush=True)
        
        for tp in sorted(tar_paths):
            if max_samples and len(self.items) >= max_samples:
                break
            with tarfile.open(tp, "r") as tar:
                by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
                for member in tar.getmembers():
                    if "." not in member.name:
                        continue
                    key, ext = member.name.rsplit(".", 1)
                    if key not in by_key:
                        by_key[key] = {}
                    by_key[key][ext] = member

                for key, files in by_key.items():
                    if "wav" in files and "json" in files:
                        f_json = tar.extractfile(files["json"])
                        if f_json is None:
                            continue
                        meta = json.load(f_json)
                        labels = meta.get("label", [])
                        f_wav = tar.extractfile(files["wav"])
                        if f_wav is None:
                            continue
                        wav_bytes = f_wav.read()
                        self.items.append((wav_bytes, labels))
                        if max_samples and len(self.items) >= max_samples:
                            break
                            
        print(f"[AudioSet20k] Loaded {len(self.items):,} samples in {time.time()-t0:.2f}s", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_bytes, labels = self.items[idx]
        try:
            wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = AF.resample(wav, sr, 16000)
            if wav.shape[1] < TARGET_AUDIO_LEN:
                wav = F.pad(wav, (0, TARGET_AUDIO_LEN - wav.shape[1]))
            else:
                wav = wav[:, :TARGET_AUDIO_LEN]
        except Exception:
            wav = torch.zeros(1, TARGET_AUDIO_LEN, dtype=torch.float32)

        # Create multi-hot label vector (size 527)
        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for l in labels:
            if 0 <= l < NUM_CLASSES:
                target[l] = 1.0

        return wav, target


def audioset_collate_fn(batch):
    wavs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return wavs, targets


@torch.no_grad()
def extract_dataset_features(
    dataset: Dataset,
    encoder: AudioEncoder,
    predictor: AVJepaPredictor | None,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=audioset_collate_fn
    )
    all_features: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    
    t0 = time.time()
    n_done = 0
    total = len(dataset)

    for wavs, targets in loader:
        wavs = wavs.to(device)
        # WavJEPA extraction
        ambient_feats = encoder.encode(wavs)  # (B, 996, 768)
        
        if predictor is not None:
            B, T, D = ambient_feats.shape
            tbins = torch.linspace(0, 511, T).long().unsqueeze(0).repeat(B, 1).to(device)
            feats = {"ambient": ambient_feats}
            tbins_dict = {"ambient": tbins}
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                # World-State representation (1024-d)
                ws = predictor.encode_world_state(feats, tbins_dict)
            reprs = ws.float().cpu()
        else:
            # Baseline: mean-pooled raw WavJEPA features (768-d)
            reprs = ambient_feats.mean(dim=1).float().cpu()

        all_features.append(reprs)
        all_targets.append(targets)
        n_done += wavs.shape[0]
        if n_done % (batch_size * 10) == 0 or n_done == total:
            elapsed = time.time() - t0
            print(f"  extracted {n_done}/{total} ({n_done/elapsed:.1f} samples/s)", flush=True)

    features_tensor = torch.cat(all_features, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    return features_tensor, targets_tensor


def compute_map(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes mean Average Precision (mAP) over 527 AudioSet classes."""
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()
    aps = []
    for c in range(targets_np.shape[1]):
        if targets_np[:, c].sum() > 0:
            ap = sklearn.metrics.average_precision_score(targets_np[:, c], probs[:, c])
            aps.append(ap)
    return float(np.mean(aps)) * 100.0 if aps else 0.0


def compute_topk_acc(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Computes Top-k multi-label accuracy."""
    topk_preds = logits.topk(k, dim=1).indices.cpu()
    targets_cpu = targets.cpu()
    hits = 0
    for i in range(logits.shape[0]):
        gt_classes = set(torch.where(targets_cpu[i] > 0)[0].tolist())
        pred_classes = set(topk_preds[i].tolist())
        if len(gt_classes & pred_classes) > 0:
            hits += 1
    return (hits / logits.shape[0]) * 100.0


def train_linear_probe(
    train_feats: torch.Tensor,
    train_targets: torch.Tensor,
    test_feats: torch.Tensor,
    test_targets: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[str, float]:
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset_tr = torch.utils.data.TensorDataset(train_feats, train_targets)
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)

    print(f"\n[LinearProbe] Training probe ({feat_dim}d → {NUM_CLASSES}d) for {epochs} epochs...", flush=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            
        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / len(dataset_tr)
            print(f"  Epoch {epoch:02d}/{epochs:02d} - BCE Loss: {avg_loss:.4f}", flush=True)

    print(f"[LinearProbe] Probe training complete in {time.time()-t0:.2f}s", flush=True)

    # Evaluation on Test set
    probe.eval()
    with torch.no_grad():
        test_logits = []
        for i in range(0, test_feats.shape[0], batch_size):
            bx = test_feats[i : i + batch_size].to(device)
            test_logits.append(probe(bx).cpu())
        test_logits = torch.cat(test_logits, dim=0)

    map_score = compute_map(test_logits, test_targets)
    top1_acc = compute_topk_acc(test_logits, test_targets, k=1)
    top5_acc = compute_topk_acc(test_logits, test_targets, k=5)

    return {
        "mAP": round(map_score, 2),
        "Top-1 Acc": round(top1_acc, 2),
        "Top-5 Acc": round(top5_acc, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="checkpoints/m2_fusion_20k_best/step19000_peak.pt",
        help="Path to M2 predictor checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/utkarsh/raid2-data/audioset_20k",
        help="Path to AudioSet-20K directory",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=None, help="Limit train samples for fast testing")
    parser.add_argument("--limit-test", type=int, default=None, help="Limit test samples for fast testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("M2 Pipeline Evaluation on AudioSet-20K")
    print(f"Device: {device} | Checkpoint: {args.ckpt}")
    print("=" * 65)

    # 1. Load AudioSet-20K Train & Test datasets
    train_tars = glob.glob(os.path.join(args.data_dir, "balanced", "train", "*.tar"))
    test_tars = glob.glob(os.path.join(args.data_dir, "balanced", "test", "*.tar"))

    if not train_tars or not test_tars:
        raise FileNotFoundError(f"No tar files found in {args.data_dir}")

    ds_train = AudioSet20kDataset(train_tars, max_samples=args.limit_train)
    ds_test = AudioSet20kDataset(test_tars, max_samples=args.limit_test)

    # 2. Load WavJEPA AudioEncoder
    encoder = AudioEncoder(WAVJEPA_BASE_REPO, n_channels=1, device=str(device))

    # 3. Load M2 Predictor
    predictor_cfg = AVJepaConfig(d_model=1024, depth=8, heads=8)
    predictor = AVJepaPredictor(predictor_cfg).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["model"], strict=False)
        print(f"\n[M2] Loaded predictor from {args.ckpt} (step {ckpt.get('step', 'N/A')})")
    else:
        print(f"\n[M2] WARNING: Checkpoint {args.ckpt} not found! Using initialized weights.")
    predictor.eval()

    # 4. Extract M2 World-State Representations
    print("\n--- Extracting M2 World-State Representations ---")
    train_m2_feats, train_targets = extract_dataset_features(ds_train, encoder, predictor, device)
    test_m2_feats, test_targets = extract_dataset_features(ds_test, encoder, predictor, device)

    # 5. Extract WavJEPA Baseline Representations (No M2 Predictor Trunk)
    print("\n--- Extracting WavJEPA Baseline Representations ---")
    train_wav_feats, _ = extract_dataset_features(ds_train, encoder, None, device)
    test_wav_feats, _ = extract_dataset_features(ds_test, encoder, None, device)

    # 6. Evaluate Linear Probes
    print("\n" + "=" * 65)
    print("RESULTS 1: WavJEPA Baseline (Audio-only Encoder)")
    print("-" * 65)
    res_wav = train_linear_probe(
        train_wav_feats, train_targets, test_wav_feats, test_targets, device, epochs=args.epochs, lr=args.lr
    )
    for metric, val in res_wav.items():
        print(f"  {metric:<12}: {val:.2f}%")

    print("\n" + "=" * 65)
    print("RESULTS 2: M2 Joint Audio-Visual Predictor (World-State)")
    print("-" * 65)
    res_m2 = train_linear_probe(
        train_m2_feats, train_targets, test_m2_feats, test_targets, device, epochs=args.epochs, lr=args.lr
    )
    for metric, val in res_m2.items():
        print(f"  {metric:<12}: {val:.2f}%")
    print("=" * 65)

    # Save summary results
    results_summary = {
        "checkpoint": args.ckpt,
        "n_train": len(ds_train),
        "n_test": len(ds_test),
        "wavjepa_baseline": res_wav,
        "m2_predictor": res_m2,
    }
    out_file = os.path.join(PROJECT_ROOT, "data", "m2_audioset20k_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved evaluation summary to {out_file}")


if __name__ == "__main__":
    main()
