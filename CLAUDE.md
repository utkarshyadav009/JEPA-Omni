# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda activate jepa-omni   # Python 3.11; see requirements.txt for packages
export HF_TOKEN=<token>    # required for gated checkpoints (EmbeddingGemma, Llama)
```

## Commands

```bash
# Smoke tests — each file is directly runnable; exercises one module on CPU/GPU
python -m models.losses
python -m models.predictor
python -m models.vision_encoder        # downloads V-JEPA 2 ViT-L (~1.3 GB first run)
python -m models.text_target
python -m models.spine_m1              # M0 gate: prints a loss, shapes correct

# Latency / VRAM probe (run before tuning batch size)
python probe_latency.py --frames 64 --res 256 --iters 30
python probe_gpu.py
python probe_wavjepa.py                # verifies WavJEPA loads and shapes are correct

# Feature extraction (run ONCE before training; skip_encoder must be True in config)
python scripts/extract_features.py    # M1: vision features for MSR-VTT / VATEX
python scripts/extract_features_av.py # M2: AV features for VGGSound → /dev/shm/jepa_m2_cache

# M1 training and eval
python train_m1.py --config configs/m1.yaml
torchrun --nproc_per_node=2 train_m1.py --config configs/m1.yaml
python eval_m1.py --config configs/m1.yaml --ckpt checkpoints/m1/best.pt

# M2 training
python train_m2.py --config configs/m2.yaml
torchrun --nproc_per_node=2 train_m2.py --config configs/m2.yaml
python train_m2.py --config configs/m2.yaml --max-steps 200  # smoke test

# Diagnostics
python scripts/cross_modal_diagnostic.py
python scripts/assert_sigreg_grad.py
python scripts/cavmae_retrieval.py
```

## Architecture

### Milestone structure
- **M1 (done):** Offline vision-text spine. Frozen V-JEPA 2 ViT-L → trainable predictor → InfoNCE alignment against a frozen text target. Achieved V→T R@1 = 22.5 on MSR-VTT+VATEX with feature caching.
- **M2 (in progress):** Joint audio-visual predictor. Cross-modal masked latent prediction: mask one modality over a ~50% time window, predict its frozen latents from the other. World-State vector is attention-pooled output of the full (unmasked) sequence.

### Model components (`models/`)

| File | Frozen? | Role |
|---|---|---|
| `vision_encoder.py` | Yes | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), BF16. Interface: `encode(frames_list) → (B, N, 1024)` |
| `text_target.py` | Yes | EmbeddingGemma (1536-d, faithful) or MiniLM (384-d, smoke). Interface: `encode(texts) → (B, D)` |
| `predictor.py` | Trainable | Maps encoder output to shared dim. Modes: `mlp` (best default), `transformer`, `llama_last8` |
| `spine_m1.py` | Partial | Assembles M1 spine. `forward() → (loss, metrics)`. `embed_video()` / `embed_text()` for eval |
| `audio_encoder.py` | Yes | WavJEPA base+nat wrapper. Input: `(B, 1, n_samples)` raw 16kHz PCM. Output: `(B, T, 768)` |
| `av_jepa_predictor.py` | Trainable | M2 joint predictor. `AVJepaConfig` + `AVJepaPredictor`. Produces World-State vector (never L2-norm it — SIGReg targets N(0,I)) |
| `sigreg.py` | — | Anti-collapse regularization alongside the prediction loss |
| `losses.py` | — | `info_nce`, `info_nce_with_queue`, `compute_siglip_loss` |

### Data pipeline (`data/`)
- `video_text_dataset.py` — real-time decoding via `torchcodec`; slow for training
- `cached_feature_dataset.py` — reads pre-computed BF16 tensors; **use this for M1 training**
- `av_cached_dataset.py` — AV cached features for M2; reads from `/dev/shm/jepa_m2_cache`

### Configs (`configs/`)
- `m1.yaml` — M1 defaults; two profiles: **smoke** (MiniLM + mlp, no HF license) vs **faithful** (EmbeddingGemma + llama_last8, gated)
- `m1_scale.yaml` — scaled M1 variant
- `m2.yaml` — M2 joint AV training; feature cache at `/dev/shm/jepa_m2_cache`

## Critical engineering notes

**Feature caching is mandatory for real training.** Running V-JEPA 2 live during training causes OOM across DDP workers and 2.6 min/step. Set `model.skip_encoder: true` in the config and point `data` at the cache dir. Extract once with `scripts/extract_features.py` (M1) or `scripts/extract_features_av.py` (M2).

**InfoNCE requires large batches.** Small batches (≤8) collapse or overfit even with a MoCo queue (`queue_size`). Batch 256+ with cached features is what drove M1 from R@1=16.8 to 22.5. The MoCo queue helps on single-GPU but isn't a substitute for real batch scale.

**World-State must not be L2-normalised.** `AVJepaPredictor.encode_world_state()` returns un-normalised vectors. SIGReg targets N(0,I), which is incompatible with the unit sphere.

**WavJEPA loading workaround.** The standard `AutoModel.from_pretrained` path hits a `transformers 5.x` bug (`all_tied_weights_keys` AttributeError). `audio_encoder.py` loads the model class directly from the HF snapshot with manual safetensor loading — do not change this.

**Authoritative files — do not recreate or modify:** `models/`, `configs/`, `scripts/probe_latency.py`, `requirements.txt`, `scripts/train.sbatch`. Exact class names: `VisionEncoder`, `TextTarget`, `Predictor`, `SpineM1`, `SpineConfig`, `info_nce`, `AVJepaPredictor`, `AVJepaConfig`, `sigreg`.

## Standing rule: falsifier tracking across M4 stages

After EVERY stage that touches the M3 connector or the M4b speech projector
(joint training, LoRA, M4c, anything downstream) — re-run BOTH standalone
falsifiers and append a row to `checkpoints/falsifier_tracking.md` before
considering that stage done:
  - **M3**: word-overlap F1 normal/swapped/zeroed + semantic cosine, no
    speech stream present (`scripts/m4_joint_eval.py`'s M3-standalone path).
  - **M4b**: semantic cosine normal / swapped-vs-target / swapped-vs-donor,
    no M3 stream present (same script's M4b-standalone path).

Frozen-LLM baselines (pre-any-joint-training): M3 F1 0.471/0.268/0.274,
cos 0.724. M4b cos 0.517/0.152/0.517. The risk this rule guards against is
**compounding drift across stages**, not any single stage's hit in
isolation — the first joint-exposure fine-tune already cost −9% M3 F1 and
−19% M4b cosine on its own; a few more stages each costing that much would
add up to something worth stopping for, even if no single stage looks alarming.

## Milestone gates

- **M0:** `python -m models.spine_m1` prints a finite loss and correct shapes.
- **M1:** Video→text R@1 within 5 pts of SigLIP2 baseline (32.5); abort/switch encoder if >10 pts worse. **Status: PASSED** (Run 5: R@1 = 22.5 with MLP + batch 256 + MSR-VTT+VATEX cached).
- **M2:** `effective_rank(world_state)` stays above a threshold (collapse = rank collapses); prediction loss decreases steadily.

## Data locations

- MSR-VTT videos: `/home/jovyan/work/data/msrvtt/video`
- MSR-VTT annotations: `msrvtt_train_7k.json`, `msrvtt_test_1k.json`
- VATEX / VGGSound: `/home/utkarsh/data/vggsound`
- M2 AV feature cache: `/dev/shm/jepa_m2_cache` (RAM disk, ~1 TiB available)
- CAV-MAE checkpoint: `~/models/cav-mae/audio_model.25.pth`
