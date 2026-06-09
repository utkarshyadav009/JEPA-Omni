# JEPA-Omni — M0 / M1 scaffold

Frozen V-JEPA 2 ViT-L → trainable predictor → text-embedding alignment via InfoNCE
(VL-JEPA reproduction). No audio, no streaming, no full-duplex yet.

## Layout
```
jepa_omni/
  models/
    vision_encoder.py   # frozen V-JEPA 2 ViT-L wrapper (written)
    text_target.py      # frozen Y-encoder + trainable projection (written)
    predictor.py        # trainable head: mlp | transformer (written)
    losses.py           # bidirectional InfoNCE + diagnostics (written)
    spine_m1.py         # assembles the spine, forward -> loss (written)
  data/
    video_text_dataset.py   # <-- AGENT TASK
  scripts/
    probe_latency.py    # V-JEPA fwd latency/VRAM probe (written)
    train.sbatch        # SLURM template (adjust to your cluster)
  configs/m1.yaml       # grounded defaults + faithful overrides (written)
  train_m1.py           # <-- AGENT TASK
  eval_m1.py            # <-- AGENT TASK
  requirements.txt
```

## Run order
```bash
# 0. env (server / GPU node)
conda create -n jepa-omni python=3.11 -y && conda activate jepa-omni
pip install -r requirements.txt

# 1. module smoke tests (CPU/GPU, no data) — each file is runnable standalone
python -m models.losses
python -m models.predictor
python -m models.vision_encoder     # downloads V-JEPA 2 ViT-L (~1.3 GB)
python -m models.text_target
python -m models.spine_m1           # M0 GATE: prints a loss, shapes correct

# 2. the M5-gating number
python scripts/probe_latency.py --frames 64 --res 256 --iters 30

# 3. train (after the dataset + train_m1.py exist)
sbatch scripts/train.sbatch
python eval_m1.py --config configs/m1.yaml --ckpt /scratch/$USER/jepa_omni/ckpt/m1/best.pt
```

## Profiles
- **Smoke (default):** ungated — MiniLM text target + `mlp` predictor. Runs with no HF license.
- **Faithful (VL-JEPA):** `text_backbone: embeddinggemma` + `predictor_mode: llama_last8`.
  Both gated; set `HF_TOKEN` and accept the licenses.
