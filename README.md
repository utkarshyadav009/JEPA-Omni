# JEPA-Omni
A Audio Visual Congruent JEPA model connected with an LLM for world modelling

## Milestone M1 — video/text alignment

A learnable "spine" aligns video and text on top of frozen SigLIP2 towers
(residual adapters + learnable SigLIP logit scale/bias). At initialisation the
adapters are the identity, so the spine reproduces the SigLIP2 zero-shot
retrieval baseline before any training.

Install dependencies (a matching FFmpeg is required for `torchcodec`):

```bash
pip install -r requirements.txt
```

Point `configs/m1.yaml` (the `data:` section) at your MSR-VTT copy, then:

```bash
# Sanity-check the data pipeline on a tiny subset
python -m data.video_text_dataset \
    --video-dir /data/msrvtt/videos \
    --annotation /data/msrvtt/test_videodatainfo.json --limit 8

# Train (single process or 2-GPU DDP)
python train_m1.py --config configs/m1.yaml
torchrun --nproc_per_node=2 train_m1.py --config configs/m1.yaml
sbatch scripts/train_m1.sbatch                       # SLURM

# Evaluate retrieval + the M1 gate
python eval_m1.py --config configs/m1.yaml --checkpoint checkpoints/m1/best.pt

# Probe vision encode latency on synthetic clips
python scripts/probe_latency.py --config configs/m1.yaml --batch-size 8
```

- **M0 gate** (printed during training): loss must decrease over the first few
  hundred steps.
- **M1 gate** (printed during eval): video→text R@1 within 5 pts of the SigLIP2
  baseline passes; >10 pts worse aborts (switch encoder).
