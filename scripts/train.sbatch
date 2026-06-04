#!/bin/bash
#SBATCH --job-name=jepa-m1
#SBATCH --gres=gpu:2          # 2x H100  (CHECK your cluster's gres syntax: gpu:h100:2 etc.)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# --- cluster-specific: adjust module names to `module avail` on YOUR cluster ---
module load cuda || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jepa-omni

export HF_HOME=/scratch/$USER/hf_cache
# export HF_TOKEN=...   # needed only for gated models (Llama-3.2-1B / EmbeddingGemma)

# single-node 2-GPU run
torchrun --standalone --nproc_per_node=2 train_m1.py --config configs/m1.yaml
