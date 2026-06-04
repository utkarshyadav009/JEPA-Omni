#!/bin/bash
#SBATCH --job-name=jepa-omni-m1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CONFIG=${1:-configs/m1.yaml}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
MASTER_PORT=${MASTER_PORT:-29500}

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port="${MASTER_PORT}" \
    train_m1.py --config "${CONFIG}"
