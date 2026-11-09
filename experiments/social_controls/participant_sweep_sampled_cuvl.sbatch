#!/bin/bash
#SBATCH --job-name=participant-sampled
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=participant-sweep-sampled-%j.log
#SBATCH --error=participant-sweep-sampled-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="/tmp/llm-psychology-matplotlib-${SLURM_JOB_ID}"
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$PYTHON" -m experiments.social_controls.participant_sweep --sample --output-dir "$PWD/results/social_controls/participant_sweep/sampled_job_${SLURM_JOB_ID}"
