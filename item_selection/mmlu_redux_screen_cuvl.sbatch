#!/bin/bash
#SBATCH --job-name=redux-screen
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=redux-screen-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
case "$SLURM_ARRAY_TASK_ID" in
    0) model=qwen ;;
    1) model=llama ;;
    *) exit 1 ;;
esac
.venv/bin/python -m item_selection.mmlu_redux screen --model "$model"
