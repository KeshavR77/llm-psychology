#!/bin/bash
#SBATCH --job-name=redux-behavior
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=redux-behavior-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
case "$SLURM_ARRAY_TASK_ID" in
    0) dataset=mmlu; style=base ;;
    1) dataset=mmlu; style=explain ;;
    2) dataset=bbh; style=base ;;
    3) dataset=bbh; style=explain ;;
    *) exit 1 ;;
esac
.venv/bin/python run_behavioral.py --dataset "$dataset" --style "$style" --seed 42 --resume
