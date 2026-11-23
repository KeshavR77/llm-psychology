#!/bin/bash
#SBATCH --job-name=redux-figures
#SBATCH --partition=cuvl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=redux-figures-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
.venv/bin/python data_analysis/bootstrap_qwen_results.py --dataset mmlu
.venv/bin/python data_analysis/plot_qwen_participants.py --dataset mmlu
