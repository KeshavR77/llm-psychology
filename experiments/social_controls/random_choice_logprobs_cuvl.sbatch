#!/bin/bash
#SBATCH --job-name=random-choice-logprobs
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=random-choice-logprobs-%j.log
#SBATCH --error=random-choice-logprobs-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON="$PWD/.venv/bin/python"
REFERENCE_DIR="${1:?Pass the completed choice-logprobs run directory as the first argument}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$PYTHON" -m experiments.social_controls.random_choice_logprobs \
    --reference-dir "$REFERENCE_DIR" \
    --output-dir "$PWD/results/social_controls/random_choice_logprobs/job_${SLURM_JOB_ID}"
