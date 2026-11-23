#!/bin/bash
#SBATCH --job-name=participant-role-logprobs
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=participant-role-logprobs-%j.log
#SBATCH --error=participant-role-logprobs-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON="$PWD/.venv/bin/python"
REFERENCE_DIR="${1:?Pass the original choice-logprobs run directory}"
RANDOM_REFERENCE_DIR="${2:?Pass the original random-choice-logprobs run directory}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$PYTHON" -m experiments.social_controls.participant_role_logprobs \
    --reference-dir "$REFERENCE_DIR" \
    --random-reference-dir "$RANDOM_REFERENCE_DIR" \
    --output-dir "$PWD/results/social_controls/participant_role_logprobs/job_${SLURM_JOB_ID}"
