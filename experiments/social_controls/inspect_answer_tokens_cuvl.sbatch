#!/bin/bash
#SBATCH --job-name=inspect-answer-tokens
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=inspect-answer-tokens-%j.log
#SBATCH --error=inspect-answer-tokens-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHON="$PWD/.venv/bin/python"
CHOICE_REFERENCE="${1:?Pass the choice-logprobs run directory}"
RANDOM_REFERENCE="${2:?Pass the random-choice-logprobs run directory}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$PYTHON" -m experiments.social_controls.inspect_answer_tokens \
    --choice-reference "$CHOICE_REFERENCE" --random-reference "$RANDOM_REFERENCE" \
    --output-dir "$PWD/results/social_controls/inspect_answer_tokens/job_${SLURM_JOB_ID}"
