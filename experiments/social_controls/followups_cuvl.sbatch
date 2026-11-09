#!/bin/bash
#SBATCH --job-name=social-followup
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=social-followup-%j.log
#SBATCH --error=social-followup-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
case "${FOLLOWUP:?set FOLLOWUP to neutral_label or three_lines}" in
    neutral_label)
        "$PYTHON" -m experiments.social_controls.run --experiment speaker_free --speaker-free-format plain-labeled --output-dir results/social_controls/followups/neutral_label/mmlu
        ;;
    three_lines)
        "$PYTHON" -m experiments.social_controls.run --lines 3 --output-dir results/social_controls/followups/three_lines/mmlu
        ;;
    *) printf 'Unknown follow-up: %s\n' "$FOLLOWUP" >&2; exit 1 ;;
esac
