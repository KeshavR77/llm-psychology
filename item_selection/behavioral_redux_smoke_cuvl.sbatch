#!/bin/bash
#SBATCH --job-name=redux-smoke
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=redux-behavior-smoke-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path
import torch
from multi_actor import load_model, run_experiment
from pipeline_config import REDUX_DATA

assert torch.cuda.is_available(), "CUDA is unavailable"
model, device = load_model("qwen")
with tempfile.TemporaryDirectory(prefix="redux-behavior-smoke-") as tmp:
    for dataset, mode, explain in (("mmlu", "1", False), ("mmlu", "10", True), ("bbh", "1", False)):
        result = run_experiment("qwen", dataset, mode, explain=explain, model=model, device=device,
                                dataset_file=REDUX_DATA if dataset == "mmlu" else None,
                                limit=2, output_file=Path(tmp) / f"{dataset}-{mode}.json", seed=42)
        assert result["attempts"] == 2 and len(result["results"]) == 2
print("Behavioral GPU smoke test passed")
PY
