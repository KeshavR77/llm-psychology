#!/bin/bash
#SBATCH --job-name=social-controls-test
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=social-controls-test-%j.log
#SBATCH --error=social-controls-test-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON="$PWD/.venv/bin/python"
JOB_DIR="$PWD/results/social_controls/smoke/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

"$PYTHON" -c 'import torch, transformers, accelerate; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0)); print("torch:", torch.__version__, "transformers:", transformers.__version__)'
"$PYTHON" -m unittest -v experiments.social_controls.test_social_controls
"$PYTHON" -m experiments.social_controls.run --dry-run > "$JOB_DIR/preview.json"
"$PYTHON" -m experiments.social_controls.run --limit 2 --output-dir "$JOB_DIR"
"$PYTHON" - "$JOB_DIR/seed_42_lines_7" <<'PY'
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
manifest = json.loads((root / "run.json").read_text())
assert manifest["status"] == "complete" and manifest["sample_count"] == 2
for control in ("speaker_free", "single_participant"):
    for name in ("behavioral.json", "analysis.json", "summary.csv", "summary.md", "diffmeans.pt"):
        assert (root / control / name).is_file(), (control, name)
    vectors = torch.load(root / control / "diffmeans.pt", map_location="cpu", weights_only=True)["diffmeans"]
    assert vectors and all(torch.isfinite(v).all() for v in vectors.values())
for condition in ("no_context", "multiple_participants"):
    assert (root / "shared" / f"{condition}.json").is_file()
assert (root / "activation_summary.json").is_file()
assert (root / "activation_summary.csv").is_file()
print(f"GPU smoke outputs verified: {root}")
PY
