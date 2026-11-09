#!/bin/bash
#SBATCH --job-name=paper-baseline-n8
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=paper-baseline-%j.log
#SBATCH --error=paper-baseline-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON="$PWD/.venv/bin/python"
OUTPUT_PARENT="$PWD/results/social_controls/original_baseline"
OUTPUT_DIR="$OUTPUT_PARENT/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_PARENT"
mkdir "$OUTPUT_DIR"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
import torch, transformers
assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
status = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=True).stdout
cache_ref = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/refs/main"
manifest = {"job_id": os.environ["SLURM_JOB_ID"], "repository_revision": revision,
            "repository_dirty": bool(status), "runner": "multi_actor.py", "model": "Qwen/Qwen3-8B",
            "cached_model_ref": cache_ref.read_text().strip() if cache_ref.is_file() else None,
            "dataset": "mmlu", "mode": "8", "explain": False, "sampling": True,
            "seed": 42, "max_new_tokens": 50, "max_attempts": 10, "batch_size": 16,
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "dataset_sha256": hashlib.sha256(Path("data/combined_mmlu_correct.json").read_bytes()).hexdigest()}
(Path(sys.argv[1]) / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
"$PYTHON" multi_actor.py --model qwen --dataset mmlu --mode 8 --seed 42 --output-file "$OUTPUT_DIR/qwen_mmlu_8_base.json"
"$PYTHON" - "$OUTPUT_DIR/qwen_mmlu_8_base.json" <<'PY'
import json, sys
from pathlib import Path
new = json.loads(Path(sys.argv[1]).read_text())
paper = json.loads(Path('results/behavioral/mmlu/qwen_mmlu_8_base.json').read_text())
assert len(new['results']) == len(paper['results']) == 100
assert all(a['question'] == b['question'] and a['prompt'] == b['prompt'] and a['main_wrong'] == b['main_wrong'] for a,b in zip(new['results'], paper['results'])), 'The prompts do not match the saved paper run'
print('Verified all 100 prompts and wrong targets match the saved paper run.')
print(f"Conformity: {new['conformity_rate']:.3f}; accuracy: {new['accuracy']:.3f}; parseable: {new['parseable']}/100")
PY
