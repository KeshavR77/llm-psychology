"""Paths for the original pipeline's MMLU-Redux experiment series."""
from pathlib import Path

REDUX_DATA = Path("data/mmlu_redux_500.json")
REDUX_ROOT = Path("results/mmlu_redux")
BEHAVIORAL_DIR = REDUX_ROOT / "behavioral"
LAYER_SWEEP_DIR = REDUX_ROOT / "layer_sweep"
PROJECTION_DIR = REDUX_ROOT / "projection"
STEERING_DIR = REDUX_ROOT / "activation_patching"
GENERATED_DIR = REDUX_ROOT / "generated"
FIGURES_DIR = REDUX_ROOT / "figures"
REFERENCE_VECTOR = LAYER_SWEEP_DIR / "qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_500_diffmeans.pt"
