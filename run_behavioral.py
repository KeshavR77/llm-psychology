"""Run original Qwen behavioral conditions in the Redux namespace."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from pipeline_config import BEHAVIORAL_DIR, REDUX_DATA
from multi_actor import DATA_DIR, load_model, run_experiment

MODEL = "qwen"
DATASETS = ["mmlu", "bbh"]
MODES = [str(i) for i in range(1, 11)] + ["qd", "da"]
EXPLAINS = [False, True]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="Run one dataset (default: both).")
    parser.add_argument("--style", choices=["base", "explain"], help="Run one prompt style (default: both).")
    parser.add_argument("--seed", type=int, default=None, help="Set the generation seed for each condition.")
    parser.add_argument("--resume", action="store_true", help="Skip matching completed conditions; fail if an existing result differs.")
    args = parser.parse_args(argv)

    datasets = [args.dataset] if args.dataset else DATASETS
    explains = [args.style == "explain"] if args.style else EXPLAINS
    combos = list(itertools.product(datasets, MODES, explains))
    sources = {dataset: REDUX_DATA if dataset == "mmlu" else DATA_DIR / "combined_bbh_correct.json" for dataset in datasets}
    source_bytes = {dataset: path.read_bytes() for dataset, path in sources.items()}
    expected = {dataset: (hashlib.sha256(data).hexdigest(), len(json.loads(data))) for dataset, data in source_bytes.items()}
    pending = []
    for dataset, mode, explain in combos:
        style = "explain" if explain else "base"
        output = BEHAVIORAL_DIR / dataset / f"{MODEL}_{dataset}_{mode}_{style}.json"
        if args.resume and output.exists():
            result = json.loads(output.read_text())
            digest, count = expected[dataset]
            if (result.get("dataset_sha256") != digest or result.get("attempts") != count or
                    len(result.get("results", [])) != count or result.get("model") != MODEL or
                    result.get("dataset") != dataset or result.get("mode") != mode or
                    result.get("seed") != args.seed):
                raise ValueError(f"Existing result does not match this run: {output}")
            print(f"[skip] {output}", flush=True)
        else:
            pending.append((dataset, mode, explain, output))
    if not pending:
        print("All requested conditions are complete.")
        return

    model, device = load_model(MODEL)
    for i, (dataset, mode, explain, output) in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] model={MODEL} dataset={dataset} mode={mode} explain={explain}", flush=True)
        run_experiment(MODEL, dataset, mode, explain=explain, model=model, device=device,
                       dataset_file=sources[dataset] if dataset == "mmlu" else None,
                       output_file=output, seed=args.seed)
    print("\nAll requested Qwen experiments complete.")


if __name__ == "__main__":
    main()
