"""Sweep prior-participant counts using the new raw-prompt behavioral protocol."""

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

from .analyze import percentile, write_json
from .prompts import ROOT, DATA_FILE, prepare_items
from .run import positive_int, run_inference


def summarize_sweep(conditions, bootstrap_samples=10_000, seed=42):
    """Use the same bootstrapped item indices for every participant count."""
    counts = sorted(conditions)
    if not counts or not conditions[counts[0]]:
        raise ValueError("At least one nonempty condition is required")
    reference = conditions[counts[0]]
    identities = [(r["item_id"], r["ground_truth"], r["wrong_target"]) for r in reference]
    if len({identity[0] for identity in identities}) != len(identities):
        raise ValueError("Item IDs must be unique")
    for n in counts:
        if [(r["item_id"], r["ground_truth"], r["wrong_target"]) for r in conditions[n]] != identities:
            raise ValueError("Items, order, and wrong targets must match across conditions")

    def metrics(rows, indices):
        parseable = sum(rows[i]["parseable"] for i in indices)
        correct = sum(rows[i]["is_correct"] for i in indices)
        adopted = sum(rows[i]["adopts_wrong_target"] for i in indices)
        return {
            "conformity": adopted / parseable if parseable else None,
            "wrong_target_adoption_all_items": adopted / len(indices),
            "accuracy": correct / parseable if parseable else None,
            "parse_rate": parseable / len(indices),
        }

    indices = list(range(len(reference)))
    observed = {n: metrics(conditions[n], indices) for n in counts}
    samples = {n: {metric: [] for metric in observed[n]} for n in counts}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        draw = rng.choices(indices, k=len(indices))
        for n in counts:
            for metric, value in metrics(conditions[n], draw).items():
                if value is not None:
                    samples[n][metric].append(value)
    summaries = []
    for n in counts:
        rows = conditions[n]
        entry = {"participants": n, "prior_answer_lines": n - 1, "items": len(rows),
                 "parseable": sum(r["parseable"] for r in rows),
                 "parse_failures": sum(not r["parseable"] for r in rows),
                 "correct": sum(r["is_correct"] for r in rows),
                 "wrong_target_adoptions": sum(r["adopts_wrong_target"] for r in rows)}
        for metric, value in observed[n].items():
            entry[metric] = value
            entry[f"{metric}_ci_low"] = percentile(samples[n][metric], 0.025)
            entry[f"{metric}_ci_high"] = percentile(samples[n][metric], 0.975)
            entry[f"{metric}_valid_bootstrap_samples"] = len(samples[n][metric])
        summaries.append(entry)
    return summaries


def save_summaries(output_dir, summaries):
    write_json(output_dir / "summary.json", {
        "description": "Participants = prior named answer lines + model; the model is not explicitly assigned a participant role.",
        "conformity_denominator": "parseable outputs, matching the paper plot",
        "wrong_target_adoption_all_items_denominator": "all selected items; parse failures count as no adoption",
        "bootstrap": "paired item bootstrap, 10000 resamples, seed 42",
        "conditions": summaries,
    })
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["participants"] for r in summaries]
    y = [r["conformity"] if r["conformity"] is not None else float("nan") for r in summaries]
    lo = [r["conformity_ci_low"] if r["conformity_ci_low"] is not None else float("nan") for r in summaries]
    hi = [r["conformity_ci_high"] if r["conformity_ci_high"] is not None else float("nan") for r in summaries]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(x, y, color="#1f77b4", marker="o", markersize=3.5, linewidth=1.8)
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.14, linewidth=0)
    ax.set_title("MMLU")
    ax.set_xlabel("Participants (including model)")
    ax.set_ylabel("Conformity rate (parseable responses)")
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"conformity_by_participants.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true", help="Sample using the model generation configuration instead of greedy decoding")
    parser.add_argument("--participants", type=int, nargs="+", choices=range(2, 11),
                        default=list(range(2, 11)), help="Participant counts including the model (default: 2 through 10)")
    args = parser.parse_args(argv)
    if len(args.participants) != len(set(args.participants)):
        parser.error("participant counts must not be repeated")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "dataset": str(DATA_FILE), "model": "Qwen/Qwen3-8B",
                "seed": args.seed, "limit": args.limit, "participants": args.participants,
                "model_is_explicit_participant": False, "prompt_format": "new raw answer-only",
                "sampling": args.sample, "generation_attempts": 1, "max_new_tokens": 50,
                "activation_collection": False, "bootstrap_samples": 10_000, "bootstrap_seed": 42}
    write_json(args.output_dir / "run.json", manifest)

    import torch
    import transformers
    from multi_actor import MODEL_IDS, load_model, parse_answer

    torch.manual_seed(args.seed)
    runner, _ = load_model("qwen")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    manifest.update({"model": MODEL_IDS["qwen"], "model_commit": getattr(runner.model.config, "_commit_hash", None),
                     "tokenizer": runner.tokenizer.name_or_path,
                     "tokenizer_commit": runner.tokenizer.init_kwargs.get("_commit_hash"),
                     "model_dtype": str(next(runner.model.parameters()).dtype),
                     "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                     "sampling_parameters": {name: getattr(runner.generation_config, name, None)
                                             for name in ("temperature", "top_k", "top_p")} if args.sample else None,
                     "repository_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                     "repository_dirty": bool(status.stdout) if status.returncode == 0 else None})
    write_json(args.output_dir / "run.json", manifest)
    conditions = {}
    for n in args.participants:
        items = prepare_items(lines=n - 1, seed=args.seed, limit=args.limit)
        print(f"Participants {n} ({n - 1} prior answers)", flush=True)
        results, _ = run_inference(runner, items, ("multiple_participants",), parse_answer,
                                   capture_activations=False, do_sample=args.sample)
        conditions[n] = results["multiple_participants"]
        write_json(args.output_dir / f"participants_{n}.json", {"participants": n, "results": conditions[n]})
    summaries = summarize_sweep(conditions)
    save_summaries(args.output_dir, summaries)
    manifest.update({"status": "complete", "sample_count": len(items), "item_ids": [i["item_id"] for i in items]})
    write_json(args.output_dir / "run.json", manifest)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
