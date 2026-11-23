"""Score independently random prior answers using the plain-answer Qwen protocol."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .analyze import write_json
from .choice_logprobs import make_items, make_prompts, score_items, summarize
from .prompts import DATA_FILE, LETTERS
from .run import positive_int


def make_random_items(lines=7, seed=42, random_seed=1729, limit=None):
    items = make_items(lines=lines, seed=seed, limit=limit)
    rng = random.Random(random_seed)
    for item in items:
        answers = [rng.choice(LETTERS) for _ in range(lines)]
        item["peer_answers"] = answers
        item["prompts"] = {"random_participants": make_prompts(item, lines, answers)["multiple_participants"]}
    return items


def compare_to_reference(random_rows, reference, items, lines, seed, bootstrap_samples=10_000):
    """Pair by item ID and verify that the saved reference prompts are identical."""
    expected = make_items(lines=lines, seed=seed, limit=len(items))
    aligned = {}
    for condition in ("no_context", "multiple_participants"):
        rows = reference[condition]
        by_id = {row["item_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Reference contains duplicate item IDs")
        aligned[condition] = []
        for item in expected:
            row = by_id.get(item["item_id"])
            if row is None or any(row[key] != item[key] for key in ("ground_truth", "wrong_target")):
                raise ValueError("Reference items or target answers do not match")
            if row["prompt"] != item["prompts"][condition]:
                raise ValueError("Reference prompt differs from this experiment's prompt")
            aligned[condition].append(row)
    if [row["item_id"] for row in random_rows] != [item["item_id"] for item in expected]:
        raise ValueError("Random and reference item order does not match")

    def paired(first_name, first, second_name, second):
        result = summarize({"no_context": first, "multiple_participants": second},
                           bootstrap_samples=bootstrap_samples)
        return {
            "conditions": {first_name: result["conditions"]["no_context"],
                           second_name: result["conditions"]["multiple_participants"]},
            "difference": f"{second_name}_minus_{first_name}",
            "paired_differences": result["paired_differences_multiple_participants_minus_no_context"],
            "bootstrap_samples": result["bootstrap_samples"],
            "bootstrap_seed": result["bootstrap_seed"],
        }

    baseline = aligned["no_context"]
    unanimous = aligned["multiple_participants"]
    eligible = [i for i, row in enumerate(baseline) if row["is_correct"]]
    return {
        "random_minus_no_context": paired("no_context", baseline, "random_participants", random_rows),
        "unanimous_wrong_minus_random": paired("random_participants", random_rows, "unanimous_wrong", unanimous),
        "no_context_correct_items": len(eligible),
        "wrong_target_choices_among_no_context_correct": {
            "random_participants": sum(random_rows[i]["adopts_wrong_target"] for i in eligible),
            "unanimous_wrong": sum(unanimous[i]["adopts_wrong_target"] for i in eligible),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=positive_int, default=7)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the matched wrong-answer targets")
    parser.add_argument("--random-seed", type=int, default=1729, help="Independent seed for participant answers")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--output-dir", type=Path, help="New output directory, required except for --dry-run")
    parser.add_argument("--reference-dir", type=Path, help="Completed choice_logprobs run for paired comparisons")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    items = make_random_items(args.lines, args.seed, args.random_seed, args.limit)
    if args.dry_run:
        print(json.dumps({"sample_count": len(items), "items": items}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless using --dry-run")

    reference_manifest, reference = None, None
    dataset_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    if args.reference_dir:
        reference_manifest = json.loads((args.reference_dir / "run.json").read_text())
        reference = json.loads((args.reference_dir / "results.json").read_text())
        if (reference_manifest.get("status") != "complete" or
            reference_manifest.get("dataset_sha256") != dataset_hash or
            reference_manifest.get("seed") != args.seed or reference_manifest.get("lines") != args.lines):
            raise ValueError("Reference run has different data, target seed, participant count, or is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    from multi_actor import MODEL_IDS, load_model

    runner, _ = load_model("qwen")
    manifest = {
        "status": "running", "model": MODEL_IDS["qwen"],
        "model_commit": getattr(runner.model.config, "_commit_hash", None),
        "tokenizer": runner.tokenizer.name_or_path,
        "tokenizer_commit": runner.tokenizer.init_kwargs.get("_commit_hash"),
        "model_dtype": str(next(runner.model.parameters()).dtype),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": runner.device, "gpu": torch.cuda.get_device_name(0) if runner.device.startswith("cuda") else None,
        "dataset": str(DATA_FILE), "dataset_sha256": dataset_hash,
        "sample_count": len(items), "item_ids": [item["item_id"] for item in items],
        "seed": args.seed, "random_seed": args.random_seed, "lines": args.lines,
        "limit": args.limit, "batch_size": args.batch_size,
        "prior_answers": "independent uniform draws from A-D, including the correct answer",
        "scoring": "next-token log probabilities for space-prefixed A, B, C, D after Your answer:",
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
    }
    if reference_manifest:
        for key in ("model", "model_commit", "model_dtype", "tokenizer"):
            if manifest[key] != reference_manifest.get(key):
                raise ValueError(f"Reference {key} does not match this run")
    write_json(args.output_dir / "run.json", manifest)
    results, token_ids = score_items(runner, items, args.batch_size)
    if reference_manifest and token_ids != reference_manifest.get("answer_token_ids"):
        raise ValueError("Answer token IDs differ from the reference run")
    rows = results["random_participants"]
    summary = {
        "item_count": len(rows),
        "random_participants": {
            "accuracy": sum(row["is_correct"] for row in rows) / len(rows),
            "wrong_target_choice_rate": sum(row["adopts_wrong_target"] for row in rows) / len(rows),
            "mean_wrong_target_probability_given_abcd": sum(
                row["answer_probabilities_given_abcd"][row["wrong_target"]] for row in rows) / len(rows),
        },
        "probability_note": "Probabilities given ABCD are renormalized over only the four answer tokens.",
    }
    if reference:
        summary["paired_comparisons"] = compare_to_reference(rows, reference, items, args.lines, args.seed)
    write_json(args.output_dir / "results.json", results)
    write_json(args.output_dir / "summary.json", summary)
    manifest.update({"status": "complete", "answer_token_ids": token_ids})
    write_json(args.output_dir / "run.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
