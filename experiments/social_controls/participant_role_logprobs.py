"""Test explicit quiz participation with no responses, random peers, and wrong peers."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .analyze import write_json
from .choice_logprobs import make_items, score_items, summarize
from .prompts import DATA_FILE, ROOT
from .random_choice_logprobs import make_random_items
from .run import positive_int


def role_prefix(lines):
    return f"You are Participant {lines + 1} in a quiz with {lines + 1} participants.\n\n"


def make_role_items(lines=7, seed=42, random_seed=1729, limit=None):
    items = make_items(lines=lines, seed=seed, limit=limit)
    random_items = make_random_items(lines=lines, seed=seed, random_seed=random_seed, limit=limit)
    prefix = role_prefix(lines)
    for item, random_item in zip(items, random_items):
        if any(item[key] != random_item[key] for key in ("item_id", "ground_truth", "wrong_target")):
            raise ValueError("Random and unanimous conditions must use the same items and targets")
        item["random_peer_answers"] = random_item["peer_answers"]
        item["prompts"] = {
            "role_no_responses": prefix + item["prompts"]["no_context"],
            "role_random_participants": prefix + random_item["prompts"]["random_participants"],
            "role_unanimous_wrong": prefix + item["prompts"]["multiple_participants"],
        }
    return items


def align_references(items, reference, random_reference, lines):
    prefix = role_prefix(lines)
    sources = {
        "no_context": (reference["no_context"], "role_no_responses"),
        "random_participants": (random_reference["random_participants"], "role_random_participants"),
        "unanimous_wrong": (reference["multiple_participants"], "role_unanimous_wrong"),
    }
    aligned = {}
    for name, (rows, role_condition) in sources.items():
        by_id = {row["item_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate item IDs in {name} reference")
        aligned[name] = []
        for item in items:
            row = by_id.get(item["item_id"])
            if row is None or any(row[key] != item[key] for key in ("ground_truth", "wrong_target")):
                raise ValueError(f"Items or target answers do not match the {name} reference")
            if prefix + row["prompt"] != item["prompts"][role_condition]:
                raise ValueError(f"Prompt other than role prefix differs from the {name} reference")
            if name == "random_participants" and row["peer_answers"] != item["random_peer_answers"]:
                raise ValueError("Random participant answers do not match the reference")
            aligned[name].append(row)
    return aligned


def paired(first_name, first, second_name, second, bootstrap_samples=10_000):
    result = summarize({"no_context": first, "multiple_participants": second},
                       bootstrap_samples=bootstrap_samples)
    return {
        "conditions": {first_name: result["conditions"]["no_context"],
                       second_name: result["conditions"]["multiple_participants"]},
        "difference": f"{second_name}_minus_{first_name}",
        "paired_differences": result["paired_differences_multiple_participants_minus_no_context"],
        "reference_correct_items": result["baseline_correct_items"],
        "wrong_target_choices_among_reference_correct": result["peer_wrong_target_choices_among_baseline_correct"],
        "wrong_target_rate_among_reference_correct": result["peer_wrong_target_rate_among_baseline_correct"],
        "bootstrap_samples": result["bootstrap_samples"], "bootstrap_seed": result["bootstrap_seed"],
    }


def summarize_role(results, references=None, bootstrap_samples=10_000):
    no_responses = results["role_no_responses"]
    random_peers = results["role_random_participants"]
    wrong_peers = results["role_unanimous_wrong"]
    comparisons = {
        "random_vs_no_responses": paired("role_no_responses", no_responses, "role_random_participants", random_peers, bootstrap_samples),
        "unanimous_vs_no_responses": paired("role_no_responses", no_responses, "role_unanimous_wrong", wrong_peers, bootstrap_samples),
        "unanimous_vs_random": paired("role_random_participants", random_peers, "role_unanimous_wrong", wrong_peers, bootstrap_samples),
    }
    if references is not None:
        for name, role_condition in (("no_context", "role_no_responses"),
                                     ("random_participants", "role_random_participants"),
                                     ("unanimous_wrong", "role_unanimous_wrong")):
            comparisons[f"role_effect_{name}"] = paired(name, references[name], role_condition,
                                                         results[role_condition], bootstrap_samples)
    return {
        "item_count": len(no_responses), "comparisons": comparisons,
        "probability_note": "Probabilities given ABCD are renormalized over only the four answer tokens.",
        "role_baseline_note": "The no-responses condition still assigns the model a role in a multi-participant quiz.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=positive_int, default=7)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--reference-dir", type=Path, help="Completed no-role no-peer/unanimous run")
    parser.add_argument("--random-reference-dir", type=Path, help="Completed no-role random-peer run")
    parser.add_argument("--output-dir", type=Path, help="New output directory, required except for --dry-run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.reference_dir) != bool(args.random_reference_dir):
        parser.error("Both reference directories must be supplied together")
    items = make_role_items(args.lines, args.seed, args.random_seed, args.limit)
    if args.dry_run:
        print(json.dumps({"sample_count": len(items), "items": items}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless using --dry-run")

    dataset_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    reference_manifests = []
    references = None
    if args.reference_dir:
        for path in (args.reference_dir, args.random_reference_dir):
            manifest = json.loads((path / "run.json").read_text())
            if (manifest.get("status") != "complete" or manifest.get("dataset_sha256") != dataset_hash or
                manifest.get("seed") != args.seed or manifest.get("lines") != args.lines):
                raise ValueError(f"Reference {path} has mismatched data or configuration, or is incomplete")
            reference_manifests.append(manifest)
        if reference_manifests[1].get("random_seed") != args.random_seed:
            raise ValueError("Random-answer seed differs from the reference")
        references = align_references(items, json.loads((args.reference_dir / "results.json").read_text()),
                                      json.loads((args.random_reference_dir / "results.json").read_text()), args.lines)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    from multi_actor import MODEL_IDS, load_model

    runner, _ = load_model("qwen")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    manifest = {
        "status": "running", "model": MODEL_IDS["qwen"],
        "model_commit": getattr(runner.model.config, "_commit_hash", None),
        "tokenizer": runner.tokenizer.name_or_path,
        "tokenizer_commit": runner.tokenizer.init_kwargs.get("_commit_hash"),
        "model_dtype": str(next(runner.model.parameters()).dtype),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "device": runner.device, "gpu": torch.cuda.get_device_name(0) if runner.device.startswith("cuda") else None,
        "repository_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "repository_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "dataset": str(DATA_FILE), "dataset_sha256": dataset_hash,
        "sample_count": len(items), "item_ids": [item["item_id"] for item in items],
        "seed": args.seed, "random_seed": args.random_seed, "lines": args.lines,
        "limit": args.limit, "batch_size": args.batch_size, "role_prefix": role_prefix(args.lines),
        "conditions": list(items[0]["prompts"]),
        "scoring": "next-token log probabilities for space-prefixed A, B, C, D after Your answer:",
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        "random_reference_dir": str(args.random_reference_dir) if args.random_reference_dir else None,
    }
    for reference in reference_manifests:
        for key in ("model", "model_commit", "model_dtype", "tokenizer"):
            if manifest[key] != reference.get(key):
                raise ValueError(f"Reference {key} does not match this run")
    write_json(args.output_dir / "run.json", manifest)
    results, token_ids = score_items(runner, items, args.batch_size)
    if any(token_ids != reference.get("answer_token_ids") for reference in reference_manifests):
        raise ValueError("Answer token IDs differ from the reference runs")
    summary = summarize_role(results, references)
    summary["role_baseline_note"] = (f"The no-responses condition still says the model is one of {args.lines + 1} participants.")
    write_json(args.output_dir / "results.json", results)
    write_json(args.output_dir / "summary.json", summary)
    manifest.update({"status": "complete", "answer_token_ids": token_ids})
    write_json(args.output_dir / "run.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
