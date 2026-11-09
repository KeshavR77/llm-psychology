import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .activations import add_activations, capture_residuals, make_diffmeans
from .analyze import analyze_run, write_json
from .prompts import DATA_FILE, ROOT, prepare_items, selected_conditions


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Isolated MMLU repetition and speaker-attribution controls.")
    parser.add_argument("--experiment", choices=["speaker_free", "single_participant", "both"], default="both")
    parser.add_argument("--lines", type=positive_int, default=7, help="Number of answer lines, not total participants")
    parser.add_argument("--speaker-free-format", choices=["xml", "plain-labeled"], default="xml",
                        help="Use bare XML answers or neutral Answer: <letter> lines for the speaker-free condition")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "social_controls" / "mmlu",
                        help="Parent directory; a seed_<seed>_lines_<lines> run directory is appended")
    parser.add_argument("--dry-run", action="store_true", help="Print all selected prompts as JSON without loading weights or writing outputs")
    parser.add_argument("--overwrite", action="store_true", help="Replace the selected run directory and all its previous outputs")
    return parser.parse_args(argv)


def run_inference(runner, items, conditions, parse_answer, capture_activations=True, do_sample=False):
    import torch

    results = {name: [] for name in conditions}
    sums = {name: {} for name in conditions}
    tokenizer = runner.tokenizer
    runner.model.eval()
    with torch.inference_mode():
        for index, item in enumerate(items):
            for condition in conditions:
                prompt = item["prompts"][condition]
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {key: value.to(runner.device) for key, value in inputs.items()}
                token_ids = inputs["input_ids"][0].tolist()
                position = len(token_ids) - 1
                if capture_activations:
                    with capture_residuals(runner.model, position) as activations:
                        runner.model(**inputs, use_cache=False)
                    if len(activations) != len(runner.model.model.layers):
                        raise RuntimeError("Not all transformer layers produced activations")
                    add_activations(sums[condition], activations)
                output = runner.generate(**inputs, do_sample=do_sample, max_new_tokens=50)
                response_ids = output["sequences"][0, len(token_ids):].tolist()
                response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer = parse_answer("mmlu", response)
                parseable = answer in "ABCD" and len(answer) == 1
                results[condition].append({
                    **{key: value for key, value in item.items() if key != "prompts"},
                    "condition": condition,
                    "prompt": prompt,
                    "prompt_token_ids": token_ids,
                    "sequence_length": len(token_ids),
                    "position": "last_prompt_token",
                    "position_index": position,
                    "position_token": tokenizer.convert_ids_to_tokens(token_ids[position]),
                    "position_token_id": token_ids[position],
                    "raw_response": response,
                    "response_token_ids": response_ids,
                    "model_answer": answer,
                    "parseable": parseable,
                    "is_correct": parseable and answer == item["ground_truth"],
                    "adopts_wrong_target": parseable and answer == item["wrong_target"],
                    "generation_attempts": 1,
                })
            print(f"[{index + 1}/{len(items)}] {item['item_id'][:12]}", flush=True)
    return results, sums


def main(argv=None):
    args = parse_args(argv)
    conditions = selected_conditions(args.experiment)
    items = prepare_items(args.lines, args.seed, args.limit, speaker_free_format=args.speaker_free_format)
    if args.dry_run:
        preview = [{**item, "prompts": {name: item["prompts"][name] for name in conditions}} for item in items]
        print(json.dumps({"conditions": conditions, "sample_count": len(items), "items": preview}, indent=2))
        return

    run_dir = args.output_dir / f"seed_{args.seed}_lines_{args.lines}"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run outputs already exist: {run_dir}. Use --overwrite to replace this run.")

    import torch
    from multi_actor import MODEL_IDS, load_model, parse_answer

    torch.manual_seed(args.seed)
    runner, _ = load_model("qwen")
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    (run_dir / "shared").mkdir(parents=True, exist_ok=True)
    controls = [name for name in conditions if name not in ("no_context", "multiple_participants")]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    manifest = {
        "status": "running", "experiment": args.experiment, "controls": controls,
        "model": "qwen", "model_name": MODEL_IDS["qwen"],
        "tokenizer": runner.tokenizer.name_or_path,
        "model_commit": getattr(runner.model.config, "_commit_hash", None),
        "tokenizer_commit": runner.tokenizer.init_kwargs.get("_commit_hash"),
        "model_dtype": str(next(runner.model.parameters()).dtype),
        "torch_version": str(torch.__version__),
        "repository_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "repository_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "dataset": "mmlu", "data_file": str(DATA_FILE), "sample_count": len(items),
        "item_ids": [item["item_id"] for item in items], "lines": args.lines,
        "seed": args.seed, "limit": args.limit, "conditions": conditions,
        "speaker_free_format": args.speaker_free_format,
        "input_encoding": "raw", "do_sample": False, "max_new_tokens": 50,
        "generation_attempts": 1, "activation_kind": "residual",
        "position": "last_prompt_token", "bootstrap_samples": 10_000, "bootstrap_seed": 42,
    }
    write_json(run_dir / "run.json", manifest)
    results, sums = run_inference(runner, items, conditions, parse_answer)
    for condition, records in results.items():
        directory = run_dir / ("shared" if condition in ("no_context", "multiple_participants") else condition)
        directory.mkdir(exist_ok=True)
        name = f"{condition}.json" if directory.name == "shared" else "behavioral.json"
        write_json(directory / name, {"condition": condition, "results": records})
    for control in controls:
        vectors = make_diffmeans(sums["multiple_participants"], sums[control], len(items))
        torch.save({
            **manifest,
            "status": "complete",
            "direction": f"multiple_participants_minus_{control}",
            "layers": sorted(vectors), "diffmeans": vectors,
        }, run_dir / control / "diffmeans.pt")
    analyze_run(run_dir)
    manifest["status"] = "complete"
    write_json(run_dir / "run.json", manifest)
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
