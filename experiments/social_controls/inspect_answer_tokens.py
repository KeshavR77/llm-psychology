"""Inspect competing next tokens in completed choice-scoring prompts."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from .analyze import write_json
from .choice_logprobs import answer_token_ids
from .run import positive_int


def load_inputs(choice_dir, random_dir, limit):
    manifests = [json.loads((path / "run.json").read_text()) for path in (choice_dir, random_dir)]
    for manifest in manifests:
        if manifest.get("status") != "complete":
            raise ValueError("Both reference runs must be complete")
    for key in ("model", "model_commit", "model_dtype", "tokenizer", "dataset_sha256", "lines", "seed", "answer_token_ids"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"Reference runs disagree on {key}")
    choices = json.loads((choice_dir / "results.json").read_text())
    randoms = json.loads((random_dir / "results.json").read_text())
    conditions = {"no_context": choices["no_context"], "unanimous_wrong": choices["multiple_participants"],
                  "random_participants": randoms["random_participants"]}
    if limit:
        conditions = {name: rows[:limit] for name, rows in conditions.items()}
    identities = [[(row["item_id"], row["ground_truth"], row["wrong_target"]) for row in rows]
                  for rows in conditions.values()]
    if not identities[0] or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Reference conditions must have matching ordered questions and wrong targets")
    return conditions, manifests


def diagnose(runner, conditions, batch_size=8, top_k=10):
    import torch

    tokenizer = runner.tokenizer
    jobs = [(name, row) for name, rows in conditions.items() for row in rows]
    answer_ids = answer_token_ids(tokenizer, (row["prompt"] for _, row in jobs))
    bare_ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"}
    if any(len(ids) != 1 for ids in bare_ids.values()):
        raise ValueError("Bare answer letters must each be one token")
    bare_ids = {letter: ids[0] for letter, ids in bare_ids.items()}
    special_ids = {label: tokenizer.encode(text, add_special_tokens=False) for label, text in
                   (("space", " "), ("newline", "\n"))}
    output = {name: [] for name in conditions}
    runner.model.eval()
    with torch.inference_mode():
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            inputs = tokenizer([row["prompt"] for _, row in batch], return_tensors="pt", padding=True,
                               add_special_tokens=False)
            inputs = {key: value.to(runner.device) for key, value in inputs.items()}
            logits = runner.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probabilities, k=top_k, dim=-1)
            for pos, (condition, row) in enumerate(batch):
                p = probabilities[pos]
                top = [{"token_id": int(token), "token": tokenizer.convert_ids_to_tokens([int(token)])[0],
                        "text": tokenizer.decode([int(token)], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False),
                        "probability": float(value)}
                       for token, value in zip(indices[pos].cpu().tolist(), values[pos].cpu().tolist())]
                mass = lambda ids: float(p[list(ids)].sum().item())
                output[condition].append({
                    "item_id": row["item_id"], "prompt": row["prompt"],
                    "model_answer": row["model_answer"], "recorded_abcd_mass": row["abcd_probability_mass"],
                    "abcd_mass": mass(answer_ids.values()), "bare_abcd_mass": mass(bare_ids.values()),
                    "special_token_probabilities": {name: float(p[ids[0]].item()) if len(ids) == 1 else None
                                                    for name, ids in special_ids.items()},
                    "top_tokens": top,
                })
            print(f"Inspected {min(start + batch_size, len(jobs))}/{len(jobs)} prompts", flush=True)
    return output, answer_ids, bare_ids


def summarize(output):
    summary = {}
    for name, rows in output.items():
        top_counts = collections.Counter(row["top_tokens"][0]["text"] for row in rows)
        summary[name] = {
            "item_count": len(rows),
            "median_abcd_mass": statistics.median(row["abcd_mass"] for row in rows),
            "median_bare_abcd_mass": statistics.median(row["bare_abcd_mass"] for row in rows),
            "median_space_probability": statistics.median(row["special_token_probabilities"]["space"] for row in rows),
            "median_newline_probability": statistics.median(row["special_token_probabilities"]["newline"] for row in rows),
            "top_next_tokens": [{"text": text, "count": count} for text, count in top_counts.most_common(15)],
            "max_abs_difference_from_recorded_abcd_mass": max(abs(row["abcd_mass"] - row["recorded_abcd_mass"]) for row in rows),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice-reference", type=Path, required=True)
    parser.add_argument("--random-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--top-k", type=positive_int, default=10)
    args = parser.parse_args(argv)
    conditions, manifests = load_inputs(args.choice_reference, args.random_reference, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from multi_actor import load_model
    runner, _ = load_model("qwen")
    for manifest in manifests:
        if (getattr(runner.model.config, "_commit_hash", None) != manifest.get("model_commit") or
            str(next(runner.model.parameters()).dtype) != manifest.get("model_dtype") or
            runner.tokenizer.name_or_path != manifest.get("tokenizer")):
            raise ValueError("Model, precision, or tokenizer differs from reference run")
    output, answer_ids, bare_ids = diagnose(runner, conditions, args.batch_size, args.top_k)
    if any(answer_ids != manifest["answer_token_ids"] for manifest in manifests):
        raise ValueError("Answer token IDs differ from reference runs")
    summary = summarize(output)
    write_json(args.output_dir / "results.json", output)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "run.json", {"status": "complete", "model": manifests[0]["model"],
        "model_commit": manifests[0]["model_commit"], "dataset_sha256": manifests[0]["dataset_sha256"],
        "choice_reference": str(args.choice_reference), "random_reference": str(args.random_reference),
        "answer_token_ids": answer_ids, "bare_answer_token_ids": bare_ids, "top_k": args.top_k,
        "batch_size": args.batch_size, "limit": args.limit})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
