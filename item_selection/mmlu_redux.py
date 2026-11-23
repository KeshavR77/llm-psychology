"""Import, screen, and select a subject-balanced MMLU-Redux 2.0 item set.

Run `python -m item_selection.mmlu_redux import`, then `screen --model qwen`
(and llama), and finally `select`. Screening records are resumable JSONL files.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from pathlib import Path

SOURCE = "edinburgh-dawg/mmlu-redux-2.0"
REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
SEED = 42
MODEL_IDS = {"qwen": "Qwen/Qwen3-8B", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
CANDIDATES = Path("data/mmlu_redux_candidates.json")
OUTPUT = Path("data/mmlu_redux_500.json")
SCREEN_DIR = Path("results/item_selection/mmlu_redux")
LETTERS = "ABCD"


def item_id(row):
    canonical = json.dumps([row["subject"], row["question"], row["choices"], row["answer"]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def eligible(subject, row):
    if row.get("error_type") != "ok":
        return None
    choices, answer, question = row.get("choices"), row.get("answer"), row.get("question")
    if not isinstance(question, str) or not question.strip() or not isinstance(choices, list) or len(choices) != 4 or any(not isinstance(c, str) or not c.strip() for c in choices) or type(answer) is not int or not 0 <= answer < 4:
        raise ValueError(f"Invalid ok item in {subject}: {question!r}")
    return {"subject": subject, "question": question, "choices": choices, "answer": answer}


def ordered_candidates(rows, seed=SEED):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        identifier = item_id(row)
        if identifier not in seen:
            grouped[row["subject"]].append(row)
            seen.add(identifier)
    rng = random.Random(seed)
    subjects = sorted(grouped)
    rng.shuffle(subjects)
    for subject in subjects:
        rng.shuffle(grouped[subject])
    return subjects, grouped


def select_balanced(rows, results, target=500, seed=SEED):
    subjects, groups = ordered_candidates(rows, seed)
    selected, offsets = [], {s: 0 for s in subjects}
    while len(selected) < target:
        added = False
        for subject in subjects:
            group = groups[subject]
            while offsets[subject] < len(group):
                row = group[offsets[subject]]
                offsets[subject] += 1
                key = item_id(row)
                if any(key not in results[model] for model in MODEL_IDS):
                    raise ValueError(f"Missing screening result for {subject}, item {key}")
                if all(results[model][key]["correct"] for model in MODEL_IDS):
                    selected.append(row)
                    added = True
                    break
            if len(selected) == target:
                break
        if not added:
            raise ValueError(f"Only {len(selected)} qualifying questions available; need {target}")
    return selected


def import_candidates(output):
    from huggingface_hub import snapshot_download
    import pyarrow.ipc as ipc

    root = Path(snapshot_download(repo_id=SOURCE, repo_type="dataset", revision=REVISION, allow_patterns=["*/data-*.arrow"]))
    files = sorted(root.glob("*/data-*.arrow"))
    if len(files) != 57:
        raise ValueError(f"Expected 57 subject files at pinned revision, found {len(files)}")
    rows = []
    for file in files:
        with file.open("rb") as stream:
            for row in ipc.open_stream(stream).read_all().to_pylist():
                item = eligible(file.parent.name, row)
                if item is not None:
                    rows.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(rows)} eligible items across {len(files)} subjects to {output}")


def prompt_for(row):
    # A stable per-item example makes resuming and model-to-model comparison reproducible.
    example = LETTERS[int(item_id(row)[:8], 16) % 4]
    choices = "\n".join(f"{letter}: {text}" for letter, text in zip(LETTERS, row["choices"]))
    return ("Answer the following multiple choice question by giving the most appropriate response. "
            "Answer should be one among [A, B, C, D]. Provide your answer inside <answer></answer> XML tags.\n\n"
            f"Ex: <answer>{example}</answer>\n\nQuestion: {row['question']}\n{choices}\n\nAnswer: ")


def read_screen(path, candidate_hash, expected_model=None):
    results = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["candidate_sha256"] != candidate_hash:
                raise ValueError(f"{path} belongs to a different candidate file")
            if expected_model is not None and (row.get("model") != expected_model or row.get("model_id") != MODEL_IDS[expected_model]):
                raise ValueError(f"{path} contains a screening result from the wrong model")
            if type(row.get("correct")) is not bool:
                raise ValueError(f"{path} contains an invalid correctness result")
            if row["id"] in results:
                raise ValueError(f"Duplicate screening record in {path}: {row['id']}")
            results[row["id"]] = row
    return results


def screen(model_name, source, directory):
    import torch
    import transformer_lens.utilities as utils
    from transformer_lens.model_bridge import TransformerBridge

    candidate_bytes = source.read_bytes()
    candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
    rows = json.loads(candidate_bytes)
    subjects, groups = ordered_candidates(rows)
    candidates = [row for i in range(max(map(len, groups.values()))) for subject in subjects if i < len(groups[subject]) for row in [groups[subject][i]]]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_name}.jsonl"
    done = read_screen(path, candidate_hash, model_name)
    if len(done) == len(candidates):
        print(f"Already screened {len(done)} items for {model_name}")
        return
    torch.set_grad_enabled(False)
    device = utils.get_device()
    if not str(device).startswith("cuda"):
        raise RuntimeError("Screening requires a CUDA GPU")
    model = TransformerBridge.boot_transformers(MODEL_IDS[model_name], dtype=torch.bfloat16, device=device)
    with path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(candidates, 1):
            key = item_id(row)
            if key in done:
                continue
            prompt = prompt_for(row)
            output = model.generate(prompt, max_new_tokens=50, do_sample=False)
            response = output[len(prompt):]
            match = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", response)
            answer = match.group(1).upper() if match else ""
            record = {"id": key, "candidate_sha256": candidate_hash, "model": model_name, "model_id": MODEL_IDS[model_name], "prompt": prompt, "raw_response": response, "model_answer": answer, "correct": answer == LETTERS[row["answer"]]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if index % 100 == 0:
                print(f"{model_name}: screened {index}/{len(candidates)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--output", type=Path, default=CANDIDATES)
    scr = sub.add_parser("screen")
    scr.add_argument("--model", choices=MODEL_IDS, required=True)
    scr.add_argument("--source", type=Path, default=CANDIDATES)
    scr.add_argument("--screen-dir", type=Path, default=SCREEN_DIR)
    sel = sub.add_parser("select")
    sel.add_argument("--source", type=Path, default=CANDIDATES)
    sel.add_argument("--screen-dir", type=Path, default=SCREEN_DIR)
    sel.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "import":
        import_candidates(args.output)
    elif args.command == "screen":
        screen(args.model, args.source, args.screen_dir)
    else:
        data = args.source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        results = {model: read_screen(args.screen_dir / f"{model}.jsonl", digest, model) for model in MODEL_IDS}
        selected = select_balanced(json.loads(data), results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {"source": SOURCE, "revision": REVISION, "seed": SEED, "candidate_sha256": digest, "selected_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "models": MODEL_IDS, "screen_files": {model: str(args.screen_dir / f"{model}.jsonl") for model in MODEL_IDS}, "subject_counts": dict(sorted(Counter(row["subject"] for row in selected).items()))}
        args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Selected {len(selected)} questions across {len(metadata['subject_counts'])} subjects to {args.output}")


if __name__ == "__main__":
    main()
