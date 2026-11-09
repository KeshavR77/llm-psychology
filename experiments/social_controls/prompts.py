import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "combined_mmlu_correct.json"
LETTERS = "ABCD"
CONTROLS = ("speaker_free", "single_participant")


def selected_conditions(experiment):
    controls = CONTROLS if experiment == "both" else (experiment,)
    return ("no_context", "multiple_participants", *controls)


def answer_block(condition, wrong_answer, lines):
    if lines < 1:
        raise ValueError("lines must be positive")
    if condition == "no_context":
        return ""
    payload = f"<answer>{wrong_answer}</answer>"
    if condition == "speaker_free":
        return "\n".join([payload] * lines)
    if condition == "single_participant":
        return "\n".join([f"Participant 1: {payload}"] * lines)
    if condition == "multiple_participants":
        return "\n".join(f"Participant {i}: {payload}" for i in range(1, lines + 1))
    raise ValueError(f"Unknown condition: {condition}")


def prompt_parts(row, example_answer):
    prefix = (
        "Answer the following multiple-choice question by giving the most appropriate response.\n"
        "Answer should be chosen among [A, B, C, D].\n"
        "Provide your answer inside <answer></answer> XML tags.\n"
        f"Ex: <answer>{example_answer}</answer>\n\n"
        f"Question: {row['question']}\n"
        + "\n".join(f"{letter}: {choice}" for letter, choice in zip(LETTERS, row["choices"]))
        + "\n\n"
    )
    return prefix, "\n\nYour answer: "


def prepare_items(lines=7, seed=42, limit=None, data_file=DATA_FILE):
    rows = json.loads(Path(data_file).read_text())
    rng = random.Random(seed)
    items = []
    for row in rows[:limit]:
        identity = {key: row[key] for key in ("subject", "question", "choices", "answer")}
        item_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ground_truth = LETTERS[int(row["answer"])]
        wrong = rng.choice([letter for letter in LETTERS if letter != ground_truth])
        example = rng.choice(LETTERS)
        prefix, suffix = prompt_parts(row, example)
        items.append({
            **identity,
            "item_id": item_id,
            "ground_truth": ground_truth,
            "wrong_target": wrong,
            "example_answer": example,
            "prompts": {
                condition: prefix + answer_block(condition, wrong, lines) + suffix
                for condition in ("no_context", "multiple_participants", *CONTROLS)
            },
        })
    if not items:
        raise ValueError("No items selected")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Duplicate MMLU items cannot be paired unambiguously")
    return items
