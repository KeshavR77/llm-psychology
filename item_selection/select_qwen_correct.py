"""Build a Qwen-correct MMLU set from a no-peer behavioral run."""

import argparse
import json
from pathlib import Path


def select_correct(summary):
    if summary.get("model") != "qwen" or summary.get("dataset") != "mmlu" or summary.get("n") != 1:
        raise ValueError("Input must be a Qwen MMLU no-peer (mode 1) behavioral result")
    fields = ("subject", "question", "choices", "answer")
    return [{field: row[field] for field in fields}
            for row in summary["results"] if row.get("is_correct") is True]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.result.resolve() == args.output.resolve():
        parser.error("Output must differ from the input result")
    rows = select_correct(json.loads(args.result.read_text(encoding="utf-8")))
    if not rows:
        parser.error("The baseline contains no correctly answered questions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(rows)} Qwen-correct questions to {args.output}")


if __name__ == "__main__":
    main()
