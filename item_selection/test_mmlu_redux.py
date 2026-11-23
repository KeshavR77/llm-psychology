import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from item_selection.mmlu_redux import eligible, item_id, ordered_candidates, read_screen, select_balanced


def question(subject, number):
    return {"subject": subject, "question": f"{subject} {number}?", "choices": ["a", "b", "c", "d"], "answer": 0}


def screenings(rows, wrong=()):
    return {model: {item_id(row): {"correct": item_id(row) not in wrong} for row in rows} for model in ("qwen", "llama")}


class ReduxSelectionTest(unittest.TestCase):
    def test_annotations_and_validation(self):
        row = {**question("s", 1), "error_type": "ok"}
        self.assertEqual(eligible("s", row), question("s", 1))
        self.assertIsNone(eligible("s", {**row, "error_type": "wrong_groundtruth"}))
        with self.assertRaises(ValueError):
            eligible("s", {**row, "answer": 4})
        with self.assertRaises(ValueError):
            eligible("s", {**row, "choices": ["a"]})

    def test_500_balanced_reproducible_and_both_models_correct(self):
        rows = [question(str(subject), i) for subject in range(57) for i in range(12)]
        results = screenings(rows)
        subjects, grouped = ordered_candidates(rows)
        first = grouped[subjects[0]][0]
        results["llama"][item_id(first)]["correct"] = False
        chosen = select_balanced(rows, results)
        self.assertEqual(len(chosen), 500)
        self.assertEqual(chosen, select_balanced(rows, results))
        self.assertNotIn(first, chosen)
        self.assertEqual(set(Counter(row["subject"] for row in chosen).values()), {8, 9})
        self.assertEqual(len({item_id(row) for row in chosen}), 500)

    def test_exhausted_subject_redistributes_and_insufficient_fails(self):
        rows = [question("small", 0)] + [question("large", i) for i in range(5)]
        results = screenings(rows)
        selected = select_balanced(rows, results, target=5)
        self.assertEqual(Counter(row["subject"] for row in selected), {"small": 1, "large": 4})
        with self.assertRaisesRegex(ValueError, "Only 6 qualifying"):
            select_balanced(rows, results, target=7)

    def test_screen_records_with_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen.jsonl"
            row = {"id": "1", "candidate_sha256": "hash", "model": "qwen",
                   "model_id": "Qwen/Qwen3-8B", "correct": True, "prompt": "one\u0085two"}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_screen(path, "hash", "qwen")["1"], row)

    def test_missing_screen_or_mismatched_source_fails(self):
        rows = [question("s", 0)]
        results = screenings(rows)
        del results["llama"][item_id(rows[0])]
        with self.assertRaisesRegex(ValueError, "Missing screening"):
            select_balanced(rows, results, target=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.jsonl"
            path.write_text(json.dumps({"id": "1", "candidate_sha256": "old"}) + "\n")
            with self.assertRaisesRegex(ValueError, "different candidate"):
                read_screen(path, "new")
            path.write_text(json.dumps({"id": "1", "candidate_sha256": "new", "model": "qwen", "model_id": "Qwen/Qwen3-8B", "correct": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "wrong model"):
                read_screen(path, "new", "llama")


class ReduxPipelineTest(unittest.TestCase):
    def test_activation_loader_uses_fixed_items_without_fallback(self):
        from layer_sweep import load_questions
        rows = [question("s", i) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mmlu_redux_500.json").write_text(json.dumps(rows))
            self.assertEqual(load_questions(root, "mmlu", 3, 42), rows)
            self.assertEqual(load_questions(root, "mmlu", None, 123), rows)
            with self.assertRaisesRegex(ValueError, "fewer than requested"):
                load_questions(root, "mmlu", 4, 42)
            (root / "mmlu_redux_500.json").unlink()
            with self.assertRaises(FileNotFoundError):
                load_questions(root, "mmlu", 3, 42)

    def test_behavioral_runner_uses_one_redux_set(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        import run_behavioral
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "mmlu_redux_500.json"
            dataset.write_text("[]")
            with patch.object(run_behavioral, "REDUX_DATA", dataset), \
                 patch.object(run_behavioral, "BEHAVIORAL_DIR", Path(tmp) / "results"), \
                 patch.object(run_behavioral, "load_model", return_value=(object(), "cpu")), \
                 patch.object(run_behavioral, "run_experiment") as run, redirect_stdout(StringIO()):
                run_behavioral.main([])
            self.assertEqual(run.call_count, 48)
            for call in run.call_args_list:
                data = call.kwargs["dataset_file"]
                self.assertEqual(data, dataset if call.args[1] == "mmlu" else None)
                self.assertTrue(str(call.kwargs["output_file"]).startswith(tmp))

    def test_behavioral_resume_skips_matching_and_rejects_stale(self):
        import hashlib
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        import run_behavioral
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "items.json"
            source.write_text(json.dumps([question("s", 1)]))
            output = root / "behavioral/mmlu/qwen_mmlu_1_base.json"
            output.parent.mkdir(parents=True)
            result = {"model": "qwen", "dataset": "mmlu", "mode": "1", "seed": 42,
                      "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "attempts": 1, "results": [{}]}
            output.write_text(json.dumps(result))
            with patch.object(run_behavioral, "REDUX_DATA", source), \
                 patch.object(run_behavioral, "BEHAVIORAL_DIR", root / "behavioral"), \
                 patch.object(run_behavioral, "load_model", return_value=(object(), "cpu")), \
                 patch.object(run_behavioral, "run_experiment") as run, redirect_stdout(StringIO()):
                run_behavioral.main(["--dataset", "mmlu", "--style", "base", "--seed", "42", "--resume"])
                self.assertEqual(run.call_count, 11)
                result["dataset_sha256"] = "wrong"
                output.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    run_behavioral.main(["--dataset", "mmlu", "--style", "base", "--seed", "42", "--resume"])


if __name__ == "__main__":
    unittest.main()
