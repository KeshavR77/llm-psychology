import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from item_selection.select_qwen_correct import select_correct
from multi_actor import run_experiment


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": [1]}
        return {"input_ids": torch.ones((len(text), 1), dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        return [str(token) for token in ids]

    def decode(self, ids, **kwargs):
        return "<answer>A</answer>"


class FakeModel:
    tokenizer = FakeTokenizer()

    def generate(self, input_ids, **kwargs):
        return {"sequences": torch.cat((input_ids, torch.ones_like(input_ids)), dim=1)}


class MmluExpansionTest(unittest.TestCase):
    @unittest.skipUnless(Path("data/mmlu_test.json").is_file(), "Optional full MMLU dataset is not imported")
    def test_imported_data_contains_fixed_paper_items(self):
        full = json.loads(Path("data/mmlu_test.json").read_text())
        fixed = json.loads(Path("data/combined_mmlu_correct.json").read_text())
        def key(row):
            return (row["subject"], row["question"], tuple(row["choices"]), row["answer"])
        self.assertEqual(len(full), 14042)
        self.assertEqual(len({row["subject"] for row in full}), 57)
        self.assertTrue(all(len(row["choices"]) == 4 and 0 <= row["answer"] < 4 for row in full))
        self.assertTrue({key(row) for row in fixed}.issubset({key(row) for row in full}))

    def test_qwen_baseline_selection_and_peer_run(self):
        rows = [
            {"subject": "x", "question": "One?", "choices": ["a", "b", "c", "d"], "answer": 0},
            {"subject": "x", "question": "Two?", "choices": ["a", "b", "c", "d"], "answer": 1},
            {"subject": "x", "question": "Three?", "choices": ["a", "b", "c", "d"], "answer": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps(rows))
            baseline = run_experiment("qwen", "mmlu", "1", model=FakeModel(), device="cpu",
                                      dataset_file=source, limit=2, output_file=Path(directory) / "baseline.json")
            self.assertEqual((baseline["attempts"], baseline["correct"]), (2, 1))
            self.assertEqual(baseline["dataset_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            selected = select_correct(baseline)
            self.assertEqual(selected, rows[:1])
            source.write_text(json.dumps(selected))
            peer = run_experiment("qwen", "mmlu", "10", model=FakeModel(), device="cpu",
                                  dataset_file=source, output_file=Path(directory) / "peer.json")
            self.assertEqual(peer["attempts"], 1)
            self.assertEqual(peer["results"][0]["question"], "One?")
            self.assertIn("Participant 1:", peer["results"][0]["prompt"])
            with self.assertRaises(ValueError):
                select_correct(peer)
            with self.assertRaises(ValueError):
                run_experiment("qwen", "mmlu", "1", model=FakeModel(), dataset_file=source)


if __name__ == "__main__":
    unittest.main()
