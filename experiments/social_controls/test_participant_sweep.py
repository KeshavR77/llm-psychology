import contextlib
import io
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from .participant_sweep import main, save_summaries, summarize_sweep
from .prompts import ROOT, prepare_items
from .run import run_inference
from .test_social_controls import HAS_TORCH, fake_parse, make_fake_runner, record


class ParticipantSweepTests(unittest.TestCase):
    def test_saved_control_prompts_match_sweep(self):
        for lines, relative in [(7, "mmlu/seed_42_lines_7"),
                                (3, "followups/three_lines/mmlu/seed_42_lines_3")]:
            path = ROOT / "results/social_controls" / relative / "shared/multiple_participants.json"
            if not path.is_file():
                continue
            saved = json.loads(path.read_text())["results"]
            items = prepare_items(lines=lines)
            self.assertEqual([r["item_id"] for r in saved], [r["item_id"] for r in items])
            self.assertEqual([r["prompt"] for r in saved], [r["prompts"]["multiple_participants"] for r in items])
            self.assertEqual([r["wrong_target"] for r in saved], [r["wrong_target"] for r in items])

    def test_metrics_pairing_and_figure_output(self):
        rows = [record("a", "multiple_participants", "B"),
                record("b", "multiple_participants", "A"),
                record("c", "multiple_participants", "")]
        conditions = {2: rows, 3: [dict(r) for r in rows]}
        results = summarize_sweep(conditions, bootstrap_samples=200)
        for result in results:
            self.assertEqual(result["parseable"], 2)
            self.assertEqual(result["parse_failures"], 1)
            self.assertEqual(result["conformity"], 0.5)
            self.assertEqual(result["wrong_target_adoption_all_items"], 1 / 3)
            self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(results[0]["conformity_ci_low"], results[1]["conformity_ci_low"])
        self.assertEqual(results[0]["conformity_ci_high"], results[1]["conformity_ci_high"])
        with self.assertRaises(ValueError):
            summarize_sweep({2: rows, 3: rows[::-1]}, bootstrap_samples=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_summaries(root, results)
            for name in ("summary.json", "summary.csv", "conformity_by_participants.png", "conformity_by_participants.pdf"):
                self.assertTrue((root / name).is_file())

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required")
    def test_sampled_sweep_uses_sampling_and_records_settings(self):
        import torch
        import sys

        runner = make_fake_runner()
        runner.generation_config = types.SimpleNamespace(temperature=0.6, top_k=20, top_p=0.95)
        calls = []
        def generate(input_ids, attention_mask=None, do_sample=False, max_new_tokens=None):
            calls.append(do_sample)
            self.assertEqual(max_new_tokens, 50)
            response = torch.tensor([[ord(c) for c in "<answer>B</answer>"]])
            return {"sequences": torch.cat([input_ids, response], dim=1)}
        runner.generate = generate
        fake_module = types.SimpleNamespace(MODEL_IDS={"qwen": "fake-qwen"},
                                            load_model=lambda name: (runner, "cpu"), parse_answer=fake_parse)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"multi_actor": fake_module}):
            output_dir = Path(tmp) / "sampled"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--sample", "--limit", "1", "--output-dir", str(output_dir)])
            manifest = json.loads((output_dir / "run.json").read_text())
            self.assertEqual(calls, [True] * 9)
            self.assertEqual(runner.model.prefill_calls, 0)
            self.assertTrue(manifest["sampling"])
            self.assertEqual(manifest["sampling_parameters"], {"temperature": 0.6, "top_k": 20, "top_p": 0.95})
            self.assertEqual(manifest["generation_attempts"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(json.loads((output_dir / "summary.json").read_text())["conditions"]), 9)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required")
    def test_sampled_single_participant_count(self):
        import sys
        import torch

        runner = make_fake_runner()
        runner.generation_config = types.SimpleNamespace(temperature=0.6, top_k=20, top_p=0.95)
        calls = []
        def generate(input_ids, attention_mask=None, do_sample=False, max_new_tokens=None):
            calls.append(do_sample)
            response = torch.tensor([[ord(c) for c in "<answer>B</answer>"]])
            return {"sequences": torch.cat([input_ids, response], dim=1)}
        runner.generate = generate
        fake_module = types.SimpleNamespace(MODEL_IDS={"qwen": "fake-qwen"},
                                            load_model=lambda name: (runner, "cpu"), parse_answer=fake_parse)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"multi_actor": fake_module}):
            output_dir = Path(tmp) / "single"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--sample", "--participants", "8", "--limit", "1", "--output-dir", str(output_dir)])
            manifest = json.loads((output_dir / "run.json").read_text())
            self.assertEqual(calls, [True])
            self.assertEqual(manifest["participants"], [8])
            self.assertEqual(runner.model.prefill_calls, 0)
            self.assertTrue((output_dir / "participants_8.json").is_file())
            self.assertFalse((output_dir / "participants_7.json").exists())

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required")
    def test_behavioral_inference_skips_activation_prefill(self):
        import torch

        runner = make_fake_runner()
        def generate(input_ids, attention_mask=None, do_sample=True, max_new_tokens=None):
            self.assertFalse(do_sample)
            self.assertEqual(max_new_tokens, 50)
            self.assertEqual(runner.model.prefill_calls, 0)
            self.assertTrue(all(not layer._forward_hooks for layer in runner.model.model.layers))
            response = torch.tensor([[ord(c) for c in "<answer>B</answer>"]])
            return {"sequences": torch.cat([input_ids, response], dim=1)}
        runner.generate = generate
        with contextlib.redirect_stdout(io.StringIO()):
            results, sums = run_inference(runner, prepare_items(lines=7, limit=1),
                                           ("multiple_participants",), fake_parse, capture_activations=False)
        self.assertEqual(runner.model.prefill_calls, 0)
        self.assertEqual(sums["multiple_participants"], {})
        self.assertEqual(results["multiple_participants"][0]["model_answer"], "B")


if __name__ == "__main__":
    unittest.main()
