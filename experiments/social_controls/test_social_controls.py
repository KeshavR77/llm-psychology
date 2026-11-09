import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from .activations import add_activations, capture_residuals, make_diffmeans
from .analyze import align_records, summarize
from .prompts import ROOT, answer_block, prepare_items, prompt_parts, selected_conditions
from .run import main, parse_args, run_inference


HAS_TORCH = importlib.util.find_spec("torch") is not None


class PromptTests(unittest.TestCase):
    def test_matched_blocks_and_wrappers(self):
        item = prepare_items(limit=1)[0]
        prefix, suffix = prompt_parts(item, item["example_answer"])
        payload = f"<answer>{item['wrong_target']}</answer>"
        for condition, prompt in item["prompts"].items():
            block = answer_block(condition, item["wrong_target"], 7)
            self.assertEqual(prompt, prefix + block + suffix)
            self.assertTrue(prompt.endswith("Your answer: "))
            if condition == "no_context":
                self.assertEqual(block, "")
            else:
                self.assertEqual(block.count(payload), 7)
                self.assertEqual(len(block.splitlines()), 7)
            if condition == "speaker_free":
                self.assertEqual(block.splitlines(), [payload] * 7)
            elif condition == "single_participant":
                self.assertEqual(block.splitlines(), [f"Participant 1: {payload}"] * 7)
            elif condition == "multiple_participants":
                self.assertEqual(block.splitlines(), [f"Participant {i}: {payload}" for i in range(1, 8)])

    def test_one_line_identity(self):
        item = prepare_items(lines=1, limit=1)[0]
        self.assertEqual(item["prompts"]["multiple_participants"], item["prompts"]["single_participant"])

    def test_reproducibility_order_and_stable_ids(self):
        items = prepare_items()
        self.assertEqual(len(items), 100)
        self.assertEqual(items[:3], prepare_items(limit=3))
        changed_seed = prepare_items(seed=9)
        self.assertEqual([row["item_id"] for row in items], [row["item_id"] for row in changed_seed])
        for item in items:
            self.assertNotEqual(item["ground_truth"], item["wrong_target"])

    def test_dry_run_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-S", "-m", "experiments.social_controls.run", "--dry-run", "--output-dir", tmp],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            preview = json.loads(result.stdout)
            self.assertEqual(preview["sample_count"], 100)
            self.assertEqual(len(preview["items"]), 100)
            self.assertEqual(len(preview["items"][0]["prompts"]), 4)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cli_validation_and_condition_selection(self):
        for flag in ("--lines", "--limit"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "0"])
        self.assertEqual(selected_conditions("speaker_free"), ("no_context", "multiple_participants", "speaker_free"))


def record(item_id, condition, answer, truth="A", target="B"):
    return {
        "item_id": item_id, "condition": condition, "ground_truth": truth,
        "wrong_target": target, "model_answer": answer, "parseable": answer in ("A", "B", "C", "D"),
        "is_correct": answer == truth, "adopts_wrong_target": answer == target,
        "sequence_length": 20,
    }


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.reference = [record(str(i), "multiple_participants", answer) for i, answer in enumerate(("B", "", "A", "B"))]
        self.control = [record(str(i), "speaker_free", answer) for i, answer in enumerate(("A", "B", "", "B"))]
        self.baseline = [record(str(i), "no_context", answer) for i, answer in enumerate(("A", "A", "C", ""))]

    def test_pairing_by_id_and_invalid_sets(self):
        aligned = align_records(self.reference, self.control[::-1], self.baseline[::-1])
        self.assertEqual(aligned[1], self.control)
        for invalid in (self.control[:-1], self.control + self.control[:1]):
            with self.assertRaises(ValueError):
                align_records(self.reference, invalid, self.baseline)
        inconsistent = [dict(row, wrong_target="C") for row in self.control]
        with self.assertRaises(ValueError):
            align_records(self.reference, inconsistent, self.baseline)

    def test_denominators_and_baseline_filter(self):
        summary = summarize(self.reference, self.control, self.baseline, bootstrap_samples=30)
        reference = summary["conditions"]["multiple_participants"]
        self.assertEqual(reference["counts"]["parse_failures"], 1)
        self.assertEqual(reference["counts"]["baseline_correct_items"], 2)
        self.assertEqual(reference["counts"]["baseline_correct_parse_failures"], 1)
        observed = {key: value["value"] for key, value in reference["metrics"].items()}
        self.assertEqual(observed, {"accuracy": 1 / 3, "parse_rate": 3 / 4,
                                    "wrong_target_adoption": 1 / 2, "baseline_correct_adoption": 1 / 2})

    def test_bootstrap_is_paired_and_reproducible(self):
        control = [dict(row, condition="speaker_free") for row in self.reference]
        summary = summarize(self.reference, control[::-1], self.baseline, bootstrap_samples=100)
        self.assertEqual(summary, summarize(self.reference, control[::-1], self.baseline, bootstrap_samples=100))
        for value in summary["paired_differences"].values():
            self.assertEqual((value["value"], value["ci_low"], value["ci_high"]), (0, 0, 0))

    def test_empty_conditional_denominators(self):
        baseline = [dict(row, is_correct=False) for row in self.baseline]
        summary = summarize(self.reference, self.control, baseline, bootstrap_samples=10)
        value = summary["paired_differences"]["baseline_correct_adoption"]
        self.assertIsNone(value["value"])
        self.assertIsNone(value["ci_low"])
        self.assertEqual(value["valid_bootstrap_samples"], 0)
        json.dumps(summary, allow_nan=False)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for tensor and inference tests")
class ActivationTests(unittest.TestCase):
    def test_accumulation_sign_mean_and_loader_compatibility(self):
        import torch
        from steering import load_vector

        reference, control = {}, {}
        for _ in range(2):
            add_activations(reference, {2: torch.full((4096,), 3, dtype=torch.float16),
                                        7: torch.full((4096,), -1, dtype=torch.float16)})
            add_activations(control, {2: torch.ones(4096), 7: torch.ones(4096)})
        vectors = make_diffmeans(reference, control, 2)
        self.assertEqual(set(vectors), {2, 7})
        self.assertEqual(reference[2].dtype, torch.float32)
        torch.testing.assert_close(vectors[2], torch.full((4096,), 2.0))
        torch.testing.assert_close(vectors[7], torch.full((4096,), -2.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diffmeans.pt"
            torch.save({"diffmeans": vectors}, path)
            torch.testing.assert_close(load_vector(path, 7), vectors[7])
        with self.assertRaises(ValueError):
            add_activations(reference, {2: torch.ones(4096)})
        with self.assertRaises(ValueError):
            add_activations({}, {0: torch.tensor([float("nan")])})

    def test_capture_final_position_and_cleanup_on_failure(self):
        import torch

        model = make_fake_runner().model
        inputs = torch.tensor([[1, 2, 3]])
        with capture_residuals(model, 2) as acts:
            model(input_ids=inputs)
        self.assertEqual(set(acts), {0, 1})
        torch.testing.assert_close(acts[0], torch.full((4,), 4.0))
        torch.testing.assert_close(acts[1], torch.full((4,), 6.0))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with capture_residuals(model, 2):
                raise RuntimeError("intentional")
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_runner_shares_inputs_and_evaluates_shared_conditions_once(self):
        runner = make_fake_runner()
        items = prepare_items(lines=1, limit=2)
        with contextlib.redirect_stdout(io.StringIO()):
            results, sums = run_inference(runner, items, selected_conditions("both"), fake_parse)
        self.assertEqual(runner.generate_calls, 8)
        self.assertEqual(runner.model.prefill_calls, 8)
        for rows in results.values():
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["generation_attempts"] == 1 for row in rows))
            self.assertTrue(all(row["position_index"] == row["sequence_length"] - 1 for row in rows))
        import torch
        for vector in make_diffmeans(sums["multiple_participants"], sums["single_participant"], 2).values():
            torch.testing.assert_close(vector, torch.zeros(4))

    def test_two_item_end_to_end_and_overwrite_guard(self):
        import torch

        fake_module = types.SimpleNamespace(MODEL_IDS={"qwen": "fake-qwen"},
                                            load_model=lambda name: (make_fake_runner(), "cpu"), parse_answer=fake_parse)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"multi_actor": fake_module}):
            args = ["--limit", "2", "--lines", "1", "--output-dir", tmp]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            root = Path(tmp) / "seed_42_lines_1"
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "complete")
            for name in ("speaker_free", "single_participant"):
                for output in ("behavioral.json", "analysis.json", "summary.csv", "summary.md", "diffmeans.pt"):
                    self.assertTrue((root / name / output).is_file())
                artifact = torch.load(root / name / "diffmeans.pt", weights_only=True)
                self.assertEqual(artifact["position"], "last_prompt_token")
                self.assertTrue(all(torch.isfinite(vector).all() for vector in artifact["diffmeans"].values()))
            with self.assertRaises(FileExistsError):
                main(args)
            with contextlib.redirect_stdout(io.StringIO()):
                main(args + ["--experiment", "speaker_free", "--overwrite"])
            self.assertFalse((root / "single_participant").exists())


def fake_parse(dataset, response):
    match = re.search(r"<answer>([A-D])</answer>", response)
    return match.group(1) if match else ""


def make_fake_runner():
    import torch

    class Block(torch.nn.Module):
        def __init__(self, increment, as_tuple=False):
            super().__init__()
            self.increment = increment
            self.as_tuple = as_tuple

        def forward(self, hidden):
            hidden = hidden + self.increment
            return (hidden,) if self.as_tuple else hidden

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(1, as_tuple=True), Block(2)])
            self.config = types.SimpleNamespace(_commit_hash=None)
            self.prefill_calls = 0

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            self.last_inputs = (input_ids, attention_mask)
            self.prefill_calls += 1
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
            for block in self.model.layers:
                output = block(hidden)
                hidden = output[0] if isinstance(output, tuple) else output
            return hidden

    class Tokenizer:
        name_or_path = "fake-tokenizer"
        init_kwargs = {}

        def __call__(self, prompt, return_tensors):
            ids = torch.tensor([[ord(char) % 128 for char in prompt]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, ids, **kwargs):
            return "".join(chr(value) for value in ids)

        def convert_ids_to_tokens(self, value):
            return str(value)

    class Runner:
        device = "cpu"

        def __init__(self):
            self.model = Model()
            self.tokenizer = Tokenizer()
            self.generate_calls = 0

        def generate(self, input_ids, attention_mask=None, do_sample=True, max_new_tokens=None):
            assert input_ids is self.model.last_inputs[0]
            assert attention_mask is self.model.last_inputs[1]
            assert not do_sample and max_new_tokens == 50
            assert all(not layer._forward_hooks for layer in self.model.model.layers)
            self.generate_calls += 1
            output = torch.tensor([[ord(char) for char in "<answer>B</answer>"]])
            return {"sequences": torch.cat([input_ids, output], dim=1)}

    return Runner()


if __name__ == "__main__":
    unittest.main()
