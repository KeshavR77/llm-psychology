import random
import unittest
from types import SimpleNamespace

from .choice_logprobs import make_items, score_items
from .random_choice_logprobs import compare_to_reference, make_random_items
from .test_choice_logprobs import FakeModel, FakeTokenizer


class RandomBaselineTests(unittest.TestCase):
    def test_independent_uniform_answers_and_matched_questions(self):
        items = make_random_items(limit=10)
        original = make_items(limit=10)
        generator = random.Random(1729)
        for item, reference in zip(items, original):
            self.assertEqual(item["peer_answers"], [generator.choice("ABCD") for _ in range(7)])
            self.assertEqual((item["item_id"], item["wrong_target"]),
                             (reference["item_id"], reference["wrong_target"]))
            prompt = item["prompts"]["random_participants"]
            for index, answer in enumerate(item["peer_answers"], 1):
                self.assertIn(f"Participant {index}: {answer}\n", prompt)
            self.assertNotIn("<answer>", prompt)
            self.assertTrue(prompt.endswith("Your answer:"))
        self.assertEqual(items, make_random_items(limit=10))
        self.assertTrue(any(item["ground_truth"] in item["peer_answers"] for item in items))

    def test_scores_only_random_prompts_and_preserves_answers(self):
        items = make_random_items(limit=2)
        runner = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeModel(), device="cpu")
        results, _ = score_items(runner, items, batch_size=2)
        self.assertEqual(list(results), ["random_participants"])
        for row, item in zip(results["random_participants"], items):
            self.assertEqual(row["model_answer"], "D")
            self.assertEqual(row["peer_answers"], item["peer_answers"])

    def test_comparison_pairs_and_checks_reference_prompts(self):
        items = make_random_items(limit=2)
        expected = make_items(limit=2)
        def record(item, prompt, correct):
            return {"item_id": item["item_id"], "ground_truth": item["ground_truth"],
                    "wrong_target": item["wrong_target"], "prompt": prompt,
                    "is_correct": correct, "adopts_wrong_target": not correct,
                    "answer_probabilities_given_abcd": {item["wrong_target"]: 0.1 if correct else 0.8}}
        random_rows = [record(item, item["prompts"]["random_participants"], i == 0)
                       for i, item in enumerate(items)]
        reference = {condition: [record(item, item["prompts"][condition], condition == "no_context")
                                 for item in expected]
                     for condition in ("no_context", "multiple_participants")}
        comparison = compare_to_reference(random_rows, reference, items, 7, 42, bootstrap_samples=20)
        self.assertEqual(comparison["no_context_correct_items"], 2)
        self.assertEqual(comparison["wrong_target_choices_among_no_context_correct"],
                         {"random_participants": 1, "unanimous_wrong": 2})
        self.assertEqual(comparison["unanimous_wrong_minus_random"]["paired_differences"]["accuracy"]["value"], -0.5)
        reference["multiple_participants"][0]["prompt"] += "changed"
        with self.assertRaisesRegex(ValueError, "Reference prompt"):
            compare_to_reference(random_rows, reference, items, 7, 42, bootstrap_samples=20)


if __name__ == "__main__":
    unittest.main()
