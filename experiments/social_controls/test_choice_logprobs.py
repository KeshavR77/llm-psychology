import math
import unittest

from .choice_logprobs import answer_token_ids, make_items, make_prompts, score_items, summarize
from .prompts import prepare_items


class PromptTests(unittest.TestCase):
    def test_approved_prompts_and_original_wrong_targets(self):
        item = make_items(limit=1)[0]
        original = prepare_items(limit=1)[0]
        self.assertEqual((item["item_id"], item["wrong_target"]),
                         (original["item_id"], original["wrong_target"]))
        base = item["prompts"]["no_context"]
        peer = item["prompts"]["multiple_participants"]
        block = "Other participants' responses:\n" + "\n".join(
            f"Participant {i}: {item['wrong_target']}" for i in range(1, 8)) + "\n\n"
        self.assertEqual(peer.replace(block, ""), base)
        self.assertTrue(base.endswith("\n\nYour answer:"))
        self.assertNotIn("<answer>", peer)
        self.assertNotEqual(item["ground_truth"], item["wrong_target"])
        self.assertEqual(len(make_items()), 100)

    def test_variable_participant_count(self):
        item = make_items(lines=2, limit=1)[0]
        prompt = make_prompts(item, 2)["multiple_participants"]
        self.assertIn("Participant 2:", prompt)
        self.assertNotIn("Participant 3:", prompt)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if len(text) >= 2 and text[-2] == " " and text[-1] in "ABCD":
            return [ord(c) + 200 for c in text[:-2]] + ["ABCD".index(text[-1]) + 1]
        return [ord(c) + 200 for c in text]

    def __call__(self, prompts, return_tensors, padding, add_special_tokens):
        import torch
        ids = [self.encode(prompt) for prompt in prompts]
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([[0] * (width - len(row)) + row for row in ids]),
                "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids])}


class FakeModel:
    def eval(self):
        pass

    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
        import torch
        from types import SimpleNamespace
        assert not use_cache and logits_to_keep == 1
        # A vocabulary token outside A-D is most likely; D must still win among the four choices.
        logits = torch.tensor([10.0, 1.0, 2.0, 3.0, 4.0]).repeat(len(input_ids), 1).unsqueeze(1)
        return SimpleNamespace(logits=logits)


class ScoreTests(unittest.TestCase):
    def test_selects_highest_abcd_and_saves_full_vocabulary_probabilities(self):
        from types import SimpleNamespace
        runner = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeModel(), device="cpu")
        items = make_items(limit=2)
        results, ids = score_items(runner, items, batch_size=3)
        self.assertEqual(ids, {"A": 1, "B": 2, "C": 3, "D": 4})
        for records in results.values():
            for row in records:
                self.assertEqual(row["model_answer"], "D")
                self.assertAlmostEqual(sum(row["answer_probabilities_given_abcd"].values()), 1.0, places=6)
                self.assertLess(row["abcd_probability_mass"], 0.01)
                self.assertAlmostEqual(math.exp(row["answer_log_probabilities"]["D"]),
                                       math.exp(4) / sum(math.exp(x) for x in (10, 1, 2, 3, 4)), places=7)

    def test_rejects_multi_token_answer(self):
        class BadTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == " A":
                    return [1, 2]
                return super().encode(text, add_special_tokens=add_special_tokens)
        with self.assertRaisesRegex(ValueError, "single token"):
            answer_token_ids(BadTokenizer(), ["Your answer:"])

    def test_paired_summary_and_mismatched_items(self):
        def row(item_id, correct, adopted, wrong_prob):
            return {"item_id": item_id, "ground_truth": "A", "wrong_target": "B", "is_correct": correct,
                    "adopts_wrong_target": adopted, "answer_probabilities_given_abcd": {"B": wrong_prob}}
        baseline = [row("1", True, False, 0.1), row("2", False, True, 0.8)]
        peers = [row("1", False, True, 0.7), row("2", False, True, 0.9)]
        results = {"no_context": baseline, "multiple_participants": peers}
        summary = summarize(results, bootstrap_samples=30)
        self.assertEqual(summary["baseline_correct_items"], 1)
        self.assertEqual(summary["peer_wrong_target_rate_among_baseline_correct"], 1.0)
        diffs = summary["paired_differences_multiple_participants_minus_no_context"]
        self.assertEqual(diffs["accuracy"]["value"], -0.5)
        self.assertEqual(diffs["wrong_target_choice_rate"]["value"], 0.5)
        self.assertAlmostEqual(diffs["mean_wrong_target_probability_given_abcd"]["value"], 0.35)
        with self.assertRaisesRegex(ValueError, "same items"):
            summarize({"no_context": baseline, "multiple_participants": peers[::-1]})


if __name__ == "__main__":
    unittest.main()
