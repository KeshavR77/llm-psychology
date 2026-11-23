import unittest
from types import SimpleNamespace

from .choice_logprobs import make_items, score_items
from .participant_role_logprobs import align_references, make_role_items, role_prefix, summarize_role
from .random_choice_logprobs import make_random_items
from .test_choice_logprobs import FakeModel, FakeTokenizer


class RolePromptTests(unittest.TestCase):
    def test_role_sentence_is_only_change_to_all_conditions(self):
        items = make_role_items(limit=5)
        originals = make_items(limit=5)
        randoms = make_random_items(limit=5)
        prefix = "You are Participant 8 in a quiz with 8 participants.\n\n"
        self.assertEqual(role_prefix(7), prefix)
        for role, original, random in zip(items, originals, randoms):
            self.assertEqual(role["prompts"]["role_no_responses"], prefix + original["prompts"]["no_context"])
            self.assertEqual(role["prompts"]["role_unanimous_wrong"], prefix + original["prompts"]["multiple_participants"])
            self.assertEqual(role["prompts"]["role_random_participants"], prefix + random["prompts"]["random_participants"])
            self.assertEqual(role["random_peer_answers"], random["peer_answers"])
            self.assertEqual(role["wrong_target"], original["wrong_target"])
        self.assertEqual(len(make_role_items()), 100)
        self.assertEqual(role_prefix(3), "You are Participant 4 in a quiz with 4 participants.\n\n")

    def test_reference_pairing_and_prompt_mismatches(self):
        role_items = make_role_items(limit=2)
        items = make_items(limit=2)
        random_items = make_random_items(limit=2)
        def record(item, condition):
            return {**{k: v for k, v in item.items() if k != "prompts"}, "prompt": item["prompts"][condition]}
        references = {name: [record(item, name) for item in items][::-1]
                      for name in ("no_context", "multiple_participants")}
        random_reference = {"random_participants": [record(item, "random_participants") for item in random_items]}
        aligned = align_references(role_items, references, random_reference, 7)
        self.assertEqual([row["item_id"] for row in aligned["no_context"]],
                         [item["item_id"] for item in role_items])
        random_reference["random_participants"][0]["prompt"] += "wrong"
        with self.assertRaisesRegex(ValueError, "Prompt other than role prefix"):
            align_references(role_items, references, random_reference, 7)

    def test_role_conditions_scored_and_paired(self):
        runner = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeModel(), device="cpu")
        items = make_role_items(limit=2)
        results, _ = score_items(runner, items, batch_size=4)
        self.assertEqual(set(results), {"role_no_responses", "role_random_participants", "role_unanimous_wrong"})
        # The fake model returns identical logits in every condition.
        references = {"no_context": results["role_no_responses"],
                      "random_participants": results["role_random_participants"],
                      "unanimous_wrong": results["role_unanimous_wrong"]}
        summary = summarize_role(results, references, bootstrap_samples=20)
        self.assertEqual(len(summary["comparisons"]), 6)
        for comparison in summary["comparisons"].values():
            self.assertEqual(comparison["paired_differences"]["accuracy"]["value"], 0)
            self.assertEqual(comparison["paired_differences"]["wrong_target_choice_rate"]["value"], 0)


if __name__ == "__main__":
    unittest.main()
