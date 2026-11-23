import unittest
from types import SimpleNamespace

from .inspect_answer_tokens import diagnose, summarize


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text in "ABCD" and len(text) == 1:
            return [10 + "ABCD".index(text)]
        if text == " ":
            return [14]
        if text == "\n":
            return [15]
        if len(text) >= 2 and text[-2] == " " and text[-1] in "ABCD":
            return ([] if len(text) == 2 else [20]) + [1 + "ABCD".index(text[-1])]
        return [20]

    def __call__(self, texts, **kwargs):
        import torch
        return {"input_ids": torch.full((len(texts), 1), 20),
                "attention_mask": torch.ones((len(texts), 1), dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        return [f"token-{i}" for i in ids]

    def decode(self, ids, **kwargs):
        return " " if ids == [14] else f"token-{ids[0]}"


class FakeModel:
    def eval(self):
        pass

    def __call__(self, input_ids, **kwargs):
        import torch
        logits = torch.zeros((len(input_ids), 1, 21))
        logits[:, :, 14] = 7
        logits[:, :, 2] = 5
        return SimpleNamespace(logits=logits)


class InspectAnswerTokensTests(unittest.TestCase):
    def test_tracks_top_tokens_and_candidate_mass(self):
        runner = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeModel(), device="cpu")
        row = {"item_id": "one", "prompt": "Your answer:", "model_answer": "B", "abcd_probability_mass": 0.1}
        results, spaced, bare = diagnose(runner, {"random_participants": [row]}, top_k=3)
        item = results["random_participants"][0]
        self.assertEqual(spaced, {"A": 1, "B": 2, "C": 3, "D": 4})
        self.assertEqual(bare, {"A": 10, "B": 11, "C": 12, "D": 13})
        self.assertEqual(item["top_tokens"][0]["text"], " ")
        self.assertGreater(item["special_token_probabilities"]["space"], item["abcd_mass"])
        self.assertGreater(item["abcd_mass"], item["bare_abcd_mass"])
        summary = summarize(results)["random_participants"]
        self.assertEqual(summary["top_next_tokens"], [{"text": " ", "count": 1}])


if __name__ == "__main__":
    unittest.main()
