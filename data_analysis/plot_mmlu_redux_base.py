"""Plot the completed MMLU-Redux base runs without waiting for other conditions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/mmlu_redux/behavioral/mmlu"
OUTPUT = ROOT / "results/mmlu_redux/figures"


def main():
    rng = np.random.default_rng(42)
    rates = {"accuracy": [], "conformity": []}
    intervals = {"accuracy": [], "conformity": []}
    expected_hash = None
    expected_items = None
    for n in range(1, 11):
        result = json.loads((RUNS / f"qwen_mmlu_{n}_base.json").read_text())
        rows = result["results"]
        keys = [(r["subject"], r["question"], tuple(r["choices"]), r["answer"]) for r in rows]
        if expected_hash is None:
            expected_hash, expected_items = result["dataset_sha256"], keys
        elif result["dataset_sha256"] != expected_hash or keys != expected_items:
            raise ValueError("Participant runs do not use the same items in the same order")
        parsed = np.array([bool(r.get("model_answer")) for r in rows])
        correct = np.array([bool(r.get("is_correct")) for r in rows])
        conforms = np.array([r.get("conforms") is True for r in rows])
        indices = rng.integers(0, len(rows), size=(10_000, len(rows)))
        denominator = parsed[indices].sum(axis=1)
        for key, values in (("accuracy", correct), ("conformity", conforms)):
            if key == "conformity" and n == 1:
                rates[key].append(np.nan)
                intervals[key].append((np.nan, np.nan))
                continue
            rates[key].append(float(values.sum() / parsed.sum()))
            samples = np.divide(values[indices].sum(axis=1), denominator,
                                out=np.full(len(denominator), np.nan), where=denominator > 0)
            intervals[key].append(tuple(np.nanpercentile(samples, [2.5, 97.5])))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True, sharey=True)
    for ax, key, label in zip(axes, ("accuracy", "conformity"), ("Accuracy", "Wrong-peer answer adoption")):
        x = np.arange(1, 11)
        y = np.array(rates[key]) * 100
        bands = np.array(intervals[key]) * 100
        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, bands[:, 0], bands[:, 1], alpha=0.18)
        ax.set_title(label)
        ax.set_xlabel("Total participants (including Qwen)")
        ax.set_xticks(x)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Percent of parseable responses")
    fig.suptitle("MMLU-Redux: base prompts, 500 screened questions")
    fig.text(0.5, 0.005, "Shaded bands: 95% item bootstrap intervals", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"mmlu_redux_base_participants.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUTPUT}")


if __name__ == "__main__":
    main()
