import argparse
import csv
import json
import random
from pathlib import Path


METRICS = ("accuracy", "parse_rate", "wrong_target_adoption", "baseline_correct_adoption")
INTERPRETATION = (
    "Descriptive activation contrasts: explicit attribution versus unattributed assertions, "
    "and distinct speakers versus one repeated speaker. These do not establish a pure "
    "conformity mechanism. Token lengths are reported without artificial padding."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def align_records(reference, control, baseline):
    reference_ids = [row["item_id"] for row in reference]
    if len(set(reference_ids)) != len(reference_ids) or not reference_ids:
        raise ValueError("Reference requires unique, nonempty item IDs")
    aligned = [reference]
    for rows in (control, baseline):
        by_id = {row["item_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(reference_ids):
            raise ValueError("Conditions must contain exactly the same unique item IDs")
        ordered = [by_id[item_id] for item_id in reference_ids]
        for left, right in zip(reference, ordered):
            if any(left[key] != right[key] for key in ("ground_truth", "wrong_target")):
                raise ValueError("Paired items have inconsistent answers")
        aligned.append(ordered)
    return aligned


def metrics(rows, baseline, indices):
    count = len(indices)
    parseable = sum(rows[i]["parseable"] for i in indices)
    correct = sum(rows[i]["is_correct"] for i in indices)
    adopted = sum(rows[i]["adopts_wrong_target"] for i in indices)
    baseline_correct = [i for i in indices if baseline[i]["is_correct"]]
    return {
        "accuracy": correct / parseable if parseable else None,
        "parse_rate": parseable / count if count else None,
        "wrong_target_adoption": adopted / count if count else None,
        "baseline_correct_adoption": (
            sum(rows[i]["adopts_wrong_target"] for i in baseline_correct) / len(baseline_correct)
            if baseline_correct else None
        ),
    }


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def estimate(value, samples):
    return {
        "value": value,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "valid_bootstrap_samples": len(samples),
    }


def counts(rows, baseline):
    subset = [row for row, base in zip(rows, baseline) if base["is_correct"]]
    return {
        "items": len(rows),
        "parseable": sum(row["parseable"] for row in rows),
        "parse_failures": sum(not row["parseable"] for row in rows),
        "correct": sum(row["is_correct"] for row in rows),
        "wrong_target_adoptions": sum(row["adopts_wrong_target"] for row in rows),
        "baseline_correct_items": len(subset),
        "baseline_correct_parse_failures": sum(not row["parseable"] for row in subset),
        "baseline_correct_wrong_target_adoptions": sum(row["adopts_wrong_target"] for row in subset),
    }


def summarize(reference, control, baseline, bootstrap_samples=10_000, seed=42):
    reference, control, baseline = align_records(reference, control, baseline)
    conditions = {"multiple_participants": reference, control[0]["condition"]: control, "no_context": baseline}
    indices = list(range(len(reference)))
    observed = {name: metrics(rows, baseline, indices) for name, rows in conditions.items()}
    samples = {name: {metric: [] for metric in METRICS} for name in conditions}
    difference_samples = {metric: [] for metric in METRICS}
    control_name = control[0]["condition"]
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        draw = rng.choices(indices, k=len(indices))
        estimates = {name: metrics(rows, baseline, draw) for name, rows in conditions.items()}
        for name in conditions:
            for metric in METRICS:
                value = estimates[name][metric]
                if value is not None:
                    samples[name][metric].append(value)
        for metric in METRICS:
            a, b = estimates["multiple_participants"][metric], estimates[control_name][metric]
            if a is not None and b is not None:
                difference_samples[metric].append(a - b)

    differences = {}
    for metric in METRICS:
        a, b = observed["multiple_participants"][metric], observed[control_name][metric]
        differences[metric] = estimate(a - b if a is not None and b is not None else None,
                                       difference_samples[metric])
    length_deltas = [a["sequence_length"] - b["sequence_length"] for a, b in zip(reference, control)]
    return {
        "interpretation": INTERPRETATION,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metric_denominators": {
            "accuracy": "parseable outputs in each condition",
            "parse_rate": "all selected items",
            "wrong_target_adoption": "all selected items; parse failures count as no adoption",
            "baseline_correct_adoption": "items correct in the new no-context baseline; parse failures count as no adoption",
        },
        "conditions": {
            name: {"counts": counts(rows, baseline), "metrics": {
                metric: estimate(observed[name][metric], samples[name][metric]) for metric in METRICS
            }} for name, rows in conditions.items()
        },
        "difference_direction": f"multiple_participants_minus_{control_name}",
        "paired_differences": differences,
        "sequence_length_difference": {
            "direction": "multiple_participants minus control",
            "min": min(length_deltas), "max": max(length_deltas),
            "mean": sum(length_deltas) / len(length_deltas),
        },
    }


def write_summary(directory, summary):
    write_json(directory / "analysis.json", summary)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "metric", "value", "ci_low", "ci_high", "valid_bootstrap_samples"])
        writer.writeheader()
        for condition, data in summary["conditions"].items():
            for metric, result in data["metrics"].items():
                writer.writerow({"condition": condition, "metric": metric, **result})
        for metric, result in summary["paired_differences"].items():
            writer.writerow({"condition": summary["difference_direction"], "metric": metric, **result})

    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"

    lines = ["# Social attribution control", "", INTERPRETATION, "",
             "| Condition | Metric | Estimate | Paired item-bootstrap 95% interval |",
             "| --- | --- | ---: | --- |"]
    for name, data in summary["conditions"].items():
        for metric, result in data["metrics"].items():
            lines.append(f"| {name} | {metric} | {fmt(result['value'])} | [{fmt(result['ci_low'])}, {fmt(result['ci_high'])}] |")
    for metric, result in summary["paired_differences"].items():
        lines.append(f"| {summary['difference_direction']} | {metric} | {fmt(result['value'])} | [{fmt(result['ci_low'])}, {fmt(result['ci_high'])}] |")
    lines.extend(["", "Metric denominators:", ""])
    lines.extend(f"- {key}: {value}." for key, value in summary["metric_denominators"].items())
    lines.extend(["", "Counts and parse failures:", ""])
    lines.extend(f"- {name}: {json.dumps(data['counts'])}" for name, data in summary["conditions"].items())
    lines.extend(["", f"Sequence-length differences: {json.dumps(summary['sequence_length_difference'])}", ""])
    (directory / "summary.md").write_text("\n".join(lines))


def analyze_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    reference = json.loads((run_dir / "shared" / "multiple_participants.json").read_text())["results"]
    baseline = json.loads((run_dir / "shared" / "no_context.json").read_text())["results"]
    vectors = {}
    import torch

    for control in manifest["controls"]:
        directory = run_dir / control
        records = json.loads((directory / "behavioral.json").read_text())["results"]
        write_summary(directory, summarize(reference, records, baseline))
        vectors[control] = torch.load(directory / "diffmeans.pt", map_location="cpu", weights_only=True)["diffmeans"]
    layers = sorted(vectors[manifest["controls"][0]])
    layer_rows = []
    for layer in layers:
        row = {"layer": layer}
        for control, mapping in vectors.items():
            row[f"{control}_norm"] = float(mapping[layer].norm().item())
        if len(vectors) == 2:
            a, b = (vectors[name][layer].float() for name in manifest["controls"])
            denominator = a.norm() * b.norm()
            row["cosine_similarity"] = float((a @ b / denominator).clamp(-1, 1).item()) if denominator > 0 else None
        layer_rows.append(row)
    write_json(run_dir / "activation_summary.json", {"interpretation": INTERPRETATION, "layers": layer_rows})
    with (run_dir / "activation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)


def main():
    parser = argparse.ArgumentParser(description="Rebuild social-control summaries without model inference.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()
