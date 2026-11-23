# Mechanistic Interpretability of Conformity in LLMs

Code, data, and results for the paper *Mechanistic Interpretability of
Conformity in LLMs*. We study peer-majority conformity in `Qwen/Qwen3-8B` on
MMLU and BIG-Bench Hard, find a late-middle-layer residual-stream DiffMean
direction associated with incorrect consensus pressure, and show that steering
along it both increases and reduces conformity.

## Repository Layout

```
data/                        Fixed paper subsets and MMLU-Redux item sets
item_selection/              Scripts that built data/ (no-context answering + intersection)
multi_actor.py               Conformity prompts, generation, and scoring (core library)
run_behavioral.py            Runs all 48 behavioral conditions
layer_sweep.py               DiffMean directions and activation-oracle queries
oracle_utils.py              Activation-oracle helpers
project_experiment_activations.py   Layer-23 condition-mean projections
analyze_qwen_diffmean_cosines.py    Layer-wise DiffMean cosine structure
top_projection_examples.py   Items with the largest projection onto a DiffMean
steering.py                  Residual-stream steering hooks
run_steering.py              Per-layer steering sweep (Experiment 1)
run_ablation.py              All-layer single-vector steering (Experiment 2)
data_analysis/               Bootstrap statistics, paper plots, oracle/projection summaries
results/
  mmlu_redux/                New pipeline outputs, isolated from paper results
  behavioral/{mmlu,bbh}/     48 behavioral runs (n=1..10, QD, DA; base and explain)
  layer_sweep/               DiffMean tensors and activation-oracle outputs
  projection/                Layer-23 projection results
  activation_patching/       Steering figures and all-layer steering runs
paper/                       LaTeX source, figures, and generated tables
```

## Paper-to-Code Map

| Paper result | Code | Checked-in output |
| --- | --- | --- |
| Figs. 1-3, conformity statistics (Sec. 3) | `run_behavioral.py`, `data_analysis/bootstrap_qwen_results.py`, `data_analysis/plot_qwen_participants.py` | `results/behavioral/`, `paper/generated/`, `paper/figures/` |
| DiffMean directions (Sec. 4) | `layer_sweep.py` | `results/layer_sweep/qwen/*_diffmeans.pt` |
| Activation oracles, Fig. 4 (Sec. 4) | `layer_sweep.py --run-oracle`, `data_analysis/summarize_activation_oracle_outputs.py` | `results/layer_sweep/layer_sweep_{residual,delta}_*.json`, `paper/generated/qwen_activation_oracle_summary.json` |
| Projection analysis, Table 1 (Sec. 4) | `project_experiment_activations.py`, `data_analysis/projection_correlation_analysis.py` | `results/projection/`, `paper/generated/qwen_projection_*` |
| Layer-wise cosine structure, Fig. 5 (Sec. 4) | `analyze_qwen_diffmean_cosines.py` | values in `paper/sections/qwen_activation_projection_section.tex` |
| Per-layer steering heatmap, Fig. 6 (Sec. 5) | `run_steering.py` | `paper/figures/fig2_heatmap_dconf.png`, `results/activation_patching/figures/` |
| All-layer steering, Table 2 (Sec. 5) | `run_ablation.py` | `results/activation_patching/qwen_{mmlu,bbh}_ablation/` |

The checked-in outputs and paper describe the historical 100-item study. The
commands below now produce a separate Redux study and do not update the paper.
The raw per-layer, per-alpha steering sweep JSONs (~480 MB) are not checked in;
regenerate them with `run_steering.py` as shown below.

## Setup

Use Python 3.11 or newer. A GPU is expected for Qwen3-8B model runs. Run every
command from the repository root.

```bash
uv sync
source .venv/bin/activate
```

If Hugging Face requires credentials for model or adapter downloads:

```bash
huggingface-cli login
```

## Compute and Runtime

The historical paper experiments were run on a node with 2x NVIDIA H100 GPUs.
The larger Redux runs require more compute; the times below are historical.
Approximate wall-clock runtimes are:

- Behavioral inference: 1 hour.
- Activation patching and steering: 2.5 hours.
- Activation caching, DiffMean construction, and activation-oracle runs: 1 hour.
- All remaining analyses and figure generation: under 30 minutes each.

## Licenses and Upstream Assets

This repository's code and derived paper artifacts are released under the MIT
License; see `LICENSE`. Upstream assets retain their own licenses and terms:

- `Qwen/Qwen3-8B`: Apache-2.0.
- `meta-llama/Llama-3.1-8B-Instruct` (item selection only): Llama 3.1 Community License.
- MMLU (`hendrycks/test`): MIT License.
- [MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0): CC-BY-4.0.
- BIG-Bench Hard (`suzgunmirac/BIG-Bench-Hard`): MIT License.
- `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`: used as a
  Hugging Face adapter checkpoint under its upstream model-card terms; the
  release does not redistribute the adapter weights.

## Item Selection

The original pipeline now uses `data/mmlu_redux_500.json`: 500 questions from
MMLU-Redux 2.0, selected from the `ok` annotations and answered correctly
without peer context by both Qwen and Llama. Selection shuffles items within
each subject with seed 42, then draws one eligible item per subject per round.
The import pins the upstream revision; the selection manifest records hashes,
model identifiers, and subject counts. Screening is greedy using the original
no-context selection prompt; screening JSONL files retain prompts and responses.
Llama requires access to its gated Hugging Face model.

```bash
python -m item_selection.mmlu_redux import
python -m item_selection.mmlu_redux screen --model qwen
python -m item_selection.mmlu_redux screen --model llama
python -m item_selection.mmlu_redux select
```

Screening requires a CUDA GPU and can be resumed after interruption. On CU-VL,
`sbatch --array=0-1 item_selection/mmlu_redux_screen_cuvl.sbatch` runs the two
models separately; run `select` only after both jobs complete. Screening covers
the entire eligible pool and can take several hours. Each screening record includes
the candidate-file hash to prevent mixing results from different imports.

The checked-in historical `data/combined_mmlu_correct.json` and
`data/combined_bbh_correct.json` hold the 100 MMLU and 100 BBH paper items. Each item was answered
correctly, with no social context, by both `Qwen/Qwen3-8B` and
`meta-llama/Llama-3.1-8B-Instruct`. To rebuild them:

```bash
python item_selection/build_item_sets.py
```

This answers successive 200-question batches with
`item_selection/answer_without_context.py` and appends the intersection of the
two models' correct answers until each set has 100 items. Per-batch outputs go
to `results/item_selection/` (not tracked).

The optional, locally generated `data/mmlu_test.json` contains all 14,042 questions from the
[`cais/mmlu` test split](https://huggingface.co/datasets/cais/mmlu), shuffled
with seed 42 so a prefix covers multiple subjects. It includes the original
100 questions and retains 27 exact duplicate records present upstream. These
questions have **not** been screened for Qwen or Llama correctness. To create
the untracked file from the pinned upstream revision and checksum, run:

```bash
python item_selection/import_mmlu.py
```

For a larger Qwen experiment, first run a no-peer baseline, select the items
Qwen answered correctly, and then run the peer condition on that same set:

```bash
python multi_actor.py --model qwen --dataset mmlu --mode 1 --seed 42 \
  --data-file data/mmlu_test.json --limit 1000 \
  --output-file results/item_selection/qwen_mmlu_1000_baseline.json
python item_selection/select_qwen_correct.py results/item_selection/qwen_mmlu_1000_baseline.json \
  --output data/mmlu_qwen_1000_correct.json
python multi_actor.py --model qwen --dataset mmlu --mode 10 --seed 42 \
  --data-file data/mmlu_qwen_1000_correct.json \
  --output-file results/item_selection/qwen_mmlu_1000_mode10.json
```

Change the limit or omit it to use the full test split. These selected items
are correct for that particular Qwen baseline run; unlike the paper's fixed
set they have not been screened with Llama. Runs on the unscreened test split
or the Qwen-only selection should not be compared as if they used the original
100-item population. Custom datasets and limits require an explicit output
file so they cannot silently replace the existing behavioral results. Result
metadata records the input file and its SHA-256 hash.

## Behavioral Experiments

Run all 48 behavioral conditions (written to `results/mmlu_redux/behavioral/{mmlu,bbh}/`):

```bash
python run_behavioral.py
```

For restartable cluster runs, pass `--seed 42 --resume`; `--dataset mmlu` or
`--dataset bbh` and `--style base` or `--style explain` select a shard. On
CU-VL, `sbatch --array=0-3 item_selection/behavioral_redux_cuvl.sbatch` runs
all four shards. After the MMLU shards succeed, run
`sbatch item_selection/behavioral_redux_analysis_cuvl.sbatch` to generate MMLU
bootstrap tables and figures. To analyze both datasets, use the commands below
without the `--dataset mmlu` option after all four shards complete.

Regenerate one condition:

```bash
python multi_actor.py --model qwen --dataset mmlu --mode 10
python multi_actor.py --model qwen --dataset bbh --mode da --explain
```

Generate Redux behavioral tables, confidence intervals, and figures:

```bash
python data_analysis/bootstrap_qwen_results.py --dataset mmlu
python data_analysis/plot_qwen_participants.py --dataset mmlu
```

Outputs:

- `results/mmlu_redux/generated/qwen_bootstrap_results.json`
- `results/mmlu_redux/generated/qwen_condition_bootstrap.csv`
- `results/mmlu_redux/generated/qwen_contrast_tests.csv`
- `results/mmlu_redux/generated/qwen_trend_bootstrap.csv`
- `results/mmlu_redux/figures/qwen_conformity_by_participants.{pdf,png}`
- `results/mmlu_redux/figures/qwen_accuracy_by_participants.{pdf,png}`
- `results/mmlu_redux/figures/qwen_mitigation_conformity.{pdf,png}`

## Repetition and Speaker Attribution Controls

These isolated experiments use the fixed 100-item MMLU subset and raw,
answer-only prompts. Seven identical wrong answers are attributed to seven
participants, left unattributed, or attributed repeatedly to Participant 1.
Instructions, questions, answer examples, and the final response cue are shared.
A no-context baseline is evaluated once, as is the multiple-participant reference.
Existing behavioral and layer-sweep experiments are unchanged.

Preview every selected prompt without model dependencies or output files:

```bash
python -m experiments.social_controls.run --dry-run
```

Run both comparisons, or select one independently:

```bash
python -m experiments.social_controls.run
python -m experiments.social_controls.run --experiment speaker_free
python -m experiments.social_controls.run --experiment single_participant
```

For a formatting control, `--experiment speaker_free --speaker-free-format plain-labeled`
replaces each speaker-free `<answer>B</answer>` line with `Answer: B`; the
participant and baseline prompts stay unchanged. Use `--speaker-free-format answer-tagged`
for `Answer: <answer>B</answer>`, and `--no-activations` for behavioral-only runs.
Use a separate `--output-dir`
to avoid colliding with the default-format run. `--lines 3` tests three answers
per answer-present condition. These changes can also be previewed with `--dry-run`.

Defaults are Qwen3-8B, seven answer lines, all 100 local MMLU items in dataset
order, and seed 42. Generation is greedy with one attempt and a 50-token limit;
malformed responses are retained. Use `--lines`, `--limit`, and `--seed` to change
the run. A two-item GPU smoke run, followed by the full experiment, can use:

```bash
python -m experiments.social_controls.run --limit 2 --output-dir /tmp/social_controls_smoke
python -m experiments.social_controls.run
```

Outputs default to `results/social_controls/mmlu/seed_42_lines_7/`:

- `run.json` records configuration, item IDs, model/tokenizer identifiers, and repository revision.
- `shared/` contains the no-context and multiple-participant behavioral records.
- `speaker_free/` and `single_participant/` each contain `behavioral.json`,
  `analysis.json`, `summary.csv`, `summary.md`, and `diffmeans.pt`.
- `activation_summary.{json,csv}` reports layer-wise norms and, when both
  comparisons are run, cosine similarities between the two directions. Cosine
  similarity is undefined (`null` in JSON) for a zero-norm direction.

The summaries report parse failures, accuracy among parseable outputs, wrong-target
adoption among all items, and adoption among items correct in the new no-context
baseline. Parse failures count as no adoption and remain in adoption denominators.
Intervals use 10,000 paired item-bootstrap resamples with seed 42. Empty denominators
are reported as undefined, with the number of valid bootstrap resamples saved.

DiffMeans are the mean **multiple-participant minus control** residual vectors
over all selected items, captured at the final raw-prompt token. The saved prompts,
token IDs, positions, and sequence lengths expose length differences; no padding
is added. The `diffmeans` mapping is compatible with existing vector loaders.
These are descriptive contrasts of explicit attribution and distinct-source
agreement, not established pure conformity directions. Steering and paper
integration are separate follow-up work.

`--output-dir` changes the parent directory; the seed/line subdirectory is always
appended. Existing nonempty run directories are rejected. `--overwrite` replaces
that entire run directory, including previous experiment selections. Rebuild
summaries without rerunning inference using:

```bash
python -m experiments.social_controls.analyze results/social_controls/mmlu/seed_42_lines_7
python -m unittest experiments.social_controls.test_social_controls
```

The tests use synthetic tensors and a tiny fake runner; they do not download model
weights. Tensor tests require PyTorch. A real Qwen GPU smoke run is still required
to verify the deployment environment before a full run.

### Plain-answer choice scoring

`experiments.social_controls.choice_logprobs` compares a no-peer prompt with a
prompt containing seven named participants who all give the same wrong answer.
The prompts end in `Your answer:` and contain no XML tags. The default inputs are
the original 100 MMLU items, and the seeded wrong choices match the existing
social controls. No text is generated: Qwen scores the next tokens ` A`, ` B`,
` C`, and ` D`, and the highest-scoring one is the selected answer.

```bash
python -m experiments.social_controls.choice_logprobs --dry-run --limit 1
python -m experiments.social_controls.choice_logprobs --output-dir results/social_controls/choice_logprobs/my_run
sbatch experiments/social_controls/choice_logprobs_cuvl.sbatch
```

The Slurm script writes to a new `results/social_controls/choice_logprobs/job_<id>`
directory. `--lines`, `--limit`, `--seed`, and `--batch-size` adjust the run.
The output contains a manifest, both prompts and per-item answer log probabilities,
and a summary of accuracy, wrong-answer choices, baseline-correct adoption, and
paired bootstrap differences. The choice probabilities are renormalized over A–D;
the separately recorded A–D probability mass is relative to the full vocabulary.
This is a new prompt and scoring protocol; its outcomes are not directly comparable
to the previous generated-answer runs.

For a random-answer baseline using the same seven participant lines, each answer
is drawn independently and uniformly from A–D, including the correct answer. The
draws use a separate seed (`--random-seed`, default 1729), and are saved per item.
The original wrong-answer targets are retained to support paired comparisons:

```bash
python -m experiments.social_controls.random_choice_logprobs --dry-run --limit 1
sbatch experiments/social_controls/random_choice_logprobs_cuvl.sbatch \
  results/social_controls/choice_logprobs/my_run
```

Pass the completed choice-scoring run directory to Slurm, or use
`--reference-dir` with the Python command. Slurm outputs use their job IDs;
replace `my_run` with the actual output directory when using Slurm. The output includes paired comparisons
against both the no-peer and unanimous-wrong conditions; it checks the reference
data, prompts, model, and answer token IDs before comparing. Random answers can
agree, disagree, or happen to be correct.

The participant-role variant prepends `You are Participant 8 in a quiz with 8
participants.` to each prompt. It scores no prior responses, the same randomly
chosen peer responses, and the same unanimous wrong responses. The no-responses
condition still assigns the model a role in a quiz with other participants.
Its paired summaries also compare each condition to the corresponding no-role
run, checking that the role sentence is the only prompt change:

```bash
python -m experiments.social_controls.participant_role_logprobs --dry-run --limit 1
sbatch experiments/social_controls/participant_role_logprobs_cuvl.sbatch \
  results/social_controls/choice_logprobs/my_run \
  results/social_controls/random_choice_logprobs/my_run
```

Replace `my_run` with the completed random-choice output directory as well.
The script uses seven prior participants by default; `--lines` adjusts the
participant count in both the role sentence and peer prompts. For paired
comparisons, both reference directories must use the same questions, seeds,
participant count, model, and answer tokens.

## DiffMean Directions and Activation Oracles

The main activation direction is the MMLU `n=8`
`all_wrong_minus_random_answers` residual DiffMean at the assistant-start token.
The paper also uses the MMLU `n=10` residual DiffMean as a projection check.

Recompute the DiffMean tensors (written to `results/mmlu_redux/layer_sweep/qwen/`):

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 500 --activation-kind residual --position assistant_start --baseline random_answers
python layer_sweep.py --models qwen --dataset mmlu --mode 10 --limit 500 --activation-kind residual --position assistant_start --baseline random_answers
python layer_sweep.py --models qwen --dataset bbh --mode 4 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers
```

Recompute the activation-oracle runs used in the paper:

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 500 --activation-kind residual --position assistant_start --baseline random_answers --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 500 --activation-kind residual --position assistant_start --baseline no_participants --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 500 --activation-kind delta --position assistant_start --baseline random_answers --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 500 --activation-kind delta --position assistant_start --baseline no_participants --run-oracle
```

Summarize oracle outputs for the paper:

```bash
python data_analysis/summarize_activation_oracle_outputs.py
```

Output: `results/mmlu_redux/generated/qwen_activation_oracle_summary.json`

## Projection Analysis

Projection analysis computes mean layer-23 assistant-start residual activations
for each behavioral condition in `results/mmlu_redux/behavioral/` and projects each
condition mean onto the MMLU DiffMean direction.

```bash
python project_experiment_activations.py \
  --reference results/mmlu_redux/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_500_diffmeans.pt \
  --layer 23

python project_experiment_activations.py \
  --reference results/mmlu_redux/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_10_base_residual_assistant_start_all_wrong_minus_random_answers_n10_500_diffmeans.pt \
  --layer 23

python data_analysis/projection_correlation_analysis.py
```

Outputs:

- `results/mmlu_redux/projection/qwen_experiment_mean_activation_projection_layer23_onto_*.{json,md}`
- `results/mmlu_redux/generated/qwen_projection_correlation_summary.json`
- `results/mmlu_redux/generated/qwen_projection_correlations.csv`
- `results/mmlu_redux/generated/qwen_projection_prompt_deltas.csv`

The projection script caches activations under `results/mmlu_redux/projection/cache/`.
Pass `--overwrite` to recompute activations from model forward passes.

## Layer-Wise Cosine Structure

The layer-wise cosine structure section uses cosine similarities between
DiffMean vectors, primarily the MMLU `n=8` residual DiffMeans.

```bash
python analyze_qwen_diffmean_cosines.py --neighbor-k 5 --top-k 25
```

Outputs (in `results/mmlu_redux/diffmean_cosine/qwen_layer_sweep/`, not tracked):
`analysis.md`, `layer_topk_neighbors.{csv,json}`, and `layer_matrices/`. The
plotted values in Fig. 5 are copied from the layer-neighbor results into
`paper/sections/qwen_activation_projection_section.tex`.

## Activation Steering

Fig. 6 comes from the single-layer steering sweep, and Table 2 from the
all-layer runs in `results/mmlu_redux/activation_patching/qwen_*_ablation/`.

Single-layer steering sweep (Experiment 1), using the MMLU `n=8` DiffMean:

```bash
VECTOR=results/mmlu_redux/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_500_diffmeans.pt

python run_steering.py --model qwen --dataset mmlu --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir results/mmlu_redux/activation_patching/qwen_mmlu --seed 42
python run_steering.py --model qwen --dataset bbh --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir results/mmlu_redux/activation_patching/qwen_bbh --seed 42
```

All-layer single-vector steering (Experiment 2) with the layer-23 MMLU DiffMean:

```bash
python run_ablation.py --dataset mmlu --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir results/mmlu_redux/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset mmlu --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir results/mmlu_redux/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset bbh --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir results/mmlu_redux/activation_patching/qwen_bbh_ablation --seed 42
python run_ablation.py --dataset bbh --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir results/mmlu_redux/activation_patching/qwen_bbh_ablation --seed 42
```

## Build the Paper

```bash
cd paper
latexmk -pdf main.tex
```

The main paper source is `paper/main.tex`; section files are in
`paper/sections/`.
