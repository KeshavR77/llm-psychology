# Social attribution control

Descriptive activation contrasts: explicit attribution versus unattributed assertions, and distinct speakers versus one repeated speaker. These do not establish a pure conformity mechanism. Token lengths are reported without artificial padding.

| Condition | Metric | Estimate | Paired item-bootstrap 95% interval |
| --- | --- | ---: | --- |
| multiple_participants | accuracy | 0.6566 | [0.5600, 0.7475] |
| multiple_participants | parse_rate | 0.9900 | [0.9700, 1.0000] |
| multiple_participants | wrong_target_adoption | 0.3400 | [0.2500, 0.4302] |
| multiple_participants | baseline_correct_adoption | 0.3299 | [0.2371, 0.4255] |
| speaker_free | accuracy | 0.0204 | [0.0000, 0.0515] |
| speaker_free | parse_rate | 0.9800 | [0.9500, 1.0000] |
| speaker_free | wrong_target_adoption | 0.9600 | [0.9200, 0.9900] |
| speaker_free | baseline_correct_adoption | 0.9588 | [0.9167, 0.9899] |
| no_context | accuracy | 1.0000 | [1.0000, 1.0000] |
| no_context | parse_rate | 0.9700 | [0.9300, 1.0000] |
| no_context | wrong_target_adoption | 0.0000 | [0.0000, 0.0000] |
| no_context | baseline_correct_adoption | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_speaker_free | accuracy | 0.6362 | [0.5400, 0.7298] |
| multiple_participants_minus_speaker_free | parse_rate | 0.0100 | [0.0000, 0.0300] |
| multiple_participants_minus_speaker_free | wrong_target_adoption | -0.6200 | [-0.7100, -0.5200] |
| multiple_participants_minus_speaker_free | baseline_correct_adoption | -0.6289 | [-0.7245, -0.5312] |

Metric denominators:

- accuracy: parseable outputs in each condition.
- parse_rate: all selected items.
- wrong_target_adoption: all selected items; parse failures count as no adoption.
- baseline_correct_adoption: items correct in the new no-context baseline; parse failures count as no adoption.

Counts and parse failures:

- multiple_participants: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 65, "wrong_target_adoptions": 34, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 32}
- speaker_free: {"items": 100, "parseable": 98, "parse_failures": 2, "correct": 2, "wrong_target_adoptions": 96, "baseline_correct_items": 97, "baseline_correct_parse_failures": 2, "baseline_correct_wrong_target_adoptions": 93}
- no_context: {"items": 100, "parseable": 97, "parse_failures": 3, "correct": 97, "wrong_target_adoptions": 0, "baseline_correct_items": 97, "baseline_correct_parse_failures": 0, "baseline_correct_wrong_target_adoptions": 0}

Sequence-length differences: {"direction": "multiple_participants minus control", "min": 12, "max": 12, "mean": 12.0}
