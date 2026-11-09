# Social attribution control

Descriptive activation contrasts: explicit attribution versus unattributed assertions, and distinct speakers versus one repeated speaker. These do not establish a pure conformity mechanism. Token lengths are reported without artificial padding.

| Condition | Metric | Estimate | Paired item-bootstrap 95% interval |
| --- | --- | ---: | --- |
| multiple_participants | accuracy | 0.6566 | [0.5600, 0.7475] |
| multiple_participants | parse_rate | 0.9900 | [0.9700, 1.0000] |
| multiple_participants | wrong_target_adoption | 0.3400 | [0.2500, 0.4302] |
| multiple_participants | baseline_correct_adoption | 0.3299 | [0.2371, 0.4255] |
| single_participant | accuracy | 0.8889 | [0.8200, 0.9495] |
| single_participant | parse_rate | 0.9900 | [0.9700, 1.0000] |
| single_participant | wrong_target_adoption | 0.1100 | [0.0500, 0.1800] |
| single_participant | baseline_correct_adoption | 0.1134 | [0.0521, 0.1818] |
| no_context | accuracy | 1.0000 | [1.0000, 1.0000] |
| no_context | parse_rate | 0.9700 | [0.9300, 1.0000] |
| no_context | wrong_target_adoption | 0.0000 | [0.0000, 0.0000] |
| no_context | baseline_correct_adoption | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_single_participant | accuracy | -0.2323 | [-0.3200, -0.1515] |
| multiple_participants_minus_single_participant | parse_rate | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_single_participant | wrong_target_adoption | 0.2300 | [0.1500, 0.3200] |
| multiple_participants_minus_single_participant | baseline_correct_adoption | 0.2165 | [0.1383, 0.3030] |

Metric denominators:

- accuracy: parseable outputs in each condition.
- parse_rate: all selected items.
- wrong_target_adoption: all selected items; parse failures count as no adoption.
- baseline_correct_adoption: items correct in the new no-context baseline; parse failures count as no adoption.

Counts and parse failures:

- multiple_participants: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 65, "wrong_target_adoptions": 34, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 32}
- single_participant: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 88, "wrong_target_adoptions": 11, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 11}
- no_context: {"items": 100, "parseable": 97, "parse_failures": 3, "correct": 97, "wrong_target_adoptions": 0, "baseline_correct_items": 97, "baseline_correct_parse_failures": 0, "baseline_correct_wrong_target_adoptions": 0}

Sequence-length differences: {"direction": "multiple_participants minus control", "min": 0, "max": 0, "mean": 0.0}
