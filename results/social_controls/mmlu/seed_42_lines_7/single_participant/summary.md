# Social attribution control

Descriptive activation contrasts: explicit attribution versus unattributed assertions, and distinct speakers versus one repeated speaker. These do not establish a pure conformity mechanism. Token lengths are reported without artificial padding.

| Condition | Metric | Estimate | Paired item-bootstrap 95% interval |
| --- | --- | ---: | --- |
| multiple_participants | accuracy | 0.2222 | [0.1429, 0.3093] |
| multiple_participants | parse_rate | 0.9900 | [0.9700, 1.0000] |
| multiple_participants | wrong_target_adoption | 0.7700 | [0.6800, 0.8500] |
| multiple_participants | baseline_correct_adoption | 0.7629 | [0.6737, 0.8454] |
| single_participant | accuracy | 0.8182 | [0.7374, 0.8900] |
| single_participant | parse_rate | 0.9900 | [0.9700, 1.0000] |
| single_participant | wrong_target_adoption | 0.1800 | [0.1100, 0.2600] |
| single_participant | baseline_correct_adoption | 0.1856 | [0.1122, 0.2653] |
| no_context | accuracy | 1.0000 | [1.0000, 1.0000] |
| no_context | parse_rate | 0.9700 | [0.9300, 1.0000] |
| no_context | wrong_target_adoption | 0.0000 | [0.0000, 0.0000] |
| no_context | baseline_correct_adoption | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_single_participant | accuracy | -0.5960 | [-0.6939, -0.4949] |
| multiple_participants_minus_single_participant | parse_rate | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_single_participant | wrong_target_adoption | 0.5900 | [0.4900, 0.6900] |
| multiple_participants_minus_single_participant | baseline_correct_adoption | 0.5773 | [0.4792, 0.6771] |

Metric denominators:

- accuracy: parseable outputs in each condition.
- parse_rate: all selected items.
- wrong_target_adoption: all selected items; parse failures count as no adoption.
- baseline_correct_adoption: items correct in the new no-context baseline; parse failures count as no adoption.

Counts and parse failures:

- multiple_participants: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 22, "wrong_target_adoptions": 77, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 74}
- single_participant: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 81, "wrong_target_adoptions": 18, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 18}
- no_context: {"items": 100, "parseable": 97, "parse_failures": 3, "correct": 97, "wrong_target_adoptions": 0, "baseline_correct_items": 97, "baseline_correct_parse_failures": 0, "baseline_correct_wrong_target_adoptions": 0}

Sequence-length differences: {"direction": "multiple_participants minus control", "min": 0, "max": 0, "mean": 0.0}
