# Social attribution control

Descriptive activation contrasts: explicit attribution versus unattributed assertions, and distinct speakers versus one repeated speaker. These do not establish a pure conformity mechanism. Token lengths are reported without artificial padding.

| Condition | Metric | Estimate | Paired item-bootstrap 95% interval |
| --- | --- | ---: | --- |
| multiple_participants | accuracy | 0.2222 | [0.1429, 0.3093] |
| multiple_participants | parse_rate | 0.9900 | [0.9700, 1.0000] |
| multiple_participants | wrong_target_adoption | 0.7700 | [0.6800, 0.8500] |
| multiple_participants | baseline_correct_adoption | 0.7629 | [0.6737, 0.8454] |
| speaker_free | accuracy | 0.1400 | [0.0800, 0.2100] |
| speaker_free | parse_rate | 1.0000 | [1.0000, 1.0000] |
| speaker_free | wrong_target_adoption | 0.8600 | [0.7900, 0.9200] |
| speaker_free | baseline_correct_adoption | 0.8557 | [0.7835, 0.9192] |
| no_context | accuracy | 1.0000 | [1.0000, 1.0000] |
| no_context | parse_rate | 0.9700 | [0.9300, 1.0000] |
| no_context | wrong_target_adoption | 0.0000 | [0.0000, 0.0000] |
| no_context | baseline_correct_adoption | 0.0000 | [0.0000, 0.0000] |
| multiple_participants_minus_speaker_free | accuracy | 0.0822 | [0.0119, 0.1568] |
| multiple_participants_minus_speaker_free | parse_rate | -0.0100 | [-0.0300, 0.0000] |
| multiple_participants_minus_speaker_free | wrong_target_adoption | -0.0900 | [-0.1700, -0.0200] |
| multiple_participants_minus_speaker_free | baseline_correct_adoption | -0.0928 | [-0.1717, -0.0204] |

Metric denominators:

- accuracy: parseable outputs in each condition.
- parse_rate: all selected items.
- wrong_target_adoption: all selected items; parse failures count as no adoption.
- baseline_correct_adoption: items correct in the new no-context baseline; parse failures count as no adoption.

Counts and parse failures:

- multiple_participants: {"items": 100, "parseable": 99, "parse_failures": 1, "correct": 22, "wrong_target_adoptions": 77, "baseline_correct_items": 97, "baseline_correct_parse_failures": 1, "baseline_correct_wrong_target_adoptions": 74}
- speaker_free: {"items": 100, "parseable": 100, "parse_failures": 0, "correct": 14, "wrong_target_adoptions": 86, "baseline_correct_items": 97, "baseline_correct_parse_failures": 0, "baseline_correct_wrong_target_adoptions": 83}
- no_context: {"items": 100, "parseable": 97, "parse_failures": 3, "correct": 97, "wrong_target_adoptions": 0, "baseline_correct_items": 97, "baseline_correct_parse_failures": 0, "baseline_correct_wrong_target_adoptions": 0}

Sequence-length differences: {"direction": "multiple_participants minus control", "min": 42, "max": 42, "mean": 42.0}
