# Fresh-question user-benefit study, 2026-10-09

## Completion and scope

All 168/168 planned calls were independently checked on 24 new graph units. Model: qwen3:1.7b, fixed archived digest; new runtime Ollama0.40.0. Two reader seeds are within-question repeats, not 48 independent questions. This is an extension to the earlier reproduction, not a new run of the original five-trace intervention.

## Observed user benefit

The predefined nonempty exact-in-both-seeds capacity criterion was not met. Source size reduction alone is not counted as a task-solving benefit.

The original8K reference was never sent with a truncated source and is not scored as zero accuracy. Executed original readers instead used32K context and an8K output allowance; compact capacity readers used8K context and a4K output allowance. Their comparison changes both representation and resource settings and evaluates deployment packages, not an isolated causal compression effect.

## Complete executed-arm results

| Stratum | Arm | F1 | EM | Full tokens/answer | Complete wall seconds/answer | Valid/24 | Output-truncated |
|---|---|---:|---:|---:|---:|---:|---:|
| capacity | original | 0.0102 | 0.0000 | 16521.67 | 60.81 | 14/24 | 10 |
| capacity | compact | 0.0833 | 0.0833 | 5905.79 | 25.43 | 9/24 | 15 |
| capacity | prefix | 0.1897 | 0.0417 | 9632.62 | 28.59 | 12/24 | 12 |
| short | original | 0.4908 | 0.2083 | 6582.04 | 24.99 | 21/24 | 3 |
| short | compact | 0.6661 | 0.2917 | 2911.00 | 11.41 | 24/24 | 0 |
| short | prefix | 0.8195 | 0.5417 | 5449.17 | 15.14 | 24/24 | 0 |

Prefix cost includes the entire single1024-cap source pass plus one reader; it is not divided by the two reader repeats. Preparation CPU time and inverse dictionary storage are separate. Wall times are local observations including runtime behavior, not server-independent speed guarantees.

## Paired graph-level contrasts

| Stratum | Contrast | Mean F1 difference | Descriptive95% interval |
|---|---|---:|---|
| capacity | compact-original | 0.0732 | [-0.0171, 0.1987] |
| capacity | prefix-compact | 0.1064 | [-0.0686, 0.2868] |
| short | compact-original | 0.1753 | [-0.0841, 0.3930] |
| short | prefix-compact | 0.1534 | [0.0394, 0.2769] |

The primary short-input comparison is compact-direct minus original-direct. The prefix comparisons are exploratory. An interval crossing zero is not proof of equivalence or improvement. No outcome-driven extension or sample filtering was performed.

## Does the short trace itself help?

- capacity: prefix-minus-direct F1=0.1064; full preparation-inclusive token ratio=1.631. This must be distinguished from benefits attributable to reversible labels. No placement mechanism is isolated because there is no matched append arm.
- short: prefix-minus-direct F1=0.1534; full preparation-inclusive token ratio=1.872. This must be distinguished from benefits attributable to reversible labels. No placement mechanism is isolated because there is no matched append arm.

## Evidence and limitations

Exact raw HTTP bytes, write-ahead payloads and fixed selections are in `logs/user_benefit_20261009`. Independent source-round-trip and rescore evidence is in `paper/evidence_user_benefit_20261009`. A CPU graph oracle is used only to verify labels and score outputs, never to choose mappings or fill model traces. Conventional algorithms can solve these graph tasks directly; this study probes model behavior and does not show that language models are the best production graph solvers. No natural-document transfer, cross-question amortization, energy, monetary or third-party execution benefit is established.

A favorable user-benefit result should be integrated into the manuscript only with its complete failure counts, resource contract and independent audit. This report alone does not certify arXiv readiness.

## Complete source-pass outcomes

All 24 source passes are retained: 22 length-terminated, 2 normal-stop valid visible answers, 2 exact qualified visible answers. Source thinking is reused without excluding failed source answers; separate visible answers are not placed in the trace.
