# Independent confirmation of single-trace reuse

## Scope and completion

All 264/264 calls were independently audited on 24 fresh graph units, with two reader seeds averaged within each graph. Discovery cases are excluded. Runtime: Ollama0.40.0; model qwen3:1.7b with frozen digest. No pooling with earlier runs.

## Prespecified primary result

Primary accuracy-benefit criterion met: True. This refers only to prefix versus complete reversible compact-direct input on the selected graph benchmark.

| Arm | F1 | Exact match | Full tokens/answer | Full wall seconds/answer | Valid/48 | Length terminated |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.3176 | 0.0417 | 6640.67 | 28.61 | 43/48 | 5 |
| original_long | 0.3668 | 0.0833 | 7149.88 | 33.81 | 45/48 | 3 |
| compact | 0.6542 | 0.2292 | 3823.25 | 19.54 | 42/48 | 6 |
| prefix | 0.7703 | 0.3750 | 5344.46 | 16.46 | 47/48 | 1 |
| append | 0.7130 | 0.3333 | 5345.67 | 16.60 | 46/48 | 2 |

Prefix and append include the entire one-source preparation pass per answer, without amortizing it over the two repeats. Original-long allows 16384 generated tokens; original and compact allow8192; each trace package allows1024 source plus7168 reader tokens. Input lengths and actual compute are not matched. CPU recoding time and dictionary bytes are archived separately. Wall time is an observation on this computer, not an energy or monetary claim.

## Graph-level contrasts

| Contrast | F1 difference [descriptive95% interval] | Two-sided sign-flip p | Full token ratio [95% interval] |
|---|---|---:|---|
| prefix-compact | 0.1161 [0.0437, 0.1914] | 0.00624 | 1.398 [1.170, 1.667] |
| prefix-append | 0.0573 [-0.0321, 0.1705] | 0.36592 | 1.000 [0.942, 1.061] |
| prefix-original | 0.4527 [0.3320, 0.5656] | 0.00001 | 0.805 [0.724, 0.896] |
| prefix-original_long | 0.4034 [0.2931, 0.5087] | 0.00002 | 0.747 [0.635, 0.877] |
| append-compact | 0.0588 [-0.0600, 0.1695] | 0.35848 | 1.398 [1.172, 1.678] |
| compact-original | 0.3366 [0.1932, 0.4759] | 0.00027 | 0.576 [0.466, 0.711] |

24 graph units, two seeds averaged within unit; primary fixed before responses; descriptive intervals and sign-flip exchangeability assumption; secondary comparisons unadjusted for multiplicity; complete source preparation charged to each reused answer; no energy or monetary cost claim.

The primary rule requires a positive mean F1 difference, a positive lower descriptive bootstrap limit and two-sided paired sign-flip p<=.05. The sign-flip test assumes exchangeability of paired differences under its null. All other contrasts are secondary and unadjusted for multiplicity. A confidence interval crossing zero does not prove equivalence. There is no outcome-driven extension of the frozen sample.

## Boundaries and evidence

The previous discovery study failed its nonempty exact-in-both-seeds8K capacity criterion; this confirmation does not reverse that result. No natural-document transfer, universal reasoning benefit, third-party execution attestation, or lower monetary cost has been established. Conventional CPU graph algorithms directly solve these tasks: the benchmark diagnoses language-model behavior rather than proving production superiority over algorithms.

Write-ahead requests and exact response bytes: logs/trace_benefit_confirmation_20261010. Independent payload, source-round-trip and rescore evidence: paper/evidence_trace_benefit_confirmation_20261010. Incorrect and truncated source traces are retained. Every completed reader and failure is included.

## Complete source-pass outcomes

All 24 source passes are retained: 22 length-terminated, 2 normal-stop valid visible answers, 0 exact qualified visible answers. Source thinking is reused without excluding failed source answers; separate visible answers are not placed in the trace.
