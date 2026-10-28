# Diagnostic conclusion: format compliance is a material confound

## Results

These are post-interruption development diagnostics, not new confirmatory results. Original frozen scores, missingness bounds and the failed advance gate are unchanged.

In 18 known prefix outputs, 11 were strictly valid and exact, five hit the output limit, and two stopped normally but included surrounding explanation. Direct had 18 known outputs: five strict exact, two valid incorrect, nine normal malformed, and two output-limit outcomes. Format failures are therefore uneven across arms.

A separate visible-only sensitivity parser accepts exactly one line matching `Final Answer: [unquoted node identifiers]` inside surrounding prose, only for normally stopped responses. It rejects ambiguous multiple lines, malformed identifiers and all length terminations. It never reads thinking for an answer and never replaces primary scoring. Frozen audited gold sets supply scoring labels.

On the same 18 graph/repetition pairs with prefix and each control present:

| Control | Prefix sensitive exact | Control sensitive exact |
|---|---:|---:|
| Direct | 13/18 | 13/18 |
| Question first | 13/18 | 13/18 |
| Re2 | 13/18 | 15/18 |
| Task cue | 13/18 | 10/18 |
| Paid neutral prefix | 13/18 | 10/18 |
| Append | 13/18 | 11/18 |

Repeated seeds are not independent graphs. These incomplete descriptive comparisons establish neither equivalence nor superiority. They show the apparent strict-score advantage over direct and question-first is not robust to this reasonable visible-answer parsing sensitivity. Output format compliance is a material alternative explanation. Re2's observed count is higher, but not a confirmed superiority result.

## Decision

Do not rerun the unchanged 1024-token-source configuration to fill its missing table. Its reliability gate is already unattainable, and its apparent correctness advantage over the principal simple baselines disappears in this diagnostic. Do not relax the original gate or retrofit parser changes as a successful experiment.

Next development work should separate semantic correctness, format validity and output truncation. Freeze both strict interface compliance and the single-unambiguous-visible-answer metric prospectively. Prefer fixing output enforcement equally across all arms before attributing changes to reasoning reuse. Increasing output allowance is not automatically an efficiency solution: the extra source and reader tokens remain paid.

Any new confirmation must use previously unselected canonical graphs and a batch-safe protocol. No new inference was necessary to reach this bounded negative conclusion. It does not refute Trace as State, longer or better source states, other models, or multi-query reuse.

## Reproducible artifacts

- `diagnose_partial_state_20261005.py` and `PARTIAL_STATE_FAILURE_DIAGNOSIS_20261005.json`.
- `format_sensitivity_20261005.py` and `FORMAT_SENSITIVITY_20261005.json`.
- The independent audit and frozen analysis in `partial_state_study_20261004`.
