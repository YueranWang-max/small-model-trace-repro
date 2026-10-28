# Hybrid local-model trace replication — design draft, not frozen

This is preparation for a fresh experiment, not an amendment to the active source-qualification study. No calls or outcome-based selections have been made under this design. Final runtime feasibility, runner and analysis must be frozen before any call. Source-performance gate failure remains failure; it is not a necessary condition for examining the trace mechanism (see GATE_SCOPE_AND_CONTROL_CORRECTION_20261004.md).

## Main question

Does the same native five-trace bundle work better before the full graph context than after it in a local small model, and does any gain survive stronger controls and complete deployment cost accounting?

## Planned cases and resource envelope

- Sixteen fresh official long BFS graphs: first16unused distinct BFS cases in catalogue interval[14000,16000). Exclude the58earlier graphs,16bridge graphs and8completed-qualification graphs, including failed/unknown/unreached cases. Compare canonical edge multisets, preserving duplicates. No selection by answers or model scores on the new cases. Independently recompute every selected label. This revision is informed by the completed eight-case pilot and preflight data availability; the former four-stratum draft was never frozen/run and is preserved in hybrid_trace_superseded_four_strata_draft. See HYBRID_TRACE_SCOPE_REVISION_20261004.md. It does not support claims about parents or short graphs.
- qwen3.5:4b fixed digest and runtime, official general-thinking profile from the fixed archived model card. Prospective context131072, source and reader output allowance32768; all source and reader arms use the same context allocation. A separately logged one-token calibration verifies actual rendering/model/context before scored calls; include its cost.
- Five source seeds202610120–124 and five reader seeds202610130–134, fixed across cases. Source5plus five reader conditions×5repeats gives30calls per graph,480planned scored calls plus calibration. This count is stated before any results; it is not480independent graph cases.
- Confirm actual rendered tokens plus output reserve and64-token margin fit before each request. If the fixed five-trace bundle exceeds the resource envelope, retain the case and source costs. A prospective operational budget-decline rule must distinguish not-sent deployment failures from unknown transmitted calls. Do not silently clip further, reduce trace count, drop the case or change context after seeing results. Exact handling remains to be implemented and frozen.

## Five source traces, one matched bundle

Generate from full source+question, save raw native thinking and visible answers separately. Serialize only native thinking in fixed source order, using the existing original PD/delimiters/labels and head50000-character cap per trace. Do not filter wrong or length-terminated source traces. The identical serializedT is used by pre/post. Record individual clipping, full bundle hash and exact retained characters/tokens. Visible source answers are excluded fromT.

Split the full official prompt at its finalOperation boundary: x=context/head; q=Operation tail including terminal-format suffix. Verify [x,q] exactly reconstructs the source request. Every reader ends with the sameq.

| Reader arm | Prompt | What it controls |
|---|---|---|
| fresh source-only | [x,q] | Another sample at the same reader budget |
| Question First | [q,x,q] | Task conditioning before context; graph once |
| Re2 | [x,q,x,q] | Context repetition inside one request |
| trace prefix | [T,x,q] | Native trace state before context |
| trace append | [x,T,q] | Same native state after context |

Fresh source-only is not the paper's Re2. Keep the term Question First distinct from Answer Feedback. Rotate reader arm order by a frozen deterministic schedule, use identical reader seeds across arms, and record full payloads before transmission. No retries or unlogged warmups. Calibration or model/input mismatch stops the run while preserving all records.

## Cheap source-answer controls

Use both SetMode@5 and NodeMajority@5 as explicitly defined programmatic diagnostics. The original paper's Appendix E, Figure 6 caption specifies element-wise votes at a threshold of three; NodeMajority follows that rule. SetMode is an additional different control. They require all five source calls and no new reader call. The initial draft incorrectly called the original rule vague; that statement was corrected after checking the appendix.

- SetMode: canonical node sets from strictly valid source final lines; most frequent set, ties resolved by earliest source index. All invalid means abstention scored0.
- NodeMajority: nodes present in at least3of the5source outputs; invalid outputs contribute no votes. The program returns its aggregate set, including an empty set when all sources are invalid. That empty set scores normally; disclose all-invalid cases and the always-empty baseline. Unknown/wrong nodes remain false positives; no gold-based cleaning. This matches the previously audited local five-trace vote implementation; the earlier session summary incorrectly labeled that implementation as SetMode.
- Oracle@5 is retrospective privileged diagnosis, never a deployable baseline. Always-empty is separately reported.

## Units, scoring and uncertainty

Sixteen graph clusters are the units; keep all source/reader repetitions and controls together in resampling. Strict normal-stop visible terminal node-setF1/EM; no thinking salvage. Output-budget failures and malformed answers stay in the denominator. Distinguish known client deployment failures, unqualified model replies, not-sent budget declines and interrupted unknown outcomes.

Proposed comparison family: prefix against append, fresh source-only, Question First, Re2, SetMode and NodeMajority. Declare all six before outcomes. Report mean differences and graph-level descriptive stratified bootstrap intervals, plus exact paired sign tests on non-tied graph differences withHolm adjustment across the six tests. The sign test concerns direction across graph pairs, not a distribution-free test of meanF1; state that distinction. No across-stage cherry-picked significance. Freeze exact tie tolerance, bootstrap seed/count, strata, missingness treatment and practical decision criteria in executable analysis before launching. This draft alone is not preregistration.

## Costs and claims

One trace deployment pays for all five source calls plus one reader. Average reader repetitions estimate the expected single-reader cost; never divide producer cost by five. A source-only, Question First or Re2 deployment pays for its own single call. Programmatic vote deployments pay all five sources. Report full research ledger, dictionary/preparation/calibration and unknown costs separately. Token counts do not establish speed or energy improvement; repeated-prompt caching can affect actual runtime.

An ordering effect, better accuracy, reliable output and cost reduction are separate claims. Weak-source rescue can be informative even if the source gate failed; positive point estimates on a few graphs are not innovation. A robust positive result still needs independent confirmation; a bounded negative result should state model, task, representation, context and output budgets precisely.
