# Research consequence of the five-source cost bound — not a frozen experiment

The five-trace migration study has terminated with a runtime-qualification interruption; its original records remain unchanged. Its single-prediction total-token criterion cannot pass under complete qualified execution because five source inputs alone exceed the fresh-call maximum. This note identifies a different cost-generating structure to test later; it is not an outcome or a novelty claim.

## Candidate: an early native state under one total budget

Instead of paying for five completed solutions and then compressing their traces, allocate part of a fixed per-question logical token allowance to **one short native-thinking source pass**, preserve its actual partial thinking, and spend the remainder on a fresh reader with that state and the full original source. An output-length-terminated source is an intentional partial computational state, not a successful source answer. Keep its raw termination and visible-output score; do not salvage a source answer from thinking. The reader must normally stop with a valid terminal visible answer.

Let the direct full-source input count be I and its allowance C. Freeze a common total allowance B=I+C. For a partial source call, record its actual input/output token cost S. For the independently rendered state-plus-source reader, let the intended input count be J. The reader's allowance is no more than

    max(0, B - S - J).

No extra reader call is possible when that remaining allowance is nonpositive. Any such decline remains a planned deployment failure with source cost retained. It is incorrect to divide costs by repetitions or to raise the direct baseline's cap after observing outcomes. Full trace compression after a completed expensive source cannot refund its generation bill.

## Necessary controls and exact limits

Full-source single pass, Question First and Re2 must also use the same common total allowance with their own rendered input counts subtracted, and remain in the denominator when they cannot fit. The same actual partial trace goes into matched prefix/append arms. A cheap task-cue-only prefix and an equal-size irrelevant/cross-case state control are needed to distinguish useful partial computation from additional instruction or generic scaffolding. Cross-case state assignment must be deterministic and frozen without answer-based matching; it must exclude visible source answers and gold information.

The short source's allowance, serialization/clipping, seeds, native interface, total-budget rule, strict scoring, stopping rule and success criteria must be finalized before future calls. Adaptive remaining output allowance depends only on incurred cost and intended input size, never on gold answers or observed accuracy. Report token feasibility as well as normal output, node-setF1/EM, full cost and graph-cluster uncertainty. No small improvement on existing cases counts as confirmation.

This is related to Trace as State, fixed-work inference allocation and bounded-state carryover; Markovian Thinking provides a conceptual link but its trained carryover result does not establish this zero-shot proposal. The proposal retains the original source and does not require lossless semantic compression. A beneficial result would require new independent cases and strong controls, followed by separate confirmation. The current fixed five-trace experiment is necessary evidence about the original migration question and is not replaced by this easier-to-budget proposal.

## Remaining question

Will a short partial state expose useful task/frontier information early enough to reduce the later reasoning needed to solve the graph? It may instead omit crucial work, amplify mistakes, or shorten the reader budget so far that accuracy declines. No outcome currently establishes either answer. Attention-selected copied original evidence is a separate lower-generation-cost candidate, requiring genuine access to internal relevance signals and measured preparation costs.

## 2026-10-04 follow-up: feasibility and control accounting

This is still an unfrozen design note; no new inference has been launched.

A short-state deployment reads the full source twice. If its source input is the same I as direct, its generated output is K, and its separately rendered reader input is J, the reader output allowance is exactly `C - K - J` under `B = I + C`. In particular, paying for source generation and then discarding its visible output does not remove those output tokens from K. The partial-state arm is infeasible if `K + J >= C`; it must remain a declined planned case. The intended J must be computed from the complete rendered request, not from adding standalone token counts for source and trace, because boundary tokenization and instructions can change the count. Input/context headroom is an additional qualification check, independent of total logical token budget.

A nominal short source cap therefore does not guarantee a fair or feasible reader. Before a future freeze, calculate intended counts on unseen cases and explicit upper bounds for retained-state serialization; choose one fixed source cap without reading future answer scores. Actual remaining allowance can use actual incurred K and actual intended J, as predeclared, but cannot depend on source correctness. State length, intentional source termination, reader budget, and deployment decline frequency must all be reported alongside accuracy.

The matched prefix and append arms may reuse one physical source trace in the research workflow, but each single-prediction deployment pays the entire source cost. Experimental sharing is not deployment amortization. Donor traces for cross-case controls also have a production cost: if testing a deployable donor-state method, charge that production in full unless genuine deployment reuse is independently demonstrated. If a cross-case trace is only a mechanistic negative control, explicitly label its accuracy comparison and avoid treating its token advantage from a free experimental artifact as deployment evidence. A deterministic irrelevant-text control can avoid model generation, but its serialization/preparation and full reader input still count; content length and exact token matching need predeclared rules.

All controls need an explicit own deployment ledger under the same B. For direct, task cue and Question First, subtract their actual rendered input from B to obtain the output allowance. For Re2, subtract the duplicated full-prompt input rather than granting the original direct output cap on top. When comparing efficiency under a shared maximum budget, distinguish an accuracy result at equal allowance from a reduction in observed full cost; spending less than a permitted cap is not an accuracy guarantee. These accounting rules do not establish that partial state will improve performance.

## Offline new-case preparation

`prepare_partial_state_feasibility_draft_20261004.py` has executed without inference or answer scoring. It excludes all 98 graph identities in prior official frozen manifests, including the unexecuted selections of the interrupted five-trace study. Its draft selects the next 16 long BFS cases, rows 522–537, by the same catalogue length interval [14000,16000). Corrected official-template/tokenizer counts for direct, question-first, Re2 and a task cue, plus explicitly synthetic state fixtures, are saved in `partial_state_feasibility_draft_v2_20261004.json`. The first JSON is invalid because its initial split used an example Operation marker and omitted the actual graph; it remains archived only for diagnosis (see PARTIAL_STATE_PREFLIGHT_CORRECTION_20261004.md). This is not a frozen study, an independent selection audit, or performance evidence. The fixture reader-input count is not an upper bound on future generated-state input; the actual complete rendered request must determine feasibility before each transmission. A new source cap must be selected before scored calls, with actual source cost retained for all declines.
