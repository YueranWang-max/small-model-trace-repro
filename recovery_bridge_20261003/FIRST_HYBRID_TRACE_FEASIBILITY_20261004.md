# First five-trace deployment feasibility observation

Date: 2026-10-04. Scope: row 506 only, input-budget metadata. The frozen study is still running; no source or reader answer scores have been inspected or aggregated. This is not a completed accuracy analysis or independent final audit.

## Observed constraint

All five source calls for row 506 returned normal stop. The unchanged original-style serialization keeps the first 50,000 characters plus an ellipsis only when a source exceeds that threshold. The actual trace character lengths were 40,116; 34,511; 37,679; 36,275; and 41,308. None was clipped.

Both complete rendered reader requests (prefix and append) contain 144,677 intended input tokens under the frozen qualified tokenizer/renderer. The frozen context limit is 131,072; the common reader output cap is 32,768 with 64 tokens of headroom. Input alone exceeds context by 13,605 tokens. Input plus output reserve is 177,509, exceeding context by 46,437. Thus merely reducing the reader output reserve would not make this full input fit.

The runner persisted attempts 010 and 011 as `budget_declined`, `not_sent`, with reason `joint_trace_pair_or_individual_input_plus_output_reserve_exceeds_frozen_context`. Their raw generation usage is not fabricated: they were not transmitted, so the reader's known deployment generation cost is zero. The five actual source calls remain part of either single-prediction deployment cost. The predeclared joint policy applies to both trace arms and all five repetitions for this graph. Other controls continue under the original schedule; row 506 remains a planned case.

## Interpretation and limits

This is an observed context-feasibility boundary for this exact trace bundle and local configuration. It does not establish that the traces are correct or useful, that every remaining graph will decline, or that trace reuse fails when a trace fits. A decline has no reader accuracy observation; deployment-level scoring/missingness rules must follow the frozen analysis specification. Do not silently discard declined graphs to form a favorable conditional result.

An unchanged per-trace character threshold is not a portable context guarantee. Tokenization, generated length and full request formatting determine actual feasibility. A future bounded-state arm should freeze a complete-request token budget and retention policy before observing scores; head versus tail selection and shorter generation may lose useful information and require controls. It must be a separately labelled adaptation, not a rewrite of this original-style frozen migration study. A lower-input representation or runtime cache is also not proof that deleted evidence remains sufficient.

## Provenance

Metadata certificate: `first_hybrid_trace_feasibility_20261004.json`, containing bundle and attempt hashes, trace lengths, input counts and decline states; no answer text. Original records remain in `hybrid_five_trace_replication/attempt_010.json`, `attempt_011.json`, and `bundle_00.json`. Final independent audit must reconstruct these counts and frozen dependency hashes after termination.
