# Local Qwen3.5 five-trace migration protocol — preparation, not frozen

This file is not the runner's final launch protocol. Finish offline qualification, copy the finalized protocol to `hybrid_five_trace_replication_protocol.md`, set the analysis specification status to `frozen_before_first_call`, and freeze all listed dependencies and planned jobs before the first call. No inference has been made under this design. Current source performance does not determine whether the trace experiment is allowed; preserve that study's frozen gate verdict separately.

## Question and scope

Does native five-trace reuse improve local small-model graph reasoning, and does prefix placement outperform an identical appended bundle after controlling for task conditioning, repeated source and source-answer voting? All conditions preserve the original graph. This is a bounded migration of Zou and Tang's inference method and serializer, not a claim to reproduce their frontier-model256K setting. The model, data length strata, runtime and output budget differ. Source removal, paper fingerprints and natural-document generalization are separate follow-ups.

## Data and exclusions

Use the archived official750-row GraphWalks parquet, SHA256 `54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516`. Select the first16unused distinct BFS canonical edge multisets in catalogue character interval[14000,16000), preserving duplicate-edge multiplicities for graph identity. Exclude all58historical official-stage graphs,16model-bridge graphs and8completed source-qualification graphs, including unknown, failed and unreached cases. Select without model outcomes on the new graphs and recompute all16gold answers from edges. A separate auditor reconstructs selection from original frozen archives and uses queue distances for BFS rather than the runner's frontier method. Catalogue characters and actual complete prompt characters are different measures; report both. The task focus is pilot-informed: earlier short tasks/long parents showed a ceiling, and the four-stratum preflight lacked a fourth unused short BFS. Preserve the superseded unfrozen draft and HYBRID_TRACE_SCOPE_REVISION_20261004.md; do not claim this task-family decision was independent of pilot outcomes. This study does not support parents/short-task/document generalization.

## Model, runtime and resources

Use `qwen3.5:4b`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, Ollama0.35.1, actual native thinking, official fixed general-thinking profile: temperature1, top_p.95, top_k20, min_p0, presence_penalty1.5, frequency_penalty0, repeat_penalty1, mirostat0. Use context131072, output allowance32768 for all source and reader arms, input plus full output reserve plus64tokens no greater than131072. This resident context is unqualified until the logged one-output-token calibration returns. Keep its raw result and cost even if qualification fails. All requests are independent two-message system/user chats without tools, multimodal inputs or conversational history.

Use the archived official Qwen3.5 tokenizer and fixed template. Independent audit renders with Jinja, separate from the runner's narrow runtime-renderer translation. For every returned request, verify actual prompt count, native done flag, model name and resident digest/context. An unqualified result or client transport error stops the study with all remaining slots explicitly unreached, without retry. The solver HTTP timeout is7200seconds. An observation timeout is not a terminal experiment failure; wait for the original process. A prior independently audited transport interruption does not by itself forbid this separate freshly calibrated study; all prior returned responses and calibration must be qualified. Never overlap scored inference with other model calls or heavy auditing/rendering/compilation.

## Planned ledger

There are481planned slots:1calibration plus16graphs times30scored slots. Each graph has5source calls with seeds202610120–124 and5repeats of each of5reader arms with seeds202610130–134. Calibration seed202610119. The reader arm order is the cyclic rotation of `[fresh, question_first, re2, prefix, append]` by `(case_index+repeat_index)%5`. All sources for a graph precede readers. Five repetitions are not five independent graphs. Persist the complete deterministic plan, source/calibration payloads, model metadata, dependency hashes and labels before calibration. Dynamic reader payloads and bundle hashes are constructed after five sources and persisted before reader transmission. Persist each attempted payload before HTTP and the raw response before fetching resident-model metadata. No warmups, retries, favorable response replacement or additional calls outside the ledger.

## State and prompt boundaries

Take all five native thinking strings in source order, never their separate visible answers. Keep wrong and output-length-terminated source traces. Use original AppendixC warning, delimiters and trace labels. A reasoning string longer than50000characters becomes its first50000characters followed by `...`, then leading/trailing whitespace is stripped. Record clipping for each source, exact serialized bundle and SHA256. No further content selection or clipping.

The complete source user message is the released graph prompt plus the original answer-format suffix. Split immediately before its final `Operation:` to define the context head and question tail. Verify recombination exactly reconstructs the source prompt. Readers use:

| Arm | User prompt |
|---|---|
| fresh | Complete source prompt once, `[x,q]` |
| question_first | Question tail, newline, complete source prompt, `[q,x,q]` |
| re2 | Complete source prompt, newline, identical complete source prompt, `[x,q,x,q]` |
| prefix | Serialized bundle, newline, complete source prompt, `[T,x,q]` |
| append | Context head, serialized bundle, newline, question tail, `[x,T,q]` |

All readers end with the identical question tail. Prefix and append receive byte-identicalT. Fresh source-only is not Re2. System warning and user-format suffix match original AppendixC; local model-specific generation settings differ from the original frontier-model study.

## Resource declines and missingness

Compute complete rendered input tokens and output reserve before any reader call. If either trace placement exceeds the resource envelope, jointly decline both trace placements for that graph, all five repeats. Preserve graph, five source costs, bundle and intended reader jobs; each declined slot is recorded as `budget_declined`, `not_sent`, known deployment score0 and actual solver tokens0. Never drop the graph, shrinkT, reduce trace count or change context to rescue that case. An individual non-trace reader exceeding the same bound is likewise a known unsent deployment failure. These are not unknown model answers or successful model qualifications. A transmitted client failure is a known deployment score0 with unknown model response/usage; interrupted/unreached/unqualified outcomes stay unknown with score bounds[0,1]. Report every category separately.

## Scoring and source-answer controls

Primary node-setF1 and secondary EM use only the terminal visible `Final Answer: [...]` line from normal-stop responses. No thinking salvage or gold-based prediction repair. Length-limited/malformed answers are known0. Valid empty sets score normally. Unknown/wrong node strings remain false positives.

NodeMajority@5 retains nodes present in at least three strictly valid source sets, invalid sources cast no votes and denominator remains five. It returns the aggregate empty set even when all sources are invalid; score normally and disclose such groups plus always-empty baseline. This is the node-wise rule specified in original AppendixE and the previously audited local implementation. SetMode@5 selects the most frequent complete valid set, ties by earliest source; if no valid source exists, abstain with score0. Do not confuse the two controls. Unknown source outcomes make vote-control results unknown until resolved. Oracle@5 is privileged retrospective diagnosis, not a deployable method or a bound on second-pass improvements.

## Analysis frozen before first call

`hybrid_trace_analysis_spec.json` and the analysis/auditor/runner implementations are hashed into the frozen archive. The graph is the clustering unit, with all repetitions and controls retained together. Report all16planned graph means and missingness bounds. Complete-pair descriptive results include explicit rows/counts and never replace the primary denominator. Single fixed long-BFS stratum,20000graph bootstrap resamples, seed202610140, percentile95% intervals, tie tolerance1e-12. Intervals are descriptive, not simultaneous confidence intervals.

Six prefix comparisons: append, fresh, question_first, re2, set_mode, node_majority. Exact two-sided paired sign tests exclude only tied graph differences, with Holm correction over all six primaryF1 tests. These tests concern direction across graph pairs, not meanF1. EM sign results are secondary descriptive, without a confirmatory multiplicity claim. No post-hoc stratum, successful-case subset or across-stage significance selection.

The preregistered practical signal requires at least+.05meanF1, a descriptive interval above zero and adjusted directionalp<.05. Ordering evidence applies prefix versus append. An accuracy candidate additionally requires these conditions against all five non-trace controls, prefixEM no lower in point estimate than each control, at least95%normal-stop parsing, and all planned outcomes known with all returned responses qualified. The EM rule is not a formal noninferiority test. A cost candidate additionally requires known full logical token cost no greater than the cheapest qualified non-trace control. These are stringent prospective screening criteria, not guarantees of sufficient statistical power. A positive pilot requires independent new-case confirmation; no single-stage outcome establishes major progress or novel methodology.

## End-to-end cost accounting and limits

A single trace prediction pays all five source calls plus one expected reader; average the five readers to estimate that one-reader cost, never divide the source bill by five. A vote prediction pays all five sources. Fresh/QuestionFirst/Re2 pay their own single expected solver call. Record complete research token ledger including calibration, unknown usage, solver client time, qualification-metadata time, selection/source-job preparation, bundle/all-reader-job preparation and successful full research workflow wall time. Shared research preparation of25reader jobs is not a measured single-deployment preparation latency and must remain explicit. Time and tokens are different metrics; runtime caching can alter latency, and no energy or money savings follow from tokens alone. The auxiliary CPU vote computation and analysis are separate from solver tokens; measure separately if making a complete latency claim.

Maintain bounded conclusions. Improvements across model families cannot establish architecture-only or parameter-count-only causation. A null small pilot cannot refute all possible small-model trace gains. A high-cost accuracy benefit and an economical deployed benefit are different outcomes. Keep R-only empirical figures and independently verified records for eventual manuscript integration.
