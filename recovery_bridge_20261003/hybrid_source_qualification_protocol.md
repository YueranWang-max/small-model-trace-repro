# Hybrid small-model source qualification — prospective protocol

Status: prepared, no generation started and no selected-case scores observed. Running this protocol is conditional on authoritative completion and independent audits of the active bridge and interrupted indexed-source study. Freeze this document, runner, dependencies, selected inputs and full requests before the first call. Do not modify a frozen run to accommodate outcomes.

## Purpose and scope

Test whether local qwen3.5:4b can solve fresh official GraphWalks inputs under its official general-thinking sampling profile and an adequate but bounded output allowance. This is source-task qualification, not a Trace as State experiment, a causal architecture comparison or an innovation claim. The earlier Qwen3 bridge retains its own protocol, failures and unknown cell020.

## Selection and generation

- Fixed official dataset SHA54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516, same750-row release. Exclude all graph identities in ten prior official-study frozen manifests and all16bridge graphs, including interrupted/unknown attempts. Preserve duplicate edges when comparing identities.
- In each interval[2000,3500),[14000,16000), select the first two remaining parents and first two remaining BFS rows in dataset order, without inspecting answers or model outcomes. Eight distinct graphs total. Independently recompute selected gold answers and match the official labels before calls.
- These intervals use the fixed dataset's catalogue prompt_chars field, not an assumption that it equals len(prompt). Preserve both catalogue and actual complete source lengths. Runtime token qualification uses the complete rendered input. The earlier bridge found differences between the two character measures; its frozen selection remains unchanged.
- One fixed model digest2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd. Full original source plus the identical bridge terminal-format instruction and system instruction. Text-only, one leading system and one user, no tools/images/history.
- thinking=True; temperature1.0, top_p.95, top_k20, min_p0, presence_penalty1.5, frequency_penalty0, repeat_penalty1, mirostat0, num_ctx65536, num_predict32768. Two fixed source seeds202610101/202610102 per graph, consecutive seeds within each graph. No retries or outcome-based exclusions.
- These settings follow the archived fixed official Qwen3.5-4B README general-thinking profile. They differ from the Qwen3 bridge; do not call a cross-study comparison a size-only or architecture-only effect. No claim of optimal local inference configuration.

## Runtime calibration

Before scored calls, make exactly one separately labeled, durably logged input-format calibration call with a short synthetic text-only request, seed202610100 and num_predict1. Its output is not a source-task observation. Require actual prompt_eval_count to equal the independently rendered official tokenizer count and resident model digest/context to match. Its request, response, token usage and elapsed time are included in research accounting. It is not an unreported warmup. A calibration failure stops this run; preserve every attempted record and freeze a separate diagnostic before any new call. Do not repeat the calibration silently.

## Outcome and stopping rules

- Require normal termination and the visible terminal line matching Final Answer:[node1,node2]. Empty valid answer allowed. Node-set F1 and exact match; wrong nodes count as false positives. No thinking salvage, no result cleaning based on gold answers.
- Length or malformed returned answers score0. Persisted client request errors score0 for deployment performance but model answer and usage remain unknown. Interrupted attempted calls remain unknown, with missingness bounds; never silently replace them.
- Before each call persist the entire payload, ordinal, expected token count and digest. Save raw response before post-call PS checks. Stop on input or residency qualification mismatch while preserving the returned response. Stop on client error; do not continue to inflate successful-call denominators. A partial run cannot pass the gate.
- HTTP timeout3600seconds per call accommodates larger output allowance; it is a client deployment limit, not intrinsic capacity. Process runs asynchronously; do not restart solely because an observation times out.

## Gate, estimates and costs

- Eight graphs are the independent units; two seeds average within graph. All16planned source calls must have known outcomes, known usage and qualified inputs/model/context. Require at least15/16normally parsed answers, mean node-set F1>=.70 and exact match>=.50. This is an interpretable-source qualification gate, not proof of a practical trace improvement. Do not change it after observing outcomes.
- If passed, freeze fresh independent graphs for native five-trace prefix/postfix/reread and question-feedback controls. Do not select only successful source graphs. If failed, report the bounded source-task result and decide the next mechanism experiment from the complete evidence.
- Report prompt/output tokens, calibration tokens, elapsed client time and server load/prefill/decode time separately. Include all attempts. Unknown usage yields a lower bound. Token comparisons between families do not imply hardware time or energy savings.
- Record selection, label checking, token counting and metadata-preparation elapsed time before frozen-file serialization separately. Previously archived acquisition/qualification work is a prerequisite, not a measured online deployment speed comparison.
- No generation from this runner while another scientific inference process is active. No heavy audits, R rendering or paper compilation during scored inference. Immutable raw records and frozen dependency hashes remain the basis of independent post-run audit.
