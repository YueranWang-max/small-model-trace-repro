# Independent model bridge, frozen before inference

Purpose: distinguish weak source-task competence from trace placement failure. This study does not itself test Trace as State and cannot establish a size-only causal effect: model weights, capacity and training may differ.

Dataset: fixed official GraphWalks revision and SHA256 in bridge_data_manifest.json. Select first four parents and first four BFS cases in each character interval [2000,3500) and [14000,16000), excluding previously used graph identities. Historical used row indices reconstructed from session: 0–15, 100–112, 400–415, 450–462. Compare graph edge-block hashes, not only indices. Verification against inaccessible historical manifests remains pending; disclose this limitation. Selection must not use answers or model outcomes. Independently recompute graph labels before inference.

16 graphs × two seeds (202610090, 202610091) × two models (qwen3:1.7b, qwen3:8b) = 64 calls. Full official source prompt plus one identical terminal format instruction. Thinking enabled. Temperature .6, top_p .95, top_k 20, min_p 0, repeat_penalty 1, num_ctx 32768, num_predict 8192. Alternate model order per graph; two consecutive seeds per model. No warmups, retries, or outcome-based case removal. Freeze selected inputs, model digests, script SHA256 and tokenizer qualification before first call. Check actual prompt_eval_count against independently rendered tokenizer count; stop on mismatch, preserving attempted ledger.

Score only normal-stop visible terminal answers matching Final Answer: [node1, node2]. Empty list valid. Wrong nodes count as false positives. Errors, missing format and length stops score zero. Never recover answers from thinking. Compute set F1 and exact match. Primary unit is graph; average two seeds within graph. Any bootstrap interval is descriptive and not adjusted for multiple exploration stages.

Source qualification gate for 8B: at least 31/32 normally parsed answers, F1 >= .70, exact match >= .50, every call input-qualified. If passed, justify an independent, new-case native five-trace/five-reader study; do not retain only successful source cases. If failed, report capability boundary without claiming the original theory refuted.

Cost: all 64 research calls; deployment comparison uses two-seed mean per graph. Report prompt and output tokens separately. Tokens across models do not establish hardware, monetary or wall-time superiority. Include preprocessing and all attempts. Unknown usage remains unknown.

Storage: independent C: recovery study. The D: indexed-source study remains interrupted, 94 persisted responses, attempt 95 response/usage unknown, attempt 96 not reached. Do not repair its primary record by silently rerunning.
