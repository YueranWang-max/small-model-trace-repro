# Next mechanism experiment — draft, not frozen

## Research question

Can native reasoning traces improve a small model's graph reasoning under a controlled context burden, and does any gain survive counting all trace-production costs?

The completed bridge remains a source-only qualification experiment. Its answers must not be relabeled as a Trace as State test. The interrupted original cell020 remains unknown. A failed global gate cannot become a passed gate through post-hoc subgroup selection.

## Two complementary directions

1. **Within-model hybrid-attention replication.** Qwen3.5-4B official fixed configuration contains24linear-attention and8full-attention layers. This motivates testing actual native traces before and after the same source. It does not establish architecture causality or expected performance. Tokenizer and local architecture qualification are prerequisites. Compare the same model and the same trace bundle across arms; a Qwen3 comparison is observational.
2. **Separate context burden from graph difficulty.** Existing longer official graph inputs also contain larger graphs. Construct fresh synthetic graph cores and append disconnected distractor components. Keep the queried core, answer, path depth and relevant edge count fixed. Randomize core placement independently of labels and use multiple distractor loads. This tests an explicitly synthetic mechanism, separately from official-benchmark transfer.

## Required controls

- Full source baseline, full-source reread, question feedback, native trace prefix[T,x,q], native trace postfix[x,T,q]. Same terminal format, source, model settings and reader budget.
- Generate five traces from the full input with the original source question. Bundle actual native thinking in fixed source order; exclude visible source answers. Retain erroneous and length-terminated source traces. Any original-paper character clipping must be explicit and identical across relevant arms.
- Produce traces once per case and reuse the identical bundle in prefix/postfix. Predeclare reader seeds and rotate arm order. Do not select bundles or cases based on success.
- Question-feedback control estimates how much improvement can arise from repeating the task itself. Reread controls estimate improvement from extra sampling without traces.
- Keep source-free or privileged-core traces outside primary replication. They can be labeled separate diagnostic arms only if predeclared.

## Units and costs

- For context-burden manipulation, the independent unit is the graph core, not each distractor length, reader seed or derived question. Bootstrap whole cores with all variants together.
- Strict score uses normal termination and the visible terminal answer; no thinking salvage. Wrong nodes remain false positives. Client errors and interrupted unknown outcomes remain distinct.
- Count all five source calls plus reader costs for one deployment. Reader averaging estimates expected single-reader cost; it does not amortize trace production across the five sources. A separately justified repeated-question deployment may report amortization with the number of questions explicit.
- Persist every attempted request before transmission; save raw response and timing before any qualification step. Missing usage yields a cost lower bound, not a complete cost estimate.
- Tokens are an input/output accounting measure. Wall time, model loading and input processing must be reported separately; no energy claim without measurement.

## Decision sequence

1. Finish and independently audit active bridge; independently audit interrupted indexed-source study and historical graph exclusions.
2. Qualify local hybrid candidate metadata, tokenizer and rendered input lengths with a separate metadata archive. Freeze any generation smoke-check as additional calls; include them in the research ledger, outside scored primary data.
3. Freeze a fresh source qualification study using official cases unused by prior studies and a stated deployment budget. Determine sample size, settings and gate before calls. Do not choose a subgroup from the present bridge based on favorable results.
4. If source qualification supports an interpretable trace experiment, freeze fresh cases and all controls. If not, report the source-ability limitation and use the synthetic controlled-context study as a bounded mechanism test, not a benchmark success.
5. Advance a practical improvement claim only with independent cases, adequate uncertainty, reliable output and complete cost accounting. A quality gain that uses many more tokens is a quality/cost tradeoff, not a cost-saving breakthrough.

This document is an exploratory design draft. It records rationale before the next study; it is not a preregistration or a change to any active frozen protocol.
