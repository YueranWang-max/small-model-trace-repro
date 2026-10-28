# Partial-state follow-up: verified primary literature and deployment implications

Date: 2026-10-04. Scope: literature/design evidence only. No new inference, installation, model download, outcome analysis, or change to the frozen five-trace study.

## 1. Self-Speculation for Faster Reasoning Models (SSR)

Primary paper: https://arxiv.org/html/2608.20359v1
Author implementation: https://github.com/Ravi-VK/SSR

SSR uses one model at two reasoning budgets: a child drafts an answer from partial reasoning while the parent continues, then the fuller context verifies proposed tokens. Its patched vLLM implementation combines prefix caching, speculative verification and draft-backed suffix reuse. The author repository reports Qwen3.5-4B latency reductions of 9.1% on LongProc 2K, 18.5% on ClassEval and 2.9% on HumanEval. These are author-reported task/runtime results, not our local measurements. The implementation requires scheduler and decoding support; issuing two ordinary Ollama calls is not an SSR reproduction.

Implication (our inference): partial reasoning reuse is established prior art. The useful open local question is whether a constrained partial state improves graph reasoning under full deployment accounting. Concurrent work may lower latency while increasing total compute; cache reuse does not erase logical API input tokens. SSR therefore motivates a separate runtime-efficiency axis, not a relaxation of our frozen logical-token cost gate.

## 2. Memory-Augmented Compression (MAC)

Primary paper: https://arxiv.org/html/2608.21265v1

MAC builds a memory bank from solved historical examples disjoint from evaluation and retrieves reasoning scaffolds for compressed inference. Its online overhead includes query tags, embeddings, search and prompt composition. Its per-query latency excludes offline bank construction on the stated basis that the bank is reused. Table 12 reports GSM8K accuracy 91.4% for CoT versus 89.3% for CoD+Memory, and end-to-end latency 3742.5 versus 3198.5 ms. Thus that table demonstrates a speed/accuracy trade-off, not strict accuracy preservation.

Implication (our inference): task-level procedural memory may avoid generating a fresh state for every question. A local test must declare a disjoint bank, charge its construction separately and show real reuse across new questions. Existing experimental repetitions do not establish deployment amortization. A cue-only control is essential when all cases share BFS structure.

## 3. Mid-Think

Primary paper: https://arxiv.org/html/2601.07036v1

Mid-Think combines end-of-thinking/newline cues with a reasoning opener to elicit intermediate-length reasoning without parameter updates. The paper studies token-level prompting behavior with Qwen3 and related reasoning models. Its mechanism is an assistant-prefix format, not simply a user instruction to be concise. The paper also has separate RL experiments; those gains cannot be attributed to the training-free inference setting.

Implication (our inference): a single-call intermediate reasoning control may be cheaper than paying for two full-source calls. Exact assistant-prefix support and output parsing need qualification on our Qwen3.5/Ollama interface before applying it. The current native think=true/false contrast does not test this format, and Qwen3 evidence does not guarantee Qwen3.5 transfer.

## 4. TrimR

Primary paper: https://arxiv.org/html/2505.17155v2

TrimR uses a pretrained instruction-tuned verifier to check answer existence and equivalence across intermediate thoughts, then triggers early termination. Agreement checks do not prove correctness. Its deployment system uses asynchronous verifier/model coordination; reported examples use much larger reasoning models and industrial hardware. A second verifier consumes resources and must appear in the cost ledger.

Implication (our inference): termination after stable intermediate conclusions is a distinct candidate from an arbitrary short cap. But fixed graph tasks already admit an exact CPU solver; using that solver as a verifier would turn the system into tool-assisted graph solving. Such an arm must include the solver-only baseline, otherwise it could mistake exact external computation for improved language-model reasoning.

## 5. ARSM

Primary paper: https://arxiv.org/html/2609.32852v1

ARSM represents agent interaction histories as Hypothesis-Action-Result chains and updates a structured state machine during generation. Its setting includes repeated environment interaction rather than one static graph question. Reported experiments use 7B-and-larger models and cluster hardware; the appendix notes that smaller-model structured-state failures can require recovery turns. The paper measures whole-process prompt plus completion tokens and distinguishes peak context size.

Implication (our inference): useful state should expose task progress, evidence and unresolved dependencies, not merely paraphrase every sentence. But preserving a structured summary is not a correctness guarantee. Any future graph frontier/state extractor must be tested for omissions, invalid node references and recovery costs without gold-conditioned selection.

## Practical ranking after this review

1. Complete the already frozen five-trace migration study and its independent audit first.
2. Compare one-call task-cue/concise-state controls against the proposed two-call partial-state deployment under a common full-cost budget. Treat these as prior-art-inspired controls rather than novelty.
3. Keep real cross-question procedural-memory reuse as a separate study with a disjoint bank and measured construction cost.
4. Treat cached continuation/speculation as a runtime port requiring its own interface qualification. It is not available merely because textual partial traces can be archived.
5. Keep attention-based source evidence replay separate from generated-state carryover and from exact external graph solving.

A methodological tension worth measuring later: placing a trace before the original source changes the initial token prefix and may limit ordinary prefix-cache reuse; placing it after the source can preserve a common source prefix but may affect the ordering benefit studied by Trace as State. This is a deployment hypothesis, not a universal cache theorem: actual renderer, cache key, scheduler and hybrid architecture determine reuse. Our current prefix/append study does not measure cache hits or attribute latency to caching.

No cited result establishes local improvement, and none completes the human research objective.
