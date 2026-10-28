# Evidence replay: implementation follow-up, not a new local result

Primary sources checked on2026-10-04: [ReContext v1](https://arxiv.org/html/2607.02509v1), [authors' repository](https://github.com/Yanjun-Zhao/ReContext), [Qwen3 attention implementation](https://github.com/Yanjun-Zhao/ReContext/blob/master/recontext/custom_modeling_qwen3.py), and [Markovian Thinker authors' publication page](https://www.microsoft.com/en-us/research/publication/the-markovian-thinker/). No code installed, model downloaded or additional inference launched during the active source study.

## Useful distinction

ReContext proposes an evidence-first alternative to generating long reasoning: query-conditioned internal attention selects positions, the corresponding original sentences are copied, then replayed with full source retained. This is directly relevant to the human's attention, pointers and retained-source proposals. It is existing related work, not our invention. Its default evidence placement is near the question after the original context; it does not by itself confirm Trace as State's prefix mechanism.

## Implementation details to preserve

Appendix B specifies an eight-token query-side cue window, selected heads, decay0.75 and two replay rounds. It materializes selected positions as deduplicated original spans, accumulates them, and reuses a context-end KV snapshot. Candidate positions are restricted to original context in its main setting. The copied evidence may influence subsequent selection without being selectable as fresh source evidence. Reported efficiency still incurs extra time: the cited CLIPPER/Llama3 example is62minutes versus44minutes for direct decoding; no blanket savings claim follows.

The public Qwen3 attention class routes requests for attention weights through eager attention. A faithful local port must check which positions/layers are materialized, cache handling and memory, rather than enabling full all-position attention tensors at long lengths. The current Ollama study archives thinking/content and usage, not internal attention scores. Calling lexical overlap or a model-written importance list “attention” would be inaccurate. The examined repository folder supplies Qwen3/Qwen3-MoE/Llama classes; a Qwen3.5 hybrid port has not been qualified.

## Next bounded test after migration evidence

If native trace reuse yields a quality signal but high source cost, compare original-span evidence with native traces. First qualify a backend that really exposes the chosen relevance signals on a small input, including tokenizer offsets and round-trip copied spans. Freeze separate fresh cases and a matched evidence block for `[E,x,q]` versus `[x,E,q]`; add full-source, Question First, equal-size random original spans and cheap query-overlap retrieval controls. Keep original source in both evidence placements. Include readout, span recovery, each replay round and final generation costs. Do not reuse the pending source or trace study as confirmation data. A pointer/index alone does not save model input unless the consumer can resolve it; dereferenced evidence still consumes tokens.

Markovian Thinker supplies a different path: reset long reasoning into chunks with a short carryover state. Its strongest reported1.5B result trains that carryover policy using RL. Zero-shot compact-state requests could be a separate pilot, but must not be described as reproducing the trained result or as preserving information by guarantee. Neither related-work theory warrants discarding unselected source safely in our current model.

This follow-up does not change the active source protocol or the prospective five-trace comparison. It is a candidate for cost-oriented follow-up once the primary small-model migration question has direct evidence.
