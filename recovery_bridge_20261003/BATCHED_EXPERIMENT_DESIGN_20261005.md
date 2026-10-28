# Independent experiment with safe batch boundaries — design draft

Status: prospective draft, not frozen, not executed. It must not be merged with or described as a continuation of the interrupted confirmatory study.

## Scientific decision before implementation

The previous 1024-token early-state configuration cannot meet its original normal-valid reliability gate even if every missing result were favorable. Do not spend another full run repeating it solely to fill its table. First diagnose output truncation and answer-format failures separately from incorrect normally completed answers. Any configuration selected from those results is development, requiring independent graphs for confirmation.

## Execution unit

Use one graph with all prespecified arms and both repetitions as a batch. Freeze the full case list, prompt construction, seeds, randomized within-repeat reader order, budgets, parser, analysis and failure handling before new responses. Graphs are independent units; repetitions are not independent observations.

At a successful batch boundary, write and atomically commit a manifest containing protocol and dependency hashes, exact request hashes, result hashes, usage, measured preparation time, runtime/model identity and completion status. Then exit normally. A later invocation verifies every committed hash and executes only the next untouched batch. No in-flight request is resumed or replayed.

An ambiguous or unqualified call terminates its batch. Mark remaining slots unknown, preserve all actual usage, and continue only according to the failure policy fixed before execution. Preserve the failed graph in the all-plan denominator. A planned pause differs from an unplanned failure; record both explicitly.

## Temperature, rest and timing

Use manual scheduled rest between completed batches; no invented GPU temperature safety threshold. Record observed device temperature where available as diagnostic metadata. Each restart performs a separately paid unscored qualification call, records cold/warm cache and model residency, and qualifies model digest, context and token accounting. Calibration costs belong to research overhead, not hidden deployment savings.

All arms for a graph should run within the same batch. Randomize reader order, retain full source costs for every state deployment, and do not compare an experimental warm reader against a cold direct baseline as evidence of latency savings. Logical token accuracy/cost is the primary target; causal latency requires a separate cache-controlled design.

## Go/no-go before starting

1. Diagnose saved length/malformed/normal-wrong outcomes without changing their original scores.
2. Verify enough unused canonical graphs exist; do not assume remaining catalogue rows constitute enough fresh cases. Exclude all previous selected identities, including unexecuted ones.
3. Select a bounded development configuration and fixed output/context budgets on cost grounds, not gold-guided trial selection.
4. Complete offline batch persistence, hash verification, failure and resume checks before paid inference.
5. Freeze a new independent confirmation protocol. Runtime pauses must never become opportunities to alter prompts or choose favorable cases.

No claim of a methodological breakthrough, an efficiency improvement, or lossless source compression follows from this execution design.
