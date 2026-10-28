# Prospective design review — simulated perspectives, not external peer review

Scope: preparation code and specification for the next local five-trace experiment. No outcomes from that experiment exist. This review does not certify the full manuscript or successful execution.

## Method perspective

The planned same-bundle prefix/append comparison preserves the original question tail and excludes visible source answers. Native thinking from all five fixed source repeats is retained, including wrong and length-limited traces, subject only to the original head50000-character serializer. Question First and full-prompt Re2 are required to distinguish task conditioning and repetition from trace-specific benefits. A new source-only reader is explicitly separate from Re2. The original paper's Appendix E resolves Majority@5 as a node-wise three-vote threshold; the draft and continuation label errors were corrected before freezing any new calls.

The source-performance gate must retain its verdict, but must not forbid a valid mechanism experiment: the original first-pass results also fall below that local gate. Conversely, high source performance can create a ceiling; no ordering effect in sixteen cases would be a bounded result, not proof that the method cannot work on any small model.

## Statistical perspective

Sixteen graphs are the units. All repeats and source controls stay within their graph cluster. Six prefix comparisons are declared prospectively and exact directional sign tests receive Holm correction. A sign test does not test a mean-F1 effect; mean differences and problem-cluster bootstrap intervals remain descriptive. EM's no-lower-point-estimate criterion is not a statistical noninferiority test. Both qualifications must be explicit in any report. The revised single long-BFS stratum cannot establish a type/length interaction or universal capacity boundary. Unknown outcomes require all-planned bounds; favorable complete-pair estimates alone cannot trigger success. Task focus is explicitly pilot-informed, while future case selection remains outcome-blind. The failed, unfrozen former four-stratum draft is preserved.

## Resource and reproducibility perspective

Context131072 and output32768 are prospective and uncalibrated. A logged calibration must verify actual prompt count, digest and resident context. Every intended payload and its hash are persisted before transmission, with no retries. An oversized trace bundle jointly declines prefix and append rather than selecting a favorable arm, clipping further or dropping a graph. Such declines are deployment failures, distinct from transmitted unknown answers. Complete request schedules, raw outputs, bundle clipping metadata and independent tokenizer rendering checks are required.

Single trace deployments pay for five sources and one expected reader; source costs cannot be divided by five. Programmatic voting also pays five sources. Solver client wall, metadata time, shared preparation and full research workflow time are separate quantities. Research-wide preparation of25reader jobs is not measured single-deployment preparation latency. Token reduction is not speed, energy or money reduction, particularly with runtime caching. Full calibrated one-off deployment latency remains a separate follow-up if positive quality evidence appears.

## Editorial synthesis

The design is suitable for a bounded local migration study after implementations, dependency hashes and resource checks are completed and frozen. It does not support calling the prior source-only bridge a falsification of Trace as State, nor calling the planned experiment a full256K frontier-model reproduction. A positive sixteen-case pilot still requires independent fresh-case confirmation before it is called important progress or a new method. The principal manuscript should distinguish method migration, source representation controls, engineering cost and the separate source-removal aspiration.

Remaining before launch: final protocol and analysis status; offline syntax/dependency/rendering/payload checks after current inference terminates; review the frozen selected cases and all schedule slots without inspecting model outcomes; terminal current source independent audit; preserve unsuccessful calibration rather than adapting context inside the same primary study. An audited prior transport interruption is not itself a veto on a separate freshly calibrated experiment; calibration and every returned prior model response must be qualified. All old unknown/unreached graph identities remain excluded, without replacing their original outcomes.
