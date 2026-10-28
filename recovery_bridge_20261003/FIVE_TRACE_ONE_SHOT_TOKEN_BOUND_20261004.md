# A structural token-cost boundary for the active five-trace design

This is an analytic result from the frozen planned requests, not an interim outcome analysis. It does not change any frozen scoring, gate, case, model setting or stopping rule. The active inference process continues unchanged.

## Statement

For one graph, let I be the qualified input-token count of the full-source system/user request, and C the common output allowance. A fresh full-source prediction costs at most I+C logical API tokens. A five-source trace prediction pays for all five source calls, so its source inputs alone cost at least5I. All source outputs and the eventual reader add nonnegative costs. Therefore, whenever

    5I > I+C, equivalently I > C/4,

the five-trace deployment cannot consume fewer total logical tokens than the one-call full-source baseline under the same cap. This remains true if the trace reader is declined for exceeding the input envelope: the five source costs have already been incurred. It also holds for source-answer voting that pays the five sources.

## Instantiation, independently readable from the frozen schedule

The active study has C=32768, so the threshold is I>8192. All16new long-BFS cases exceed it. The fresh reader has the same full-source system/user content and native-thinking rendering as the source, with a different seed that does not change input tokens. The five source jobs per case have identical expected input counts. These are independently render-qualified counts; actual qualified responses must match them. The output-budget bound is likewise checked against returned usage in the independent audit.

| Across the16planned graphs | Mean logical tokens per graph |
|---|---:|
| Five source inputs alone: lower bound |56303.4375|
| One fresh full-source call including its maximum output: upper bound |44028.6875|
| Gap before any source outputs or trace reader |12274.75|

For example, row506 has I=11246: its five source inputs cost at least56230, whereas one full-source call costs at most44014. No correct answer, shorter second-pass reasoning, trace compression after generation or stronger voting can remove that already paid input bill. All numbers come from `hybrid_five_trace_single_prediction_cost_bound_20261004.json`, which records all16case inequalities and the frozen archive SHA256 `c62070ad840162ed4a0a8391c6a189fe3696ff79d9c5f4faabbbb5a26e0ab8ab`.

## Scope and conditions

This is a bound for **one prediction paying five full-source calls**, the current output cap, qualified counts, and total logical API tokens. It is not a theorem about all trace methods, accuracy, expected utility, latency, cache misses, energy or monetary cost. Prefix cache reuse can reduce actual computation or billed cached-token rates while the logical input token counts remain unchanged. Unknown responses or violated input/output qualification cannot support a complete-cost success claim; keep their existing missingness policy.

The frozen cost-candidate criterion requires complete trace cost no greater than the cheapest qualified non-trace comparator. The fresh comparator is already cheaper than this source-input lower bound, so the cost criterion cannot pass under complete qualified execution of the current design. Preserve that verdict rather than loosening the criterion. Accuracy and ordering evidence remain substantive questions, and this full experiment is still a bounded migration of the original five-trace method.

## Consequences for the next research step

More prompt tuning within the same five-full-source/single-prediction architecture cannot yield one-shot total-token savings here. To examine efficiency, a separately frozen future study must change the cost-generating structure: fewer or cheaper source passes, attention/readout-selected original evidence without long trace generation, or one prepared state reused across genuinely different user questions. Repeated samples of the same question are not a valid substitute for multiple deployed questions when amortizing source costs. Any change must preserve the original small-model migration goal and be tested on fresh cases with full preparation/readout costs and strong baselines; this bound alone establishes no such improvement.
