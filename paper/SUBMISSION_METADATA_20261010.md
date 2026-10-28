# Submission metadata, independently confirmed extension

Author: Yueran Wang

Affiliation: School of Artificial Intelligence, Beijing Normal University

Contact: wangyueran@mail.bnu.edu.cn

Title: Improving Local Small-Model Graph Reasoning: Compact Inputs and Reused Traces

Manuscript license: arXiv.org perpetual, non-exclusive license 1.0.

No public upload performed. Account eligibility, endorsement if required, category selection and arXiv server compilation remain external steps. Local compilation does not certify acceptance.

Abstract (exact LaTeX source):

Local small models need reliable answers within modest inference budgets. We reproduce Trace as State and examine whether reasoning reuse improves graph-task accuracy after accounting for all preparation. The original five-trace studies identify context, output-budget and answer-interface boundaries without establishing a placement advantage; a larger-output direct 4B diagnostic reaches the trace accuracy ceiling with fewer tokens. We then preserve complete inputs through reversible short node labels and test a single-trace adaptation. In an independently frozen confirmation on 24 new graphs, Qwen3-1.7B F1 rises from compact-direct 0.6542 to trace-prefix 0.7703 (difference 0.1161; descriptive 95\% interval $[0.0437,0.1914]$; paired sign-flip $p=0.00624$). This accuracy benefit uses 39.8\% more total model tokens than compact direct reading. Prespecified secondary comparisons show higher accuracy and fewer preparation-inclusive tokens than original-input direct baselines, including a contemporaneous larger-output baseline, while prefix-versus-append superiority remains uncertain. A separate 8K discovery capacity criterion fails despite input fit. The findings demonstrate a bounded accuracy--token trade-off on a local 1.7B model, rather than universal cost savings or transfer to natural documents.
