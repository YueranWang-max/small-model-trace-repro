# Teacher feedback and writing-experience checklist

## One-sentence message

我们在本地小模型上检验了唐杰老师的推理前置方法，发现其收益尚未在这些配置中确立，而输出额度、上下文容量和评分规则是必须控制的复现边界。

## Traceable disposition

| Feedback | Final implementation | Evidence / limit |
|---|---|---|
| Reproduce on small models and find limits; innovation optional | Title, introduction and conclusion explicitly position a local reproduction | Two complete matched studies; no claimed architectural breakthrough |
| One sentence should tell readers what they learn | Abstract and introduction open with the practical placement question | Final readability_changes.json; main.tex abstract and first four paragraphs |
| Use R figures, not image generation | Nine empirical figures are R-rendered exports | all_embedded_figures_are_r_exports.json and three portable R release verification files |
| Review, revise, repeat until major issues addressed | Round 1 and final round 2; exactly three simulated reports each with synthesis | Round 2 specifies concerns, repairs and remaining limitations; not external referees |
| Confirm experimental truth | Independently recomputed labels/scores, request and raw-byte checks for new studies | Four complete audits; weak historical evidence excluded; hashes not proof of external execution |
| Follow Tang paper's writing logic | Question, original intervention, matched comparison, diagnostics, bounded interpretation | Same-trace prefix/append retained; no copying of its theoretical contribution |
| Introduction as funnel | Practical question -> frontier result -> local uncertainty -> what we did/found | Four focused paragraphs; no broad historical survey |
| Abstract with concrete results | Twelve questions, 240 calls/model, F1, truncation, new baseline and 5.91 token ratio | Each current mapped number checked against complete audit summaries |
| Methods like a recipe | Data selection, model digests, caps, seeds, serialization, scorer, failures, analysis | Exact versions and detailed records in evidence package; no unlogged procedures invented |
| Results focused on findings | Lead with completed studies and then diagnostic controls | Tables/figures retain detail; older exploratory studies mostly appendices |
| Discussion compare, explain, limit | Relation to original method, output/context/scoring interpretation, prospective requirements | No demonstrated trace-use mechanism; alternative explanations kept explicit |
| Short conclusion | Five short sentences summarizing question, outcome, diagnostics and scope | No new data or citations introduced |
| Learn structure without copying wording | Original question-driven text; acknowledged original intervention | Supplied experience article treated as heuristics, not universal editorial requirements |

## Remaining limits

Twelve restricted graph units, known-question sequential budget diagnostic, selected single context case, fixed source bundles, possible public-benchmark overlap unassessed, no third-party inference replication. Five reader repeats and 566 recorded calls do not remove these limits. Minimum figure text around 5.25 points and extensive exploratory appendices remain presentation trade-offs. The internal review supports a bounded preprint, not a promise of acceptance.
