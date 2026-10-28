"""Report only complete independently audited confirmation data."""
from pathlib import Path
import json,csv
p=Path(__file__).resolve().parent;e=p/'evidence_trace_benefit_confirmation_20261010'
s=json.loads((e/'summary.json').read_bytes());assert s['complete']
lines=['# Independent confirmation of single-trace reuse','','## Scope and completion','',f"All {s['known_model_calls']}/{s['planned_model_calls']} calls were independently audited on 24 fresh graph units, with two reader seeds averaged within each graph. Discovery cases are excluded. Runtime: Ollama0.40.0; model qwen3:1.7b with frozen digest. No pooling with earlier runs.",'','## Prespecified primary result','',f"Primary accuracy-benefit criterion met: {s['primary_accuracy_benefit_supported']}. This refers only to prefix versus complete reversible compact-direct input on the selected graph benchmark.",'','| Arm | F1 | Exact match | Full tokens/answer | Full wall seconds/answer | Valid/48 | Length terminated |','|---|---:|---:|---:|---:|---:|---:|']
for r in s['arms']:
 lines.append(f"| {r['arm']} | {r['f1']:.4f} | {r['em']:.4f} | {r['mean_deployment_tokens']:.2f} | {r['mean_deployment_wall_seconds']:.2f} | {r['valid_count']}/48 | {r['length_terminated']} |")
lines+=['','Prefix and append include the entire one-source preparation pass per answer, without amortizing it over the two repeats. Original-long allows 16384 generated tokens; original and compact allow8192; each trace package allows1024 source plus7168 reader tokens. Input lengths and actual compute are not matched. CPU recoding time and dictionary bytes are archived separately. Wall time is an observation on this computer, not an energy or monetary claim.','','## Graph-level contrasts','','| Contrast | F1 difference [descriptive95% interval] | Two-sided sign-flip p | Full token ratio [95% interval] |','|---|---|---:|---|']
for c in s['contrasts']:
 lo,hi=c['descriptive_interval'];tl,th=c['full_token_ratio_interval']
 lines.append(f"| {c['contrast']} | {c['f1_difference']:.4f} [{lo:.4f}, {hi:.4f}] | {c['two_sided_paired_sign_flip_p']:.5f} | {c['full_token_ratio']:.3f} [{tl:.3f}, {th:.3f}] |")
lines+=['',s['inference_limits'],'','The primary rule requires a positive mean F1 difference, a positive lower descriptive bootstrap limit and two-sided paired sign-flip p<=.05. The sign-flip test assumes exchangeability of paired differences under its null. All other contrasts are secondary and unadjusted for multiplicity. A confidence interval crossing zero does not prove equivalence. There is no outcome-driven extension of the frozen sample.','','## Boundaries and evidence','','The previous discovery study failed its nonempty exact-in-both-seeds8K capacity criterion; this confirmation does not reverse that result. No natural-document transfer, universal reasoning benefit, third-party execution attestation, or lower monetary cost has been established. Conventional CPU graph algorithms directly solve these tasks: the benchmark diagnoses language-model behavior rather than proving production superiority over algorithms.','','Write-ahead requests and exact response bytes: logs/trace_benefit_confirmation_20261010. Independent payload, source-round-trip and rescore evidence: paper/evidence_trace_benefit_confirmation_20261010. Incorrect and truncated source traces are retained. Every completed reader and failure is included.']
(p/'TRACE_BENEFIT_CONFIRMATION_RESULTS_20261010.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (e/'plot_arms.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['arm','f1','em','mean_deployment_tokens','mean_deployment_wall_seconds'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in s['arms'])

with (e/'plot_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['contrast','primary','difference','lower','upper'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for c in s['contrasts']:
  w.writerow({'contrast':c['contrast'],'primary':c['primary'],'difference':c['f1_difference'],'lower':c['descriptive_interval'][0],'upper':c['descriptive_interval'][1]})

rs=json.loads((e/'rescored_calls.json').read_bytes());src=[r for r in rs if r['arm']=='source']
report=p/'TRACE_BENEFIT_CONFIRMATION_RESULTS_20261010.md'
with report.open('a',encoding='utf-8') as f:
 f.write(f"\n## Complete source-pass outcomes\n\nAll {len(src)} sources retained: {sum(r['termination']=='length' for r in src)} length-terminated, {sum(r['valid'] for r in src)} normal-stop valid visible answers and {int(sum(r['em'] for r in src))} qualified exact visible answers. Separate source visible answers are excluded from reused traces.\n")
