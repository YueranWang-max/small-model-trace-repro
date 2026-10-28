"""Consolidate actual requests, independent seed count and artifact provenance."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
studies=['lookup_semantics','lookup_confirmation','lookup_chain','lookup_chain_replication','model_checkpoint_order','checkpoint_representation','checkpoint_binding_confirmation']
seeds=set();records=[];warmups=[];per=[]
for name in studies:
    folder=ROOT/'logs'/name;f=json.loads((folder/'frozen.json').read_text());d=json.loads((folder/'reader.json').read_text());count=0
    seeds.update(t['seed'] for t in f['tasks'])
    for row in d['rows']:
        if 'producer' in row:values=row['producer']+list(row['readers'].values())
        elif 'calls' in row:values=row['calls']
        else:values=[row]
        records.extend(values);count+=len(values)
    for value in d['warmups'].values():warmups.append(value.get('raw',value))
    per.append(dict(study=name,formal_calls=count,warmups=len(d['warmups'])))
assert len(records)==1304 and len(warmups)==14 and len(seeds)==84
assert all(v.get('raw',{}).get('schema_valid') for v in records)
formal_http=sum(v['raw']['client_attempts'] for v in records)
assets=['paper/LOCAL_OPERATOR_STATE_DISCOVERY_20261003.md','paper/CHECKPOINT_BINDING_DISCOVERY_20261003.md','paper/LOCAL_OPERATOR_STATE_REVIEW_20261003.md','paper/figures/plot_lookup_progress.R','paper/figures/plot_checkpoint_binding.R','paper/figures/data/lookup_progress.csv','paper/figures/data/checkpoint_binding.csv','paper/figures/lookup_single_confirmation.pdf','paper/figures/lookup_single_confirmation.png','paper/figures/lookup_chain_replication.pdf','paper/figures/lookup_chain_replication.png','paper/figures/checkpoint_binding.pdf','paper/figures/checkpoint_binding.png']
output=dict(date='2026-10-03',formal_calls=len(records),formal_http_attempts=formal_http,warmup_calls=len(warmups),warmup_http_attempts=sum(v.get('client_attempts',0) for v in warmups),distinct_seed_tasks=len(seeds),seed_list=sorted(seeds),studies=per,artifacts_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in assets},not_complete_original_benchmark_reproduction=True,not_general_document_innovation=True,not_end_to_end_cost_success=True)
(ROOT/'logs/research_stage_audit_20261003.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in output.items() if k not in ['artifacts_sha256','seed_list','studies']},indent=2))
