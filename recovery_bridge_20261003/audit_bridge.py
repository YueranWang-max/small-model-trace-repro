"""Run only after inference has finished; preserve missing and unknown attempts."""
import pathlib,json,re,hashlib,statistics,collections,csv
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'bridge_run'
frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'run_bridge.py').read_bytes()).hexdigest()==frozen['runner_sha256']
assert hashlib.sha256((ROOT/'bridge_protocol.md').read_bytes()).hexdigest()==frozen['protocol_sha256']
graph_blocks=[c['prompt'].split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0] for c in frozen['cases']]
assert len(graph_blocks)==16 and len(set(graph_blocks))==16,'Graph-cluster independence failed; do not report 16 independent graphs.'
records=[]
for ordinal,job in enumerate(frozen['schedule'],1):
    p=OUT/f'attempt_{ordinal:03d}.json'
    if not p.exists():records.append(dict(ordinal=ordinal,model=job['model'],row_index=job['row_index'],status='not_reached',usage_known=False));continue
    rec=json.loads(p.read_text(encoding='utf-8'));assert rec['job']==job
    if rec['status'] in ['not_yet_attempted','attempting_once']:
        records.append(dict(ordinal=ordinal,model=job['model'],row_index=job['row_index'],status=rec['status'],usage_known=False));continue
    raw=rec.get('response',{});content=raw.get('message',{}).get('content','').strip();last=content.splitlines()[-1] if content else ''
    match=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',last.strip())
    valid=bool(match and raw.get('done_reason')=='stop' and not raw.get('error'))
    pred={x.strip() for x in match.group(1).split(',') if x.strip()} if valid else set();gold=set(job['gold'])
    f1=(2*len(pred&gold)/(len(pred)+len(gold)) if pred or gold else 1) if valid else 0
    known=isinstance(raw.get('prompt_eval_count'),int) and isinstance(raw.get('eval_count'),int)
    records.append(dict(ordinal=ordinal,model=job['model'],row_index=job['row_index'],status=rec['status'],valid=valid,f1=f1,em=int(valid and pred==gold),prediction=sorted(pred),usage_known=known,prompt_tokens=raw.get('prompt_eval_count'),output_tokens=raw.get('eval_count'),input_qualified=rec.get('input_qualified',False),resident_qualified=rec.get('resident_qualified',False)))
summary={}
for model in frozen['digests']:
    group=[r for r in records if r['model']==model];scored=[r for r in group if 'f1' in r]
    summary[model]=dict(planned=len(group),persisted=len(scored),normally_parsed=sum(r['valid'] for r in scored),mean_f1=statistics.mean(r['f1'] for r in scored) if scored else None,mean_em=statistics.mean(r['em'] for r in scored) if scored else None,known_tokens_lower_bound=sum(r['prompt_tokens']+r['output_tokens'] for r in scored if r['usage_known']),unknown_usage=sum(not r['usage_known'] for r in group),all_input_qualified=all(r.get('input_qualified') and r.get('resident_qualified') for r in group))
    s=summary[model];s['source_qualification_gate']=s['persisted']==32 and s['normally_parsed']>=31 and s['mean_f1']>=.70 and s['mean_em']>=.50 and s['all_input_qualified']
result=dict(summary=summary,records=records,complete=(OUT/'completion.json').exists(),historical_manifest_verification=frozen['historical_manifest_verification'],main_manuscript_updated=False)
(ROOT/'bridge_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
with (ROOT/'bridge_plot_data.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['model','row_index','f1','em','normal_fraction','mean_tokens'])
    for model in summary:
        for index in sorted({r['row_index'] for r in records}):
            rs=[r for r in records if r['model']==model and r['row_index']==index]
            if len(rs)!=2 or any('f1' not in r for r in rs):continue
            writer.writerow([model,index,statistics.mean(r['f1'] for r in rs),statistics.mean(r['em'] for r in rs),statistics.mean(r['valid'] for r in rs),statistics.mean(r['prompt_tokens']+r['output_tokens'] for r in rs) if all(r['usage_known'] for r in rs) else ''])
