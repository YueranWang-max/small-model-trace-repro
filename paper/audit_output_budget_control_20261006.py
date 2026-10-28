"""Independent fixed output-budget audit; no runner/solver/scorer imports."""
from pathlib import Path
import json,hashlib,csv,sys
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_confirmatory_20261006 import gold,rescore
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/output_budget_control_20261006'
BASE=ROOT/'logs/official_graph_paired4b_20261006'
REPORT=ROOT/'paper/evidence_output_budget_control_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(path,rows):
    if rows:
        with path.open('w',encoding='utf-8',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    fr=json.loads((OUT/'frozen.json').read_bytes());base=json.loads((BASE/'frozen.json').read_bytes())
    for frozen in [fr,base]:
        for n,h in frozen['dependencies'].items():assert sha(ROOT/n.replace(chr(92),'/'))==h
    assert fr['selected']==base['selected'] and fr['output_cap']==5*base['source_cap']+base['reader_cap']==28672
    assert fr['server_version']['version']=='0.35.1' and fr['digest']==base['identity']['digest']
    assert len(fr['jobs'])==60 and len(fr['selected'])==12
    tokenizer=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
    system='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
    suffix='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
    records=[];baseline=[];unknown=[];old_unknown=[];usage=[];unknown_usage=[];manifest=[];wall=[]
    for k,job in enumerate(fr['jobs']):
        i,j=divmod(k,5);case=fr['selected'][i];answer=gold(case);assert answer==set(case['answer_nodes'])
        assert job['graph_index']==i and job['repeat']==j and job['row_index']==case['row_index']
        assert job['seed']==base['reader_seeds'][j]==202610071+j
        assert job['key']==f'g{i:02d}_directlong_{j}' and job['old_direct_key']==f'g{i:02d}_direct_{j}'
        assert job['user']==case['prompt']+'\n'+suffix
        rendered='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+job['user']+' /think<|im_end|>\n<|im_start|>assistant\n'
        n=len(tokenizer.encode(rendered,add_special_tokens=False).ids)
        assert n==job['expected_input_tokens'] and n+28672+64<=32768
        oldpath=BASE/(job['old_direct_key']+'.json');oldraw=None;oldvalues=None
        if oldpath.exists():
            oldentry=json.loads(oldpath.read_bytes());manifest.append({'path':str(oldpath.relative_to(ROOT)),'sha256':sha(oldpath)})
            if oldentry['completion']=='returned':
                oldraw=oldentry['result']['raw'];rawpath=BASE/(job['old_direct_key']+'.response.json')
                assert json.loads(rawpath.read_bytes())==oldraw and sha(rawpath)==oldentry['result']['raw_response_sha256']
                manifest.append({'path':str(rawpath.relative_to(ROOT)),'sha256':sha(rawpath)})
                assert oldentry['request']['messages']==[{'role':'system','content':system},{'role':'user','content':job['user']}]
                assert oldentry['request']['options']['num_predict']==8192 and oldraw['prompt_eval_count']==n
                assert oldentry['result']['qualified'] and oldraw['model']==fr['model']
                assert any(m.get('digest')==fr['digest'] and m.get('context_length')==32768 for m in oldentry['result']['runtime_ps']['models'])
                oldvalues=rescore(oldraw,answer)
                baseline.append({'row_index':case['row_index'],'repeat':j,**oldvalues,'input_tokens':n,'output_tokens':oldraw['eval_count'],'done_reason':oldraw['done_reason']})
            else:old_unknown.append(job['old_direct_key'])
        else:old_unknown.append(job['old_direct_key'])
        path=OUT/(job['key']+'.json')
        if not path.exists():unknown.append(job['key']);continue
        entry=json.loads(path.read_bytes());manifest.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        request=entry['request'];assert entry['request_sha256']==hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        assert request=={'model':fr['model'],'messages':[{'role':'system','content':system},{'role':'user','content':job['user']}],'stream':False,'think':True,'keep_alive':'10m','options':{'num_ctx':32768,'num_predict':28672,'seed':job['seed'],'temperature':.6,'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}}
        assert entry['expected_input_tokens']==n
        result=entry.get('result',{});raw=result.get('raw',{});body=OUT/(job['key']+'.response.json')
        if body.exists():
            parsed=json.loads(body.read_bytes());assert not raw or parsed==raw;raw=parsed
            if result:assert sha(body)==result['raw_response_sha256']
            manifest.append({'path':str(body.relative_to(ROOT)),'sha256':sha(body)})
        errorbody=OUT/(job['key']+'.http_error.txt')
        if errorbody.exists():manifest.append({'path':str(errorbody.relative_to(ROOT)),'sha256':sha(errorbody)})
        if 'wall_sec' in result:wall.append(result['wall_sec'])
        if type(raw.get('prompt_eval_count')) is int and type(raw.get('eval_count')) is int:
            usage.append({'key':job['key'],'input_tokens':raw['prompt_eval_count'],'output_tokens':raw['eval_count']})
        elif entry.get('transmission')=='attempting_once':unknown_usage.append(job['key'])
        qualified=entry.get('completion')=='returned' and not result.get('error') and result.get('qualified') and raw.get('done') is True and raw.get('model')==fr['model'] and raw.get('prompt_eval_count')==n and type(raw.get('eval_count')) is int and 0<=raw['eval_count']<=28672 and n+raw['eval_count']<=32768
        if not qualified:unknown.append(job['key']);continue
        assert any(m.get('digest')==fr['digest'] and m.get('context_length')==32768 for m in result['runtime_ps']['models'])
        values=rescore(raw,answer);oldthinking=((oldraw or {}).get('message') or {}).get('thinking','') or ''
        longthinking=(raw.get('message') or {}).get('thinking','') or ''
        match=longthinking.startswith(oldthinking.rstrip()) if oldthinking.rstrip() else None
        records.append({'key':job['key'],'row_index':case['row_index'],'problem_type':case['problem_type'],'repeat':j,**values,'input_tokens':n,'output_tokens':raw['eval_count'],'done_reason':raw['done_reason'],'old_f1':oldvalues['f1'] if oldvalues else None,'old_em':oldvalues['em'] if oldvalues else None,'old_done_reason':(oldraw or {}).get('done_reason'),'thinking_prefix_match':match})
    complete=len(records)==60 and not unknown and len(baseline)==60 and not old_unknown
    summary={'complete':complete,'planned_new_calls':60,'qualified_new':len(records),'unknown_new':unknown,'old_baseline_known':len(baseline),'old_baseline_unknown':old_unknown,'known_new_research_tokens':sum(r['input_tokens']+r['output_tokens'] for r in usage),'unknown_attempted_usage':unknown_usage,'observed_new_request_wall_seconds':sum(wall),'f1_all_plan_bounds':[sum(r['f1'] for r in records)/60,(sum(r['f1'] for r in records)+len(unknown))/60],'em_all_plan_bounds':[sum(r['em'] for r in records)/60,(sum(r['em'] for r in records)+len(unknown))/60],'scope':'Same-question sequential diagnostic selected after truncation; not fresh confirmation or equal-compute control','all_60_planned_payload_and_tokenization_checks':'passed','maximum_generated_allowance':28672}
    graphs=[]
    if complete:
        stored=json.loads((OUT/'results.json').read_bytes());assert [r['key'] for r in stored]==[j['key'] for j in fr['jobs']]
        for r in records:
            original=next(v for v in stored if v['key']==r['key'])
            assert all(original['score'][m]==r[m] for m in ['valid','f1','em'])
        for case in fr['selected']:
            v=[r for r in records if r['row_index']==case['row_index']];assert len(v)==5
            mean=lambda key:float(np.mean([r[key] for r in v]))
            graphs.append({'row_index':case['row_index'],'problem_type':case['problem_type'],'f1':mean('f1'),'em':mean('em'),'old_f1':mean('old_f1'),'old_em':mean('old_em'),'deployment_tokens':mean('input_tokens')+mean('output_tokens')})
        differences=[np.array([g['f1']-g['old_f1'] for g in graphs if g['problem_type']==t]) for t in ['parents','bfs']]
        rng=np.random.default_rng(202610067);draws=sum(d[rng.integers(0,6,size=(20000,6))].mean(axis=1) for d in differences)/2
        summary.update(f1=float(np.mean([g['f1'] for g in graphs])),em=float(np.mean([g['em'] for g in graphs])),valid_count=sum(r['valid'] for r in records),length_terminated=sum(r['done_reason']=='length' for r in records),mean_deployment_tokens=float(np.mean([g['deployment_tokens'] for g in graphs])),long_minus_old_direct={'mean_difference':float(np.mean(np.concatenate(differences))),'lower':float(np.quantile(draws,.025)),'upper':float(np.quantile(draws,.975))},old_length_count=sum(r['old_done_reason']=='length' for r in records),old_length_thinking_prefix_match_count=sum(r['old_done_reason']=='length' and r['thinking_prefix_match'] is True for r in records))
    write_csv(REPORT/'audited_calls.csv',records);write_csv(REPORT/'old_direct_calls.csv',baseline);write_csv(REPORT/'graph_summary.csv',graphs)
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
