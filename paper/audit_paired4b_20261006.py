"""Independent post-run audit. No runner, oracle or scorer imports."""
from pathlib import Path
from collections import deque
import csv, hashlib, json, re, random
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/official_graph_paired4b_20261006'
REPORT=ROOT/'paper/evidence_paired4b_20261006'
digest=lambda b:hashlib.sha256(b).hexdigest()

def gold(row):
    graph,operation=row['prompt'].rsplit('Operation:',1)
    edges=re.findall(r'^([A-Za-z0-9]+) -> ([A-Za-z0-9]+)\s*$',graph,re.M)
    if row['problem_type']=='parents':
        node=re.search(r'Find the parents of node (\w+)\.',operation)[1]
        return {u for u,v in edges if v==node and u!=node}
    node=re.search(r'BFS from node (\w+)',operation)[1]
    depth=int(re.search(r'exactly depth (\d+)',operation)[1])
    adj={}
    for u,v in edges:adj.setdefault(u,[]).append(v)
    distances={node:0};queue=deque([node])
    while queue:
        u=queue.popleft()
        if distances[u]>=depth:continue
        for v in adj.get(u,[]):
            if v not in distances:distances[v]=distances[u]+1;queue.append(v)
    return {u for u,d in distances.items() if d==depth}

def rescore(raw,answer):
    content=(raw.get('message') or {}).get('content','') or ''
    lines=content.rstrip().split('\n')
    match=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',lines[-1].strip())
    valid=bool(match) and raw.get('done_reason')=='stop'
    predicted={s.strip() for s in match[1].split(',') if s.strip()} if match else set()
    f1=2*len(predicted&answer)/(len(predicted)+len(answer)) if predicted or answer else 1.
    return {'valid':valid,'em':float(valid and predicted==answer),'f1':f1 if valid else 0.}

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    for name,h in fr['dependencies'].items():assert digest((ROOT/name.replace(chr(92),"/")).read_bytes())==h,name
    data=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet'
    assert digest(data.read_bytes())==fr['data_sha256']
    dataset=pq.read_table(data).to_pylist()
    tokenizer=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
    system='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
    warning='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
    used=set()
    for name,h in fr['exclusion_archives'].items():
        path=ROOT/name.replace(chr(92),"/");assert digest(path.read_bytes())==h
        for row in json.loads(path.read_text(encoding='utf-8-sig')).get('selected',[]):
            if isinstance(row,dict) and 'prompt' in row:
                used.add(digest(row['prompt'].rsplit('Operation:',1)[0].encode()))
    rng_selection=random.Random(202610060);reselected=[]
    for typ in ['parents','bfs']:
        seen=set(used);eligible=[]
        for i,row in enumerate(dataset):
            h=digest(row['prompt'].rsplit('Operation:',1)[0].encode())
            if row['problem_type']==typ and 2000<=row['prompt_chars']<3500 and h not in seen:
                eligible.append({'row_index':i,'graph_head_sha256':h,**row});seen.add(h)
        assert eligible==fr['candidates'][typ]
        chosen=rng_selection.sample(eligible,6);reselected.extend(chosen)
        used.update(row['graph_head_sha256'] for row in chosen)
    rng_selection.shuffle(reselected);assert reselected==fr['selected']
    expected_orders=[]
    for row in reselected:
        block=[]
        for j in range(5):
            order=['pre','post','direct'];rng_selection.shuffle(order);block.append(order)
        expected_orders.append(block)
    assert expected_orders==fr['orders']
    manifest=[];records=[];graphs=[]
    complete=True
    for i,row in enumerate(fr['selected']):
        original=dataset[row['row_index']]
        assert original['prompt']==row['prompt'] and original['answer_nodes']==row['answer_nodes']
        answer=gold(row);assert answer==set(row['answer_nodes'])
        source=[];by_arm={'pre':[],'post':[],'direct':[]}
        x=row['prompt']+'\nReturn exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
        # All reader payloads are checked against their saved source thinking;
        # the stored request itself is preserved rather than inferred as evidence.
        source_thinking=[]
        for arm in ['source','pre','post','direct']:
            for j in range(5):
                key=f'g{i:02d}_{arm}_{j}';path=OUT/(key+'.json')
                if not path.exists():complete=False;continue
                entry=json.loads(path.read_text(encoding='utf-8'))
                manifest.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path.read_bytes())})
                assert entry['request_sha256']==digest(json.dumps(entry['request'],sort_keys=True).encode())
                req=entry['request'];assert req['model']=='qwen3:4b' and req['think'] is True
                assert req['options']['num_ctx']==32768
                assert req['options']['num_predict']==(4096 if arm=='source' else 8192)
                assert req['options']['seed']==fr['source_seeds' if arm=='source' else 'reader_seeds'][j]
                assert req['options']=={'num_ctx':32768,'num_predict':4096 if arm=='source' else 8192,'seed':fr['source_seeds' if arm=='source' else 'reader_seeds'][j],'temperature':.6,'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}
                assert req['stream'] is False and req['keep_alive']=='10m'
                assert req['messages'][0]=={'role':'system','content':system} and req['messages'][1]['role']=='user' and len(req['messages'])==2
                # Independently rendered count, not merely the runner's declared count.
                rendered='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+req['messages'][1]['content']+' /think<|im_end|>\n<|im_start|>assistant\n'
                independent_input=len(tokenizer.encode(rendered,add_special_tokens=False).ids)
                assert independent_input==entry['expected_input_tokens']
                assert independent_input+req['options']['num_predict']+64<=32768
                if entry['completion']!='returned':complete=False;continue
                result=entry['result'];assert not result['error'],key
                rawpath=OUT/(key+'.response.json');body=rawpath.read_bytes();raw=json.loads(body)
                assert raw==result['raw'] and digest(body)==result['raw_response_sha256']
                manifest.append({'path':str(rawpath.relative_to(ROOT)),'sha256':digest(body)})
                assert raw.get('model')=='qwen3:4b' and raw.get('prompt_eval_count')==entry['expected_input_tokens']
                assert any(m.get('digest')==fr['identity']['digest'] and m.get('context_length')==32768
                           for m in result['runtime_ps'].get('models',[])),key
                count=raw.get('prompt_eval_count');outcount=raw.get('eval_count')
                assert isinstance(count,int) and isinstance(outcount,int)
                assert raw.get('done') is True and 0<=outcount<=req['options']['num_predict'] and count+outcount<=32768
                value=rescore(raw,answer)
                rec={'key':key,'row_index':row['row_index'],'problem_type':row['problem_type'],
                     'arm':arm,'repeat':j,**value,'input_tokens':count,'output_tokens':outcount,
                     'wall_sec':result['wall_sec']}
                records.append(rec)
                if arm=='source':
                    assert req['messages'][1]['content']==x
                    source.append(rec);source_thinking.append((raw.get('message') or {}).get('thinking','') or '')
                else:
                    if arm=='direct':assert req['messages'][1]['content']==x
                    by_arm[arm].append(rec)
                    if len(source_thinking)==5 and arm!='direct':
                        saved=req['messages'][1]['content']
                        components=[warning,'<first_run_reasoning_traces>']
                        for k,t in enumerate(source_thinking,1):
                            clipped=t[:50000]+'...' if len(t)>50000 else t
                            components.append(f'[Trace {k}] '+clipped.strip())
                        trace='\n'.join(components+['</first_run_reasoning_traces>'])
                        head,operation=x.rsplit('Operation:',1)
                        expected=trace+'\n'+x if arm=='pre' else head+trace+'\nOperation:'+operation
                        assert saved==expected,key
                        for k,s in enumerate(source_thinking,1):
                            clipped=s[:50000]+'...' if len(s)>50000 else s
                            assert f'[Trace {k}] '+clipped.strip() in saved
        if len(source)==5 and all(len(v)==5 for v in by_arm.values()):
            for arm,values in by_arm.items():
                graphs.append({'row_index':row['row_index'],'problem_type':row['problem_type'],'arm':arm,
                    'f1':float(np.mean([r['f1'] for r in values])),
                    'em':float(np.mean([r['em'] for r in values])),
                    'valid_count':sum(r['valid'] for r in values),
                    'source_tokens':sum(r['input_tokens']+r['output_tokens'] for r in source),
                    'mean_reader_tokens':float(np.mean([r['input_tokens']+r['output_tokens'] for r in values])),
                    'deployment_tokens':(sum(r['input_tokens']+r['output_tokens'] for r in source) if arm!='direct' else 0)+float(np.mean([r['input_tokens']+r['output_tokens'] for r in values]))})
    summary={'complete':complete and len(records)==240,'qualified_returned_records':len(records),
             'planned':240,'graph_units_complete':len(graphs)//3,
             'scope':'Consistency of independently rescored local raw responses, not external attestation',
             'research_tokens_known':sum(r['input_tokens']+r['output_tokens'] for r in records),
             'research_observed_wall_seconds':sum(r['wall_sec'] for r in records),
             'frozen_sha256':digest((OUT/'frozen.json').read_bytes()),
             'full_payload_sampling_and_token_count_checks':'passed for every returned record; no source/scorer imports'}
    summary.update(empty_gold_graphs=sum(not row['answer_nodes'] for row in fr['selected']),
                   always_empty_baseline= sum(not row['answer_nodes'] for row in fr['selected'])/12,
                   source_valid_count=sum(r['valid'] for r in records if r['arm']=='source'))
    if summary['complete']:
        stored=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
        assert len(stored)==240 and len({r['key'] for r in stored})==240
        expected_sequence=[]
        for i in range(12):
            expected_sequence.extend(f'g{i:02d}_source_{j}' for j in range(5))
            for j in range(5):expected_sequence.extend(f'g{i:02d}_{arm}_{j}' for arm in fr['orders'][i][j])
        assert [r['key'] for r in stored]==expected_sequence
        for rec in records:
            original=next(r for r in stored if r['key']==rec['key'])
            assert all(original[k]==rec[k] for k in ['row_index','arm','repeat'])
            for metric in ['valid','f1','em']:assert original['score'][metric]==rec[metric]
        arms=[]
        for arm in ['pre','post','direct']:
            v=[g for g in graphs if g['arm']==arm]
            arms.append({'arm':arm,'f1':float(np.mean([g['f1'] for g in v])),
                         'em':float(np.mean([g['em'] for g in v])),
                         'valid_count':sum(g['valid_count'] for g in v),
                         'mean_deployment_tokens':float(np.mean([g['deployment_tokens'] for g in v]))})
        rng=np.random.default_rng(202610066);contrasts=[]
        for comparator in ['post','direct']:
            differences=[];strata={}
            for typ in ['parents','bfs']:
                ids=[r['row_index'] for r in fr['selected'] if r['problem_type']==typ]
                dif=[]
                for idx in ids:
                    a=next(g['f1'] for g in graphs if g['row_index']==idx and g['arm']=='pre')
                    b=next(g['f1'] for g in graphs if g['row_index']==idx and g['arm']==comparator)
                    dif.append(a-b)
                differences.append(np.array(dif));strata[typ]=float(np.mean(dif))
            draws=sum(np.mean(d[rng.integers(0,len(d),size=(20000,len(d)))],axis=1) for d in differences)/2
            lo,hi=np.quantile(draws,[.025,.975])
            contrasts.append({'contrast':'pre-'+comparator,'mean_difference':float(np.mean(np.concatenate(differences))),
                              'lower':float(lo),'upper':float(hi),'stratum_means':strata})
        summary.update(arms=arms,contrasts=contrasts)
        summary['strata']=[{'problem_type':typ,'arm':arm,
                            'f1':float(np.mean([g['f1'] for g in graphs if g['problem_type']==typ and g['arm']==arm])),
                            'em':float(np.mean([g['em'] for g in graphs if g['problem_type']==typ and g['arm']==arm]))}
                           for typ in ['parents','bfs'] for arm in ['pre','post','direct']]
        write_csv(REPORT/'arm_summary.csv',arms)
        write_csv(REPORT/'contrasts.csv',[{k:v for k,v in c.items() if k!='stratum_means'} for c in contrasts])
    if records:write_csv(REPORT/'audited_calls.csv',records)
    if graphs:write_csv(REPORT/'graph_summary.csv',graphs)
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
