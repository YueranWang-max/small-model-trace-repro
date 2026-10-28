"""Independent inverse-transform, graph, terminal-output and request audit."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_identity_recode'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def graph_answer(p):
    # The previous audit helper only accepts lowercase/hash node names. This
    # independent queue parser must also accept the compact uppercase alphabet.
    from collections import deque
    head,op=p['prompt'].rsplit('Operation:',1);body=head.split('Here is the graph to operate on:',1)[1]
    edges=[tuple(line.strip().split(' -> ')) for line in body.splitlines() if re.fullmatch(r'[0-9A-Za-z]+ -> [0-9A-Za-z]+',line.strip())]
    if p['problem_type']=='parents':
        target=re.search(r'parents of node (\w+)',op)[1];return {u for u,v in edges if v==target and u!=target}
    start=re.search(r'BFS from node (\w+)',op)[1];depth=int(re.search(r'exactly depth (\d+)',op)[1]);adj={}
    for u,v in edges:adj.setdefault(u,[]).append(v)
    dist={start:0};queue=deque([start])
    while queue:
        u=queue.popleft()
        if dist[u]>=depth:continue
        for v in adj.get(u,[]):
            if v not in dist:dist[v]=dist[u]+1;queue.append(v)
    return {v for v,d in dist.items() if d==depth and v!=start}
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==48;assert len({(r['row_index'],r['arm'],r['repeat']) for r in rs})==48
    assert all(h(ROOT/n)==v for n,v in fr['dependencies'].items());dp=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet';assert h(dp)==fr['data_sha256'];data=pq.read_table(dp).to_pylist();tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    chosen=[]
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            n=0
            for i,p in enumerate(data):
                digest=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or digest in used:continue
                used.add(digest);chosen.append(i);n+=1
                if n==2:break
    assert [p['row_index'] for p in fr['selected']]==[chosen[j] for j in [0,2,4,6,1,3,5,7]]
    qualified=[];cases=[]
    for i,p in enumerate(fr['selected']):
        pid=p['row_index'];assert p['prompt']==data[pid]['prompt'];gold=graph_answer(p);assert gold==set(p['answer_nodes']);x=p['prompt']+'\n'+fr['format_suffix'];row={'row_index':pid,'type':p['problem_type'],'length_bin':p['length_bin'][0],'gold_size':len(gold),'compact_dictionary_json_bytes':p['compact_dictionary_json_bytes'],'preprocess_wall_sec':p['preprocess_wall_sec']}
        actual=[r for r in rs if r['row_index']==pid];assert [(r['arm'],r['repeat']) for r in actual]==[(arm,j) for j in range(2) for arm in fr['orders'][i][j]]
        for arm in ['original','compact','long_alias']:
            mp=p['maps'][arm];inv=p['inverse'][arm];assert inv=={v:k for k,v in mp.items()} and len(inv)==len(mp)
            assert set(mp)==set(re.findall(r'\b[0-9a-f]{10}\b',x))
            transformed=re.sub(r'\b[0-9a-f]{10}\b',lambda m:mp[m[0]],x);assert transformed==p['prompts'][arm]
            pat=re.compile(r'\b(?:'+'|'.join(sorted(map(re.escape,inv),key=len,reverse=True))+r')\b');assert pat.sub(lambda m:inv[m[0]],transformed)==x
            assert {inv[n] for n in graph_answer({'prompt':transformed,'problem_type':p['problem_type']})}==gold
            scores=[];a=[r for r in actual if r['arm']==arm]
            for r in a:
                q=r['request'];assert q['model']==fr['identity']['name'] and q['think'] is True and q['messages'][1]['content']==transformed and q['options']=={'num_ctx':32768,'num_predict':8192,'seed':fr['seeds'][r['repeat']],**fr['regime']}
                assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
                c=(r['raw'].get('message') or {}).get('content','') or '';m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',c.rstrip().split('\n')[-1].strip());pred=[s.strip() for s in m[1].split(',') if s.strip()] if m else [];unknown=[n for n in pred if n not in inv];mapped=[inv[n] for n in pred if n in inv];valid=bool(m) and not r['error'] and r['raw'].get('done_reason')=='stop' and not unknown;aa=set(mapped);f=2*len(aa&gold)/(len(aa)+len(gold)) if aa or gold else 1.
                expected={'prediction':pred,'restored_prediction':mapped,'unknown_prediction_nodes':unknown,'valid':bool(valid),'em':float(valid and aa==gold),'f1':f if valid else 0.};assert expected==r['score'];scores.append(expected)
                rendered='<|im_start|>system\n'+q['messages'][0]['content']+'<|im_end|>\n<|im_start|>user\n'+transformed+' /think<|im_end|>\n<|im_start|>assistant\n';nt=len(tok.encode(rendered,add_special_tokens=False).ids);assert nt==r['expected_input_tokens'] and nt+8192+64<=32768
                ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==q and ledger['result']['raw']==r['raw'] and r['attempts']==1
                runtime=[v for v in r.get('runtime_ps',{}).get('models',[]) if v.get('digest')==fr['identity']['digest']];qualified.append(bool(r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768))
            row[arm+'_f1']=sum(s['f1'] for s in scores)/2;row[arm+'_em']=sum(s['em'] for s in scores)/2;row[arm+'_valid']=sum(s['valid'] for s in scores);row[arm+'_input_tokens']=sum(r['raw'].get('prompt_eval_count',0) for r in a)/2;row[arm+'_output_tokens']=sum(r['raw'].get('eval_count',0) for r in a)/2
        cases.append(row)
    ledgers=list(OUT.glob('attempt_*.json'));assert len(ledgers)==48 and all(json.loads(p.read_text(encoding='utf-8'))['completion']=='returned' for p in ledgers)
    rng=np.random.default_rng(202610064);draw=rng.integers(0,8,size=(20000,8));summaries=[]
    for arm in ['original','compact','long_alias']:
        a=[r for r in rs if r['arm']==arm];v=np.array([p[arm+'_f1'] for p in cases]);ci=np.quantile(v[draw].mean(axis=1),[.025,.975]);summaries.append({'arm':arm,'graphs':8,'repeats':2,'f1':float(v.mean()),'em':sum(p[arm+'_em'] for p in cases)/8,'valid':sum(r['score']['valid'] for r in a),'responses':16,'length':sum(r['raw'].get('done_reason')=='length' for r in a),'tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a),'input_tokens':sum(r['raw'].get('prompt_eval_count',0) for r in a),'output_tokens':sum(r['raw'].get('eval_count',0) for r in a),'f1_lo':float(ci[0]),'f1_hi':float(ci[1])})
    contrasts=[]
    for arm in ['original','long_alias']:
        d=np.array([p['compact_f1']-p[arm+'_f1'] for p in cases]);ci=np.quantile(d[draw].mean(axis=1),[.025,.975]);contrasts.append({'contrast':'compact-'+arm,'delta':float(d.mean()),'lo':float(ci[0]),'hi':float(ci[1]),'positive_graphs':int(sum(d>0))})
    by={s['arm']:s for s in summaries};gate=all(qualified) and by['compact']['valid']==16 and all(c['delta']>=.10 and c['positive_graphs']>=4 for c in contrasts) and by['compact']['em']>=max(by['original']['em'],by['long_alias']['em']) and by['compact']['tokens']<=by['original']['tokens']
    result={'summary':summaries,'case_table':cases,'contrasts':contrasts,'followup_gate':gate,'qualified_calls':sum(qualified),'planned_calls':48,'known_tokens':sum(s['tokens'] for s in summaries),'unknown_usage_calls':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in rs),'always_empty_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/8,'compact_dictionary_bytes_total':sum(p['compact_dictionary_json_bytes'] for p in fr['selected']),'preprocess_wall_sec_total':sum(p['preprocess_wall_sec'] for p in fr['selected'])};save(OUT/'summary.json',result);save(OUT/'independent_audit.json',{'passed':True,'qualified_calls':sum(qualified),'calls':48,'bijective_prompt_roundtrip':True,'independent_graph_equivalence':True,'strict_unknown_node_failure':True,'selection_reconstructed':True,'assets':{str(p.relative_to(ROOT)):h(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledgers}})
    for name,rows in [('official_graph_identity_recode',summaries),('official_graph_identity_recode_cases',cases)]:
        with (ROOT/'paper/figures/data'/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
