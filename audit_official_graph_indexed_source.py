"""Independent full edge-multiset, operation, spelling and mode factorial audit."""
import csv,hashlib,json,re
from collections import Counter,deque
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_indexed_source';MARKER='Here is the graph to operate on:'
ARMS=['original_on','original_off','compact_on','compact_off','indexed_on','indexed_off']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def read_graph(prompt,indexed=False):
    head,op=prompt.rsplit('Operation:',1);prefix,body=head.split(MARKER,1);edges=[];nodes=set();prose=[]
    for line in body.splitlines():
        s=line.strip()
        if not s:continue
        if indexed:
            m=re.fullmatch(r'(IN|OUT) (\w+): \[(.*)\]',s)
            if m:
                nodes.add(m[2]);other=[n.strip() for n in m[3].split(',') if n.strip()]
                edges.extend((n,m[2]) if m[1]=='IN' else (m[2],n) for n in other);continue
        else:
            parts=s.split(' -> ')
            if len(parts)==2 and all(re.fullmatch(r'\w+',n) for n in parts):edges.append(tuple(parts));continue
        prose.append(s)
    graph_nodes={n for e in edges for n in e}
    if indexed:assert nodes==graph_nodes
    return edges,graph_nodes,prefix,op,prose
def answer(edges,operation,typ):
    if typ=='parents':
        target=re.search(r'Find the parents of node (\w+)\.',operation)[1];return {u for u,v in edges if v==target and u!=target}
    start=re.search(r'BFS from node (\w+)',operation)[1];depth=int(re.search(r'exactly depth (\d+)',operation)[1]);adj={}
    for u,v in edges:adj.setdefault(u,[]).append(v)
    dist={start:0};queue=deque([start])
    while queue:
        u=queue.popleft()
        if dist[u]>=depth:continue
        for v in adj.get(u,[]):
            if v not in dist:dist[v]=dist[u]+1;queue.append(v)
    return {n for n,d in dist.items() if d==depth and n!=start}
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==96 and len({(r['row_index'],r['arm'],r['repeat']) for r in rs})==96;assert all(sha(ROOT/n)==v for n,v in fr['dependencies'].items())
    dp=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet';assert sha(dp)==fr['data_sha256'];data=pq.read_table(dp).to_pylist();tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace','official_graph_identity_recode','official_graph_reasoning_mode']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    chosen=[]
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(data):
                digest=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or digest in used:continue
                used.add(digest);chosen.append(i);count+=1
                if count==2:break
    assert [p['row_index'] for p in fr['selected']]==[chosen[j] for j in [0,2,4,6,1,3,5,7]];qualified=[];cases=[]
    for i,p in enumerate(fr['selected']):
        pid=p['row_index'];assert p['prompt']==data[pid]['prompt'];x=(p['prompt']+'\n'+fr['format_suffix']).replace(MARKER,fr['glossary']+'\n'+MARKER,1);assert x==p['prompts']['original'];ee,nn,prefix,op,prose=read_graph(x);gold=answer(ee,op,p['problem_type']);assert gold==set(p['answer_nodes']);assert p['index_kind']==fr['task_index_route'][p['problem_type']]
        row={'row_index':pid,'type':p['problem_type'],'length_bin':p['length_bin'][0],'gold_size':len(gold),'edges_including_duplicates':len(ee),'self_loops':sum(u==v for u,v in ee),'compact_dictionary_json_bytes':p['compact_dictionary_json_bytes'],'preprocess_wall_sec':p['preprocess_wall_sec']}
        for enc in ['original','compact','indexed']:
            inv=p['inverse']['original' if enc=='original' else 'compact'];mp=p['maps']['original' if enc=='original' else 'compact'];assert inv=={v:k for k,v in mp.items()};prompt=p['prompts'][enc];edges,nodes,pp,qq,text=read_graph(prompt,enc=='indexed')
            restore=lambda s:re.sub(r'\b(?:'+'|'.join(sorted(map(re.escape,inv),key=len,reverse=True))+r')\b',lambda m:inv[m[0]],s)
            assert Counter((inv[u],inv[v]) for u,v in edges)==Counter(ee) and {inv[n] for n in nodes}==nn and restore(pp)==prefix and restore(qq)==op and text==prose
            assert {inv[n] for n in answer(edges,qq,p['problem_type'])}==gold
            if enc=='compact':assert restore(prompt)==x
        records=[r for r in rs if r['row_index']==pid];assert [(r['arm'],r['repeat']) for r in records]==[(arm,j) for j in range(2) for arm in fr['orders'][i][j]]
        for r in records:
            enc,mode=r['arm'].rsplit('_',1);flag=mode=='on';q=r['request'];user=p['prompts'][enc];inv=p['inverse']['original' if enc=='original' else 'compact'];assert q['messages']==[{'role':'system','content':fr['system']},{'role':'user','content':user}] and q['think'] is flag and q['model']==fr['identity']['name'] and q['options']=={'num_ctx':32768,'num_predict':8192,'seed':fr['seeds'][r['repeat']],**fr['regime']}
            assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest();content=(r['raw'].get('message') or {}).get('content','') or '';m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',content.rstrip().split('\n')[-1].strip());pred=[n.strip() for n in m[1].split(',') if n.strip()] if m else [];mapped=[inv[n] if n in inv else '<unknown-node:'+n+'>' for n in pred];unknown=[n for n in pred if n not in inv];valid=bool(m) and not r['error'] and r['raw'].get('done_reason')=='stop';a=set(mapped);f=2*len(a&gold)/(len(a)+len(gold)) if a or gold else 1.;s={'prediction':pred,'restored_prediction':mapped,'unknown_prediction_nodes':unknown,'valid':bool(valid),'em':float(valid and a==gold),'f1':f if valid else 0.};assert s==r['score']
            rendered='<|im_start|>system\n'+fr['system']+'<|im_end|>\n<|im_start|>user\n'+user+(' /think' if flag else ' /no_think')+'<|im_end|>\n<|im_start|>assistant\n'+('' if flag else '<think>\n\n</think>\n\n');nt=len(tok.encode(rendered,add_special_tokens=False).ids);assert nt==r['expected_input_tokens'] and nt+8192+64<=32768
            ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==q and ledger['result']['raw']==r['raw'] and r['attempts']==1
            runtime=[v for v in r.get('runtime_ps',{}).get('models',[]) if v.get('digest')==fr['identity']['digest']];qualified.append(bool(r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768))
        for arm in ARMS:
            a=[r for r in records if r['arm']==arm]
            for metric in ['f1','em']:row[arm+'_'+metric]=sum(r['score'][metric] for r in a)/2
            row[arm+'_valid']=sum(r['score']['valid'] for r in a);row[arm+'_tokens']=sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a)/2
        cases.append(row)
    ledgers=list(OUT.glob('attempt_*.json'));assert len(ledgers)==96 and all(json.loads(p.read_text(encoding='utf-8'))['completion']=='returned' for p in ledgers);rng=np.random.default_rng(202610082);draw=rng.integers(0,8,size=(20000,8));summaries=[]
    for arm in ARMS:
        a=[r for r in rs if r['arm']==arm];v=np.array([p[arm+'_f1'] for p in cases]);ci=np.quantile(v[draw].mean(axis=1),[.025,.975]);summaries.append({'arm':arm,'graphs':8,'repeats':2,'f1':float(v.mean()),'em':sum(p[arm+'_em'] for p in cases)/8,'valid':sum(r['score']['valid'] for r in a),'responses':16,'length':sum(r['raw'].get('done_reason')=='length' for r in a),'deployment_tokens':sum(p[arm+'_tokens'] for p in cases),'input_tokens':sum(r['raw'].get('prompt_eval_count',0) for r in a),'output_tokens':sum(r['raw'].get('eval_count',0) for r in a),'unknown_prediction_responses':sum(bool(r['score']['unknown_prediction_nodes']) for r in a),'f1_lo':float(ci[0]),'f1_hi':float(ci[1])})
    contrasts=[]
    for arm in ['original_on','original_off','compact_off','indexed_on']:
        d=np.array([p['indexed_off_f1']-p[arm+'_f1'] for p in cases]);ci=np.quantile(d[draw].mean(axis=1),[.025,.975]);contrasts.append({'contrast':'indexed_off-'+arm,'delta':float(d.mean()),'lo':float(ci[0]),'hi':float(ci[1]),'positive_graphs':int(sum(d>0))})
    by={s['arm']:s for s in summaries};candidate=by['indexed_off'];gate=all(qualified) and candidate['valid']>=15 and all(candidate['f1']>=by[a]['f1']+.10 and candidate['em']>=by[a]['em'] for a in ['original_on','original_off']) and candidate['f1']>=by['compact_off']['f1']+.05 and contrasts[2]['positive_graphs']>=4 and candidate['deployment_tokens']<=by['original_off']['deployment_tokens']
    result={'summary':summaries,'case_table':cases,'contrasts':contrasts,'followup_gate':bool(gate),'qualified_calls':sum(qualified),'planned_calls':96,'known_tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in rs),'unknown_usage_calls':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in rs),'always_empty_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/8,'dictionary_bytes_total':sum(p['compact_dictionary_json_bytes'] for p in fr['selected']),'preprocess_wall_sec_total':sum(p['preprocess_wall_sec'] for p in fr['selected'])};save(OUT/'summary.json',result);save(OUT/'independent_audit.json',{'passed':True,'calls':96,'qualified_calls':sum(qualified),'full_edge_counter_and_selfloops_preserved':True,'node_universe_preserved':True,'operation_and_prose_preserved':True,'independent_queue_semantics':True,'selection_reconstructed':True,'unknown_nodes_are_false_positives':True,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledgers}})
    for tail,rows in [('',summaries),('_cases',cases)]:
        with (ROOT/'paper/figures/data'/('official_graph_indexed_source'+tail+'.csv')).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='case_table'},indent=2))
if __name__=='__main__':main()
