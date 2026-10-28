"""Independent audit and descriptive analysis; only run after all model calls."""
import csv,hashlib,itertools,json,re
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_long_boundary'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def terminal_score(r,gold):
    content=(r['raw'].get('message') or {}).get('content','') or '';line=content.rstrip().split('\n')[-1].strip();m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',line)
    valid=bool(m) and r['error'] is None and r['raw'].get('done_reason')=='stop';pred=set(x.strip() for x in m[1].split(',') if x.strip()) if m else set();gold=set(gold);f=(2*len(pred&gold)/(len(pred)+len(gold)) if pred or gold else 1.) if valid else 0.
    assert r['score']['valid']==valid and r['score']['em']==float(valid and pred==gold) and abs(r['score']['f1']-f)<1e-12
def graph_answer(p):
    h,q=p['prompt'].rsplit('Operation:',1);s=h.split('Here is the graph to operate on:',1)[1];edges=[line.split(' -> ') for line in s.splitlines() if re.fullmatch(r'[0-9a-z]+ -> [0-9a-z]+',line.strip())]
    if p['problem_type']=='parents':
        target=re.search(r'parents of node (\w+)',q)[1];return {u for u,v in edges if v==target and u!=target}
    start=re.search(r'BFS from node (\w+)',q)[1];d=int(re.search(r'exactly depth (\d+)',q)[1]);adj={}
    for u,v in edges:adj.setdefault(u,[]).append(v)
    # An independent queue/distance implementation, rather than runner frontier sets.
    from collections import deque
    dist={start:0};queue=deque([start])
    while queue:
        u=queue.popleft()
        if dist[u]>=d:continue
        for v in adj.get(u,[]):
            if v not in dist:dist[v]=dist[u]+1;queue.append(v)
    return {v for v,k in dist.items() if k==d and v!=start}
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==32;index={(r['row_index'],r['arm']):r for r in rs};assert len(index)==32
    assert all(v==sha(ROOT/n) for n,v in fr['dependencies'].items());assert fr['data_sha256']==sha(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet')
    data=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist();used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    selection=[]
    for typ in ['parents','bfs']:
        count=0
        for i,p in enumerate(data):
            h=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
            if p['problem_type']!=typ or not 7000<=p['prompt_chars']<9000 or h in used:continue
            selection.append(i);used.add(h);count+=1
            if count==2:break
    assert [p['row_index'] for p in fr['selected']]==[selection[0],selection[2],selection[1],selection[3]]
    tokenizer=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));qualification=[];trace_rows=[]
    pd='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
    for i,p in enumerate(fr['selected']):
        x=p['prompt'];head,op=x.rsplit('Operation:',1);assert p['prompt']==data[p['row_index']]['prompt'] and set(p['answer_nodes'])==graph_answer(p)
        high=index[(p['row_index'],'source_16384')];s=(high['raw'].get('message') or {}).get('thinking','') or '';s=s[:50000]+'...' if len(s)>50000 else s;T=pd+'\n<first_run_reasoning_traces>\n[Trace 1] '+s.strip()+'\n</first_run_reasoning_traces>'
        expected_order=['source_'+str(c) for c in fr['source_orders'][i]]+fr['reader_orders'][i];assert [r['arm'] for r in rs if r['row_index']==p['row_index']]==expected_order
        for arm in expected_order:
            r=index[(p['row_index'],arm)];terminal_score(r,p['answer_nodes']);req=r['request'];base=arm.removeprefix('check_');cap=int(arm.split('_')[1]) if arm.startswith('source_') else 8192
            expected=x if arm.startswith('source_') or base=='reread' else T+'\n'+x if base=='pre' else head+T+'\nOperation:'+op
            system=high['request']['messages'][0]['content']+(fr['check_system_suffix'] if arm.startswith('check_') else '')
            assert req['messages']==[{'role':'system','content':system},{'role':'user','content':expected}]
            assert req['model']==fr['identity']['name'] and req['think'] is True and req['options']=={'temperature':0.,'top_p':.9,'num_ctx':40960,'num_predict':cap}
            assert r['request_sha256']==hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
            rendered='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+expected+' /think<|im_end|>\n<|im_start|>assistant\n';n=len(tokenizer.encode(rendered,add_special_tokens=False).ids)
            assert n==r['expected_input_tokens'] and r['context_guard']==(n+cap+64<=40960)
            ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']!='pending' and ledger['request']==req and ledger['result']['raw']==r['raw'] and ledger['result']['error']==r['error']
            assert r['attempts']==int(r['context_guard'])
            runtime=[m for m in r.get('runtime_ps',{}).get('models',[]) if m.get('digest')==fr['identity']['digest']];ok=r['context_guard'] and r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=40960
            qualification.append({'row_index':p['row_index'],'arm':arm,'expected_input':n,'api_input':r['raw'].get('prompt_eval_count'),'context_guard':r['context_guard'],'runtime_context':runtime[0].get('context_length') if runtime else None,'qualified':ok,'within_official_32k_planned_positions':n+cap+64<=32768,'within_official_32k_actual_positions':n+r['raw'].get('eval_count',32768)+64<=32768})
            if not arm.startswith('source_'):assert r['trace_sha256']==hashlib.sha256(T.encode()).hexdigest() and r['source_request_sha256']==high['request_sha256']
        low=index[(p['row_index'],'source_8192')];lo=(low['raw'].get('message') or {}).get('thinking','') or '';hi=(high['raw'].get('message') or {}).get('thinking','') or ''
        trace_rows.append({'row_index':p['row_index'],'type':p['problem_type'],'gold_size':len(p['answer_nodes']),'low_thinking_prefix_of_high':bool(lo) and hi.startswith(lo),'trace_chars':len(T),'source8192_f1':low['score']['f1'],'source16384_f1':high['score']['f1'],'source16384_em':high['score']['em'],**{a+'_f1':index[(p['row_index'],a)]['score']['f1'] for a in ['pre','post','reread','check_pre','check_post','check_reread']}})
    ledger_paths=list(OUT.glob('attempt_*.json'));assert len(ledger_paths)==32 and all(json.loads(p.read_text(encoding='utf-8'))['completion']!='pending' for p in ledger_paths)
    sources=[r for r in rs if r['arm']=='source_16384'];token=lambda a:sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a);summaries=[]
    arms=['source_8192','source_16384','reread','pre','post','check_reread','check_pre','check_post'];idlist=[p['row_index'] for p in fr['selected']]
    for arm in arms:
        a=[index[(i,arm)] for i in idlist];src=[index[(i,'source_16384')] for i in idlist];reader=token(a);is_source=arm.startswith('source_');has_trace=arm in ['pre','post','check_pre','check_post']
        summaries.append({'arm':arm,'n':4,'f1':sum(r['score']['f1'] for r in a)/4,'em':sum(r['score']['em'] for r in a)/4,'valid':sum(r['score']['valid'] for r in a),'length':sum(r['raw'].get('done_reason')=='length' for r in a),'errors':sum(bool(r['error']) for r in a),'input_qualified':sum(q['qualified'] for q in qualification if q['arm']==arm),'reader_tokens':reader,'two_pass_tokens':reader+(token(sources) if not is_source else 0),'minimum_deployment_tokens':reader+(token(sources) if has_trace else 0),'corrected_source_errors':sum(s['score']['em']==0 and r['score']['em']==1 for s,r in zip(src,a)),'introduced_errors':sum(s['score']['em']==1 and r['score']['em']==0 for s,r in zip(src,a)),'inherited_prediction_sets':sum(set(s['score']['prediction'])==set(r['score']['prediction']) for s,r in zip(src,a)),'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in a)})
    for s in summaries:s['always_empty_baseline_f1_em']=sum(not p['answer_nodes'] for p in fr['selected'])/4
    # A malformed/missing visible answer is not an actual inherited empty set.
    for s in summaries:
        s['inherited_prediction_sets']=sum(index[(i,'source_16384')]['score']['valid'] and index[(i,s['arm'])]['score']['valid'] and set(index[(i,'source_16384')]['score']['prediction'])==set(index[(i,s['arm'])]['score']['prediction']) for i in idlist)
    by={s['arm']:s for s in summaries};candidates={}
    for name in ['pre','post','reread']:
        a,b=by['check_'+name],by[name];candidates['check_'+name]=all(q['qualified'] for q in qualification) and all(r['score']['valid'] for r in rs if not r['arm'].startswith('source_')) and a['f1']>=b['f1'] and a['em']>=b['em'] and a['corrected_source_errors']>=2 and a['reader_tokens']<=b['reader_tokens']
    result={'summary':summaries,'candidate_for_new_case_followup':candidates,'qualified_calls':sum(q['qualified'] for q in qualification),'planned_calls':32,'actual_http_attempts':sum(r['attempts'] for r in rs),'source16384_errors':sum(r['score']['em']==0 for r in sources),'always_empty_baseline_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/4,'case_table':trace_rows,'input_qualification':qualification,'total_known_tokens':token(rs),'unknown_usage_calls':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in rs),'known_tokens_are_lower_bound':any(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in rs),'all_expected_calls_ledgered':True,'historical_interruption_not_erased':True}
    save(OUT/'summary.json',result)
    audit={'passed':True,'raw_visible_scores_recomputed':32,'selection_reconstructed':True,'independent_queue_graph_labels':4,'same_T_x_q_verified':True,'write_ahead_ledgers_complete':32,'qualification_count':sum(q['qualified'] for q in qualification),'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledger_paths}}
    save(OUT/'independent_audit.json',audit)
    for name,rows in [('official_graph_long_boundary',summaries),('official_graph_long_boundary_cases',trace_rows)]:
        with (ROOT/'paper/figures/data'/(name+'.csv')).open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'summary':summaries,'qualified':result['qualified_calls'],'candidates':candidates,'always_empty':result['always_empty_baseline_f1_em'],'total_known_tokens':result['total_known_tokens']},indent=2))
if __name__=='__main__':main()
