"""Five-repeat analysis clustered by graph; raw-response and five-T audit."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from audit_official_graph_long_boundary import graph_answer,terminal_score
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_five_trace'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==80;ix={(r['row_index'],r['arm'],r['repeat']):r for r in rs};assert len(ix)==80
    assert all(v==sha(ROOT/n) for n,v in fr['dependencies'].items());data=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist();assert fr['data_sha256']==sha(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet');tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    chosen=[]
    for typ in ['parents','bfs']:
        n=0
        for i,p in enumerate(data):
            h=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
            if p['problem_type']!=typ or not 2000<=p['prompt_chars']<3500 or h in used:continue
            chosen.append(i);used.add(h);n+=1
            if n==2:break
    assert [p['row_index'] for p in fr['selected']]==[chosen[0],chosen[2],chosen[1],chosen[3]]
    pd='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.';qualified=[];cases=[]
    for i,p in enumerate(fr['selected']):
        pid=p['row_index'];assert p['prompt']==data[pid]['prompt'] and set(p['answer_nodes'])==graph_answer(p);x=p['prompt']+'\n'+fr['original_format_suffix'];head,op=x.rsplit('Operation:',1);sources=[ix[(pid,'source',j)] for j in range(5)];parts=[pd,'<first_run_reasoning_traces>']
        for j,s in enumerate(sources):
            text=(s['raw'].get('message') or {}).get('thinking','') or '';text=text[:50000]+'...' if len(text)>50000 else text;parts.append('[Trace '+str(j+1)+'] '+text.strip())
        T='\n'.join(parts+['</first_run_reasoning_traces>']);row={'row_index':pid,'type':p['problem_type'],'gold_size':len(p['answer_nodes']),'trace_chars':len(T),'source_valid':sum(r['score']['valid'] for r in sources),'source_f1':sum(r['score']['f1'] for r in sources)/5,'source_em':sum(r['score']['em'] for r in sources)/5}
        count=Counter(node for r in sources for node in set(r['score']['prediction']) if r['score']['valid']);majority={s for s,n in count.items() if n>=3};gold=set(p['answer_nodes']);row['majority_f1']=2*len(majority&gold)/(len(majority)+len(gold)) if majority or gold else 1.;row['majority_em']=float(majority==gold);row['oracle_f1']=max(r['score']['f1'] for r in sources);row['oracle_em']=max(r['score']['em'] for r in sources)
        expected=[('source',j) for j in range(5)]+[(arm,j) for j in range(5) for arm in fr['orders'][i][j]];assert [(r['arm'],r['repeat']) for r in rs if r['row_index']==pid]==expected
        for arm,j in expected:
            r=ix[(pid,arm,j)];terminal_score(r,p['answer_nodes']);q=r['request'];user=x if arm in ['source','reread'] else T+'\n'+x if arm=='pre' else head+T+'\nOperation:'+op;cap=4096 if arm=='source' else 8192;seed=(fr['source_seeds'] if arm=='source' else fr['reader_seeds'])[j]
            assert q['messages'][1]['content']==user and q['messages'][0]==sources[0]['request']['messages'][0] and q['model']==fr['identity']['name'] and q['think'] is True and q['options']=={'num_ctx':32768,'num_predict':cap,'seed':seed,**fr['regime']}
            assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest();system=q['messages'][0]['content'];rendered='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+' /think<|im_end|>\n<|im_start|>assistant\n';n=len(tok.encode(rendered,add_special_tokens=False).ids);assert n==r['expected_input_tokens'] and n+cap+64<=32768
            ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==q and ledger['result']['raw']==r['raw'] and r['attempts']==1
            runtime=[m for m in r.get('runtime_ps',{}).get('models',[]) if m.get('digest')==fr['identity']['digest']];qualified.append({'row_index':pid,'arm':arm,'repeat':j,'qualified':r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768})
            if arm!='source':assert r['trace_sha256']==hashlib.sha256(T.encode()).hexdigest() and r['source_request_sha256s']==[s['request_sha256'] for s in sources]
        for arm in ['pre','post','reread']:
            a=[ix[(pid,arm,j)] for j in range(5)];row[arm+'_f1']=sum(r['score']['f1'] for r in a)/5;row[arm+'_em']=sum(r['score']['em'] for r in a)/5;row[arm+'_valid']=sum(r['score']['valid'] for r in a)
        cases.append(row)
    ledgers=list(OUT.glob('attempt_*.json'));assert len(ledgers)==80 and all(json.loads(p.read_text(encoding='utf-8'))['completion']=='returned' for p in ledgers);summaries=[];src=[r for r in rs if r['arm']=='source'];tokens=lambda a:sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a);src_tokens=tokens(src)
    for arm in ['source','reread','pre','post']:
        a=[r for r in rs if r['arm']==arm];cost=tokens(a);summaries.append({'arm':arm,'graphs':4,'repeats':5,'f1':sum(p[arm+'_f1'] for p in cases)/4,'em':sum(p[arm+'_em'] for p in cases)/4,'valid':sum(r['score']['valid'] for r in a),'responses':20,'length':sum(r['raw'].get('done_reason')=='length' for r in a),'reader_tokens':cost,'single_prediction_deployment_tokens':src_tokens+cost/5 if arm in ['pre','post'] else cost/5,'whole_five_reader_path_tokens':src_tokens+cost if arm in ['pre','post','reread'] else src_tokens,'producer_plus_mean_reader_tokens':src_tokens+cost/5 if arm!='source' else src_tokens,'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in a)})
    baselines={'majority_f1':sum(p['majority_f1'] for p in cases)/4,'majority_em':sum(p['majority_em'] for p in cases)/4,'majority_deployment_tokens':src_tokens,'oracle_f1':sum(p['oracle_f1'] for p in cases)/4,'oracle_em':sum(p['oracle_em'] for p in cases)/4,'oracle_deployable':False,'always_empty_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/4};rng=np.random.default_rng(202610040);draw=rng.integers(0,4,size=(20000,4));comparisons=[]
    for arm in ['post','reread']:
        diff=np.array([p['pre_f1']-p[arm+'_f1'] for p in cases]);ci=np.quantile(diff[draw].mean(axis=1),[.025,.975]);comparisons.append({'contrast':'pre-'+arm,'delta':float(diff.mean()),'lo':float(ci[0]),'hi':float(ci[1]),'positive_graphs':int(sum(diff>0)),'unit':'4 graphs after5 repeats; descriptive only'})
    for s in summaries:
        v=np.array([p[s['arm']+'_f1'] for p in cases]);ci=np.quantile(v[draw].mean(axis=1),[.025,.975]);s['f1_lo']=float(ci[0]);s['f1_hi']=float(ci[1]);s['majority_f1']=baselines['majority_f1']
    by={r['arm']:r for r in summaries};gate=all(q['qualified'] for q in qualified) and all(r['score']['valid'] for r in rs if r['arm']!='source') and all(c['delta']>=.05 and c['positive_graphs']>=2 for c in comparisons) and by['pre']['em']>=max(by['post']['em'],by['reread']['em'])
    result={'summary':summaries,'case_table':cases,'source_baselines':baselines,'paired_graph_differences':comparisons,'followup_quality_gate':gate,'qualified_calls':sum(q['qualified'] for q in qualified),'planned_calls':80,'known_tokens':tokens(rs),'unknown_usage_calls':sum(s['unknown_usage'] for s in summaries),'graph_units':4,'no_repeat_pseudoreplication':True,'source_cap_lower_than_reader_cap':True};save(OUT/'summary.json',result);save(OUT/'independent_audit.json',{'passed':True,'calls':80,'qualified_calls':sum(q['qualified'] for q in qualified),'same_five_reasoning_only_T_verified':True,'graph_selection_reconstructed':True,'original_format_suffix_consistent':True,'score_units':4,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledgers}})
    for name,rows in [('official_graph_five_trace',summaries),('official_graph_five_trace_cases',cases)]:
        with (ROOT/'paper/figures/data'/(name+'.csv')).open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'summary':summaries,'baselines':baselines,'contrasts':comparisons,'quality_gate':gate,'qualified_calls':result['qualified_calls'],'known_tokens':result['known_tokens']},indent=2))
if __name__=='__main__':main()
