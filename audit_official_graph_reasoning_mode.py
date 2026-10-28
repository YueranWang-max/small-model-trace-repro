"""Mode-isolated, graph-clustered accuracy and complete deployment cost audit."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from audit_official_graph_long_boundary import graph_answer,terminal_score
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_reasoning_mode'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==64 and len({(r['row_index'],r['arm'],r['repeat']) for r in rs})==64;assert all(sha(ROOT/n)==v for n,v in fr['dependencies'].items())
    dp=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet';assert sha(dp)==fr['data_sha256'];data=pq.read_table(dp).to_pylist();used=set();tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace','official_graph_identity_recode']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    chosen=[]
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(data):
                digest=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or digest in used:continue
                used.add(digest);chosen.append(i);count+=1
                if count==4:break
    expected=[chosen[j] for i in range(4) for j in [i,4+i,8+i,12+i]];assert [p['row_index'] for p in fr['selected']]==expected
    qualified=[];cases=[];ix={(r['row_index'],r['arm'],r['repeat']):r for r in rs}
    for i,p in enumerate(fr['selected']):
        pid=p['row_index'];assert p['prompt']==data[pid]['prompt'] and graph_answer(p)==set(p['answer_nodes']);user=p['prompt']+'\n'+fr['format_suffix'];row={'row_index':pid,'type':p['problem_type'],'length_bin':p['length_bin'][0],'gold_size':len(p['answer_nodes'])};a=[r for r in rs if r['row_index']==pid];assert [(r['arm'],r['repeat']) for r in a]==[(arm,j) for j in range(2) for arm in fr['orders'][i][j]]
        for r in a:
            terminal_score(r,p['answer_nodes']);q=r['request'];flag=r['arm']=='on';assert q['think'] is flag and q['model']==fr['identity']['name'] and q['messages'][1]['content']==user and q['messages'][0]==a[0]['request']['messages'][0] and q['options']=={'num_ctx':32768,'num_predict':8192,'seed':fr['seeds'][r['repeat']],**fr['regime']}
            assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest();rendered='<|im_start|>system\n'+q['messages'][0]['content']+'<|im_end|>\n<|im_start|>user\n'+user+(' /think' if flag else ' /no_think')+'<|im_end|>\n<|im_start|>assistant\n'+('' if flag else '<think>\n\n</think>\n\n');nt=len(tok.encode(rendered,add_special_tokens=False).ids);assert nt==r['expected_input_tokens'] and nt+8192+64<=32768
            ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==q and ledger['result']['raw']==r['raw'] and r['attempts']==1
            runtime=[v for v in r.get('runtime_ps',{}).get('models',[]) if v.get('digest')==fr['identity']['digest']];qualified.append(bool(r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768))
        for arm in ['on','off','route']:
            selected=fr['deployment_route'][p['problem_type']] if arm=='route' else arm;rr=[ix[(pid,selected,j)] for j in range(2)]
            for metric in ['f1','em']:row[arm+'_'+metric]=sum(r['score'][metric] for r in rr)/2
            row[arm+'_valid']=sum(r['score']['valid'] for r in rr);row[arm+'_length']=sum(r['raw'].get('done_reason')=='length' for r in rr);row[arm+'_tokens']=sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in rr)/2
        row['oracle_f1']=max(row['on_f1'],row['off_f1']);cases.append(row)
    ledgers=list(OUT.glob('attempt_*.json'));assert len(ledgers)==64 and all(json.loads(p.read_text(encoding='utf-8'))['completion']=='returned' for p in ledgers)
    rng=np.random.default_rng(202610072);draw=rng.integers(0,16,size=(20000,16));summaries=[]
    for arm in ['on','off','route']:
        v=np.array([p[arm+'_f1'] for p in cases]);ci=np.quantile(v[draw].mean(axis=1),[.025,.975]);summaries.append({'arm':arm,'graphs':16,'repeats':2,'f1':float(v.mean()),'em':sum(p[arm+'_em'] for p in cases)/16,'valid':sum(p[arm+'_valid'] for p in cases),'responses':32,'length':sum(p[arm+'_length'] for p in cases),'deployment_tokens':sum(p[arm+'_tokens'] for p in cases),'f1_lo':float(ci[0]),'f1_hi':float(ci[1])})
    comparisons=[]
    for a,b in [('off','on'),('route','on'),('route','off')]:
        d=np.array([p[a+'_f1']-p[b+'_f1'] for p in cases]);ci=np.quantile(d[draw].mean(axis=1),[.025,.975]);comparisons.append({'contrast':a+'-'+b,'delta':float(d.mean()),'lo':float(ci[0]),'hi':float(ci[1]),'positive_graphs':int(sum(d>0)),'negative_graphs':int(sum(d<0))})
    groups=[]
    for typ in ['parents','bfs']:
        for lo in [2000,7000]:
            gg=[p for p in cases if p['type']==typ and p['length_bin']==lo]
            for arm in ['on','off']:
                groups.append({'type':typ,'length_bin':lo,'arm':arm,'graphs':len(gg),'f1':sum(p[arm+'_f1'] for p in gg)/len(gg),'em':sum(p[arm+'_em'] for p in gg)/len(gg),'deployment_tokens':sum(p[arm+'_tokens'] for p in gg),'valid':sum(p[arm+'_valid'] for p in gg)})
    by={s['arm']:s for s in summaries};gates={}
    for arm in ['off','route']:
        s=by[arm];on=by['on'];gate=all(qualified) and s['valid']>=31 and s['f1']>=on['f1']-.05 and s['em']>=on['em'] and s['deployment_tokens']<=.7*on['deployment_tokens'];dominated=arm=='route' and by['off']['f1']>=s['f1'] and by['off']['em']>=s['em'] and by['off']['deployment_tokens']<=s['deployment_tokens'] and (by['off']['f1']>s['f1'] or by['off']['deployment_tokens']<s['deployment_tokens']);gates[arm]={'exploratory_followup_gate':bool(gate and not dominated),'dominated_by_always_off':bool(dominated)}
    result={'summary':summaries,'case_table':cases,'groups':groups,'comparisons':comparisons,'gates':gates,'qualified_calls':sum(qualified),'planned_calls':64,'known_tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in rs),'unknown_usage_calls':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in rs),'always_empty_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/16,'oracle_mean_f1':sum(p['oracle_f1'] for p in cases)/16,'oracle_is_not_deployed':True};save(OUT/'summary.json',result);save(OUT/'independent_audit.json',{'passed':True,'calls':64,'qualified_calls':sum(qualified),'only_think_flag_changed_between_paired_requests':True,'different_service_thinking_markers_counted':True,'task_route_prespecified':True,'selection_reconstructed':True,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledgers}})
    for tail,rows in [('',summaries),('_cases',cases),('_groups',groups)]:
        with (ROOT/'paper/figures/data'/('official_graph_reasoning_mode'+tail+'.csv')).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['case_table','groups']},indent=2))
if __name__=='__main__':main()
