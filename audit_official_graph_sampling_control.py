"""Recompute matched sampler data from raw visible responses and request ledgers."""
import csv,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from audit_official_graph_long_boundary import graph_answer,terminal_score
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_sampling_control'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));assert len(rs)==16;index={(r['row_index'],r['regime'],r['seed']):r for r in rs};assert len(index)==16
    assert all(v==sha(ROOT/n) for n,v in fr['dependencies'].items());data=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist();assert fr['data_sha256']==sha(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet');tokenizer=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
    selected=[]
    for typ in ['parents','bfs']:
        count=0
        for i,p in enumerate(data):
            h=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
            if p['problem_type']!=typ or not 7000<=p['prompt_chars']<9000 or h in used:continue
            selected.append(i);used.add(h);count+=1
            if count==2:break
    assert [p['row_index'] for p in fr['selected']]==[selected[0],selected[2],selected[1],selected[3]]
    qualified=[];cases=[]
    for j,p in enumerate(fr['selected']):
        i=p['row_index'];assert p['prompt']==data[i]['prompt'] and set(p['answer_nodes'])==graph_answer(p)
        actual=[(r['regime'],r['seed']) for r in rs if r['row_index']==i];assert actual==[tuple(x) for x in fr['orders'][j]]
        row={'row_index':i,'type':p['problem_type'],'gold_size':len(p['answer_nodes'])}
        for regime in ['greedy','recommended']:
            a=[index[(i,regime,seed)] for seed in fr['seeds']]
            row[regime+'_f1']=sum(r['score']['f1'] for r in a)/2;row[regime+'_em']=sum(r['score']['em'] for r in a)/2;row[regime+'_valid']=sum(r['score']['valid'] for r in a)
            for r in a:
                terminal_score(r,p['answer_nodes']);req=r['request'];assert req['model']==fr['identity']['name'] and req['think'] is True and req['messages'][1]['content']==p['prompt']
                opts={'num_ctx':32768,'num_predict':8192,'seed':r['seed'],**fr['regimes'][regime]};assert req['options']==opts
                assert r['request_sha256']==hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
                system=req['messages'][0]['content'];x='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+p['prompt']+' /think<|im_end|>\n<|im_start|>assistant\n';n=len(tokenizer.encode(x,add_special_tokens=False).ids);assert n==r['expected_input_tokens'] and n+8192+64<=32768
                ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==req and ledger['result']['raw']==r['raw'] and ledger['result']['error']==r['error'] and r['attempts']==1
                runtime=[m for m in r.get('runtime_ps',{}).get('models',[]) if m.get('digest')==fr['identity']['digest']];qualified.append({'row_index':i,'regime':regime,'seed':r['seed'],'qualified':r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768,'api_input_matches':r['api_input_matches']})
            assert a[0]['request']['messages']==index[(i,'recommended' if regime=='greedy' else 'greedy',a[0]['seed'])]['request']['messages']
        cases.append(row)
    ledgers=list(OUT.glob('attempt_*.json'));assert len(ledgers)==16 and all(json.loads(p.read_text(encoding='utf-8'))['completion']=='returned' for p in ledgers)
    summaries=[]
    for regime in ['greedy','recommended']:
        a=[r for r in rs if r['regime']==regime];summaries.append({'regime':regime,'graphs':4,'repeats_per_graph':2,'responses':8,'f1':sum(p[regime+'_f1'] for p in cases)/4,'em':sum(p[regime+'_em'] for p in cases)/4,'valid':sum(r['score']['valid'] for r in a),'length':sum(r['raw'].get('done_reason')=='length' for r in a),'errors':sum(bool(r['error']) for r in a),'known_tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a),'qualified':sum(q['qualified'] for q in qualified if q['regime']==regime),'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in a)})
    g,s=summaries;gate=all(q['qualified'] for q in qualified) and s['valid']==8 and all(p['recommended_f1']>=p['greedy_f1'] for p in cases) and s['em']>=g['em'] and sum(p['recommended_f1']>p['greedy_f1'] for p in cases)>=2 and s['known_tokens']<=g['known_tokens']
    result={'summary':summaries,'case_table':cases,'qualified':sum(q['qualified'] for q in qualified),'candidate_gate':gate,'always_empty_baseline_f1_em':sum(not p['answer_nodes'] for p in fr['selected'])/4,'graph_unit_not_repeat_unit':True,'not_temperature_only_ablation':True,'raw_history_preserved':True,'unknown_usage_calls':sum(s['unknown_usage'] for s in summaries)};save(OUT/'summary.json',result);save(OUT/'independent_audit.json',{'passed':True,'calls':16,'paired_graphs':4,'input_qualification':qualified,'selection_reconstructed':True,'visible_scores_recomputed':16,'ledgered_responses':16,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']+ledgers}})
    for name,rows in [('official_graph_sampling_control',summaries),('official_graph_sampling_control_cases',cases)]:
        with (ROOT/'paper/figures/data'/(name+'.csv')).open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
