"""Separate partial audit. Preserve the interrupted primary records and never impute missing cells."""
import json,re,hashlib
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from audit_official_graph_indexed_source import read_graph,answer,ARMS,MARKER
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_indexed_source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    assert len(rs)==94 and all(sha(ROOT/n)==v for n,v in fr['dependencies'].items())
    planned=[(p['row_index'],arm,j) for i,p in enumerate(fr['selected']) for j in range(2) for arm in fr['orders'][i][j]]
    assert [(r['row_index'],r['arm'],r['repeat']) for r in rs]==planned[:94]
    pending=json.loads((OUT/'attempt_095.json').read_text(encoding='utf-8'))
    assert pending['completion']=='pending' and pending['transmission']=='attempting_once' and 'result' not in pending
    pending_pid,pending_arm,pending_repeat=planned[94]
    pending_case=next(p for p in fr['selected'] if p['row_index']==pending_pid)
    pending_enc,pending_mode=pending_arm.rsplit('_',1)
    expected_pending_request={'model':fr['identity']['name'],'messages':[{'role':'system','content':fr['system']},{'role':'user','content':pending_case['prompts'][pending_enc]}],'think':pending_mode=='on','stream':False,'options':{'num_ctx':32768,'num_predict':8192,'seed':fr['seeds'][pending_repeat],**fr['regime']}}
    assert pending['request']==expected_pending_request
    assert pending['request_sha256']==hashlib.sha256(json.dumps(expected_pending_request,sort_keys=True).encode()).hexdigest()
    assert not (OUT/'attempt_096.json').exists()
    dp=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet';assert sha(dp)==fr['data_sha256']
    data=pq.read_table(dp).to_pylist();tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
    cases=[];qualified=[]
    for i,p in enumerate(fr['selected']):
        assert p['prompt']==data[p['row_index']]['prompt']
        x=(p['prompt']+'\n'+fr['format_suffix']).replace(MARKER,fr['glossary']+'\n'+MARKER,1)
        assert x==p['prompts']['original'];ee,nn,prefix,op,prose=read_graph(x);gold=answer(ee,op,p['problem_type']);assert gold==set(p['answer_nodes'])
        for enc in ['original','compact','indexed']:
            inv=p['inverse']['original' if enc=='original' else 'compact'];mp=p['maps']['original' if enc=='original' else 'compact'];assert inv=={v:k for k,v in mp.items()}
            edges,nodes,pp,qq,text=read_graph(p['prompts'][enc],enc=='indexed')
            restore=lambda s:re.sub(r'\b(?:'+'|'.join(sorted(map(re.escape,inv),key=len,reverse=True))+r')\b',lambda m:inv[m[0]],s)
            assert Counter((inv[u],inv[v]) for u,v in edges)==Counter(ee) and {inv[n] for n in nodes}==nn
            assert restore(pp)==prefix and restore(qq)==op and text==prose and {inv[n] for n in answer(edges,qq,p['problem_type'])}==gold
            if enc=='compact':assert restore(p['prompts'][enc])==x
        records=[r for r in rs if r['row_index']==p['row_index']]
        for r in records:
            enc,mode=r['arm'].rsplit('_',1);flag=mode=='on';q=r['request'];user=p['prompts'][enc];inv=p['inverse']['original' if enc=='original' else 'compact']
            assert q['messages']==[{'role':'system','content':fr['system']},{'role':'user','content':user}] and q['think'] is flag and q['model']==fr['identity']['name']
            assert q['options']=={'num_ctx':32768,'num_predict':8192,'seed':fr['seeds'][r['repeat']],**fr['regime']}
            assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
            text=(r['raw'].get('message') or {}).get('content','') or '';m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',text.rstrip().split('\n')[-1].strip())
            pred=[n.strip() for n in m[1].split(',') if n.strip()] if m else [];mapped=[inv[n] if n in inv else '<unknown-node:'+n+'>' for n in pred];unknown=[n for n in pred if n not in inv]
            valid=bool(m) and not r['error'] and r['raw'].get('done_reason')=='stop';aset=set(mapped);f1=(2*len(aset&gold)/(len(aset)+len(gold)) if aset or gold else 1) if valid else 0
            assert dict(prediction=pred,restored_prediction=mapped,unknown_prediction_nodes=unknown,valid=bool(valid),em=float(valid and aset==gold),f1=f1)==r['score']
            rendered='<|im_start|>system\n'+fr['system']+'<|im_end|>\n<|im_start|>user\n'+user+(' /think' if flag else ' /no_think')+'<|im_end|>\n<|im_start|>assistant\n'+('' if flag else '<think>\n\n</think>\n\n')
            count=len(tok.encode(rendered,add_special_tokens=False).ids);assert count==r['expected_input_tokens'] and count+8192+64<=32768
            ledger=json.loads((ROOT/r['attempt_ledger']).read_text(encoding='utf-8'));assert ledger['completion']=='returned' and ledger['request']==q and ledger['result']['raw']==r['raw'] and r['attempts']==1
            runtime=[v for v in r.get('runtime_ps',{}).get('models',[]) if v.get('digest')==fr['identity']['digest']]
            qualified.append(bool(r['api_input_matches'] and len(runtime)==1 and runtime[0].get('context_length',0)>=32768))
        cases.append(dict(row_index=p['row_index'],type=p['problem_type'],length_bin=p['length_bin'],gold_size=len(gold)))
    summary=[]
    for arm in ARMS:
        a=[r for r in rs if r['arm']==arm];missing=16-len(a)
        row=dict(arm=arm,planned_responses=16,known_responses=len(a),missing_cells=missing,normal_parse=sum(r['score']['valid'] for r in a),length_stops=sum(r['raw'].get('done_reason')=='length' for r in a),unknown_node_responses=sum(bool(r['score']['unknown_prediction_nodes']) for r in a),known_token_lower_bound=sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a))
        for metric in ['f1','em']:
            total=sum(r['score'][metric] for r in a);row[metric+'_available_mean']=total/len(a);row[metric+'_missingness_bounds']=[total/16,(total+missing)/16]
        summary.append(row)
    by={r['arm']:r for r in summary};candidate=by['indexed_off']
    robust_failures=[]
    for arm in ['original_on','original_off']:
        if candidate['f1_missingness_bounds'][1]<by[arm]['f1_missingness_bounds'][0]+.10:robust_failures.append('Cannot meet F1 +.10 over '+arm+' even under favorable missing outcomes')
        if candidate['em_missingness_bounds'][1]<by[arm]['em_missingness_bounds'][0]:robust_failures.append('Cannot meet nonlower EM over '+arm)
    if candidate['f1_missingness_bounds'][1]<by['compact_off']['f1_missingness_bounds'][0]+.05:robust_failures.append('Cannot meet F1 +.05 over compact_off')
    if candidate['normal_parse']<15:robust_failures.append('Indexed_off reliability criterion fails')
    result=dict(partial_audit_passed=True,primary_complete=False,saved_calls=94,planned_calls=96,qualified_saved_calls=sum(qualified),attempt_095_outcome_and_usage='unknown',attempt_096='not_reached',missing_cells=planned[94:],summary=summary,cases=cases,known_token_lower_bound=sum(r['known_token_lower_bound'] for r in summary),robust_quality_gate_failures=robust_failures,followup_gate=False if robust_failures else 'not_determined_due_to_missingness',main_manuscript_updated=False,selection_reconstruction_pending=True,assets={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json']+list(OUT.glob('attempt_*.json'))})
    (OUT/'interrupted_independent_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['assets','cases']},indent=2))
if __name__=='__main__':main()
