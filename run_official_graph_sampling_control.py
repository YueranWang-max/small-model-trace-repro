"""Frozen decoding-regime control on new official graphs; no trace feedback."""
import itertools,json,time,urllib.request,urllib.error
from pathlib import Path
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,OPENER,sha,save,score,oracle
from run_official_graph_long_boundary import input_count
OUT=ROOT/'logs/official_graph_sampling_control'
PROTOCOL=ROOT/'protocol_official_graph_sampling_control_20261003.md'
MODEL='qwen3:1.7b';CTX=32768;CAP=8192
SEEDS=[202610037,202610038]
REGIMES={'greedy':{'temperature':0.,'top_p':.9,'top_k':40,'min_p':0.,'repeat_penalty':1.},'recommended':{'temperature':.6,'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}}
def call(user,regime,seed):
    opts={'num_ctx':CTX,'num_predict':CAP,'seed':seed,**REGIMES[regime]};req={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],'stream':False,'think':True,'options':opts};n=input_count(SYSTEM,user);assert n+CAP+64<=CTX
    ordinal=len(list(OUT.glob('attempt_*.json')))+1;ledger=OUT/f'attempt_{ordinal:03d}.json';entry={'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),'expected_input_tokens':n,'transmission':'not_yet_attempted','completion':'pending'};save(ledger,entry);entry['transmission']='attempting_once';save(ledger,entry);raw={};err=None;body=None;t=time.perf_counter()
    try:
        rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(rq,timeout=900) as h:raw=json.load(h)
    except urllib.error.HTTPError as e:err=repr(e);body=e.read().decode('utf-8',errors='replace')
    except Exception as e:err=repr(e)
    result={'request':req,'request_sha256':entry['request_sha256'],'raw':raw,'error':err,'http_error_body':body,'attempts':1,'wall_sec':time.perf_counter()-t,'expected_input_tokens':n,'max_input_output_tokens':n+CAP,'api_input_matches':raw.get('prompt_eval_count')==n,'attempt_ledger':str(ledger.relative_to(ROOT))}
    try:
        with OPENER.open('http://127.0.0.1:11434/api/ps',timeout=15) as h:result['runtime_ps']=json.load(h)
    except Exception as e:result['runtime_ps_error']=repr(e)
    entry.update({'completion':'returned','result':result});save(ledger,entry);return result
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_native_probe.py','run_official_graph_long_boundary.py','run_official_graph_native_order.py','data/qwen3_tokenizer_official/tokenizer.json','data/qwen3_tokenizer_official/README.md','data/qwen3_tokenizer_official/generation_config.json']}
    if not (OUT/'frozen.json').exists():
        gc=json.loads((ROOT/'data/qwen3_tokenizer_official/generation_config.json').read_text(encoding='utf-8'));assert gc['temperature']==.6 and gc['top_p']==.95 and gc['top_k']==20
        rows=pq.read_table(DATA).to_pylist();used=set()
        for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary']:
            for r in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(r['prompt'].rsplit('Operation:',1)[0].encode()))
        ps=[]
        for typ in ['parents','bfs']:
            count=0
            for i,r in enumerate(rows):
                h=sha(r['prompt'].rsplit('Operation:',1)[0].encode())
                if r['problem_type']!=typ or not 7000<=r['prompt_chars']<9000 or h in used:continue
                assert set(oracle(r))==set(r['answer_nodes']);ps.append({'row_index':i,'graph_head_sha256':h,**r});used.add(h);count+=1
                if count==2:break
            assert count==2
        ps=[ps[0],ps[2],ps[1],ps[3]];base=list(itertools.product(['greedy','recommended'],SEEDS));orders=[base[i:]+base[:i] for i in range(4)]
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        identity=next(m for m in inventory['models'] if m['name']==MODEL);assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
        save(OUT/'frozen.json',{'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':ps,'regimes':REGIMES,'seeds':SEEDS,'orders':orders,'ctx':CTX,'cap':CAP,'identity':identity,'model_inventory':inventory,'source':'Official Qwen3-1.7B model card/generation_config, recommendation excludes greedy thinking','selection':'first2 each type long-bin disjoint graph heads from all previous official runs','same_repeat_penalty_in_both_regimes':1.,'not_a_single_parameter_ablation':True,'old_implicit_repeat_penalty_not_recreated':True})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert fr['dependencies']==deps and fr['data_sha256']==sha(DATA.read_bytes())
    assert not (OUT/'results.json').exists() and not list(OUT.glob('attempt_*.json'))
    records=[]
    for i,p in enumerate(fr['selected']):
        for regime,seed in fr['orders'][i]:
            r={'model':MODEL,'row_index':p['row_index'],'regime':regime,'seed':seed,**call(p['prompt'],regime,seed)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
            print(i+1,p['row_index'],regime,seed,r['raw'].get('done_reason'),r['score'],'input_match',r['api_input_matches'],flush=True)
    print('Complete16 matched decoding-regime calls',flush=True)
if __name__=='__main__':main()
