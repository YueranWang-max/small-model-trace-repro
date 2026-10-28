"""Longer official-graph native budget/order/checking factorial pilot."""
import itertools,json,time,urllib.request,urllib.error
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from run_official_graph_native_probe import ROOT,DATA,OPENER,SYSTEM,sha,save,score,oracle
from run_official_graph_native_order import serialize
OUT=ROOT/'logs/official_graph_long_boundary'
PROTOCOL=ROOT/'protocol_official_graph_long_boundary_20261003.md'
MODEL='qwen3:1.7b';CTX=40960
CHECK=' Treat supplied reasoning as tentative. Independently recompute the requested result from the actual graph, using a fresh visited set for BFS. Before answering, verify that each listed node is at the requested shortest-path depth or has an actual edge to the target, as appropriate. Ignore answers suggested by examples or traces when unsupported.'
TOK=ROOT/'data/qwen3_tokenizer_official/tokenizer.json'
TOKENIZER=Tokenizer.from_file(str(TOK))
def input_count(system,user):
    # Archived Ollama template inserts /think. One/two system newlines have
    # identical token counts; counts calibrated against all32 prior responses.
    text='<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+' /think<|im_end|>\n<|im_start|>assistant\n'
    return len(TOKENIZER.encode(text,add_special_tokens=False).ids)
def request(model,system,user,cap):
    req={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'stream':False,'think':True,'options':{'temperature':0.,'top_p':.9,'num_ctx':CTX,'num_predict':cap}}
    n=input_count(system,user);guard=n+cap+64<=CTX;t=time.perf_counter();raw={};error=None;body=None
    # Persist a write-ahead request ledger, so process death cannot hide an attempt.
    ordinal=len(list(OUT.glob('attempt_*.json')))+1
    ledger=OUT/f'attempt_{ordinal:03d}.json';entry={'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),'expected_input_tokens':n,'max_output_tokens':cap,'context_guard':guard,'transmission':'not_yet_attempted','completion':'pending'};save(ledger,entry)
    if not guard:error='context_guard_failed';entry['completion']='skipped_before_transmission'
    else:
        entry['transmission']='attempting_once';save(ledger,entry)
        try:
            rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
            with OPENER.open(rq,timeout=900) as h:raw=json.load(h)
        except urllib.error.HTTPError as e:error=repr(e);body=e.read().decode('utf-8',errors='replace')
        except Exception as e:error=repr(e)
        entry['completion']='returned'
    result={'request':req,'request_sha256':entry['request_sha256'],'raw':raw,'error':error,'http_error_body':body,'attempts':int(guard),'wall_sec':time.perf_counter()-t,'expected_input_tokens':n,'max_input_output_tokens':n+cap,'context_guard':guard,'api_input_matches':raw.get('prompt_eval_count')==n,'attempt_ledger':str(ledger.relative_to(ROOT))}
    try:
        with OPENER.open('http://127.0.0.1:11434/api/ps',timeout=15) as h:result['runtime_ps']=json.load(h)
    except Exception as e:result['runtime_ps_error']=repr(e)
    # The ledger owns the full response even if process death precedes matrix save.
    entry['result']=result;save(ledger,entry);return result
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_native_probe.py','run_official_graph_native_order.py','data/qwen3_tokenizer_official/tokenizer.json','logs/official_graph_tokenizer/ollama_show.json']}
    if not (OUT/'frozen.json').exists():
        show=json.loads((ROOT/'logs/official_graph_tokenizer/ollama_show.json').read_text(encoding='utf-8'));local=show['model_info']['tokenizer.ggml.tokens'];vocab=TOKENIZER.get_vocab();assert all(i<len(local) and local[i]==s for s,i in vocab.items())
        prior=json.loads((ROOT/'logs/official_graph_order_replication/results.json').read_text(encoding='utf-8'));calibration=[input_count(r['request']['messages'][0]['content'],r['request']['messages'][1]['content'])==r['raw']['prompt_eval_count'] for r in prior];assert len(calibration)==32 and all(calibration)
        rows=pq.read_table(DATA).to_pylist();used=set()
        for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication']:
            for r in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(r['prompt'].rsplit('Operation:',1)[0].encode()))
        ps=[]
        for typ in ['parents','bfs']:
            selected=[]
            for i,r in enumerate(rows):
                h=sha(r['prompt'].rsplit('Operation:',1)[0].encode())
                if r['problem_type']!=typ or not 7000<=r['prompt_chars']<9000 or h in used:continue
                assert set(oracle(r))==set(r['answer_nodes']);selected.append({'row_index':i,'graph_head_sha256':h,**r});used.add(h)
                if len(selected)==2:break
            assert len(selected)==2;ps+=selected
        ps=[ps[0],ps[2],ps[1],ps[3]]
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        identity=next(m for m in inventory['models'] if m['name']==MODEL);assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
        orders=list(itertools.permutations(['pre','post','reread']))
        arm_orders=[]
        for i in range(4):
            plain=list(orders[i]);checks=['check_'+a for a in plain];arm_orders.append([x for pair in zip(plain,checks) for x in (pair if i%2==0 else pair[::-1])])
        save(OUT/'frozen.json',{'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':ps,'model_inventory':inventory,'identity':identity,'ctx':CTX,'source_caps':[8192,16384],'source_orders':[[8192,16384],[16384,8192],[16384,8192],[8192,16384]],'reader_cap':8192,'reader_orders':arm_orders,'check_system_suffix':CHECK,'tokenizer_calibration':{'vocabulary_mapping_matches':len(vocab),'prior_api_count_matches':sum(calibration),'prior_n':32,'render_byte_identity_not_asserted':True},'selection':'first2 each type long-bin graph heads disjoint all prior official studies'})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert fr['dependencies']==deps and fr['data_sha256']==sha(DATA.read_bytes())
    assert not (OUT/'results.json').exists() and not list(OUT.glob('attempt_*.json'))
    records=[]
    for i,p in enumerate(fr['selected']):
        x=p['prompt'];head,op=x.rsplit('Operation:',1);sources={}
        for cap in fr['source_orders'][i]:
            r={'model':MODEL,'row_index':p['row_index'],'arm':'source_'+str(cap),**request(MODEL,SYSTEM,x,cap)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);sources[cap]=r;records.append(r);save(OUT/'results.json',records)
            print(i+1,p['row_index'],r['arm'],r['raw'].get('done_reason'),r['score'],'input_match',r['api_input_matches'],flush=True)
        source=sources[16384];T=serialize((source['raw'].get('message') or {}).get('thinking','') or '');prompts={'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op,'reread':x}
        for arm in fr['reader_orders'][i]:
            base=arm.removeprefix('check_');system=SYSTEM+(CHECK if arm.startswith('check_') else '')
            r={'model':MODEL,'row_index':p['row_index'],'arm':arm,'trace_sha256':sha(T.encode()),'source_request_sha256':source['request_sha256'],**request(MODEL,system,prompts[base],8192)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
            print(i+1,p['row_index'],arm,r['raw'].get('done_reason'),r['score'],'input_match',r['api_input_matches'],flush=True)
    print('Complete32 calls; no failed source excluded',flush=True)
if __name__=='__main__':main()
