"""Only the native reasoning-mode flag differs between paired official graph calls."""
import json,time,urllib.request,urllib.error
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,OPENER,sha,save,score,oracle
from run_official_graph_sampling_control import REGIMES
from run_official_graph_five_trace import FORMAT
OUT=ROOT/'logs/official_graph_reasoning_mode'
PROTOCOL=ROOT/'protocol_official_graph_reasoning_mode_20261003.md'
MODEL='qwen3:1.7b';CTX=32768;CAP=8192
TOKENIZER=None
def render(user,think):
    return '<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'+user+(' /think' if think else ' /no_think')+'<|im_end|>\n<|im_start|>assistant\n'+('' if think else '<think>\n\n</think>\n\n')
def call(user,think,seed):
    req={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],'stream':False,'think':think,'options':{'num_ctx':CTX,'num_predict':CAP,'seed':seed,**REGIMES['recommended']}}
    n=len(TOKENIZER.encode(render(user,think),add_special_tokens=False).ids);assert n+CAP+64<=CTX
    ledger=OUT/f'attempt_{len(list(OUT.glob("attempt_*.json")))+1:03d}.json';entry={'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),'expected_input_tokens':n,'transmission':'not_yet_attempted','completion':'pending'};save(ledger,entry);entry['transmission']='attempting_once';save(ledger,entry)
    raw={};err=None;body=None;t=time.perf_counter()
    try:
        rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(rq,timeout=900) as response:raw=json.load(response)
    except urllib.error.HTTPError as e:err=repr(e);body=e.read().decode('utf-8',errors='replace')
    except Exception as e:err=repr(e)
    result={'request':req,'request_sha256':entry['request_sha256'],'raw':raw,'error':err,'http_error_body':body,'attempts':1,'wall_sec':time.perf_counter()-t,'expected_input_tokens':n,'max_input_output_tokens':n+CAP,'api_input_matches':raw.get('prompt_eval_count')==n,'attempt_ledger':str(ledger.relative_to(ROOT))}
    try:
        with OPENER.open('http://127.0.0.1:11434/api/ps',timeout=15) as response:result['runtime_ps']=json.load(response)
    except Exception as e:result['runtime_ps_error']=repr(e)
    entry.update({'completion':'returned','result':result});save(ledger,entry);return result
def main():
    global TOKENIZER
    OUT.mkdir(exist_ok=True,parents=True);assert not (OUT/'frozen.json').exists() and not list(OUT.glob('attempt_*.json')) and not (OUT/'results.json').exists()
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_native_probe.py','run_official_graph_sampling_control.py','run_official_graph_five_trace.py','data/qwen3_tokenizer_official/tokenizer.json','logs/official_graph_tokenizer/ollama_show.json']}
    TOKENIZER=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'));used=set()
    for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace','official_graph_identity_recode']:
        for p in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(p['prompt'].rsplit('Operation:',1)[0].encode()))
    rows=pq.read_table(DATA).to_pylist();selected=[]
    for lo,hi in [(2000,3500),(7000,9000)]:
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(rows):
                digest=sha(p['prompt'].rsplit('Operation:',1)[0].encode())
                if p['problem_type']!=typ or not lo<=p['prompt_chars']<hi or digest in used:continue
                used.add(digest);assert set(oracle(p))==set(p['answer_nodes']);selected.append({'row_index':i,'length_bin':[lo,hi],'graph_head_sha256':digest,**p});count+=1
                if count==4:break
            assert count==4
    selected=[selected[j] for i in range(4) for j in [i,4+i,8+i,12+i]]
    with OPENER.open('http://127.0.0.1:11434/api/tags') as response:inventory=json.load(response)
    identity=next(m for m in inventory['models'] if m['name']==MODEL);assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
    fr={'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':selected,'seeds':[202610070,202610071],'orders':[[['on','off'] if (i+j)%2==0 else ['off','on'] for j in range(2)] for i in range(16)],'identity':identity,'model_inventory':inventory,'regime':REGIMES['recommended'],'ctx':CTX,'cap':CAP,'format_suffix':FORMAT,'deployment_route':{'parents':'off','bfs':'on'},'selection':'first4 per type per bin, graph-head disjoint seven earlier stages'};save(OUT/'frozen.json',fr);rs=[]
    for i,p in enumerate(selected):
        for j,seed in enumerate(fr['seeds']):
            for arm in fr['orders'][i][j]:
                r={'model':MODEL,'row_index':p['row_index'],'arm':arm,'repeat':j,'seed':seed,**call(p['prompt']+'\n'+FORMAT,arm=='on',seed)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);rs.append(r);save(OUT/'results.json',rs);print(i+1,p['row_index'],arm,j,r['raw'].get('done_reason'),r['score'],'input_match',r['api_input_matches'],flush=True)
                if not r['error']:assert r['api_input_matches'],'Stop after preserving an input calibration mismatch; do not overwrite ledger.'
    print('Complete64 paired reasoning-mode calls; 16 independent graphs.',flush=True)
if __name__=='__main__':main()
