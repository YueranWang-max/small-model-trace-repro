"""Five source traces / five native repeats, with the original graph interface."""
import itertools,json
from pathlib import Path
import pyarrow.parquet as pq
import run_official_graph_sampling_control as client
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,OPENER,sha,save,score,oracle
from run_official_graph_native_order import PD
OUT=ROOT/'logs/official_graph_five_trace'
PROTOCOL=ROOT/'protocol_official_graph_five_trace_20261003.md'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
def serialize(traces):
    a=[PD,'<first_run_reasoning_traces>']
    for i,s in enumerate(traces,1):
        if len(s)>50000:s=s[:50000]+'...'
        a.append(f'[Trace {i}] '+s.strip())
    return '\n'.join(a+['</first_run_reasoning_traces>'])
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,PROTOCOL.name,'run_official_graph_sampling_control.py','run_official_graph_long_boundary.py','run_official_graph_native_probe.py','run_official_graph_native_order.py','data/qwen3_tokenizer_official/tokenizer.json','data/qwen3_tokenizer_official/generation_config.json']}
    if not (OUT/'frozen.json').exists():
        used=set()
        for name in ['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control']:
            for r in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(sha(r['prompt'].rsplit('Operation:',1)[0].encode()))
        rows=pq.read_table(DATA).to_pylist();ps=[]
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(rows):
                h=sha(p['prompt'].rsplit('Operation:',1)[0].encode())
                if p['problem_type']!=typ or not 2000<=p['prompt_chars']<3500 or h in used:continue
                assert set(oracle(p))==set(p['answer_nodes']);ps.append({'row_index':i,'graph_head_sha256':h,**p});used.add(h);count+=1
                if count==2:break
            assert count==2
        ps=[ps[0],ps[2],ps[1],ps[3]];orders=list(itertools.permutations(['pre','post','reread']))
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        identity=next(m for m in inventory['models'] if m['name']=='qwen3:1.7b');assert identity['digest']=='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
        save(OUT/'frozen.json',{'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':ps,'source_seeds':[202610040+j for j in range(5)],'reader_seeds':[202610050+j for j in range(5)],'orders':[[list(orders[(i+j)%6]) for j in range(5)] for i in range(4)],'identity':identity,'model_inventory':inventory,'regime':client.REGIMES['recommended'],'ctx':32768,'source_cap':4096,'reader_cap':8192,'original_format_suffix':FORMAT,'trace_count':5,'reader_repeat_count':5,'selection':'first2 each type short-bin graph heads disjoint all five earlier stages'})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert fr['dependencies']==deps and fr['data_sha256']==sha(DATA.read_bytes());assert not (OUT/'results.json').exists() and not list(OUT.glob('attempt_*.json'))
    # Configure the existing durable HTTP client in this process only. No frozen
    # source file or earlier study output is changed.
    client.OUT=OUT;client.CTX=32768;records=[]
    for i,p in enumerate(fr['selected']):
        x=p['prompt']+'\n'+FORMAT;head,op=x.rsplit('Operation:',1);traces=[];source_ids=[];client.CAP=4096
        for j,seed in enumerate(fr['source_seeds']):
            r={'model':'qwen3:1.7b','row_index':p['row_index'],'arm':'source','repeat':j,'seed':seed,**client.call(x,'recommended',seed)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records);traces.append((r['raw'].get('message') or {}).get('thinking','') or '');source_ids.append(r['request_sha256']);print(i+1,p['row_index'],'source',j,r['raw'].get('done_reason'),r['score'],flush=True)
        T=serialize(traces);prompts={'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op,'reread':x};client.CAP=8192
        for j,seed in enumerate(fr['reader_seeds']):
            for arm in fr['orders'][i][j]:
                r={'model':'qwen3:1.7b','row_index':p['row_index'],'arm':arm,'repeat':j,'seed':seed,'trace_sha256':sha(T.encode()),'source_request_sha256s':source_ids,**client.call(prompts[arm],'recommended',seed)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records);print(i+1,p['row_index'],arm,j,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete80 native calls; 4 graph units, not20 independent reader cases',flush=True)
if __name__=='__main__':main()
