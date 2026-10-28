"""Frozen new-graph native order readout. Never filter by producer correctness."""
import itertools,json
from pathlib import Path
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT,DATA,OPENER,sha,save,call,score,oracle
from run_official_graph_native_order import serialize
OUT=ROOT/'logs/official_graph_order_replication'
PROTOCOL=ROOT/'protocol_official_graph_order_replication_20261003.md'
MODEL='qwen3:1.7b'
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={n:sha((ROOT/n).read_bytes()) for n in [Path(__file__).name,'run_official_graph_native_probe.py','run_official_graph_native_order.py',PROTOCOL.name]}
    if not (OUT/'frozen.json').exists():
        rows=pq.read_table(DATA).to_pylist();old=[]
        for name in ['official_graph_native_probe','official_graph_budget_extension']:
            old+=json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['selected']
        used={sha(r['prompt'].rsplit('Operation:',1)[0].encode()) for r in old};selected=[]
        for typ in ['parents','bfs']:
            n=0
            for i,r in enumerate(rows):
                h=sha(r['prompt'].rsplit('Operation:',1)[0].encode())
                if r['problem_type']!=typ or not 2000<=r['prompt_chars']<3500 or h in used:continue
                assert set(oracle(r))==set(r['answer_nodes']);selected.append({'row_index':i,'graph_head_sha256':h,**r});used.add(h);n+=1
                if n==4:break
            assert n==4
        # Alternate tasks, rather than run all parents then all BFS.
        ps=[selected[j+offset] for j in range(4) for offset in [0,4]]
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        orders=list(itertools.permutations(['pre','post','reread']))
        save(OUT/'frozen.json',{'selected':ps,'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'model_inventory':inventory,'model':MODEL,'cap':8192,'orders':[list(orders[i%6]) for i in range(8)],'selection':'first4 each type short-bin with graph-head hashes disjoint both earlier probes and each other'})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert fr['dependencies']==deps and fr['data_sha256']==sha(DATA.read_bytes())
    assert not (OUT/'results.json').exists()
    records=[]
    for i,p in enumerate(fr['selected']):
        x=p['prompt'];head,op=x.rsplit('Operation:',1)
        source={'model':MODEL,'row_index':p['row_index'],'arm':'source',**call(MODEL,x,8192)};source['score']=score(source['raw'],p['answer_nodes'],source['error']);records.append(source);save(OUT/'results.json',records)
        # Even erroneous/incomplete/empty reasoning stays as-is, with no gold correction.
        T=serialize((source['raw'].get('message') or {}).get('thinking','') or '')
        prompts={'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op,'reread':x}
        for arm in fr['orders'][i]:
            r={'model':MODEL,'row_index':p['row_index'],'arm':arm,'trace_sha256':sha(T.encode()),'source_request_sha256':source['request_sha256'],**call(MODEL,prompts[arm],8192)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
            print(i+1,p['row_index'],arm,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete32 native new-graph calls',flush=True)
if __name__=='__main__':main()
