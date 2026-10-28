"""Adaptive, separately frozen output-budget extension; prior probe unchanged."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT,DATA,MODELS,OPENER,sha,save,call,score,oracle
OUT=ROOT/'logs/official_graph_budget_extension'
PROTOCOL=ROOT/'protocol_official_graph_budget_extension_20261003.md'
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    deps={Path(__file__).name:sha(Path(__file__).read_bytes()),'run_official_graph_native_probe.py':sha((ROOT/'run_official_graph_native_probe.py').read_bytes())}
    if not (OUT/'frozen.json').exists():
        rows=pq.read_table(DATA).to_pylist();old=json.loads((ROOT/'logs/official_graph_native_probe/frozen.json').read_text(encoding='utf-8'));used={x['row_index'] for x in old['selected']};ps=[]
        for typ in ['parents','bfs']:
            i,r=next((i,r) for i,r in enumerate(rows) if i not in used and r['problem_type']==typ and 2000<=r['prompt_chars']<3500)
            assert set(oracle(r))==set(r['answer_nodes']);ps.append({'row_index':i,**r})
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        save(OUT/'frozen.json',{'selected':ps,'data_sha256':sha(DATA.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'dependencies':deps,'caps':[4096,8192],'model_inventory':inventory,'adaptive_reason':'preceding16-call interface gates all failed; choose next unused rows without score selection'})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert fr['dependencies']==deps and fr['data_sha256']==sha(DATA.read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes())
    # No resume is supported: an unexpected interruption requires a separate deviation ledger.
    assert not (OUT/'results.json').exists()
    records=[]
    for model in MODELS:
        for p in fr['selected']:
            for cap in fr['caps']:
                r={'model':model,'row_index':p['row_index'],'cap':cap,**call(model,p['prompt'],cap)}
                r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
                print(model,p['row_index'],cap,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete8 separately frozen calls',flush=True)
if __name__=='__main__':main()
