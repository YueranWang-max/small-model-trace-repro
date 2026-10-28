"""Prespecified native reasoning interface probe on unchanged official prompts."""
import hashlib,json,re,time,urllib.request,urllib.error
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'logs/official_graph_native_probe'
DATA=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet'
PROTOCOL=ROOT/'protocol_official_graph_native_probe_20261003.md'
MODELS=['qwen3.5:4b','qwen3:1.7b']
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def score(raw,gold,error=None):
    # Visible terminal line only; never salvage an answer from thinking.
    c=(raw.get('message') or {}).get('content','') or ''
    line=c.rstrip().split('\n')[-1]
    m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',line.strip())
    valid=bool(m) and not error and raw.get('done_reason')=='stop'
    pred=[a.strip() for a in m.group(1).split(',') if a.strip()] if m else []
    a,b=set(pred),set(gold)
    f=2*len(a&b)/(len(a)+len(b)) if a or b else 1.
    return {'prediction':pred,'valid':valid,'em':float(valid and a==b),'f1':f if valid else 0.}
def call(model,prompt,cap):
    req={'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}], 'stream':False,'think':True,'options':{'temperature':0.,'top_p':.9,'num_ctx':32768,'num_predict':cap}}
    t=time.perf_counter();raw={};err=None;body=None
    try:
        r=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(r,timeout=900) as h:raw=json.load(h)
    except urllib.error.HTTPError as e:err=repr(e);body=e.read().decode('utf-8',errors='replace')
    except Exception as e:err=repr(e)
    return {'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),'raw':raw,'error':err,'http_error_body':body,'attempts':1,'wall_sec':time.perf_counter()-t}
def oracle(row):
    head,operation=row['prompt'].rsplit('Operation:',1)
    text=head.split('Here is the graph to operate on:',1)[1]
    edges=re.findall(r'^([a-zA-Z0-9]+) -> ([a-zA-Z0-9]+)\s*$',text,re.M)
    if row['problem_type']=='parents':
        target=re.search(r'Find the parents of node (\w+)\.',operation)[1]
        return sorted({u for u,v in edges if v==target and u!=target})
    start=re.search(r'BFS from node (\w+)',operation)[1];depth=int(re.search(r'exactly depth (\d+)',operation)[1])
    adj={}
    for u,v in edges:adj.setdefault(u,set()).add(v)
    seen={start};front={start}
    for _ in range(depth):
        front={v for u in front for v in adj.get(u,())}-seen;seen|=front
    return sorted(front)
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if not (OUT/'frozen.json').exists():
        rows=pq.read_table(DATA).to_pylist();selected=[]
        for lo,hi in [(2000,3500),(7000,9000)]:
            for typ in ['parents','bfs']:
                i,r=next((i,r) for i,r in enumerate(rows) if r['problem_type']==typ and lo<=r['prompt_chars']<hi)
                selected.append({'row_index':i,**r})
        mismatches=[i for i,r in enumerate(rows) if set(oracle(r))!=set(r['answer_nodes'])]
        assert not mismatches,mismatches
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:models=json.load(h)
        save(OUT/'frozen.json',{'data_sha256':sha(DATA.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'runner_sha256':sha(Path(__file__).read_bytes()),'selected':selected,'oracle_audit':{'rows':len(rows),'mismatches':mismatches},'model_inventory':models,'caps':[1024,4096]})
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    assert fr['runner_sha256']==sha(Path(__file__).read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes()) and fr['data_sha256']==sha(DATA.read_bytes())
    records=json.loads((OUT/'results.json').read_text(encoding='utf-8')) if (OUT/'results.json').exists() else []
    done={(r['model'],r['row_index'],r['cap']) for r in records}
    for model in MODELS:
        for p in fr['selected']:
            for cap in fr['caps']:
                if (model,p['row_index'],cap) in done:continue
                r={'model':model,'row_index':p['row_index'],'cap':cap,**call(model,p['prompt'],cap)}
                r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
                print(model,p['row_index'],cap,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete',len(records),'native interface calls; not an order experiment.',flush=True)
if __name__=='__main__':main()
