import pathlib,json,hashlib,re,urllib.request,time,collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'bridge_run'
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
def save(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def graph(prompt):
    body=prompt.split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0]
    edges=re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})$',body,re.M)
    assert edges
    return body,edges
def recompute(row):
    text=row['prompt'];body,edges=graph(text);op=text.rsplit('Operation:',1)[1]
    if row['problem_type']=='parents':
        target=re.search(r'Find the parents of node ([0-9a-f]{10})',op).group(1)
        return {u for u,v in edges if v==target and u!=target}
    target,depth=re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',op).groups();depth=int(depth)
    adj=collections.defaultdict(list)
    for u,v in edges:adj[u].append(v)
    dist={target:0};q=collections.deque([target])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if v not in dist:dist[v]=dist[u]+1;q.append(v)
    return {v for v,d in dist.items() if d==depth and v!=target}
def request(path,payload=None,timeout=90):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=timeout) as response:return json.load(response)
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def main():
    assert not OUT.exists(),'Never overwrite or resume automatically; inspect existing ledger first.'
    rows=pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist()
    assert len(rows)==750
    assert sha(ROOT/'graphwalks_128k_and_shorter.parquet')=='54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    for row in rows:assert recompute(row)==set(row['answer_nodes'])
    prior=list(range(16))+list(range(100,113))+list(range(400,416))+list(range(450,463))
    excluded={hashlib.sha256(graph(rows[i]['prompt'])[0].encode()).hexdigest() for i in prior}
    chosen=[]
    for lo,hi in [(2000,3500),(14000,16000)]:
        for typ in ['parents','bfs']:
            matches=[i for i,r in enumerate(rows) if lo<=r['prompt_chars']<hi and r['problem_type']==typ and hashlib.sha256(graph(r['prompt'])[0].encode()).hexdigest() not in excluded]
            assert len(matches)>=4
            for i in matches[:4]:chosen.append(dict(row_index=i,bin=[lo,hi],**rows[i]))
    tok=Tokenizer.from_file(str(ROOT/'tokenizer.json'))
    models=['qwen3:1.7b','qwen3:8b'];tags=request('tags');digests={m:next(x['digest'] for x in tags['models'] if x['name']==m) for m in models}
    a=json.loads((ROOT/'bridge_metadata_qwen3_1_7b.json').read_text(encoding='utf-8'));b=json.loads((ROOT/'bridge_metadata_qwen3_8b.json').read_text(encoding='utf-8'))
    assert a['template']==b['template'] and a['model_info']['tokenizer.ggml.tokens']==b['model_info']['tokenizer.ggml.tokens']
    vocab=tok.get_vocab();assert all(a['model_info']['tokenizer.ggml.tokens'][v]==k for k,v in vocab.items())
    schedule=[]
    for n,case in enumerate(chosen):
        user=case['prompt']+'\n'+FORMAT
        rendered='<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'+user+' /think<|im_end|>\n<|im_start|>assistant\n'
        expected=len(tok.encode(rendered,add_special_tokens=False).ids);assert expected+8192+64<=32768
        for model in (models if n%2==0 else models[::-1]):
            for seed in [202610090,202610091]:schedule.append(dict(row_index=case['row_index'],model=model,seed=seed,user=user,expected_prompt_tokens=expected,gold=case['answer_nodes']))
    OUT.mkdir();save(OUT/'frozen.json',dict(protocol_sha256=sha(ROOT/'bridge_protocol.md'),runner_sha256=sha(pathlib.Path(__file__)),digests=digests,cases=chosen,schedule=schedule,historical_manifest_verification='pending D drive restoration',all_750_labels_recomputed=True))
    for ordinal,job in enumerate(schedule,1):
        path=OUT/f'attempt_{ordinal:03d}.json';record=dict(ordinal=ordinal,job=job,status='not_yet_attempted');save(path,record)
        record.update(status='attempting_once',start_time=time.time());save(path,record)
        payload=dict(model=job['model'],messages=[dict(role='system',content=SYSTEM),dict(role='user',content=job['user'])],think=True,stream=False,options=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,repeat_penalty=1,num_ctx=32768,num_predict=8192,seed=job['seed']))
        try:record['response']=request('chat',payload,900);record['status']='returned'
        except Exception as e:record.update(status='request_error',error=repr(e))
        record['end_time']=time.time();save(path,record)
        if 'response' in record:
            raw=record['response'];record['input_qualified']=raw.get('prompt_eval_count')==job['expected_prompt_tokens'];save(path,record)
            ps=request('ps');save(OUT/f'ps_{ordinal:03d}.json',ps)
            active=[m for m in ps.get('models',[]) if m.get('name')==job['model']]
            record['resident_qualified']=any(m.get('digest')==digests[job['model']] and m.get('context_length',0)>=32768 for m in active);save(path,record)
            print(ordinal,job['row_index'],job['model'],raw.get('done_reason'),raw.get('prompt_eval_count'),raw.get('eval_count'),flush=True)
            assert record['input_qualified'] and record['resident_qualified'],'Qualification failed; preserve ledger and stop.'
        else:print(ordinal,'request_error',flush=True)
    save(OUT/'completion.json',dict(saved_attempts=64,finished_at=time.time()))
if __name__=='__main__':main()
