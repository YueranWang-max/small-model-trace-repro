"""Independent archived-data audit. Run only after the inference process is terminal."""
import json,pathlib,hashlib,re,collections,csv
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment,StrictUndefined
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'hybrid_source_qualification';META=ROOT/'hybrid_candidate_qualification_20261004'
MODEL='qwen3.5:4b';DIGEST='2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
OPTIONS=dict(temperature=1.0,top_p=.95,top_k=20,min_p=0,presence_penalty=1.5,frequency_penalty=0,repeat_penalty=1,mirostat=0,num_ctx=65536,num_predict=32768)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def graph(text):
    body=text.split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0]
    edges=re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$',body,re.M)
    assert edges;return edges
def signature(text):return hashlib.sha256(json.dumps(sorted(graph(text)),separators=(',',':')).encode()).hexdigest()
def gold(case):
    edges=graph(case['prompt']);op=case['prompt'].rsplit('Operation:',1)[1]
    if case['problem_type']=='parents':
        node=re.search(r'Find the parents of node ([0-9a-f]{10})',op)[1]
        return {u for u,v in edges if v==node and u!=node}
    root,depth=re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',op).groups()
    adjacency=collections.defaultdict(set)
    for u,v in edges:adjacency[u].add(v)
    distances={root:0};queue=collections.deque([root])
    while queue:
        u=queue.popleft()
        for v in adjacency[u]:
            if v not in distances:distances[v]=distances[u]+1;queue.append(v)
    return {v for v,d in distances.items() if d==int(depth) and v!=root}
def raise_exception(message):raise ValueError(message)
def main():
    target=OUT/'independent_audit.json';assert not target.exists(),'Preserve completed audit evidence.'
    completion=json.loads((OUT/'completion.json').read_bytes())
    frozen=json.loads((OUT/'frozen.json').read_bytes())
    assert all(sha(pathlib.Path(p))==value for p,value in frozen['dependencies'].items())
    assert frozen['model']==MODEL and frozen['digest']==DIGEST and frozen['runtime']['version']=='0.35.1'
    assert sha(ROOT/'graphwalks_128k_and_shorter.parquet')=='54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    rows=pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist();assert len(rows)==750
    historical=set();historical_rows=set()
    for path in sorted((ROOT.parent/'logs').glob('official_graph_*/frozen.json')):
        archive=json.loads(path.read_bytes())
        for record in archive.get('selected',archive.get('cases',[])):
            case=record.get('paper_row',record);historical.add(signature(case['prompt']));historical_rows.add(case['row_index'])
    assert sorted(historical_rows)==frozen['historical_rows'] and len(historical)==58
    bridge=json.loads((ROOT/'bridge_run/frozen.json').read_bytes())['cases']
    assert [c['row_index'] for c in bridge]==frozen['bridge_rows']
    excluded=historical|{signature(c['prompt']) for c in bridge}
    selection=[]
    for lo,hi in [(2000,3500),(14000,16000)]:
        for kind in ['parents','bfs']:
            eligible=[i for i,r in enumerate(rows) if lo<=r['prompt_chars']<hi and r['problem_type']==kind and signature(r['prompt']) not in excluded]
            selection+=eligible[:2]
    cases=frozen['cases'];assert selection==[c['row_index'] for c in cases]
    assert len({signature(c['prompt']) for c in cases})==8
    for c in cases:
        assert {k:v for k,v in c.items() if k not in ['row_index','length_bin']}==rows[c['row_index']]
        assert gold(c)==set(c['answer_nodes'])
    env=Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols']);env.globals['raise_exception']=raise_exception
    template=env.from_string(json.loads((META/'tokenizer_config.json').read_bytes())['chat_template'])
    tok=Tokenizer.from_file(str(META/'tokenizer.json'))
    specs=[dict(phase='input_calibration',row_index=None,seed=202610100,cap=1,user='A directed graph has the nodes A and B and no edges. Find the parents of node B.\n'+FORMAT)]
    for c in cases:
        for seed in [202610101,202610102]:specs.append(dict(phase='scored_source',row_index=c['row_index'],seed=seed,cap=32768,user=c['prompt']+'\n'+FORMAT))
    assert len(specs)==len(frozen['schedule'])==17
    records=[]
    for ordinal,(spec,job) in enumerate(zip(specs,frozen['schedule']),1):
        payload=dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=spec['user'])],think=True,stream=False,options={**OPTIONS,'seed':spec['seed'],'num_predict':spec['cap']})
        rendered=template.render(messages=payload['messages'],tools=None,add_generation_prompt=True,enable_thinking=True)
        count=len(tok.encode(rendered,add_special_tokens=False).ids);assert count+spec['cap']+64<=65536
        expected=dict(ordinal=ordinal,phase=spec['phase'],row_index=spec['row_index'],request=payload,expected_prompt_tokens=count,request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
        assert expected==job
        path=OUT/f'attempt_{ordinal:03d}.json'
        r=dict(ordinal=ordinal,phase=spec['phase'],row_index=spec['row_index'],seed=spec['seed'],attempted=False,response_known=False,score_determined=False,usage_known=False,qualified=False)
        if not path.exists():records.append(r);continue
        rec=json.loads(path.read_bytes());assert rec['job']==job
        r.update(attempted=rec['transmission']=='attempting_once',completion=rec['completion'],ledger_sha256=sha(path),client_seconds=rec.get('elapsed_seconds'))
        if rec['completion']=='client_error':
            r.update(error=rec['error'],score_determined=True,f1=0.,em=0,normal_parse=False);records.append(r);continue
        if rec['completion']!='returned':records.append(r);continue
        raw=rec['raw'];r['response_known']=True
        ps=rec.get('runtime_ps',{}).get('models',[])
        input_ok=raw.get('prompt_eval_count')==count
        resident=any(m.get('digest')==DIGEST and m.get('context_length',0)>=65536 for m in ps)
        response_ok=raw.get('done') is True and raw.get('model')==MODEL and not raw.get('error')
        if 'input_qualified' in rec:assert input_ok==rec['input_qualified'] and resident==rec['resident_qualified'] and response_ok==rec['response_qualified']
        qualified=input_ok and resident and response_ok
        usage=all(isinstance(raw.get(k),int) and raw[k]>=0 for k in ['prompt_eval_count','eval_count'])
        if usage:assert raw['eval_count']<=spec['cap']
        text=(raw.get('message') or {}).get('content','') or '';line=text.rstrip().split('\n')[-1].strip()
        match=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',line)
        valid=bool(match) and raw.get('done_reason')=='stop' and not raw.get('error')
        prediction={v.strip() for v in match[1].split(',') if v.strip()} if valid else set()
        answer=gold(next(c for c in cases if c['row_index']==spec['row_index'])) if spec['phase']=='scored_source' else set()
        f1=(2*len(prediction&answer)/(len(prediction)+len(answer)) if prediction or answer else 1.) if valid else 0.
        r.update(qualified=qualified,usage_known=usage,score_determined=qualified,normal_parse=valid,f1=f1,em=int(valid and prediction==answer),prediction=sorted(prediction),done_reason=raw.get('done_reason'),prompt_tokens=raw.get('prompt_eval_count'),output_tokens=raw.get('eval_count'))
        for key in ['total_duration','load_duration','prompt_eval_duration','eval_duration']:r[key+'_seconds']=raw.get(key)/1e9 if isinstance(raw.get(key),int) else None
        records.append(r)
    scored=[r for r in records if r['phase']=='scored_source'];known=[r for r in scored if r['score_determined']];missing=16-len(known)
    bounds={m:[sum(r[m] for r in known)/16,(sum(r[m] for r in known)+missing)/16] for m in ['f1','em']}
    calibration=records[0];normal=sum(r.get('normal_parse',False) for r in known)
    gate=bool(completion['primary_complete'] and missing==0 and all(r['qualified'] and r['usage_known'] for r in scored) and calibration['qualified'] and calibration['usage_known'] and normal>=15 and bounds['f1'][0]>=.70 and bounds['em'][0]>=.50)
    cost=[r for r in records if r['usage_known']]
    result=dict(audit_passed=True,primary_complete=completion['primary_complete'],planned_scored_calls=16,planned_calibration_calls=1,known_scored_outcomes=len(known),unknown_scored_outcomes=missing,normal_parse=normal,primary_bounds=bounds,frozen_gate_passed=gate,known_research_tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in cost),unknown_attempted_usage_calls=sum(r['attempted'] and not r['usage_known'] for r in records),not_reached_calls=sum(not r['attempted'] for r in records),calibration_tokens=calibration['prompt_tokens']+calibration['output_tokens'] if calibration['usage_known'] else None,preparation_seconds=frozen['preparation_before_freeze_serialization_seconds'],records=records,limitations=['Catalogue length and full prompt length are distinct measures.','Calibration is outside scored source data but included in research cost.','Unqualified returned answers are diagnostic, not qualified primary source scores.','Model family/training/tokenizer/profile differences prevent an architecture-only or scale-only causal conclusion.','Source qualification does not establish successful trace transfer.'])
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    columns=['ordinal','phase','row_index','seed','attempted','response_known','score_determined','qualified','usage_known','normal_parse','f1','em','prompt_tokens','output_tokens','client_seconds']
    with (OUT/'audited_calls.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
