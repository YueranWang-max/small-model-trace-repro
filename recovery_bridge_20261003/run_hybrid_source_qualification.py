"""One-shot prospective source qualification. Do not launch during active inference."""
import pathlib,json,hashlib,re,collections,time,urllib.request
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parent
META=ROOT/'hybrid_candidate_qualification_20261004'
OUT=ROOT/'hybrid_source_qualification'
MODEL='qwen3.5:4b'
DIGEST='2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
OPTIONS=dict(temperature=1.0,top_p=.95,top_k=20,min_p=0,presence_penalty=1.5,
    frequency_penalty=0,repeat_penalty=1,mirostat=0,num_ctx=65536,num_predict=32768)
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def request(endpoint,payload=None,timeout=90):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+endpoint,
        data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=timeout) as r:return json.load(r)
def edges(prompt):
    block=prompt.split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0]
    es=re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$',block,re.M)
    assert es;return es
def signature(prompt):return hashlib.sha256(json.dumps(sorted(edges(prompt)),separators=(',',':')).encode()).hexdigest()
def oracle(row):
    es=edges(row['prompt']);op=row['prompt'].rsplit('Operation:',1)[1]
    if row['problem_type']=='parents':
        target=re.search(r'Find the parents of node ([0-9a-f]{10})',op)[1]
        return {u for u,v in es if v==target and u!=target}
    target,depth=re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',op).groups()
    visited={target};frontier={target}
    for _ in range(int(depth)):
        frontier={v for u,v in es if u in frontier and v not in visited};visited.update(frontier)
    return frontier-{target}
def rendered(user):
    return '<|im_start|>system\n'+SYSTEM.strip()+'<|im_end|>\n<|im_start|>user\n'+user.strip()+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
def main():
    preparation_started=time.monotonic()
    assert not OUT.exists(),'Never silently resume, overwrite, or repeat primary calls.'
    assert (ROOT/'bridge_continuation/completion.json').exists(),'Active bridge must finish first.'
    bridge_audit=json.loads((ROOT/'continued_bridge_audit.json').read_bytes())
    assert bridge_audit['known_outcomes']==63 and not bridge_audit['historical_manifest_check_pending']
    indexed_path=ROOT.parent/'logs/official_graph_indexed_source/interrupted_independent_audit.json'
    assert json.loads(indexed_path.read_bytes())['partial_audit_passed']
    assert json.loads((META/'offline_render_checks.json').read_bytes())['all_34_offline_renders_equal']
    assert json.loads((META/'padding_difference_checks.json').read_bytes())['all_checks_pass']
    assert sha(ROOT/'graphwalks_128k_and_shorter.parquet')=='54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    assert sha(META/'tokenizer.json')=='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
    rows=pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist();assert len(rows)==750
    hist=json.loads((ROOT/'historical_manifest_exclusion_check_20261004.json').read_bytes())
    bridge=json.loads((ROOT/'bridge_run/frozen.json').read_bytes())
    excluded=set(hist['bridge_graph_signatures'])|{signature(rows[i]['prompt']) for i in hist['historical_rows']}
    cases=[]
    for lo,hi in [(2000,3500),(14000,16000)]:
        for typ in ['parents','bfs']:
            available=[i for i,r in enumerate(rows) if lo<=r['prompt_chars']<hi and r['problem_type']==typ and signature(r['prompt']) not in excluded]
            assert len(available)>=2
            for index in available[:2]:cases.append(dict(row_index=index,length_bin=[lo,hi],**rows[index]))
    assert len({signature(c['prompt']) for c in cases})==8
    for c in cases:assert oracle(c)==set(c['answer_nodes'])
    tags=request('tags');version=request('version')
    assert next(m['digest'] for m in tags['models'] if m['name']==MODEL)==DIGEST
    assert version['version']=='0.35.1','Requalify changed runtime before new calls.'
    tokenizer=Tokenizer.from_file(str(META/'tokenizer.json'))
    specs=[dict(phase='input_calibration',row_index=None,seed=202610100,
        user='A directed graph has the nodes A and B and no edges. Find the parents of node B.\n'+FORMAT,cap=1)]
    for c in cases:
        for seed in [202610101,202610102]:specs.append(dict(phase='scored_source',row_index=c['row_index'],
            seed=seed,user=c['prompt']+'\n'+FORMAT,cap=32768))
    schedule=[]
    for ordinal,spec in enumerate(specs,1):
        payload=dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=spec['user'])],
            think=True,stream=False,options={**OPTIONS,'seed':spec['seed'],'num_predict':spec['cap']})
        count=len(tokenizer.encode(rendered(spec['user']),add_special_tokens=False).ids)
        assert count+spec['cap']+64<=65536
        schedule.append(dict(ordinal=ordinal,phase=spec['phase'],row_index=spec['row_index'],request=payload,
            expected_prompt_tokens=count,request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()))
    deps=[pathlib.Path(__file__),ROOT/'hybrid_source_qualification_protocol.md',
        ROOT/'continued_bridge_audit.json',indexed_path,ROOT/'historical_manifest_exclusion_check_20261004.json',
        META/'tokenizer.json',META/'config.json',META/'tokenizer_config.json',META/'official_README.md',
        META/'local_qwen3_5_4b_show.json',META/'offline_render_checks.json',META/'padding_difference_checks.json',
        META/'runtime_qwen35_renderer.go']
    OUT.mkdir()
    save(OUT/'frozen.json',dict(dependencies={str(p):sha(p) for p in deps},cases=cases,schedule=schedule,
        model=MODEL,digest=DIGEST,model_inventory=tags,runtime=version,
        historical_rows=hist['historical_rows'],bridge_rows=[c['row_index'] for c in bridge['cases']],
        all_eight_labels_recomputed=True,source_character_metadata=[dict(row_index=c['row_index'],catalogue_prompt_chars=c['prompt_chars'],actual_source_characters=len(c['prompt'])) for c in cases],preparation_before_freeze_serialization_seconds=time.monotonic()-preparation_started,gate=dict(normal_parse_min=15,planned_source_calls=16,f1_min=.70,em_min=.50)))
    for job in schedule:
        path=OUT/f"attempt_{job['ordinal']:03d}.json"
        record=dict(job=job,completion='pending',transmission='not_yet_attempted');save(path,record)
        record.update(transmission='attempting_once',started_at=time.time(),monotonic_started=time.monotonic());save(path,record)
        try:raw=request('chat',job['request'],timeout=3600)
        except Exception as error:
            record.update(completion='client_error',error=repr(error),ended_at=time.time(),elapsed_seconds=time.monotonic()-record['monotonic_started']);save(path,record)
            save(OUT/'completion.json',dict(primary_complete=False,reason='client_error',last_ordinal=job['ordinal']))
            print(job['ordinal'],'client_error',repr(error),flush=True);return
        record.update(completion='returned',raw=raw,ended_at=time.time(),elapsed_seconds=time.monotonic()-record['monotonic_started']);save(path,record)
        try:ps=request('ps')
        except Exception as error:
            record['qualification_metadata_error']=repr(error);save(path,record)
            save(OUT/'completion.json',dict(primary_complete=False,reason='metadata_error',last_ordinal=job['ordinal']));return
        record['runtime_ps']=ps
        record['input_qualified']=raw.get('prompt_eval_count')==job['expected_prompt_tokens']
        record['resident_qualified']=any(m.get('digest')==DIGEST and m.get('context_length',0)>=65536 for m in ps.get('models',[]))
        record['response_qualified']=raw.get('done') is True and raw.get('model')==MODEL and not raw.get('error')
        save(path,record)
        print(job['ordinal'],job['phase'],job['row_index'],raw.get('done_reason'),raw.get('prompt_eval_count'),raw.get('eval_count'),flush=True)
        if not (record['input_qualified'] and record['resident_qualified'] and record['response_qualified']):
            save(OUT/'completion.json',dict(primary_complete=False,reason='qualification_failed',last_ordinal=job['ordinal']));return
    save(OUT/'completion.json',dict(primary_complete=True,scored_source_calls=16,calibration_calls=1,finished_at=time.time()))
if __name__=='__main__':main()
