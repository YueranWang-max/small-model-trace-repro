"""Durable prospective replication; --prepare freezes without inference."""
import argparse, json, random, time, urllib.request, urllib.error
from pathlib import Path
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT, DATA, SYSTEM, OPENER, sha, score, oracle
from run_official_graph_long_boundary import input_count
from run_official_graph_five_trace import serialize, FORMAT

OUT = ROOT / 'logs/official_graph_confirmatory_20261006'
PROTOCOL = ROOT / 'protocol_official_graph_confirmatory_20261006.md'
MODEL = 'qwen3:1.7b'
DIGEST = '8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'

def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def get(endpoint):
    with OPENER.open('http://127.0.0.1:11434/api/' + endpoint, timeout=15) as response:
        return json.load(response)

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    names = [Path(__file__).name, PROTOCOL.name, 'run_official_graph_native_probe.py',
             'run_official_graph_five_trace.py', 'run_official_graph_native_order.py',
             'run_official_graph_sampling_control.py', 'run_official_graph_long_boundary.py',
             'data/qwen3_tokenizer_official/tokenizer.json']
    deps = {n: sha((ROOT/n).read_bytes()) for n in names}
    if (OUT/'frozen.json').exists():
        fr = json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
        assert fr['dependencies'] == deps and fr['data_sha256'] == sha(DATA.read_bytes())
        return fr
    used, exclusions = set(), {}
    for path in sorted((ROOT/'logs').glob('official_graph*/frozen.json')):
        if path.parent == OUT: continue
        exclusions[str(path.relative_to(ROOT))] = sha(path.read_bytes())
        for row in json.loads(path.read_text(encoding='utf-8-sig')).get('selected', []):
            if isinstance(row, dict) and 'prompt' in row:
                used.add(sha(row['prompt'].rsplit('Operation:',1)[0].encode()))
    rng = random.Random(202610060)
    rows = pq.read_table(DATA).to_pylist()
    candidates, selected = {}, []
    for typ in ['parents', 'bfs']:
        eligible, seen = [], set(used)
        for i,row in enumerate(rows):
            h = sha(row['prompt'].rsplit('Operation:',1)[0].encode())
            if row['problem_type'] == typ and 2000 <= row['prompt_chars'] < 3500 and h not in seen:
                eligible.append({'row_index':i,'graph_head_sha256':h,**row}); seen.add(h)
        candidates[typ] = eligible
        assert len(eligible) >= 6
        chosen = rng.sample(eligible,6)
        for row in chosen:
            assert set(oracle(row)) == set(row['answer_nodes'])
            used.add(row['graph_head_sha256'])
        selected.extend(chosen)
    rng.shuffle(selected)
    orders=[]
    for row in selected:
        block=[]
        for j in range(5):
            arms=['pre','post','direct']; rng.shuffle(arms); block.append(arms)
        orders.append(block)
    inventory=get('tags'); identity=next(m for m in inventory['models'] if m['name']==MODEL)
    assert identity['digest']==DIGEST
    fr={'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selected':selected,
        'candidates':candidates,'exclusion_archives':exclusions,'orders':orders,
        'source_seeds':list(range(202610061,202610066)),
        'reader_seeds':list(range(202610071,202610076)),
        'identity':identity,'inventory':inventory,'server_version':get('version'),
        'ctx':32768,'source_cap':4096,'reader_cap':8192,'planned_calls':240}
    save(OUT/'frozen.json',fr)
    return fr

def call(key, user, cap, seed):
    path=OUT/(key+'.json')
    if path.exists():
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['completion'] in ['returned','budget_declined'], 'Ambiguous request: '+key
        return record['result']
    if (OUT/'STOP').exists(): raise RuntimeError('STOP requested between calls')
    n=input_count(SYSTEM,user)
    req={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],
         'stream':False,'think':True,'keep_alive':'10m',
         'options':{'num_ctx':32768,'num_predict':cap,'seed':seed,'temperature':.6,
                    'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}}
    record={'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),
            'expected_input_tokens':n,'completion':'pending','transmission':'not_yet_attempted'}
    if n+cap+64>32768:
        result={'raw':{},'error':None,'budget_declined':True,'expected_input_tokens':n}
        record.update(completion='budget_declined',result=result);save(path,record);return result
    save(path,record);record['transmission']='attempting_once';save(path,record)
    start=time.perf_counter();raw={};error=None;body=b''
    try:
        rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(rq,timeout=900) as response: body=response.read()
        (OUT/(key+'.response.json')).write_bytes(body)
        raw=json.loads(body)
    except urllib.error.HTTPError as exc:
        body=exc.read();(OUT/(key+'.http_error.txt')).write_bytes(body);error=repr(exc)
    except Exception as exc: error=repr(exc)
    result={'raw':raw,'error':error,'wall_sec':time.perf_counter()-start,
            'expected_input_tokens':n,'raw_response_sha256':sha(body) if body else None,
            'runtime_ps':get('ps')}
    result['qualified']=any(m.get('digest')==DIGEST and m.get('context_length')==32768
                             for m in result['runtime_ps'].get('models',[])) and raw.get('model')==MODEL and raw.get('prompt_eval_count')==n
    record.update(completion='returned',result=result);save(path,record)
    if error or not result['qualified']: raise RuntimeError('Transport/interface failure preserved: '+key)
    time.sleep(5)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    fr=freeze()
    if args.prepare: print('Frozen 12 graph units, 240 scheduled calls',flush=True);return
    results=[]
    try:
        for i,row in enumerate(fr['selected']):
            x=row['prompt']+'\n'+FORMAT;head,op=x.rsplit('Operation:',1);traces=[];source_keys=[]
            jobs=[('source',j,s) for j,s in enumerate(fr['source_seeds'])]
            for arm,j,seed in jobs:
                key=f'g{i:02d}_{arm}_{j}';r=call(key,x,4096,seed)
                r={**r,'key':key,'row_index':row['row_index'],'arm':arm,'repeat':j}
                r['score']=score(r['raw'],row['answer_nodes'],r['error']);results.append(r)
                save(OUT/'results.json',results)
                traces.append((r['raw'].get('message') or {}).get('thinking','') or '');source_keys.append(key)
                save(OUT/'status.json',{'state':'running','recorded':len(results),'planned':240,'latest':key,'updated_unix':time.time()})
                print(key,r['score'],flush=True)
            T=serialize(traces);prompts={'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op,'direct':x}
            for j,seed in enumerate(fr['reader_seeds']):
                for arm in fr['orders'][i][j]:
                    key=f'g{i:02d}_{arm}_{j}';r=call(key,prompts[arm],8192,seed)
                    r={**r,'key':key,'row_index':row['row_index'],'arm':arm,'repeat':j,
                       'trace_sha256':sha(T.encode()),'source_keys':source_keys}
                    r['score']=None if r.get('budget_declined') else score(r['raw'],row['answer_nodes'],r['error'])
                    results.append(r);save(OUT/'results.json',results)
                    save(OUT/'status.json',{'state':'running','recorded':len(results),'planned':240,'latest':key,'updated_unix':time.time()})
                    print(key,r['score'],flush=True)
            time.sleep(60)
        save(OUT/'status.json',{'state':'completed','recorded':len(results),'planned':240,'updated_unix':time.time()})
    except Exception as exc:
        save(OUT/'status.json',{'state':'stopped_on_exception','error':repr(exc),'recorded':len(results),'planned':240,'updated_unix':time.time()})
        raise

if __name__=='__main__': main()
