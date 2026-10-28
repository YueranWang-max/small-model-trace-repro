"""Reuse exact retained traces to diagnose the previous context failure."""
import json,random,time,sys,urllib.request,urllib.error,subprocess
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'recovery_bridge_20261003'))
import run_hybrid_source_qualification as base
from run_official_graph_native_probe import score
OUT=ROOT/'logs/context_recovery_20261006'
OLD=ROOT/'recovery_bridge_20261003/hybrid_five_trace_replication'
PROTOCOL=ROOT/'protocol_context_recovery_20261006.md'
CTX=196608

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def gpu():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    return {'returncode':r.returncode,'snapshot':r.stdout.strip(),'error':r.stderr.strip()}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    deps={str(p.relative_to(ROOT)):base.sha(p) for p in [Path(__file__),PROTOCOL,OLD/'frozen.json',OLD/'bundle_00.json',ROOT/'recovery_bridge_20261003/run_hybrid_source_qualification.py',ROOT/'run_official_graph_native_probe.py',ROOT/'recovery_bridge_20261003/hybrid_candidate_qualification_20261004/tokenizer.json']}
    if (OUT/'frozen.json').exists():
        fr=json.loads((OUT/'frozen.json').read_bytes());assert fr['dependencies']==deps;return fr
    old=json.loads((OLD/'frozen.json').read_bytes());bundle=json.loads((OLD/'bundle_00.json').read_bytes())
    case=next(c for c in old['cases'] if c['row_index']==bundle['row_index'])
    tokenizer=Tokenizer.from_file(str(ROOT/'recovery_bridge_20261003/hybrid_candidate_qualification_20261004/tokenizer.json'))
    jobs=[];rng=random.Random(202610198)
    for repeat in range(5):
        block=[j for j in bundle['reader_jobs'] if j['repeat']==repeat];assert len(block)==5
        rng.shuffle(block)
        for source in block:
            job=json.loads(json.dumps(source));req=job['request']
            req['options']['num_ctx']=CTX;req['keep_alive']='10m'
            n=len(tokenizer.encode(base.rendered(req['messages'][1]['content']),add_special_tokens=False).ids)
            assert n==job['expected_prompt_tokens'] and n+32768+64<=CTX
            job.update(key=f"r{repeat}_{job['arm']}",request=req,expected_prompt_tokens=n)
            jobs.append(job)
    probe=json.loads(json.dumps(next(j for j in jobs if j['arm']=='prefix')))
    probe['request']['options'].update(num_predict=1,seed=202610199)
    probe.update(key='full_input_probe',arm='probe',repeat=-1)
    inventory=base.request('tags');assert next(m['digest'] for m in inventory['models'] if m['name']==base.MODEL)==base.DIGEST
    sources=[]
    for ordinal in bundle['source_ordinals']:
        path=OLD/f'attempt_{ordinal:03d}.json';record=json.loads(path.read_bytes())
        assert record['completion']=='returned' and all(record[k] for k in ['input_qualified','resident_qualified','response_qualified'])
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':base.sha(path),
                        'input_tokens':record['raw']['prompt_eval_count'],'output_tokens':record['raw']['eval_count']})
    fr={'dependencies':deps,'case':case,'jobs':[probe]+jobs,'original_sources':sources,
        'ctx':CTX,'reader_cap':32768,'model':base.MODEL,'digest':base.DIGEST,'inventory':inventory,
        'server_version':base.request('version'),'bundle_sha256':bundle['bundle_sha256'],
        'capacity_preflight':[{'context':c,'prefix_input':probe['expected_prompt_tokens'],'reserve':32768,'fit':probe['expected_prompt_tokens']+32768+64<=c} for c in [131072,163840,196608,262144]]}
    save(OUT/'frozen.json',fr);return fr

def execute(job):
    path=OUT/(job['key']+'.json');assert not path.exists(),'Never retransmit an existing attempt'
    if (OUT/'STOP').exists():raise RuntimeError('STOP requested between calls')
    req=job['request'];record={'job':job,'completion':'pending','transmission':'not_yet_attempted',
        'request_sha256':__import__('hashlib').sha256(json.dumps(req,sort_keys=True).encode()).hexdigest(),
        'gpu_before':gpu()}
    save(path,record);record.update(transmission='attempting_once',started_at=time.time());save(path,record)
    t=time.perf_counter();body=b'';raw={};error=None
    try:
        rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        with base.OPENER.open(rq,timeout=7200) as response:body=response.read()
        (OUT/(job['key']+'.response.json')).write_bytes(body);raw=json.loads(body)
    except urllib.error.HTTPError as exc:
        body=exc.read();(OUT/(job['key']+'.http_error.txt')).write_bytes(body);error=repr(exc)
    except Exception as exc:error=repr(exc)
    record.update(completion='returned',raw=raw,error=error,ended_at=time.time(),wall_sec=time.perf_counter()-t,
                  raw_response_sha256=__import__('hashlib').sha256(body).hexdigest() if body else None)
    save(path,record)
    try:record['runtime_ps']=base.request('ps')
    except Exception as exc:record['metadata_error']=repr(exc)
    record['gpu_after']=gpu()
    record['qualified']=not error and raw.get('done') is True and raw.get('model')==base.MODEL and raw.get('prompt_eval_count')==job['expected_prompt_tokens'] and any(m.get('digest')==base.DIGEST and m.get('context_length')==CTX for m in record.get('runtime_ps',{}).get('models',[]))
    save(path,record)
    if not record['qualified']:raise RuntimeError('Operational/interface failure retained: '+job['key'])
    time.sleep(5);return record

def main():
    fr=freeze()
    if '--prepare' in sys.argv:print('Frozen single-case context recovery: 1 full-input probe, 25 readers',flush=True);return
    assert not list(OUT.glob('r*.json')) and not (OUT/'full_input_probe.json').exists()
    records=[]
    try:
        for job in fr['jobs']:
            record=execute(job)
            if job['arm']!='probe':record['score']=score(record['raw'],fr['case']['answer_nodes'],record['error'])
            records.append(record);save(OUT/'results.json',records)
            save(OUT/'status.json',{'state':'running','recorded':len(records),'planned':26,'latest':job['key'],'updated_unix':time.time()})
            print(job['key'],record['raw'].get('done_reason'),record.get('score'),flush=True)
            if job['arm']!='probe' and sum(r['job']['repeat']==job['repeat'] for r in records)==5:time.sleep(60)
        save(OUT/'status.json',{'state':'completed','recorded':26,'planned':26,'updated_unix':time.time()})
    except Exception as exc:
        save(OUT/'status.json',{'state':'stopped_on_exception','recorded':len(records),'planned':26,'error':repr(exc),'updated_unix':time.time()});raise

if __name__=='__main__':main()
