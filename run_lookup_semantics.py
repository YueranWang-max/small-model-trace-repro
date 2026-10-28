"""Frozen single-lookup factorial diagnosis; native JSON, no answer repair."""
import hashlib, json, random, re, sys, time
from pathlib import Path
import run_controlled_step_schema as interface

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'logs/lookup_semantics'
MODELS=['qwen3:1.7b','qwen3.5:4b']
ARMS=[f'{fmt}_{scope}' for fmt in ['arrow','json','prose'] for scope in ['local','all','noise']]+['legacy_local']
SYSTEM='Perform the requested table lookup.'
def sha(b): return hashlib.sha256(b).hexdigest()
def render(t, fmt, step):
    pairs=[(f'S{k:02d}',f'S{t["maps"][step-1][str(k)]:02d}') for k in t['row_orders'][step-1]]
    if fmt=='arrow': body='; '.join(a+'->'+b for a,b in pairs)+'.'
    elif fmt=='json': body=json.dumps(dict(pairs))
    else: body='\n'.join(f'If the input is {a}, the output is {b}.' for a,b in pairs)
    return f'Step {step}:\n'+body
def prompt(t,a):
    if a=='legacy_local': fmt,scope='arrow','local'
    else: fmt,scope=a.split('_')
    steps=[t['step']] if scope=='local' else list(range(1,7))
    source='\n\n'.join(render(t,fmt,j) for j in steps)
    if scope=='noise': source=t['noise']+'\n'+source
    state=f'S{t["input"]:02d}'
    if a=='legacy_local': command=f'Current state is {state}. Apply ONLY step {t["step"]} to this current state. Do not apply earlier or later steps.'
    else: command=f'Current input key is {state}. Look up this input in Step {t["step"]} exactly ONCE and return its output. This is one function evaluation. Do NOT use the returned output as another input. Do NOT execute other steps.'
    return 'Source:\n'+source+'\nOperation: '+command+'\nReturn a JSON object with exactly one field state, chosen from S00 through S07. No explanation.'
def freeze():
    OUT.mkdir(parents=True,exist_ok=True); assert not (OUT/'frozen.json').exists()
    tasks=[]
    for i,seed in enumerate(range(6101,6109)):
        rng=random.Random(seed); maps=[]; orders=[]
        for _ in range(6):
            values=list(range(8));rng.shuffle(values);maps.append(dict(zip(map(str,range(8)),values)))
            rows=list(range(8));rng.shuffle(rows);orders.append(rows)
        step=i%6+1; current=rng.randrange(8)
        noise='\n'.join(f'Archive ticket N{j:04d} records unrelated object B{rng.randrange(1000):03d} and status H{rng.randrange(1000):03d}; these tickets do not change any step table.' for j in range(128))
        order=list(ARMS);rng.shuffle(order)
        tasks.append(dict(id=f'lookup_{seed}',seed=seed,maps=maps,row_orders=orders,step=step,input=current,answer=maps[step-1][str(current)],noise=noise,arm_order=order))
    files=['run_lookup_semantics.py','protocol_lookup_semantics_20261003.md','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    f=dict(tasks=tasks,models=MODELS,arms=ARMS,development=True,hashes={p:sha((ROOT/p).read_bytes()) for p in files})
    (OUT/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print('frozen',len(tasks),len(ARMS))
def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    rp=OUT/'reader.json';d=json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else dict(frozen_sha256=sha(fp.read_bytes()),warmups={},rows=[])
    assert d['frozen_sha256']==sha(fp.read_bytes())
    def save():rp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in f['models']:
        if model not in d['warmups']:
            g=interface.native(model,SYSTEM,'Return {"state":"S00"}.',num_ctx=32768,num_predict=256)
            d['warmups'][model]=g.raw;save()
        for t in f['tasks']:
            for a in t['arm_order']:
                if any(r['model']==model and r['id']==t['id'] and r['arm']==a for r in d['rows']):continue
                p=prompt(t,a);start=time.perf_counter()
                try:
                    g=interface.native(model,SYSTEM,p,num_ctx=32768,num_predict=256)
                    value=int(json.loads(g.raw['native_schema_content'])['state'][1:]) if g.raw['schema_valid'] else None
                    row=dict(output=value,raw=g.raw)
                except Exception as e:row=dict(output=None,error=repr(e))
                row.update(model=model,id=t['id'],arm=a,prompt_sha256=sha(p.encode()),wall_sec=time.perf_counter()-start)
                d['rows'].append(row);save()
            print(model,t['id'],'done',len(d['rows']),flush=True)
def analyze():
    f=json.loads((OUT/'frozen.json').read_text());d=json.loads((OUT/'reader.json').read_text());assert len(d['rows'])==160
    summary={}
    for model in f['models']:
        summary[model]={}
        for a in f['arms']:
            rows=[r for r in d['rows'] if r['model']==model and r['arm']==a];correct=0
            for r in rows:
                t=next(t for t in f['tasks'] if t['id']==r['id']);p=prompt(t,a);assert r['prompt_sha256']==sha(p.encode())
                fmt='arrow' if a=='legacy_local' else a.split('_')[0]
                text=render(t,fmt,t['step'])
                if fmt=='arrow': pairs=re.findall(r'(S\d+)->(S\d+)',text)
                elif fmt=='json': pairs=json.loads(text.split('\n',1)[1]).items()
                else:pairs=re.findall(r'input is (S\d+), the output is (S\d+)',text)
                mapping={int(x[1:]):int(y[1:]) for x,y in pairs};assert sorted(mapping)==list(range(8));assert mapping[t['input']]==t['answer']
                raw=r.get('raw',{});payload=dict(model=model,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9))
                if raw:assert raw['request_payload_sha256']==sha(json.dumps(payload,sort_keys=True).encode())
                correct+=r['output']==t['answer']
            summary[model][a]=dict(correct=correct,n=len(rows),tokens=sum((r.get('raw',{}).get('prompt_eval_count') or 0)+(r.get('raw',{}).get('eval_count') or 0) for r in rows),wall_sec=sum(r['wall_sec'] for r in rows),invalid=sum(not r.get('raw',{}).get('schema_valid',False) for r in rows),errors=sum('error' in r for r in rows))
    result=dict(development=True,audited_calls=len(d['rows']),models=summary)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
