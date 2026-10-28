"""Keep model-produced values; vary how current state is distinguished from history."""
import hashlib,json,random,sys,time
from pathlib import Path
import run_model_checkpoint_order as order
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/checkpoint_representation';ARMS=['history','capsule','json','input_key']
def sha(b):return hashlib.sha256(b).hexdigest()
def state(row,arm):
    value=f'S{row["producer"][-1]["output"]:02d}'
    if arm=='history':return row['state']
    if arm=='capsule':return f'Execution record:\nCompleted steps: 1 through 3. Current state: {value}. Next step: 4.'
    if arm=='json':return 'Execution record:\n'+json.dumps(dict(completed_steps=[1,2,3],current_state=value,next_step=4))
    return f'Execution record:\nCurrent input key: {value}. Next step: 4.'
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'frozen.json').exists();fp=order.OUT/'frozen.json';rp=order.OUT/'reader.json';sp=order.OUT/'summary.json';f=json.loads(fp.read_text());d=json.loads(rp.read_text());assert json.loads(sp.read_text())['audited_calls']==168
    for row in d['rows']:assert len(row['producer'])==3 and row['producer'][-1]['output'] is not None and order.checkpoint(row['producer'])==row['state']
    files=['run_checkpoint_representation.py','protocol_checkpoint_representation_20261003.md','run_model_checkpoint_order.py','run_lookup_followup.py','run_lookup_semantics.py','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    frozen=dict(tasks=f['tasks'],models=f['models'],producer_rows=d['rows'],source_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [fp,rp,sp]},hashes={p:sha((ROOT/p).read_bytes()) for p in files},adaptive_reuse=True)
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');print('frozen',len(d['rows']))
def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text());rp=OUT/'reader.json'
    for p,h in {**f['hashes'],**f['source_sha256']}.items():assert sha((ROOT/p).read_bytes())==h
    d=json.loads(rp.read_text()) if rp.exists() else dict(frozen_sha256=sha(fp.read_bytes()),warmups={},rows=[]);assert d['frozen_sha256']==sha(fp.read_bytes())
    def save():rp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in f['models']:
        if model not in d['warmups']:d['warmups'][model]=order.call(model,'Return {"state":"S00"}.',0,0);save()
        for t in f['tasks']:
            producer=next(r for r in f['producer_rows'] if (r['model'],r['id'])==(model,t['id']));arms=ARMS.copy();random.Random(t['seed']+200000).shuffle(arms)
            for arm in arms:
                if any((r['model'],r['id'],r['arm'])==(model,t['id'],arm) for r in d['rows']):continue
                p=order.prompt(t,'pre_local',state(producer,arm));v=order.call(model,p,producer['producer'][-1]['output'],4);v.update(model=model,id=t['id'],arm=arm);d['rows'].append(v);save()
            print(model,t['id'],'done',flush=True)
def analyze():
    f=json.loads((OUT/'frozen.json').read_text());d=json.loads((OUT/'reader.json').read_text());assert len(d['rows'])==96;out=dict(adaptive_reuse=True,audited_calls=0,models={})
    for p,h in {**f['hashes'],**f['source_sha256']}.items():assert sha((ROOT/p).read_bytes())==h
    for model in f['models']:
        out['models'][model]={}
        for arm in ARMS:
            rows=[r for r in d['rows'] if r['model']==model and r['arm']==arm];true=conditional=0
            for r in rows:
                t=next(t for t in f['tasks'] if t['id']==r['id']);producer=next(v for v in f['producer_rows'] if (v['model'],v['id'])==(model,t['id']));current=producer['producer'][-1]['output'];assert r['input']==current
                p=order.prompt(t,'pre_local',state(producer,arm));assert r['prompt_sha256']==sha(p.encode());raw=r.get('raw',{})
                if raw:
                    payload=dict(model=model,messages=[dict(role='system',content=order.SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=order.prior.dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==raw['request_payload_sha256']
                    if raw['schema_valid']:assert int(json.loads(raw['native_schema_content'])['state'][1:])==r['output']
                true+=r['output']==t['trajectory'][4];conditional+=r['output']==t['maps'][3][str(current)];out['audited_calls']+=1
            out['models'][model][arm]=dict(true_correct=true,conditioned_transition_correct=conditional,n=12,tokens=sum((r.get('raw',{}).get('prompt_eval_count') or 0)+(r.get('raw',{}).get('eval_count') or 0) for r in rows),wall_sec=sum(r['wall_sec'] for r in rows),invalid=sum(not r.get('raw',{}).get('schema_valid',False) for r in rows),errors=sum('error' in r for r in rows))
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
