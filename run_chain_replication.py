"""Independent new-seed replication, preserving all earlier frozen studies."""
import hashlib,json,random,sys,time
from pathlib import Path
import run_lookup_followup as prior

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/lookup_chain_replication'
def sha(b):return hashlib.sha256(b).hexdigest()
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'frozen.json').exists()
    old=ROOT/'logs/lookup_chain/frozen.json';summary=ROOT/'logs/lookup_chain/summary.json';selected=json.loads(old.read_text())['selection'];assert json.loads(summary.read_text())['audited_calls']==208
    tasks=[]
    for i,seed in enumerate(range(6401,6417)):
        t=prior.make_task(seed,i,True);t['arm_order']=['selected_chain','continuous'];random.Random(seed+100000).shuffle(t['arm_order']);tasks.append(t)
    files=['run_chain_replication.py','protocol_chain_replication_20261003.md','run_lookup_followup.py','run_lookup_semantics.py','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    f=dict(tasks=tasks,models=prior.dev.MODELS,selection=selected,adaptive_replication=True,previous_summary_sha256=sha(summary.read_bytes()),hashes={p:sha((ROOT/p).read_bytes()) for p in files});(OUT/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print('frozen',len(tasks),selected)
def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text());rp=OUT/'reader.json'
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    d=json.loads(rp.read_text()) if rp.exists() else dict(frozen_sha256=sha(fp.read_bytes()),warmups={},rows=[]);assert d['frozen_sha256']==sha(fp.read_bytes())
    def save():rp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in f['models']:
        if model not in d['warmups']:
            g=prior.dev.interface.native(model,prior.dev.SYSTEM,'Return {"state":"S00"}.',num_ctx=32768,num_predict=256);d['warmups'][model]=g.raw;save()
        for t in f['tasks']:
            for arm in t['arm_order']:
                r=next((r for r in d['rows'] if (r['model'],r['id'],r['arm'])==(model,t['id'],arm)),None)
                if r is None:r=dict(model=model,id=t['id'],arm=arm,calls=[]);d['rows'].append(r)
                count=6 if arm=='selected_chain' else 1
                while len(r['calls'])<count:
                    i=len(r['calls']);current=t['input'] if i==0 else r['calls'][-1]['output']
                    if current is None:r['terminated_invalid']=True;break
                    start=time.perf_counter();p=prior.prompt(t,arm,f['selection'][model],i+1,current);prep=time.perf_counter()-start;start=time.perf_counter()
                    try:
                        g=prior.dev.interface.native(model,prior.dev.SYSTEM,p,num_ctx=32768,num_predict=256);v=dict(output=int(json.loads(g.raw['native_schema_content'])['state'][1:]) if g.raw['schema_valid'] else None,raw=g.raw)
                    except Exception as e:v=dict(output=None,error=repr(e))
                    v.update(step=i+1,input=current,prompt_sha256=sha(p.encode()),preparation_sec=prep,wall_sec=time.perf_counter()-start);r['calls'].append(v);save()
                r['complete']=len(r['calls'])==count;save()
            print(model,t['id'],'done',flush=True)
def analyze():
    f=json.loads((OUT/'frozen.json').read_text());d=json.loads((OUT/'reader.json').read_text());assert len(d['rows'])==64
    result=dict(adaptive_replication=True,selection=f['selection'],models={},audited_calls=0,model_feedback_checks=0)
    for model in f['models']:
        result['models'][model]={}
        for arm in ['continuous','selected_chain']:
            rows=[r for r in d['rows'] if r['model']==model and r['arm']==arm];calls=[];correct=perfect=transitions=0
            for r in rows:
                t=next(t for t in f['tasks'] if t['id']==r['id']);state=t['input'];trajectory=[state]
                for mapping in t['maps']:assert sorted(map(int,mapping))==list(range(8));assert sorted(mapping.values())==list(range(8));state=mapping[str(state)];trajectory.append(state)
                assert trajectory==t['trajectory']
                for i,v in enumerate(r['calls']):
                    assert v['input']==(t['input'] if i==0 else r['calls'][i-1]['output']);result['model_feedback_checks']+=int(i>0)
                    p=prior.prompt(t,arm,f['selection'][model],v['step'],v['input']);assert sha(p.encode())==v['prompt_sha256']
                    raw=v.get('raw',{})
                    if raw:
                        payload=dict(model=model,messages=[dict(role='system',content=prior.dev.SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=prior.dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==raw['request_payload_sha256']
                        if raw['schema_valid']:assert int(json.loads(raw['native_schema_content'])['state'][1:])==v['output']
                    transitions+=v['output']==t['maps'][i][str(v['input'])];result['audited_calls']+=1;calls.append(v)
                correct+=r['complete'] and r['calls'][-1]['output']==trajectory[-1]
                perfect+=arm=='selected_chain' and r['complete'] and all(v['output']==trajectory[i+1] for i,v in enumerate(r['calls']))
            result['models'][model][arm]=dict(correct=int(correct),n=16,perfect_chains=int(perfect) if arm=='selected_chain' else None,correct_transitions=int(transitions) if arm=='selected_chain' else None,calls=len(calls),tokens=sum((v.get('raw',{}).get('prompt_eval_count') or 0)+(v.get('raw',{}).get('eval_count') or 0) for v in calls),wall_sec=sum(v['wall_sec'] for v in calls),preparation_sec=sum(v['preparation_sec'] for v in calls),invalid=sum(not v.get('raw',{}).get('schema_valid',False) for v in calls),errors=sum('error' in v for v in calls))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
