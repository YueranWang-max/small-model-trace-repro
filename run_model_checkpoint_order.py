"""Same model-produced checkpoint under order and retained-source interventions."""
import hashlib,json,random,sys,time
from pathlib import Path
import run_lookup_followup as prior
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/model_checkpoint_order'
ARMS=['pre_full','post_full','pre_local','post_local'];SYSTEM=prior.dev.SYSTEM
def sha(b):return hashlib.sha256(b).hexdigest()
def source(t,arm):
    return prior.dev.render(t,'arrow',4) if arm.endswith('local') else t['noise']+'\n'+f'Initial state: S{t["input"]:02d}.\n'+'\n\n'.join(prior.dev.render(t,'arrow',j) for j in range(1,7))
def checkpoint(calls):
    return 'Execution record:\n'+'\n'.join(f'Step {i+1} output: S{v["output"]:02d}.' for i,v in enumerate(calls))+f'\nCompleted steps: 1 through 3. Current state: S{calls[-1]["output"]:02d}. Next step: 4.'
def prompt(t,arm,state):
    trace='State:\n'+state;src='Source:\n'+source(t,arm)
    q='Operation: Use the current state in the execution record as the input key. Look up this input in Step 4 exactly ONCE. Do NOT use its output as another input. Do NOT execute any other step. Return a JSON object with exactly one field state, chosen from S00 through S07. No explanation.'
    return '\n\n'.join([trace,src,q] if arm.startswith('pre') else [src,trace,q])
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'frozen.json').exists();tasks=[]
    for i,seed in enumerate(range(6501,6513)):
        t=prior.make_task(seed,i,True);t['arm_order']=ARMS.copy();random.Random(seed+100000).shuffle(t['arm_order']);tasks.append(t)
    files=['run_model_checkpoint_order.py','protocol_model_checkpoint_order_20261003.md','run_lookup_followup.py','run_lookup_semantics.py','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    f=dict(tasks=tasks,models=prior.dev.MODELS,adaptive=True,hashes={p:sha((ROOT/p).read_bytes()) for p in files});(OUT/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print('frozen',len(tasks))
def call(model,p,current,step):
    start=time.perf_counter()
    try:
        g=prior.dev.interface.native(model,SYSTEM,p,num_ctx=32768,num_predict=256);v=dict(output=int(json.loads(g.raw['native_schema_content'])['state'][1:]) if g.raw['schema_valid'] else None,raw=g.raw)
    except Exception as e:v=dict(output=None,error=repr(e))
    v.update(input=current,step=step,prompt_sha256=sha(p.encode()),wall_sec=time.perf_counter()-start);return v
def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text());rp=OUT/'reader.json'
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    d=json.loads(rp.read_text()) if rp.exists() else dict(frozen_sha256=sha(fp.read_bytes()),warmups={},rows=[]);assert d['frozen_sha256']==sha(fp.read_bytes())
    def save():rp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in f['models']:
        if model not in d['warmups']:d['warmups'][model]=call(model,'Return {"state":"S00"}.',0,0);save()
        for t in f['tasks']:
            row=next((r for r in d['rows'] if (r['model'],r['id'])==(model,t['id'])),None)
            if row is None:row=dict(model=model,id=t['id'],producer=[],readers={});d['rows'].append(row)
            while len(row['producer'])<3:
                i=len(row['producer']);current=t['input'] if i==0 else row['producer'][-1]['output']
                if current is None:row['terminated_invalid']=True;break
                p=prior.prompt(t,'selected_chain','arrow',i+1,current);row['producer'].append(call(model,p,current,i+1));save()
            if len(row['producer'])!=3 or row['producer'][-1]['output'] is None:continue
            state=checkpoint(row['producer']);row['state']=state;row['state_sha256']=sha(state.encode());save()
            for arm in t['arm_order']:
                if arm in row['readers']:continue
                row['readers'][arm]=call(model,prompt(t,arm,state),row['producer'][-1]['output'],4);save()
            print(model,t['id'],'done',flush=True)
def analyze():
    f=json.loads((OUT/'frozen.json').read_text());d=json.loads((OUT/'reader.json').read_text());assert len(d['rows'])==24;out=dict(adaptive=True,models={},audited_calls=0,model_feedback_checks=0)
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    def audit(model,p,v):
        assert sha(p.encode())==v['prompt_sha256'];raw=v.get('raw',{})
        if raw:
            payload=dict(model=model,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=prior.dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==raw['request_payload_sha256']
            if raw['schema_valid']:assert int(json.loads(raw['native_schema_content'])['state'][1:])==v['output']
        out['audited_calls']+=1
    def metrics(calls):return dict(calls=len(calls),tokens=sum((v.get('raw',{}).get('prompt_eval_count') or 0)+(v.get('raw',{}).get('eval_count') or 0) for v in calls),wall_sec=sum(v['wall_sec'] for v in calls),invalid=sum(not v.get('raw',{}).get('schema_valid',False) for v in calls),errors=sum('error' in v for v in calls))
    for model in f['models']:
        rows=[r for r in d['rows'] if r['model']==model];m=dict(producer={},arms={});produced=[]
        for row in rows:
            t=next(t for t in f['tasks'] if t['id']==row['id']);state=t['input'];traj=[state]
            for mapping in t['maps']:state=mapping[str(state)];traj.append(state)
            assert traj==t['trajectory']
            for i,v in enumerate(row['producer']):
                assert v['input']==(t['input'] if i==0 else row['producer'][i-1]['output']);out['model_feedback_checks']+=int(i>0);audit(model,prior.prompt(t,'selected_chain','arrow',i+1,v['input']),v);produced.append(v)
            if 'state' in row:assert checkpoint(row['producer'])==row['state'] and sha(row['state'].encode())==row['state_sha256']
        m['producer']={**metrics(produced), 'checkpoint_correct':sum(len(r['producer'])==3 and r['producer'][-1]['output']==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][3] for r in rows),'n':12}
        for arm in ARMS:
            calls=[];true=conditional=0
            for row in rows:
                if arm not in row['readers']:continue
                t=next(t for t in f['tasks'] if t['id']==row['id']);v=row['readers'][arm];assert v['input']==row['producer'][-1]['output'];audit(model,prompt(t,arm,row['state']),v);calls.append(v);true+=v['output']==t['trajectory'][4];conditional+=v['output']==t['maps'][3][str(v['input'])]
                for scope in ['full','local']:assert source(t,'pre_'+scope)==source(t,'post_'+scope)
            m['arms'][arm]={**metrics(calls),'true_correct':true,'conditioned_transition_correct':conditional,'n':12}
        m['paired_order_difference']={scope:sum((r['readers'].get('pre_'+scope,{}).get('output')==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4])-(r['readers'].get('post_'+scope,{}).get('output')==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4]) for r in rows) for scope in ['full','local']};out['models'][model]=m
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
