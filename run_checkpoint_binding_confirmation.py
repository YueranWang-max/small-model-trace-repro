"""New checkpoint cohort: trace/capsule order versus explicit operation binding."""
import hashlib,json,random,re,sys
from pathlib import Path
import run_model_checkpoint_order as order
import run_checkpoint_representation as representation
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/checkpoint_binding_confirmation'
ARMS=['pre_history_local','post_history_local','pre_selected_local','post_selected_local','action']
original_prompt=order.prompt
def sha(b):return hashlib.sha256(b).hexdigest()
def current(state):
    matches=re.findall(r'Current state: S(\d+)',state);assert len(matches)==1;return int(matches[0])
def prompt(t,arm,state):
    if arm=='action':return order.prior.prompt(t,'selected_chain','arrow',4,current(state))
    if 'selected' in arm:
        selected=load_selection();state=representation.state(dict(producer=[dict(output=current(state))],state=state),selected)
    return original_prompt(t,('pre_' if arm.startswith('pre') else 'post_')+'local',state)
def load_selection():return json.loads((OUT/'frozen.json').read_text())['selection']
def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'frozen.json').exists();sp=representation.OUT/'summary.json';s=json.loads(sp.read_text());assert s['audited_calls']==96
    scores={a:sum(s['models'][m][a]['conditioned_transition_correct'] for m in s['models']) for a in ['capsule','json','input_key']};selected=max(scores,key=scores.get);tasks=[]
    for i,seed in enumerate(range(6601,6617)):
        t=order.prior.make_task(seed,i,True);t['arm_order']=ARMS.copy();random.Random(seed+300000).shuffle(t['arm_order']);tasks.append(t)
    files=['run_checkpoint_binding_confirmation.py','protocol_checkpoint_binding_confirmation_20261003.md','run_checkpoint_representation.py','run_model_checkpoint_order.py','run_lookup_followup.py','run_lookup_semantics.py','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    f=dict(tasks=tasks,models=order.prior.dev.MODELS,selection=selected,selection_scores=scores,selection_summary_sha256=sha(sp.read_bytes()),adaptive=True,hashes={p:sha((ROOT/p).read_bytes()) for p in files});(OUT/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print('frozen',selected,scores,len(tasks))
def run():
    order.OUT=OUT;order.prompt=prompt;order.run()
def analyze():
    f=json.loads((OUT/'frozen.json').read_text());d=json.loads((OUT/'reader.json').read_text());assert len(d['rows'])==32;assert d['frozen_sha256']==sha((OUT/'frozen.json').read_bytes())
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    result=dict(adaptive=True,selection=f['selection'],models={},audited_calls=0,model_feedback_checks=0)
    def audit(model,p,v):
        assert sha(p.encode())==v['prompt_sha256'];raw=v.get('raw',{})
        if raw:
            payload=dict(model=model,messages=[dict(role='system',content=order.SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=order.prior.dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==raw['request_payload_sha256']
            if raw['schema_valid']:assert int(json.loads(raw['native_schema_content'])['state'][1:])==v['output']
        result['audited_calls']+=1
    def metrics(calls):return dict(calls=len(calls),tokens=sum((v.get('raw',{}).get('prompt_eval_count') or 0)+(v.get('raw',{}).get('eval_count') or 0) for v in calls),wall_sec=sum(v['wall_sec'] for v in calls),invalid=sum(not v.get('raw',{}).get('schema_valid',False) for v in calls),errors=sum('error' in v for v in calls))
    for model in f['models']:
        rows=[r for r in d['rows'] if r['model']==model];produced=[];m=dict(producer={},arms={})
        for row in rows:
            t=next(t for t in f['tasks'] if t['id']==row['id']);rng=random.Random(t['seed']);maps=[];orders=[]
            for _ in range(6):
                values=list(range(8));rng.shuffle(values);maps.append(dict(zip(map(str,range(8)),values)));values=list(range(8));rng.shuffle(values);orders.append(values)
            assert maps==t['maps'] and orders==t['row_orders'] and rng.randrange(8)==t['input'];state=t['input'];trajectory=[state]
            for mapping in maps:state=mapping[str(state)];trajectory.append(state)
            assert trajectory==t['trajectory']
            for i,v in enumerate(row['producer']):
                assert v['input']==(t['input'] if i==0 else row['producer'][i-1]['output']);result['model_feedback_checks']+=int(i>0);audit(model,order.prior.prompt(t,'selected_chain','arrow',i+1,v['input']),v);produced.append(v)
            if 'state' in row:assert row['state']==order.checkpoint(row['producer']) and row['state_sha256']==sha(row['state'].encode())
        m['producer']={**metrics(produced),'checkpoint_correct':sum(len(r['producer'])==3 and r['producer'][-1]['output']==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][3] for r in rows),'n':16}
        for arm in ARMS:
            calls=[];true=conditional=0
            for row in rows:
                if arm not in row['readers']:continue
                t=next(t for t in f['tasks'] if t['id']==row['id']);v=row['readers'][arm];assert v['input']==current(row['state']);p=prompt(t,arm,row['state']);audit(model,p,v);calls.append(v)
                # Independently parse all eight current source pairs, not just the answer edge.
                pairs=re.findall(r'(S\d+)->(S\d+)',p);mapping={int(a[1:]):int(b[1:]) for a,b in pairs};assert mapping=={int(k):val for k,val in t['maps'][3].items()}
                true+=v['output']==t['trajectory'][4];conditional+=v['output']==t['maps'][3][str(v['input'])]
            m['arms'][arm]={**metrics(calls),'true_correct':true,'conditioned_transition_correct':conditional,'n':16}
        m['paired_differences']={arm:dict(other_only_correct=sum(r['readers'].get(arm,{}).get('output')==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4] and r['readers'].get('pre_history_local',{}).get('output')!=next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4] for r in rows),history_only_correct=sum(r['readers'].get(arm,{}).get('output')!=next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4] and r['readers'].get('pre_history_local',{}).get('output')==next(t for t in f['tasks'] if t['id']==r['id'])['trajectory'][4] for r in rows)) for arm in ARMS[1:]};result['models'][model]=m
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
