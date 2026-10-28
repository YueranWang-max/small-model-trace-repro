"""New-seed replication and model-only six-step propagation."""
import hashlib,json,random,sys,time
from pathlib import Path
import run_lookup_semantics as dev

ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def make_task(seed,index,chain):
    rng=random.Random(seed);maps=[];orders=[]
    for _ in range(6):
        values=list(range(8));rng.shuffle(values);maps.append(dict(zip(map(str,range(8)),values)))
        order=list(range(8));rng.shuffle(order);orders.append(order)
    initial=rng.randrange(8);trajectory=[initial]
    for mapping in maps:trajectory.append(mapping[str(trajectory[-1])])
    step=index%6+1
    noise='\n'.join(f'Archive ticket N{j:04d} records unrelated object B{rng.randrange(1000):03d} and status H{rng.randrange(1000):03d}; these tickets do not change any step table.' for j in range(128))
    arms=['continuous','legacy_chain','selected_chain'] if chain else ['selected_local','selected_all','selected_noise','legacy_local'];rng.shuffle(arms)
    return dict(id=f'followup_{seed}',seed=seed,maps=maps,row_orders=orders,step=step,input=initial,answer=maps[step-1][str(initial)],trajectory=trajectory,noise=noise,arm_order=arms)
def prompt(t,arm,fmt,step,current):
    t={**t,'step':step,'input':current}
    if arm=='legacy_chain' or arm=='legacy_local':return dev.prompt(t,'legacy_local')
    if arm=='selected_chain':return dev.prompt(t,fmt+'_local')
    if arm.startswith('selected_'):return dev.prompt(t,fmt+'_'+arm.split('_')[1])
    source=t['noise']+'\n'+'\n\n'.join(dev.render(t,fmt,j) for j in range(1,7))
    return 'Source:\n'+source+f'\nOperation: Initial state is S{current:02d}. Execute Step 1 through Step 6 in order. At each step, perform exactly ONE lookup using that step table and the current input. Use its output as the input to the NEXT step only. Return the state after Step 6.\nReturn a JSON object with exactly one field state, chosen from S00 through S07. No explanation.'
def freeze(phase):
    folder=ROOT/'logs'/('lookup_'+phase);folder.mkdir(parents=True,exist_ok=True);assert not(folder/'frozen.json').exists()
    sp=dev.OUT/'summary.json';summary=json.loads(sp.read_text());assert summary['audited_calls']==160
    selection={m:max(['arrow','json','prose'],key=lambda fmt:summary['models'][m][fmt+'_local']['correct']) for m in dev.MODELS}
    chain=phase=='chain';seeds=range(6301,6309) if chain else range(6201,6225)
    files=['run_lookup_followup.py','protocol_lookup_confirmation_20261003.md','run_lookup_semantics.py','run_controlled_step_schema.py','run_controlled_step_execution.py','run_controlled_state_boundary.py','repro_lib.py']
    f=dict(phase=phase,selection=selection,selection_summary_sha256=sha(sp.read_bytes()),tasks=[make_task(s,i,chain) for i,s in enumerate(seeds)],models=dev.MODELS,hashes={p:sha((ROOT/p).read_bytes()) for p in files})
    (folder/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print(phase,'frozen',selection,len(f['tasks']))
def run(phase):
    folder=ROOT/'logs'/('lookup_'+phase);fp=folder/'frozen.json';f=json.loads(fp.read_text());rp=folder/'reader.json'
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    d=json.loads(rp.read_text()) if rp.exists() else dict(frozen_sha256=sha(fp.read_bytes()),warmups={},rows=[]);assert d['frozen_sha256']==sha(fp.read_bytes())
    def save():rp.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in f['models']:
        if model not in d['warmups']:
            g=dev.interface.native(model,dev.SYSTEM,'Return {"state":"S00"}.',num_ctx=32768,num_predict=256);d['warmups'][model]=g.raw;save()
        for t in f['tasks']:
            for arm in t['arm_order']:
                row=next((r for r in d['rows'] if (r['model'],r['id'],r['arm'])==(model,t['id'],arm)),None)
                if row is None:row=dict(model=model,id=t['id'],arm=arm,calls=[]);d['rows'].append(row)
                count=6 if arm.endswith('chain') else 1
                while len(row['calls'])<count:
                    i=len(row['calls']);step=i+1 if phase=='chain' else t['step'];current=t['input'] if i==0 else row['calls'][-1]['output']
                    if current is None:row['terminated_invalid']=True;break
                    start=time.perf_counter();p=prompt(t,arm,f['selection'][model],step,current);prep=time.perf_counter()-start;start=time.perf_counter()
                    try:
                        g=dev.interface.native(model,dev.SYSTEM,p,num_ctx=32768,num_predict=256);value=int(json.loads(g.raw['native_schema_content'])['state'][1:]) if g.raw['schema_valid'] else None;call=dict(output=value,raw=g.raw)
                    except Exception as e:call=dict(output=None,error=repr(e))
                    call.update(step=step,input=current,prompt_sha256=sha(p.encode()),preparation_sec=prep,wall_sec=time.perf_counter()-start);row['calls'].append(call);save()
                row['complete']=len(row['calls'])==count;save()
            print(phase,model,t['id'],'done',flush=True)
def analyze(phase):
    folder=ROOT/'logs'/('lookup_'+phase);f=json.loads((folder/'frozen.json').read_text());d=json.loads((folder/'reader.json').read_text());assert len(d['rows'])==len(f['tasks'])*2*len(f['tasks'][0]['arm_order'])
    result=dict(phase=phase,selection=f['selection'],models={},audited_calls=0,model_feedback_checks=0)
    for model in f['models']:
        result['models'][model]={}
        for arm in f['tasks'][0]['arm_order']:
            rows=[r for r in d['rows'] if r['model']==model and r['arm']==arm];correct=perfect=transitions=0;calls=[]
            for r in rows:
                t=next(t for t in f['tasks'] if t['id']==r['id']);state=t['input'];path=[state]
                for mapping in t['maps']:assert sorted(map(int,mapping))==list(range(8));assert sorted(mapping.values())==list(range(8));state=mapping[str(state)];path.append(state)
                assert path==t['trajectory']
                for i,v in enumerate(r['calls']):
                    assert v['input']==(t['input'] if i==0 else r['calls'][i-1]['output']);result['model_feedback_checks']+=int(i>0)
                    p=prompt(t,arm,f['selection'][model],v['step'],v['input']);assert sha(p.encode())==v['prompt_sha256']
                    if v.get('raw'):
                        payload=dict(model=model,messages=[dict(role='system',content=dev.SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==v['raw']['request_payload_sha256']
                        if v['raw']['schema_valid']:assert int(json.loads(v['raw']['native_schema_content'])['state'][1:])==v['output']
                    transitions+=v['output']==t['maps'][v['step']-1][str(v['input'])];result['audited_calls']+=1;calls.append(v)
                expected=path[-1] if phase=='chain' else t['answer'];correct+=r['complete'] and r['calls'][-1]['output']==expected
                perfect+=r['complete'] and arm.endswith('chain') and all(v['output']==path[i+1] for i,v in enumerate(r['calls']))
            result['models'][model][arm]=dict(correct=int(correct),n=len(rows),perfect_chains=int(perfect) if arm.endswith('chain') else None,correct_transitions=int(transitions) if arm.endswith('chain') else None,calls=len(calls),tokens=sum((v.get('raw',{}).get('prompt_eval_count') or 0)+(v.get('raw',{}).get('eval_count') or 0) for v in calls),wall_sec=sum(v['wall_sec'] for v in calls),preparation_sec=sum(v['preparation_sec'] for v in calls),invalid=sum(not v.get('raw',{}).get('schema_valid',False) for v in calls),errors=sum('error' in v for v in calls))
        if phase=='confirmation':
            indexed={(r['id'],r['arm']):r['calls'][0]['output'] for r in d['rows'] if r['model']==model};pairs={}
            for other in ['selected_noise','selected_all','legacy_local']:
                wins=losses=0
                for t in f['tasks']:
                    left=indexed[(t['id'],'selected_local')]==t['answer'];right=indexed[(t['id'],other)]==t['answer'];wins+=left and not right;losses+=right and not left
                pairs[other]=dict(local_only_correct=wins,other_only_correct=losses)
            result['models'][model]['paired_disagreements']=pairs
    (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]](sys.argv[2])
