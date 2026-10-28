"""Check source blocks, actual checkpoint binding and request provenance."""
import csv,hashlib,json,random,re
from pathlib import Path
import run_model_checkpoint_order as order
import run_checkpoint_representation as representation
import run_checkpoint_binding_confirmation as binding
ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
results=[]
for name,expected in [('model_checkpoint_order',168),('checkpoint_representation',96),('checkpoint_binding_confirmation',256)]:
    folder=ROOT/'logs'/name;f=json.loads((folder/'frozen.json').read_text());d=json.loads((folder/'reader.json').read_text());assert d['frozen_sha256']==sha((folder/'frozen.json').read_bytes())
    for p,h in {**f['hashes'],**f.get('source_sha256',{})}.items():assert sha((ROOT/p).read_bytes())==h
    tasks={t['id']:t for t in f['tasks']};records=[];feedback=bound=0
    for t in tasks.values():
        rng=random.Random(t['seed']);maps=[];orders=[]
        for _ in range(6):
            values=list(range(8));rng.shuffle(values);maps.append(dict(zip(map(str,range(8)),values)));values=list(range(8));rng.shuffle(values);orders.append(values)
        assert maps==t['maps'] and orders==t['row_orders'] and rng.randrange(8)==t['input']
    def collect(model,t,p,v,step,full=False):
        assert sha(p.encode())==v['prompt_sha256']
        blocks=re.findall(r'Step (\d+):\n(.*?)(?=\n\nStep |\n\nState:|\n\nOperation:|\nOperation:|$)',p,re.S)
        assert [int(j) for j,_ in blocks]==(list(range(1,7)) if full else [step])
        for j,text in blocks:
            parsed={int(a):int(b) for a,b in re.findall(r'S(\d+)->S(\d+)',text)};assert parsed=={int(k):value for k,value in t['maps'][int(j)-1].items()}
        raw=v['raw'];obj=json.loads(raw['native_schema_content']);assert set(obj)=={'state'} and obj['state'] in order.prior.dev.interface.SCHEMA['properties']['state']['enum'];assert int(obj['state'][1:])==v['output'] and raw['schema_valid']
        payload=dict(model=model,messages=[dict(role='system',content=order.SYSTEM),dict(role='user',content=p)],stream=False,think=False,format=order.prior.dev.interface.SCHEMA,options=dict(temperature=0,num_ctx=32768,num_predict=256,top_p=.9));assert sha(json.dumps(payload,sort_keys=True).encode())==raw['request_payload_sha256'];records.append(v)
    if name=='checkpoint_representation':
        for row in d['rows']:
            t=tasks[row['id']];producer=next(r for r in f['producer_rows'] if (r['model'],r['id'])==(row['model'],row['id']));assert producer['state']==order.checkpoint(producer['producer']);assert row['input']==producer['producer'][-1]['output'];bound+=1
            p=order.prompt(t,'pre_local',representation.state(producer,row['arm']));collect(row['model'],t,p,row,4)
    else:
        for row in d['rows']:
            t=tasks[row['id']]
            for i,v in enumerate(row['producer']):
                assert v['input']==(t['input'] if i==0 else row['producer'][i-1]['output']);feedback+=int(i>0);collect(row['model'],t,order.prior.prompt(t,'selected_chain','arrow',i+1,v['input']),v,i+1)
            assert row['state']==order.checkpoint(row['producer']) and row['state_sha256']==sha(row['state'].encode())
            for arm,v in row['readers'].items():
                assert v['input']==row['producer'][-1]['output'];bound+=1
                p=order.prompt(t,arm,row['state']) if name=='model_checkpoint_order' else binding.prompt(t,arm,row['state'])
                collect(row['model'],t,p,v,4,full=arm.endswith('full'))
            if name=='model_checkpoint_order':
                for scope in ['full','local']:
                    pre=order.prompt(t,'pre_'+scope,row['state']);post=order.prompt(t,'post_'+scope,row['state']);assert pre.split('Operation:',1)[1]==post.split('Operation:',1)[1]
                    assert pre.startswith('State:\n'+row['state']) and post.startswith('Source:\n'+order.source(t,'post_'+scope))
    assert len(records)==expected
    result=dict(study=name,calls=len(records),http_attempts=sum(v['raw']['client_attempts'] for v in records),native_json_valid=len(records),model_feedback_checks=feedback,reader_checkpoint_binding_checks=bound,source_maps_and_frozen_code_verified=True)
    results.append(result);(folder/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(ROOT/'logs/checkpoint_audit.json').write_text(json.dumps(dict(studies=results,total_calls=sum(r['calls'] for r in results)),indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
folder=ROOT/'logs/checkpoint_binding_confirmation'
f=json.loads((folder/'frozen.json').read_text());d=json.loads((folder/'reader.json').read_text())
summary=json.loads((folder/'summary.json').read_text());diagnostics={};figure=[]
for model in f['models']:
    diagnostics[model]={}
    rows=[r for r in d['rows'] if r['model']==model]
    for arm in binding.ARMS:
        full=producer_full=checkpoint_correct=0;contingency=[]
        for row in rows:
            t=next(t for t in f['tasks'] if t['id']==row['id'])
            pf=all(v['output']==t['trajectory'][i+1] for i,v in enumerate(row['producer']))
            cc=row['producer'][-1]['output']==t['trajectory'][3]
            v=row['readers'][arm];truth=v['output']==t['trajectory'][4]
            cond=v['output']==t['maps'][3][str(v['input'])]
            producer_full+=pf;checkpoint_correct+=cc;full+=pf and truth
            contingency.append(dict(task=t['id'],producer_full=pf,checkpoint_correct=cc,true_final=truth,conditioned_transition=cond))
        diagnostics[model][arm]=dict(n=16,producer_full_correct=producer_full,checkpoint_correct=checkpoint_correct,full_four_step_correct=full,rows=contingency)
        metrics=summary['models'][model]['arms'][arm]
        figure.append(dict(model=model,arm=arm,n=16,conditioned_correct=metrics['conditioned_transition_correct'],true_correct=metrics['true_correct'],full_four_step_correct=full))
(folder/'trajectory_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
with (ROOT/'paper/figures/data/checkpoint_binding.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(figure[0]));writer.writeheader();writer.writerows(figure)
print('Four-step paths:',{m:{a:v['full_four_step_correct'] for a,v in arms.items()} for m,arms in diagnostics.items()})
