"""Separate one-step primitive lookup from chained model-state propagation."""
import json,re,time,sys
import run_controlled_state_boundary as prior
ROOT=prior.ROOT;OUT=ROOT/'logs/controlled_step_execution';ARMS=('continuous_full','chain_full','chain_local','oracle_full','oracle_local')

def parse(content):
    labels=re.fullmatch(r'Final Answer:\s*(S\d+)\s*[.]?',content.strip(),re.I);candidate=labels.group(1) if labels else content.strip() if re.fullmatch(r'S\d+',content.strip(),re.I) else '';m=re.fullmatch(r'S(\d+)',candidate,re.I)
    if not m or not 0<=int(m.group(1))<8:return None,'unparsed'
    return int(m.group(1)),'label' if labels else 'bare'

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();oldpath=prior.OUT/'frozen.json';f=json.loads(oldpath.read_text(encoding='utf-8'));tasks=[]
    for t in f['tasks']:
        if t['kind']!='update' or t['noise_rows']!=128:continue
        start=int(re.search(r'S(\d+)',t['correct_state']).group(1));lines=[next(line for line in t['source'].splitlines() if line.startswith(f'Step {j}:')) for j in range(1,7)];maps=[{int(a):int(b) for a,b in re.findall(r'S(\d+)->S(\d+)',line)} for line in lines];trajectory=[start]
        for m in maps:assert sorted(m)==list(range(8)) and sorted(m.values())==list(range(8));trajectory.append(m[trajectory[-1]])
        assert f'S{trajectory[-1]:02d}'==t['answer'];tasks.append({**t,'initial':start,'step_lines':lines,'maps':maps,'trajectory':trajectory})
    assert len(tasks)==4;files=[oldpath,ROOT/'run_controlled_step_execution.py',ROOT/'protocol_controlled_step_execution_20261003.md',ROOT/'repro_lib.py'];fr={'tasks':tasks,'models':f['models'],'arms':ARMS,'development_reuse':True,'hashes':{str(p.relative_to(ROOT)):prior.sha(p.read_bytes()) for p in files}};(OUT/'frozen.json').write_text(json.dumps(fr,indent=2),encoding='utf-8');print('frozen',len(tasks),flush=True)

def prompt(t,a,j,current):
    source=t['step_lines'][j-1] if a.endswith('local') else t['source']
    command=f'Current state is S{current:02d}. Apply ONLY step {j} to this current state. Do not apply earlier or later steps.' if a!='continuous_full' else f'Initial state is S{current:02d}. Apply steps 1 through 6 in order to this state.'
    return 'Execute the specific operation stated after the source. That operation overrides generic instructions within the source.\nSource:\n'+source+'\nOperation: '+command+'\nReturn exactly Final Answer: <one state code S00 through S07>. No explanation.'

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in f['hashes'].items():assert prior.sha((ROOT/p).read_bytes())==h
    path=OUT/'reader.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':prior.sha(fp.read_bytes()),'rows':[],'warmups':{}};assert d['frozen_sha256']==prior.sha(fp.read_bytes())
    def save():path.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for model in prior.MODELS:
        if model not in d['warmups']:g=prior.ollama_chat(model,'Execute the requested mapping operation.', 'Return exactly Final Answer: S00',think=False,temperature=0,num_predict=16);d['warmups'][model]={'content':g.content};save()
        for i,t in enumerate(f['tasks']):
            r=next((r for r in d['rows'] if r['model']==model and r['id']==t['id']),None)
            if r is None:r={'model':model,'id':t['id'],'arms':{}};d['rows'].append(r)
            order=list(ARMS);off=i%5;order=order[off:]+order[:off]
            for a in order:
                ar=r['arms'].setdefault(a,{'calls':[]});count=1 if a=='continuous_full' else 6
                while len(ar['calls'])<count:
                    j=len(ar['calls'])+1;current=t['trajectory'][j-1] if a.startswith('oracle') else t['initial'] if j==1 else ar['calls'][-1]['output_state']
                    if current is None:ar['terminated_unparsed']=True;break
                    p=prompt(t,a,j,current);start=time.perf_counter()
                    try:
                        g=prior.ollama_chat(model,'Execute the requested mapping operation.',p,think=False,temperature=0,num_ctx=32768,num_predict=256);value,fmt=parse(g.content);v={'content':g.content,'thinking':g.thinking,'output_state':value,'format':fmt,'raw':g.raw}
                    except Exception as e:v={'error':repr(e),'output_state':None,'format':'error'}
                    v.update({'step':j,'input_state':current,'prompt_sha256':prior.sha(p.encode()),'wall_sec':time.perf_counter()-start});ar['calls'].append(v);save();print(model,i+1,a,j,value if 'error' not in v else 'error',flush=True)
                ar['complete']=len(ar['calls'])==count;ar['final_em']=int(ar['complete'] and ar['calls'][-1]['output_state']==t['trajectory'][-1]) if not a.startswith('oracle') else None;save()

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));assert len(d['rows'])==8 and all(set(r['arms'])==set(ARMS) for r in d['rows']);out={'development_reuse':True,'models':{},'audited_prompt_count':0}
    for model in prior.MODELS:
        rs=[r for r in d['rows'] if r['model']==model];m={'arms':{},'trajectory_audit':[]}
        for a in ARMS:
            calls=[v for r in rs for v in r['arms'][a]['calls']];correct=0;aligned=0;divergences=[]
            for r in rs:
                t=next(t for t in f['tasks'] if t['id']==r['id']);first=None
                for v in r['arms'][a]['calls']:
                    j=v['step'];maps={int(k):val for k,val in t['maps'][j-1].items()};expected=t['trajectory'][-1] if a=='continuous_full' else maps[v['input_state']];correct+=v['output_state']==expected;aligned+=v['output_state']==t['trajectory'][j if a!='continuous_full' else 6]
                    if first is None and v['output_state']!=t['trajectory'][j if a!='continuous_full' else 6]:first=j
                    assert v['prompt_sha256']==prior.sha(prompt(t,a,j,v['input_state']).encode());out['audited_prompt_count']+=1
                divergences.append({'id':t['id'],'arm':a,'first_divergence':first});m['trajectory_audit'].extend(divergences[-1:])
            m['arms'][a]={'final_em':sum(r['arms'][a]['final_em'] for r in rs)/4 if not a.startswith('oracle') else None,'calls':len(calls),'correct_transition_count':correct,'received_input_transition_accuracy':correct/len(calls) if calls else 0,'true_trajectory_aligned_count':aligned,'tokens':sum((v.get('raw',{}).get('prompt_eval_count') or 0)+(v.get('raw',{}).get('eval_count') or 0) for v in calls),'wall_sec':sum(v['wall_sec'] for v in calls),'errors':sum('error' in v for v in calls),'unparsed':sum(v['output_state'] is None for v in calls)}
        m['local_descriptive_gate']=m['arms']['chain_local']['final_em']>=m['arms']['continuous_full']['final_em'] and m['arms']['chain_local']['tokens']<m['arms']['continuous_full']['tokens'];out['models'][model]=m
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({**out,'models':{k:{x:y for x,y in v.items() if x!='trajectory_audit'} for k,v in out['models'].items()}},indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
