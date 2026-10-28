"""Oracle intermediate states isolate consumer failures from producer errors."""
import json,re,random,time,sys,urllib.request
from pathlib import Path
from repro_lib import ollama_chat
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/controlled_state_boundary';MODELS=('qwen3:1.7b','qwen3.5:4b');ARMS=('no_state','correct_pre','correct_post','wrong_pre')
import hashlib
def sha(v):return hashlib.sha256(v).hexdigest()

def instance(seed,kind,noise):
    rng=random.Random(seed);key='RUN'+str(seed);alias=rng.randrange(8);initial=rng.randrange(8);values=rng.sample(range(100,999),8);maps=[]
    for _ in range(6):
        perm=list(range(8));rng.shuffle(perm);maps.append(perm)
    registry=[f'{key}: active object A{alias:02d}; initial state S{initial:02d}.']+[f'RUN{seed+j+100}: active object A{j:02d}; initial state S{(j+3)%8:02d}.' for j in range(8)];rng.shuffle(registry)
    core=['Registry:']+registry
    if kind=='binding':
        rows=[f'Object A{i:02d} has delivery code V{values[i]}.' for i in range(8)];rng.shuffle(rows);core+=['Object records:']+rows;answer=f'V{values[alias]}';wrong=f'V{values[(alias+1)%8]}';state=f'Active object for {key}: A{alias:02d}.';bad=f'Active object for {key}: A{(alias+1)%8:02d}.';question=f'What is the delivery code of the active object for {key}?'
    else:
        current=initial;badcurrent=(initial+1)%8
        for perm in maps:current=perm[current];badcurrent=perm[badcurrent]
        recompute=initial
        for step in range(6):recompute=maps[step][recompute]
        assert recompute==current and current!=badcurrent
        core+=['Apply steps 1 through 6 in order. At each step use exactly the mapping for the current state.']+[f'Step {j+1}: '+ '; '.join(f'S{i:02d}->S{v:02d}' for i,v in enumerate(perm))+'.' for j,perm in enumerate(maps)];answer=f'S{current:02d}';wrong=f'S{badcurrent:02d}';state=f'Initial state for {key}: S{initial:02d}.';bad=f'Initial state for {key}: S{(initial+1)%8:02d}.';question=f'For {key}, what state is reached after applying all six steps?'
    noise_rng=random.Random(seed+9923);filler=[f'Archive ticket N{i:04d} records unrelated object B{noise_rng.randrange(1000):03d} and status H{noise_rng.randrange(1000):03d}; these archive tickets do not change registry or object records.' for i in range(noise)];cut=len(filler)//2;source='\n'.join(filler[:cut]+core+filler[cut:]);assert answer!=wrong;assert f'{key}: active object A{alias:02d}; initial state S{initial:02d}.' in source
    return {'id':f'{kind}_{seed}_{noise}','base_id':f'{kind}_{seed}','seed':seed,'kind':kind,'noise_rows':noise,'source':source,'source_sha256':sha(source.encode()),'question':question,'correct_state':state,'wrong_state':bad,'answer':answer,'wrong_state_answer':wrong}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();tags=json.load(urllib.request.urlopen('http://localhost:11434/api/tags'));ids={m:next(v for v in tags['models'] if v['name']==m) for m in MODELS};tasks=[instance(seed,kind,n) for seed in range(4701,4705) for kind in ('binding','update') for n in (16,128)];assert len(tasks)==16
    for t in tasks:assert instance(t['seed'],t['kind'],t['noise_rows'])==t
    files=[ROOT/'run_controlled_state_boundary.py',ROOT/'protocol_controlled_state_boundary_20261003.md',ROOT/'repro_lib.py'];f={'tasks':tasks,'models':ids,'arms':ARMS,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}};(OUT/'frozen.json').write_text(json.dumps(f,indent=2),encoding='utf-8');print('frozen',len(tasks),flush=True)

def prompt(t,a):
    source='Source:\n'+t['source']+'\n';notes='Working state (may be wrong; verify against source):\n'+(t['wrong_state'] if a=='wrong_pre' else t['correct_state'])+'\n'
    body=source if a=='no_state' else source+notes if a=='correct_post' else notes+source
    return 'Use the source as authority. Any working notes can be wrong.\n'+body+'\nQuestion: '+t['question']+'\nOutput exactly Final Answer: <one code>, with no other text.'

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    path=OUT/'reader.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[],'warmups':{}};assert d['frozen_sha256']==sha(fp.read_bytes())
    for model in MODELS:
        if model not in d['warmups']:
            g=ollama_chat(model,'Follow source and output instructions.','Reply exactly Final Answer: ready',think=False,temperature=0,num_predict=16);d['warmups'][model]={'content':g.content,'raw':g.raw}
        for i,t in enumerate(f['tasks']):
            r=next((r for r in d['rows'] if r['model']==model and r['id']==t['id']),None)
            if r is None:r={'model':model,'id':t['id'],'base_id':t['base_id'],'kind':t['kind'],'noise_rows':t['noise_rows'],'arms':{}};d['rows'].append(r)
            order=list(ARMS);off=i%4;order=order[off:]+order[:off]
            for a in order:
                if a in r['arms']:continue
                p=prompt(t,a);start=time.perf_counter()
                try:
                    g=ollama_chat(model,'Follow source and output instructions.',p,think=False,temperature=0,num_ctx=32768,num_predict=256);matches=re.findall(r'(?im)^Final Answer:\s*([A-Za-z]\d+)\s*[.]?\s*$',g.content);v=matches[-1].upper() if matches else '';out={'content':g.content,'thinking':g.thinking,'answer':v,'em':int(v==t['answer']),'follows_wrong_state':int(v==t['wrong_state_answer']),'prompt_tokens':g.raw.get('prompt_eval_count'),'generation_tokens':g.raw.get('eval_count'),'raw':g.raw}
                except Exception as e:out={'error':repr(e),'answer':'','em':0,'follows_wrong_state':0}
                out.update({'wall_sec':time.perf_counter()-start,'prompt_sha256':sha(p.encode())});r['arms'][a]=out;path.write_text(json.dumps(d,indent=2),encoding='utf-8');print(model,i+1,a,out['em'],flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));assert len(d['rows'])==32 and all(set(r['arms'])==set(ARMS) for r in d['rows']);out={'models':{}}
    for model in MODELS:
        rs=[r for r in d['rows'] if r['model']==model];groups=[[r for r in rs if r['base_id']==b] for b in sorted({r['base_id'] for r in rs})];m={'arms':{},'paired':{},'strata':{}}
        for a in ARMS:m['arms'][a]={'em':sum(r['arms'][a]['em'] for r in rs)/16,'tokens':sum((r['arms'][a].get('prompt_tokens') or 0)+(r['arms'][a].get('generation_tokens') or 0) for r in rs),'wall_sec':sum(r['arms'][a]['wall_sec'] for r in rs),'errors':sum('error' in r['arms'][a] for r in rs),'empty_answers':sum(not r['arms'][a]['answer'] for r in rs)}
        m['wrong_state_follow_rate']=sum(r['arms']['wrong_pre']['follows_wrong_state'] for r in rs)/16
        for kind in ('binding','update'):
            for n in (16,128):
                rows=[r for r in rs if r['kind']==kind and r['noise_rows']==n];m['strata'][f'{kind}_{n}']={a:sum(r['arms'][a]['em'] for r in rows)/len(rows) for a in ARMS}
        for b in ('no_state','correct_post'):
            rng=random.Random(20261003);bs=[]
            for _ in range(10000):
                sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms']['correct_pre']['em']-r['arms'][b]['em'] for r in sample)/16)
            bs.sort();m['paired']['pre_minus_'+b]={'diff':m['arms']['correct_pre']['em']-m['arms'][b]['em'],'ci95':[bs[250],bs[9750]]}
        m['placement_gate']=m['paired']['pre_minus_correct_post']['ci95'][0]>0;m['oracle_usefulness_gate']=m['paired']['pre_minus_no_state']['ci95'][0]>0;out['models'][model]=m
    for r in d['rows']:
        t=next(t for t in f['tasks'] if t['id']==r['id'])
        for a,v in r['arms'].items():assert v['prompt_sha256']==sha(prompt(t,a).encode())
    out['reconstructed_prompt_hashes']=128;(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
