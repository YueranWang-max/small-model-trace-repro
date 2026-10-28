import json,re,time,sys,urllib.request
import run_controlled_state_boundary as prior
from analyze_controlled_state_format import recover
ROOT=prior.ROOT;OUT=ROOT/'logs/controlled_checkpoint';ARMS=('correct_pre','correct_post','wrong_pre')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();oldpath=prior.OUT/'frozen.json';old=json.loads(oldpath.read_text(encoding='utf-8'));tasks=[]
    for t in old['tasks']:
        if t['kind']!='update':continue
        initial=int(re.search(r'S(\d+)',t['correct_state']).group(1));maps=[]
        for j in range(1,7):
            line=next(line for line in t['source'].splitlines() if line.startswith(f'Step {j}:'));maps.append({int(a):int(b) for a,b in re.findall(r'S(\d+)->S(\d+)',line)})
        state=initial
        for m in maps[:3]:state=m[state]
        current=state;badcurrent=(state+1)%8
        for m in maps[3:]:current=m[current];badcurrent=m[badcurrent]
        assert f'S{current:02d}'==t['answer'] and current!=badcurrent;tasks.append({**t,'checkpoint':f'Checkpoint after completing steps 1, 2 and 3: S{state:02d}. Resume with steps 4, 5 and 6 only.','wrong_checkpoint':f'Checkpoint after completing steps 1, 2 and 3: S{(state+1)%8:02d}. Resume with steps 4, 5 and 6 only.','wrong_checkpoint_answer':f'S{badcurrent:02d}'})
    files=[oldpath,ROOT/'run_controlled_checkpoint.py',ROOT/'protocol_controlled_checkpoint_20261003.md',ROOT/'run_controlled_state_boundary.py',ROOT/'analyze_controlled_state_format.py',ROOT/'repro_lib.py'];d={'adaptive_diagnostic':True,'tasks':tasks,'models':old['models'],'hashes':{str(p.relative_to(ROOT)):prior.sha(p.read_bytes()) for p in files}};(OUT/'frozen.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print('frozen checkpoint',len(tasks),flush=True)

def prompt(t,a):
    note='Working progress state (may be wrong; verify against source):\n'+(t['wrong_checkpoint'] if a=='wrong_pre' else t['checkpoint'])+'\n';source='Source:\n'+t['source']+'\n';return 'Use source as authority. Any working notes can be wrong.\n'+(source+note if a=='correct_post' else note+source)+'\nQuestion: '+t['question']+'\nOutput exactly Final Answer: <one code>, with no other text.'

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in f['hashes'].items():assert prior.sha((ROOT/p).read_bytes())==h
    path=OUT/'reader.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':prior.sha(fp.read_bytes()),'rows':[],'warmups':{}};assert d['frozen_sha256']==prior.sha(fp.read_bytes())
    for model in prior.MODELS:
        if model not in d['warmups']:g=prior.ollama_chat(model,'Follow source and output instructions.','Reply exactly Final Answer: ready',think=False,temperature=0,num_predict=16);d['warmups'][model]={'content':g.content}
        for i,t in enumerate(f['tasks']):
            r=next((r for r in d['rows'] if r['model']==model and r['id']==t['id']),None)
            if r is None:r={'model':model,'id':t['id'],'base_id':t['base_id'],'noise_rows':t['noise_rows'],'arms':{}};d['rows'].append(r)
            order=list(ARMS);off=i%3;order=order[off:]+order[:off]
            for a in order:
                if a in r['arms']:continue
                p=prompt(t,a);start=time.perf_counter()
                try:
                    g=prior.ollama_chat(model,'Follow source and output instructions.',p,think=False,temperature=0,num_ctx=32768,num_predict=256);matches=re.findall(r'(?im)^Final Answer:\s*([A-Za-z]\d+)\s*[.]?\s*$',g.content);v={'content':g.content,'answer':matches[-1].upper() if matches else '','raw':g.raw};value,reason=recover(v);v.update({'value':value,'recovery':reason,'em':int(value==t['answer']),'follows_wrong_checkpoint':int(value==t['wrong_checkpoint_answer'])})
                except Exception as e:v={'error':repr(e),'em':0,'follows_wrong_checkpoint':0,'value':''}
                v.update({'prompt_sha256':prior.sha(p.encode()),'wall_sec':time.perf_counter()-start});r['arms'][a]=v;path.write_text(json.dumps(d,indent=2),encoding='utf-8');print(model,i+1,a,v['em'],flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));base=json.loads((prior.OUT/'format_diagnostic.json').read_text(encoding='utf-8'));assert len(d['rows'])==16 and all(set(r['arms'])==set(ARMS) for r in d['rows']);out={'adaptive_diagnostic':True,'models':{}}
    for model in prior.MODELS:
        rs=[r for r in d['rows'] if r['model']==model];old=[r for r in base['rows'] if r['model']==model and r['kind']=='update'];out['models'][model]={'checkpoint_em':{a:sum(r['arms'][a]['em'] for r in rs)/8 for a in ARMS},'archived_initial_or_no_state_em':{a:sum(r['arms'][a]['em'] for r in old)/8 for a in prior.ARMS},'wrong_checkpoint_follow_rate':sum(r['arms']['wrong_pre']['follows_wrong_checkpoint'] for r in rs)/8,'strata':{str(n):{a:sum(r['arms'][a]['em'] for r in rs if r['noise_rows']==n)/4 for a in ARMS} for n in (16,128)},'errors':sum('error' in v for r in rs for v in r['arms'].values())}
    for r in d['rows']:
        t=next(t for t in f['tasks'] if t['id']==r['id'])
        for a,v in r['arms'].items():assert v['prompt_sha256']==prior.sha(prompt(t,a).encode())
    out['verified_prompts']=48;(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
