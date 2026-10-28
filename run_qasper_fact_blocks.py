"""Cheap bounded source neighborhoods; explicitly a reused development cohort."""
import json,re,time,math,sys,random
from pathlib import Path
import run_qasper_attention_reader as reader
from run_qasper_attention_guard import family, RULES
ROOT=reader.ROOT; SOURCE=reader.SOURCE; sha=reader.sha
OUT=ROOT/'logs/qasper_fact_blocks'; ARMS=('full','elements','block','wide')
STOP=set('what which how is are do does did can was were has have the a an they their paper authors model method use used using this that of in on for to and it with by as'.split())
PRED={**RULES,'baseline':RULES['baseline']+r'|\b(?:models?|train\w*|named?|called)\b','method':r'\b(?:combin\w*|consist\w*|compos\w*|compris\w*|contain\w*|component\w*|layer\w*|network\w*|convol\w*|architect\w*|explor\w*|extend\w*|divid\w*|reduc\w*|factoriz\w*|defin\w*|refer\w*|implement\w*|propos\w*)\b|\d'}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'selected.json').exists();start=time.perf_counter()
    oldpath=ROOT/'logs/qasper_attention_guard/frozen.json';fr=json.loads(oldpath.read_text(encoding='utf-8'));tasks=[]
    for p in fr['papers']:
        for q in p['questions']:
            terms=set(re.findall(r'\b[a-z][a-z0-9-]+\b',q['question'].lower()))-STOP;fam=family(q['question']);sets={a:set() for a in ARMS};blocks=[]
            for i,para in enumerate(p['paragraphs']):
                units=[u for u in p['units'] if u['pid']==i];n=len(units);k=max(1,math.ceil(.6*n));rank=sorted(units,key=lambda u:(-sum(fr['weights'][x] for x in u['features']),u['sid']));base={u['sid'] for u in rank[:k]};seeds=[]
                for u in units:
                    overlap=len(terms & set(re.findall(r'\b[a-z][a-z0-9-]+\b',u['text'].lower())));predicate=bool(re.search(PRED.get(fam,PRED['method']),u['text'],re.I));score=overlap+int(predicate)
                    if predicate and score>0:seeds.append((score,u['sid']))
                seeds=sorted(seeds,key=lambda x:(-x[0],x[1]))[:2];chosen=set(base)
                for score,j in seeds:
                    ids=list(range(max(0,j-1),min(n,j+2)));chosen.update(ids);blocks.append({'pid':i,'seed_sid':j,'sids':ids,'rank_score':score,'family':fam})
                wide={u['sid'] for u in rank[:len(chosen)]}
                for a,ids in [('full',range(n)),('elements',base),('block',chosen),('wide',wide)]:sets[a].update((i,j) for j in ids)
            tasks.append({'paper_id':p['paper_id'],'title':p['title'],**q,'fact_neighborhoods':blocks,'arms':{a:reader.render(p,s) for a,s in sets.items()}})
    prep=time.perf_counter()-start
    files=[oldpath,SOURCE,ROOT/'run_qasper_fact_blocks.py',ROOT/'protocol_qasper_fact_blocks_20261003.md',ROOT/'run_qasper_attention_reader.py',ROOT/'run_qasper_attention_guard.py']
    frozen={**fr,'development_reuse':True,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'neighborhood_predicates':PRED};(OUT/'frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    selected={'frozen_sha256':sha((OUT/'frozen.json').read_bytes()),'scores_sha256':sha((OUT/'scores.json').read_bytes()) if (OUT/'scores.json').exists() else None,'preparation_wall_sec':prep,'tasks':tasks};(OUT/'scores.json').write_text('{}',encoding='utf-8');selected['scores_sha256']=sha((OUT/'scores.json').read_bytes());(OUT/'selected.json').write_text(json.dumps(selected,indent=2),encoding='utf-8');print('prepared',len(tasks),prep,flush=True)

def run():
    reader.OUT=OUT;reader.ARMS=ARMS;reader.run()

def analyze():
    d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));s=json.loads((OUT/'selected.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows);groups=[[r for r in rows if r['paper_id']==p] for p in sorted({r['paper_id'] for r in rows})];out={'development_reuse':True,'preparation_wall_sec':s['preparation_wall_sec'],'arms':{},'paired':{}}
    for a in ARMS:
        rs=[r['arms'][a] for r in rows];out['arms'][a]={'f1':sum(r['official_f1'] for r in rs)/12,'reader_tokens':sum((r.get('prompt_tokens') or 0)+(r.get('generation_tokens') or 0) for r in rs),'read_wall_sec':sum(r['wall_sec'] for r in rs),'errors':sum('error' in r for r in rs),'retention':sum(t['arms'][a]['retained_chars'] for t in s['tasks'])/sum(t['arms'][a]['source_chars'] for t in s['tasks'])}
    for a,b in [('block','wide'),('block','elements'),('block','full')]:
        rng=random.Random(20261003);bs=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms'][a]['official_f1']-r['arms'][b]['official_f1'] for r in sample)/12)
        bs.sort();out['paired'][a+'_minus_'+b]={'diff':out['arms'][a]['f1']-out['arms'][b]['f1'],'ci95':[bs[250],bs[9750]]}
    out['reader_token_reduction']=1-out['arms']['block']['reader_tokens']/out['arms']['full']['reader_tokens'];out['benefit_gate']=all(out['paired']['block_minus_'+b]['ci95'][0]>0 for b in ['wide','elements']);out['quality_budget_gate']=out['paired']['block_minus_full']['ci95'][0]>=-.02 and out['reader_token_reduction']>=.1 and not out['arms']['block']['errors'];(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'prepare':prepare,'run':run,'analyze':analyze}[sys.argv[1]]()
