"""Frozen cheap lexical chains on another reader development cohort."""
import json,re,time,math,sys
import run_qasper_fact_blocks as previous
from run_qasper_fact_blocks import ROOT,SOURCE,sha,reader,STOP,PRED,family
from run_qasper_paragraph_elements import spans,features
from build_qasper_relation_ledger import paragraphs
from run_qasper_contract_replication import contract
OUT=ROOT/'logs/qasper_relation_closure';ARMS=previous.ARMS
EDGE_STOP=STOP|set('we our all these those such also other more one two three first second approach information work proposed previous results data training network networks layer layers models methods'.split())
CONT=r'\b(?:consists? of|composed of|concatenat\w*|incorporat\w*|both|respectively|followed by|where|provided that|only if|the former|the latter|sub-network\w*|subnetwork\w*|component\w*)\b'

def words(text):return set(re.findall(r'\b[a-z][a-z0-9-]{2,}\b',text.lower()))-EDGE_STOP

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();start=time.perf_counter();raw=json.loads(SOURCE.read_text(encoding='utf-8'));oldpath=ROOT/'logs/qasper_attention_guard/frozen.json';old=json.loads(oldpath.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));seen=set()
    def walk(v):
        if isinstance(v,dict):
            if isinstance(v.get('paper_id'),str):seen.add(v['paper_id'])
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    for p in (ROOT/'logs').glob('*/frozen.json'):walk(json.loads(p.read_text(encoding='utf-8')))
    papers=[]
    for pid in sorted(split['heldout'],key=lambda x:sha(x.encode())):
        p=raw[pid];ps=paragraphs(p)
        if pid in seen or len(p['qas'])<2 or len(ps)>35 or sum(len(x['text']) for x in ps)>18000:continue
        papers.append({'paper_id':pid,'title':p['title'],'paragraphs':ps,'units':[{**u,'pid':i,'sid':j,'features':features(u['text'])} for i,para in enumerate(ps) for j,u in enumerate(spans(para['text']))],'questions':[{'id':q['question_id'],'question':q['question'],'kind':contract(q['question'])[0]} for q in p['qas'][:2]]})
        if len(papers)==6:break
    assert len(papers)==6;tasks=[]
    for p in papers:
        for q in p['questions']:
            terms=words(q['question']);fam=family(q['question']);sets={a:set() for a in ARMS};records=[]
            for i,para in enumerate(p['paragraphs']):
                us=[u for u in p['units'] if u['pid']==i];n=len(us);rank=sorted(us,key=lambda u:(-sum(old['weights'][x] for x in u['features']),u['sid']));base={u['sid'] for u in rank[:max(1,math.ceil(.6*n))]};pred=[bool(re.search(PRED.get(fam,PRED['method']),u['text'],re.I)) for u in us];edges=[]
                for j in range(n-1):
                    shared=sorted(words(us[j]['text']) & words(us[j+1]['text']));cont=bool(re.search(CONT,us[j+1]['text'],re.I))
                    if pred[j] and ((pred[j+1] and shared) or cont):edges.append({'from':j,'to':j+1,'shared_terms':shared,'continuation':cont})
                seeds=sorted([(len(terms&words(u['text']))+1,u['sid']) for u in us if pred[u['sid']]],key=lambda v:(-v[0],v[1]))[:1];chosen=set(base);component=set()
                if seeds:
                    component={seeds[0][1]};changed=True
                    while changed:
                        before=len(component)
                        for e in edges:
                            if e['from'] in component or e['to'] in component:component.update([e['from'],e['to']])
                        changed=before!=len(component)
                    chosen.update(range(n) if len(component)>4 else component)
                records.append({'pid':i,'seed':seeds,'edges':edges,'component':sorted(component),'full_paragraph_fallback':len(component)>4});wide={u['sid'] for u in rank[:len(chosen)]}
                for a,ids in [('full',range(n)),('elements',base),('block',chosen),('wide',wide)]:sets[a].update((i,j) for j in ids)
            tasks.append({'paper_id':p['paper_id'],'title':p['title'],**q,'relation_chains':records,'arms':{a:reader.render(p,k) for a,k in sets.items()}})
    prep=time.perf_counter()-start;files=[SOURCE,oldpath,ROOT/'logs/qasper_fingerprint/split.json',ROOT/'run_qasper_relation_closure.py',ROOT/'protocol_qasper_relation_closure_20261003.md',ROOT/'run_qasper_fact_blocks.py',ROOT/'run_qasper_attention_reader.py',ROOT/'run_qasper_attention_guard.py',ROOT/'run_qasper_paragraph_elements.py',ROOT/'build_qasper_relation_ledger.py',ROOT/'run_qasper_contract_replication.py']
    fr={'papers':papers,'excluded_paper_ids':sorted(seen),'development_reuse':False,'previously_scanned_partition':True,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'predicates':PRED,'continuation_regex':CONT,'edge_stopwords':sorted(EDGE_STOP)};(OUT/'frozen.json').write_text(json.dumps(fr,indent=2),encoding='utf-8');(OUT/'scores.json').write_text('{}',encoding='utf-8');(OUT/'selected.json').write_text(json.dumps({'frozen_sha256':sha((OUT/'frozen.json').read_bytes()),'scores_sha256':sha((OUT/'scores.json').read_bytes()),'preparation_wall_sec':prep,'tasks':tasks},indent=2),encoding='utf-8');print('prepared', [p['paper_id'] for p in papers],prep,flush=True)

def run():reader.OUT=OUT;reader.ARMS=ARMS;reader.run()
def analyze():
    previous.OUT=OUT;previous.analyze()
    p=OUT/'summary.json';d=json.loads(p.read_text(encoding='utf-8'));d['development_reuse']=False;d['previously_scanned_partition']=True;p.write_text(json.dumps(d,indent=2),encoding='utf-8')
if __name__=='__main__':{'prepare':prepare,'run':run,'analyze':analyze}[sys.argv[1]]()
