"""Frozen real-document state/routing bridge. No answer annotation in method."""
import json, random, re, sys, time, urllib.request
from pathlib import Path
from run_qasper_prefix_vs_bm25 import ROOT, sha
from run_qasper_fingerprint import prepare, evidence
from run_stageC import BM25
from rescore_qasper_demand_transfer import references, f1, normalize

OUT=ROOT/'logs/qasper_state_resume'
SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_state_resume_20261003.md'
MODELS=['qwen3:1.7b','qwen3.5:4b']
ARMS=['full','base','wide','reread','state_wide','state_adaptive']
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
ANSWER={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
STATE={'type':'object','properties':{'known':{'type':'string'},'missing_queries':{'type':'array','items':{'type':'string'},'maxItems':3},'draft':{'type':'string'}},'required':['known','missing_queries','draft'],'additionalProperties':False}

def dependencies():
    names=[Path(__file__).name,'run_qasper_prefix_vs_bm25.py','run_qasper_fingerprint.py','run_stageC.py','rescore_qasper_demand_transfer.py']
    return {n:sha((ROOT/n).read_bytes()) for n in names}

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def freeze():
    OUT.mkdir(exist_ok=True,parents=True); path=OUT/'frozen.json'; assert not path.exists()
    src=json.loads(SOURCE.read_text(encoding='utf-8')); split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    t=time.perf_counter(); ps=[]
    eligible=[pid for pid in split['heldout'] if len(src[pid]['qas'])>=2]
    for pid in eligible[24:30]:
        p=prepare(pid,src[pid]);p['questions']=p['questions'][:2];p['title']=src[pid]['title']
        offset=0
        for para in p['paragraphs']:
            para['serialized_char_start']=offset;para['text_sha256']=sha(para['text'].encode());offset+=len(para['text'])+2
        p['serialized_source_sha256']=sha('\n\n'.join(x['text'] for x in p['paragraphs']).encode());ps.append(p)
    assert len(ps)==6 and sum(len(p['questions']) for p in ps)==12
    save(path,{'source_sha256':sha(SOURCE.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'dependencies':dependencies(),'models':MODELS,'papers':ps,'prepare_sec':time.perf_counter()-t})
    print('Frozen 6 papers,12 questions,2 models,168 formal calls',flush=True)

def call(model,prompt,schema,cap):
    payload={'model':model,'messages':[{'role':'system','content':'Use only supplied research-paper evidence. Unsupported answers must be unknown. Output valid JSON.'},{'role':'user','content':prompt}], 'format':schema,'stream':False,'think':False,'options':{'temperature':0.0,'num_ctx':32768,'num_predict':cap}}
    r={'request':payload,'request_sha256':sha(json.dumps(payload,sort_keys=True).encode()),'http_attempts':1};t=time.perf_counter()
    try:
        req=urllib.request.Request('http://localhost:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(req,timeout=900) as resp:raw=json.loads(resp.read())
        r['raw']=raw;r['parsed']=json.loads(raw['message']['content'])
    except Exception as exc:r['error']=repr(exc);r['parsed']={}
    r['wall_sec']=time.perf_counter()-t;return r

def material(p,ids):return '\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n{p['paragraphs'][i]['text']}" for i in ids)

def route(p,q,state):
    ids=q['rank'][:8].copy(); idx=BM25([x['text'] for x in p['paragraphs']]);queries=state.get('missing_queries',[])
    queries=[x[:200] for x in queries[:3] if isinstance(x,str) and x.strip()] if isinstance(queries,list) else []
    ranks=[idx.search(x,len(p['paragraphs'])) for x in queries]
    for depth in range(len(p['paragraphs'])):
        for rank in ranks:
            i=rank[depth]
            if i not in ids:ids.append(i)
            if len(ids)>=12:return ids,queries
    for i in q['rank']:
        if i not in ids:ids.append(i)
        if len(ids)>=12:break
    return ids,queries

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8')); assert fr['dependencies']==dependencies() and fr['source_sha256']==sha(SOURCE.read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes())
    path=OUT/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'models':{}}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    for mi,model in enumerate(MODELS):
        block=data['models'].setdefault(model,{'records':[]})
        if 'metadata' not in block:
            with OPENER.open('http://localhost:11434/api/tags') as resp:block['metadata']=json.loads(resp.read())
        if 'warmup' not in block:block['warmup']=call(model,'Answer ready.',ANSWER,16);save(path,data)
        for pi,p in enumerate(fr['papers']):
            for qi,q in enumerate(p['questions']):
                row=next((x for x in block['records'] if x['id']==q['id']),None)
                if row is None:row={'paper_id':p['id'],'id':q['id'],'arms':{}};block['records'].append(row)
                query='Paper: '+p['title']+'\nQuestion: '+q['text']
                if 'producer' not in row:
                    prompt=query+'\nEvidence:\n'+material(p,q['rank'][:8])+'\nRecord only supported facts in known. Identify unanswered parts and emit up to three targeted search queries in missing_queries; empty list if sufficient. Give a provisional short answer in draft, or unknown. Do not invent values from table captions.'
                    row['producer']=call(model,prompt,STATE,384);save(path,data)
                if 'adaptive_ids' not in row:
                    t=time.perf_counter();row['adaptive_ids'],row['routing_queries']=route(p,q,row['producer']['parsed']);row['route_sec']=time.perf_counter()-t
                    order=ARMS.copy();random.Random(20261003+pi*100+qi+mi*10000).shuffle(order);row['arm_order']=order;save(path,data)
                for arm in row['arm_order']:
                    if arm in row['arms']:continue
                    ids=list(range(len(p['paragraphs']))) if arm=='full' else q['rank'][:8] if arm=='base' else row['adaptive_ids'] if arm=='state_adaptive' else q['rank'][:12]
                    prompt=query+'\nEvidence:\n'+material(p,ids)
                    if arm.startswith('state_'):prompt+='\nPrevious reading state (may be wrong; verify every claim):\n'+json.dumps(row['producer']['parsed'],ensure_ascii=False)+'\nUse known facts and resolve missing requirements from the evidence above.'
                    prompt+='\nReturn only the requested short answer in answer. Include all requested entities and conditions. If evidence is insufficient answer unknown. A caption without table values does not establish numeric values.'
                    result=call(model,prompt,ANSWER,768);result['paragraph_ids']=ids;row['arms'][arm]=result;save(path,data)
                    print(model,pi+1,qi+1,arm,'error' if 'error' in result else 'done',flush=True)

def analyze():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));data=json.loads((OUT/'results.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'));out={'models':{},'prepare_sec':fr['prepare_sec'],'scope':'exploratory; no untouched holdout or original benchmark claim'}
    assert fr['dependencies']==dependencies()
    for model,block in data['models'].items():
        rows=block['records'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows)
        summary={'arms':{},'paired':{},'warmup_sec':block['warmup']['wall_sec'],'route_sec':sum(r['route_sec'] for r in rows)}
        for row in rows:
            rawq=next(q for q in src[row['paper_id']]['qas'] if q['question_id']==row['id']);refs=references(rawq)
            p=next(p for p in fr['papers'] if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id']);ev=evidence(p,q,src)
            for arm,r in row['arms'].items():
                pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred
                r['official_f1']=max(f1(aligned,g) for g in refs);r['literal_f1']=max(f1(pred,g) for g in refs);r['em']=float(any(normalize(aligned)==normalize(g) for g in refs));r['evidence_all_hit']=ev.issubset(r['paragraph_ids']) if ev else None
        for arm in ARMS:
            xs=[r['arms'][arm] for r in rows];charged=[r['producer'] for r in rows]+xs if arm.startswith('state_') else xs
            summary['arms'][arm]={'f1':sum(x['official_f1'] for x in xs)/12,'em':sum(x['em'] for x in xs)/12,'wall_sec':sum(x['wall_sec'] for x in charged),'total_tokens':sum((x.get('raw',{}).get('prompt_eval_count') or 0)+(x.get('raw',{}).get('eval_count') or 0) for x in charged),'errors':sum('error' in x for x in charged),'coverage_eligible':sum(x['evidence_all_hit'] is not None for x in xs),'coverage_all_hit':sum(x['evidence_all_hit'] is True for x in xs)}
        groups=[[r for r in rows if r['paper_id']==p['id']] for p in fr['papers']]
        for a,b in [('state_adaptive','wide'),('state_wide','reread'),('state_adaptive','state_wide'),('state_adaptive','full')]:
            rng=random.Random(20261003);boot=[]
            for _ in range(10000):
                sample=[r for _ in groups for r in groups[rng.randrange(6)]];boot.append(sum(r['arms'][a]['official_f1']-r['arms'][b]['official_f1'] for r in sample)/len(sample))
            boot.sort();summary['paired'][a+'_minus_'+b]={'difference':summary['arms'][a]['f1']-summary['arms'][b]['f1'],'paper_ci95':[boot[250],boot[9750]]}
        summary['advance']=all(summary['paired'][k]['difference']>0 for k in ['state_adaptive_minus_wide','state_wide_minus_reread']) and summary['arms']['state_adaptive']['f1']>=summary['arms']['full']['f1'] and summary['arms']['state_adaptive']['total_tokens']<summary['arms']['full']['total_tokens'] and summary['arms']['state_adaptive']['errors']==0
        out['models'][model]=summary
    save(OUT/'scored_results.json',data);save(OUT/'summary.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
