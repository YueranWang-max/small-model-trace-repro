"""Fixed new-case replication; annotations only used after all reader calls."""
import json,random,re,sys,time
from pathlib import Path
from run_qasper_state_resume import ROOT,SOURCE,STATE,ANSWER,OPENER,call,prepare,material,sha,save,dependencies,references,f1,normalize,evidence
OUT=ROOT/'logs/qasper_interface_replication'
PROTOCOL=ROOT/'protocol_qasper_interface_replication_20261003.md'
MODELS=['qwen3.5:4b','qwen3:1.7b']
ARMS=['full','base','wide','base_bind','wide_bind','state_bind']
SPLIT=ROOT/'logs/qasper_fingerprint/split.json'

def binding(q):
    if re.match(r'(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b',q.lower().strip()):
        return 'Answer must be exactly Yes, No, or unknown, without any explanation.'
    return 'The answer field must contain only the requested facts or complete requested list. Omit restatement of the question, rationale, and any unrequested conditions or details. Do not copy prose from the state just because it is present.'

def hashes():return {**dependencies(),Path(__file__).name:sha(Path(__file__).read_bytes())}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists()
    src=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads(SPLIT.read_text(encoding='utf-8'));eligible=[pid for pid in split['heldout'] if len(src[pid]['qas'])>=2];old=json.loads((ROOT/'logs/qasper_state_resume/frozen.json').read_text(encoding='utf-8'));ids=eligible[30:42]
    assert len(ids)==12 and not set(ids)&{p['id'] for p in old['papers']}
    t=time.perf_counter();ps=[]
    for pid in ids:
        p=prepare(pid,src[pid]);p['title']=src[pid]['title'];p['questions']=p['questions'][:2];offset=0
        for para in p['paragraphs']:
            para['serialized_char_start']=offset;para['text_sha256']=sha(para['text'].encode());offset+=len(para['text'])+2
        p['serialized_source_sha256']=sha('\n\n'.join(x['text'] for x in p['paragraphs']).encode());ps.append(p)
    orders={}
    for mi,m in enumerate(MODELS):
        order=ARMS.copy();random.Random(202610033+mi).shuffle(order);orders[m]=order
    save(OUT/'frozen.json',{'source_sha256':sha(SOURCE.read_bytes()),'split_sha256':sha(SPLIT.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'dependencies':hashes(),'papers':ps,'orders':orders,'prepare_sec':time.perf_counter()-t,'selection':'eligible30:42 first2 questions; disjoint preceding6 papers'})
    print('Frozen12 papers24 questions336 formal calls2 warmups',flush=True)

def validate(fr):
    assert fr['dependencies']==hashes() and fr['source_sha256']==sha(SOURCE.read_bytes()) and fr['split_sha256']==sha(SPLIT.read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes())

def reader_prompt(p,q,arm,state):
    ids=list(range(len(p['paragraphs']))) if arm=='full' else q['rank'][:8] if arm.startswith('base') else q['rank'][:12]
    prompt='Paper: '+p['title']+'\nQuestion: '+q['text']+'\nEvidence:\n'+material(p,ids)
    if arm=='state_bind':prompt+='\nPrevious reading state (may be wrong; verify every claim):\n'+json.dumps(state,ensure_ascii=False)+'\nUse known facts and resolve missing requirements from the evidence above.'
    prompt+='\nReturn only the requested short answer in answer. Include all requested entities and conditions. If evidence is insufficient answer unknown. A caption without table values does not establish numeric values.'
    if arm.endswith('_bind'):prompt+='\nFinal answer binding: '+binding(q['text'])
    return prompt,ids

def producer_prompt(p,q):
    return 'Paper: '+p['title']+'\nQuestion: '+q['text']+'\nEvidence:\n'+material(p,q['rank'][:8])+'\nRecord only supported facts in known. Identify unanswered parts and emit up to three targeted search queries in missing_queries; empty list if sufficient. Give a provisional short answer in draft, or unknown. Do not invent values from table captions.'

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'));validate(fr);path=OUT/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'models':{}}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    for m in MODELS:
        b=data['models'].setdefault(m,{'records':[]})
        if 'metadata' not in b:
            with OPENER.open('http://localhost:11434/api/tags') as resp:b['metadata']=json.loads(resp.read())
        if 'warmup' not in b:b['warmup']=call(m,'Answer ready.',ANSWER,16);save(path,data)
        for pi,p in enumerate(fr['papers']):
            for qi,q in enumerate(p['questions']):
                row=next((r for r in b['records'] if r['id']==q['id']),None)
                if row is None:
                    k=(pi*2+qi)%6;o=fr['orders'][m];row={'paper_id':p['id'],'id':q['id'],'arms':{},'order':o[k:]+o[:k]};b['records'].append(row)
                if 'producer' not in row:row['producer']=call(m,producer_prompt(p,q),STATE,384);save(path,data)
                for a in row['order']:
                    if a in row['arms']:continue
                    prompt,ids=reader_prompt(p,q,a,row['producer']['parsed']);r=call(m,prompt,ANSWER,768);r['paragraph_ids']=ids;row['arms'][a]=r;save(path,data);print(m,pi+1,qi+1,a,'error' if 'error' in r else 'done',flush=True)

def analyze():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));validate(fr);data=json.loads((OUT/'results.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'));summary={'papers':12,'questions':24,'formal_calls':336,'warmups':2,'models':{},'prepare_sec':fr['prepare_sec'],'scope':'new strategy readout, not project-wide untouched holdout'}
    for m in MODELS:
        rows=data['models'][m]['records'];assert len(rows)==24 and all(set(r['arms'])==set(ARMS) for r in rows)
        for row in rows:
            p=next(p for p in fr['papers'] if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id']);gq=next(q for q in src[p['id']]['qas'] if q['question_id']==row['id']);refs=references(gq);ev=evidence(p,q,src)
            for a,r in row['arms'].items():
                pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['f1']=max(f1(aligned,g) for g in refs);r['literal_f1']=max(f1(pred,g) for g in refs);r['em']=float(any(normalize(aligned)==normalize(g) for g in refs));r['answer_words']=len(pred.split());r['boolean_question']=binding(q['text']).startswith('Answer must');r['boolean_format_ok']=pred in ('Yes','No','unknown') if r['boolean_question'] else None;r['all_evidence']=ev.issubset(r['paragraph_ids']) if ev else None
        v={'arms':{},'paired':{},'warmup_sec':data['models'][m]['warmup']['wall_sec']}
        for a in ARMS:
            xs=[r['arms'][a] for r in rows];charged=xs+[r['producer'] for r in rows] if a=='state_bind' else xs
            v['arms'][a]={'f1':sum(r['f1'] for r in xs)/24,'em':sum(r['em'] for r in xs)/24,'mean_words':sum(r['answer_words'] for r in xs)/24,'tokens':sum((r.get('raw',{}).get('prompt_eval_count') or 0)+(r.get('raw',{}).get('eval_count') or 0) for r in charged),'wall_sec':sum(r['wall_sec'] for r in charged),'errors':sum('error' in r for r in charged),'all_evidence_hits':sum(r['all_evidence'] is True for r in xs),'mapped_questions':sum(r['all_evidence'] is not None for r in xs),'boolean_questions':sum(r['boolean_question'] for r in xs),'boolean_format_ok':sum(r['boolean_format_ok'] is True for r in xs)}
        groups=[[r for r in rows if r['paper_id']==p['id']] for p in fr['papers']]
        contrasts=[('state_bind','wide_bind'),('wide_bind','wide'),('base_bind','base')]+[(a,'full') for a in ARMS if a!='full']
        for a,b in contrasts:
            rng=random.Random(202610033);boot=[]
            for _ in range(20000):
                sample=[r for _ in groups for r in groups[rng.randrange(12)]];boot.append(sum(r['arms'][a]['f1']-r['arms'][b]['f1'] for r in sample)/24)
            boot.sort();v['paired'][a+'_minus_'+b]={'difference':v['arms'][a]['f1']-v['arms'][b]['f1'],'paper_ci95':[boot[500],boot[19500]]}
        v['state_gate']=v['paired']['state_bind_minus_wide_bind']['paper_ci95'][0]>0 and v['arms']['state_bind']['errors']==0 and v['arms']['state_bind']['tokens']<=v['arms']['full']['tokens']
        v['interface_gate']=v['paired']['wide_bind_minus_wide']['paper_ci95'][0]>0
        v['compression_gates']={a:v['paired'][a+'_minus_full']['paper_ci95'][0]>-.03 and v['arms'][a]['tokens']<=.8*v['arms']['full']['tokens'] and v['arms'][a]['errors']==0 for a in ARMS if a!='full'}
        summary['models'][m]=v
    summary['advance_state']=all(v['state_gate'] for v in summary['models'].values());summary['confirm_interface']=all(v['interface_gate'] for v in summary['models'].values());save(OUT/'scored_results.json',data);save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
