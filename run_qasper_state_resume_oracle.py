"""Privileged evidence diagnostic, separate from deployable pilot."""
import json,sys
from pathlib import Path
from run_qasper_state_resume import OUT,ROOT,SOURCE,MODELS,sha,save,evidence,call,ANSWER,references,f1,normalize
DEST=ROOT/'logs/qasper_state_resume_oracle'
PROTOCOL=ROOT/'protocol_qasper_state_resume_oracle_20261003.md'

def freeze():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'frozen.json').exists()
    src=json.loads(SOURCE.read_text(encoding='utf-8'));fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));cases=[]
    for p in fr['papers']:
        for q in p['questions']:
            ev=sorted(evidence(p,q,src));cases.append({'paper_id':p['id'],'id':q['id'],'paragraph_ids':ev or list(range(len(p['paragraphs']))),'no_mapped_evidence_fallback':not bool(ev)})
    save(DEST/'frozen.json',{'pilot_frozen_sha256':sha((OUT/'frozen.json').read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'cases':cases});print('Frozen 24 privileged diagnostic calls')

def run():
    fr=json.loads((DEST/'frozen.json').read_text(encoding='utf-8'));assert fr['script_sha256']==sha(Path(__file__).read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes())
    assert fr['pilot_frozen_sha256']==sha((OUT/'frozen.json').read_bytes())
    pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));ps=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))['papers'];src=json.loads(SOURCE.read_text(encoding='utf-8'))
    path=DEST/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha((DEST/'frozen.json').read_bytes()),'records':[]}
    for model in MODELS:
        for c in fr['cases']:
            if any(r['model']==model and r['id']==c['id'] for r in data['records']):continue
            base=next(r for r in pilot['models'][model]['records'] if r['id']==c['id'])['arms']['full']['request']['messages'][1]['content']
            p=next(p for p in ps if p['id']==c['paper_id']);from run_qasper_state_resume import material
            prompt=base.replace(material(p,list(range(len(p['paragraphs'])))),material(p,c['paragraph_ids']),1)
            r=call(model,prompt,ANSWER,768);q=next(q for q in src[c['paper_id']]['qas'] if q['question_id']==c['id']);pred=r['parsed'].get('answer','');aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['official_f1']=max(f1(aligned,g) for g in references(q))
            data['records'].append({**c,'model':model,'result':r});save(path,data);print(model,c['paper_id'],'oracle done',flush=True)
    summary={}
    for model in MODELS:
        rs=[r for r in data['records'] if r['model']==model];assert len(rs)==12
        summary[model]={'f1':sum(r['result']['official_f1'] for r in rs)/12,'errors':sum('error' in r['result'] for r in rs),'fallback_questions':sum(r['no_mapped_evidence_fallback'] for r in rs),'tokens':sum((r['result'].get('raw',{}).get('prompt_eval_count') or 0)+(r['result'].get('raw',{}).get('eval_count') or 0) for r in rs),'wall_sec':sum(r['result']['wall_sec'] for r in rs)}
    save(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
