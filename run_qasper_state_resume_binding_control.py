"""Equal question-only answer-binding controls; no model-generated state."""
import json,random,sys
from pathlib import Path
from run_qasper_state_resume import ROOT,OUT,MODELS,SOURCE,ANSWER,call,sha,save,references,f1,normalize
DEST=ROOT/'logs/qasper_state_resume_binding_control'
PROTOCOL=ROOT/'protocol_qasper_state_resume_binding_control_20261003.md'

def freeze():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'frozen.json').exists()
    pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_state_resume_ablation/frozen.json').read_text(encoding='utf-8'));cases=[]
    for c in old['cases']:
        row=next(r for r in pilot['models'][c['model']]['records'] if r['id']==c['id']);suffix='\nFinal answer binding: '+c['prompts']['state_extract'].split('\nFinal answer binding: ',1)[1]
        order=['wide_extract','base_extract'];random.Random(202610032+len(cases)).shuffle(order)
        cases.append({'model':c['model'],'paper_id':c['paper_id'],'id':c['id'],'prompts':{a:row['arms'][a.split('_')[0]]['request']['messages'][1]['content']+suffix for a in order},'order':order})
    save(DEST/'frozen.json',{'pilot_sha256':sha((OUT/'results.json').read_bytes()),'ablation_frozen_sha256':sha((ROOT/'logs/qasper_state_resume_ablation/frozen.json').read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'cases':cases});print('Frozen48 equal-binding control calls')

def run():
    fr=json.loads((DEST/'frozen.json').read_text(encoding='utf-8'));assert fr['script_sha256']==sha(Path(__file__).read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes()) and fr['pilot_sha256']==sha((OUT/'results.json').read_bytes()) and fr['ablation_frozen_sha256']==sha((ROOT/'logs/qasper_state_resume_ablation/frozen.json').read_bytes())
    src=json.loads(SOURCE.read_text(encoding='utf-8'));path=DEST/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha((DEST/'frozen.json').read_bytes()),'records':[]}
    for c in fr['cases']:
        row=next((r for r in data['records'] if r['model']==c['model'] and r['id']==c['id']),None)
        if row is None:row={'model':c['model'],'paper_id':c['paper_id'],'id':c['id'],'arms':{}};data['records'].append(row)
        for a in c['order']:
            if a in row['arms']:continue
            r=call(c['model'],c['prompts'][a],ANSWER,768);q=next(q for q in src[c['paper_id']]['qas'] if q['question_id']==c['id']);pred=r['parsed'].get('answer','');aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['official_f1']=max(f1(aligned,g) for g in references(q));r['answer_words']=len(pred.split());row['arms'][a]=r;save(path,data);print(c['model'],c['paper_id'],a,'done',flush=True)
    summary={}
    for model in MODELS:
        rows=[r for r in data['records'] if r['model']==model];assert len(rows)==12
        summary[model]={a:{'f1':sum(r['arms'][a]['official_f1'] for r in rows)/12,'mean_answer_words':sum(r['arms'][a]['answer_words'] for r in rows)/12,'errors':sum('error' in r['arms'][a] for r in rows),'tokens':sum((r['arms'][a].get('raw',{}).get('prompt_eval_count') or 0)+(r['arms'][a].get('raw',{}).get('eval_count') or 0) for r in rows),'wall_sec':sum(r['arms'][a]['wall_sec'] for r in rows)} for a in ['wide_extract','base_extract']}
    save(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
