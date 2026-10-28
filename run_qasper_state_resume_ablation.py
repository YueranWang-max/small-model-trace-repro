"""Adaptive state-content/output-binding diagnostic on reused pilot cases."""
import json,random,re,sys
from pathlib import Path
from run_qasper_state_resume import ROOT,OUT,MODELS,ANSWER,sha,save,call,references,f1,normalize,SOURCE
DEST=ROOT/'logs/qasper_state_resume_ablation'
PROTOCOL=ROOT/'protocol_qasper_state_resume_ablation_20261003.md'

def freeze():
    DEST.mkdir(exist_ok=True,parents=True);assert not (DEST/'frozen.json').exists()
    data=json.loads((OUT/'results.json').read_text(encoding='utf-8'));ps=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))['papers'];cases=[]
    for model in MODELS:
        for i,row in enumerate(data['models'][model]['records']):
            p=next(p for p in ps if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id']);prompt=row['arms']['state_wide']['request']['messages'][1]['content'];state=row['producer']['parsed']
            queryonly=prompt.replace(json.dumps(state,ensure_ascii=False),json.dumps({'missing_queries':state.get('missing_queries',[])},ensure_ascii=False),1).replace('Use known facts and resolve missing requirements from the evidence above.','Resolve missing requirements from the evidence above.')
            rule='Answer must be exactly Yes, No, or unknown, without any explanation.' if re.match(r'(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b',q['text'].lower().strip()) else 'The answer field must contain only the requested facts or complete requested list. Omit restatement of the question, rationale, and any unrequested conditions or details. Do not copy prose from the state just because it is present.'
            order=['queries_only','state_extract'];random.Random(202610031+i).shuffle(order)
            cases.append({'model':model,'paper_id':p['id'],'id':q['id'],'prompts':{'queries_only':queryonly,'state_extract':prompt+'\nFinal answer binding: '+rule},'order':order})
    save(DEST/'frozen.json',{'pilot_results_sha256':sha((OUT/'results.json').read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'cases':cases});print('Frozen48 reused-case diagnostic calls')

def run():
    fr=json.loads((DEST/'frozen.json').read_text(encoding='utf-8'));assert fr['script_sha256']==sha(Path(__file__).read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes()) and fr['pilot_results_sha256']==sha((OUT/'results.json').read_bytes())
    src=json.loads(SOURCE.read_text(encoding='utf-8'));path=DEST/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha((DEST/'frozen.json').read_bytes()),'records':[]}
    for c in fr['cases']:
        row=next((r for r in data['records'] if r['model']==c['model'] and r['id']==c['id']),None)
        if row is None:row={'model':c['model'],'paper_id':c['paper_id'],'id':c['id'],'arms':{}};data['records'].append(row)
        for arm in c['order']:
            if arm in row['arms']:continue
            r=call(c['model'],c['prompts'][arm],ANSWER,768);q=next(q for q in src[c['paper_id']]['qas'] if q['question_id']==c['id']);pred=r['parsed'].get('answer','');aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['official_f1']=max(f1(aligned,g) for g in references(q));r['answer_words']=len(pred.split());row['arms'][arm]=r;save(path,data);print(c['model'],c['paper_id'],arm,'done',flush=True)
    summary={}
    for model in MODELS:
        rows=[r for r in data['records'] if r['model']==model];assert len(rows)==12
        summary[model]={a:{'f1':sum(r['arms'][a]['official_f1'] for r in rows)/12,'mean_answer_words':sum(r['arms'][a]['answer_words'] for r in rows)/12,'errors':sum('error' in r['arms'][a] for r in rows),'reader_tokens':sum((r['arms'][a].get('raw',{}).get('prompt_eval_count') or 0)+(r['arms'][a].get('raw',{}).get('eval_count') or 0) for r in rows),'reader_wall_sec':sum(r['arms'][a]['wall_sec'] for r in rows)} for a in ['queries_only','state_extract']}
    save(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
