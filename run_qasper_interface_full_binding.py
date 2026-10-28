"""Adaptive equal-format source-length diagnostic; frozen replication unchanged."""
import json,sys,random
from pathlib import Path
from run_qasper_interface_replication import ROOT,OUT,MODELS,SOURCE,ANSWER,call,sha,save,references,f1,normalize
DEST=ROOT/'logs/qasper_interface_full_binding'
PROTOCOL=ROOT/'protocol_qasper_interface_full_binding_20261003.md'

def freeze():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'frozen.json').exists();pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));cases=[]
    for m in MODELS:
        for row in pilot['models'][m]['records']:
            suffix='\nFinal answer binding: '+row['arms']['wide_bind']['request']['messages'][1]['content'].split('\nFinal answer binding: ',1)[1]
            cases.append({'model':m,'paper_id':row['paper_id'],'id':row['id'],'prompt':row['arms']['full']['request']['messages'][1]['content']+suffix})
    save(DEST/'frozen.json',{'pilot_sha256':sha((OUT/'results.json').read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'cases':cases});print('Frozen48 full-source equal-binding control calls')

def run():
    fr=json.loads((DEST/'frozen.json').read_text(encoding='utf-8'));assert fr['pilot_sha256']==sha((OUT/'results.json').read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes()) and fr['script_sha256']==sha(Path(__file__).read_bytes());src=json.loads(SOURCE.read_text(encoding='utf-8'));path=DEST/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha((DEST/'frozen.json').read_bytes()),'records':[]}
    for c in fr['cases']:
        if any(r['model']==c['model'] and r['id']==c['id'] for r in data['records']):continue
        r=call(c['model'],c['prompt'],ANSWER,768);gq=next(q for q in src[c['paper_id']]['qas'] if q['question_id']==c['id']);pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['f1']=max(f1(aligned,g) for g in references(gq));r['answer_words']=len(pred.split());data['records'].append({'model':c['model'],'paper_id':c['paper_id'],'id':c['id'],'result':r});save(path,data);print(c['model'],c['paper_id'],'full_bind','error' if 'error' in r else 'done',flush=True)
    summary={};original=json.loads((OUT/'scored_results.json').read_text(encoding='utf-8'));primary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    for m in MODELS:
        rows=[r for r in data['records'] if r['model']==m];assert len(rows)==24;xs=[r['result'] for r in rows];idx={r['id']:r for r in original['models'][m]['records']};v={'f1':sum(r['f1'] for r in xs)/24,'tokens':sum((r.get('raw',{}).get('prompt_eval_count') or 0)+(r.get('raw',{}).get('eval_count') or 0) for r in xs),'wall_sec':sum(r['wall_sec'] for r in xs),'errors':sum('error' in r for r in xs),'mean_words':sum(r['answer_words'] for r in xs)/24,'paired':{},'gates':{}}
        groups=[[r for r in rows if r['paper_id']==pid] for pid in dict.fromkeys(r['paper_id'] for r in rows)]
        for arm in ['base_bind','wide_bind','state_bind']:
            rng=random.Random(202610034);boot=[]
            for _ in range(20000):
                sample=[r for _ in groups for r in groups[rng.randrange(12)]];boot.append(sum(idx[r['id']]['arms'][arm]['f1']-r['result']['f1'] for r in sample)/24)
            boot.sort();v['paired'][arm+'_minus_full_bind']={'difference':primary['models'][m]['arms'][arm]['f1']-v['f1'],'paper_ci95':[boot[500],boot[19500]]};a=primary['models'][m]['arms'][arm];v['gates'][arm]=boot[500]>-.03 and a['tokens']<=.8*v['tokens'] and a['errors']==0 and v['errors']==0
        summary[m]=v
    save(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
