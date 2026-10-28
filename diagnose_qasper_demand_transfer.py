"""Descriptive post hoc diagnostics, not a new selection rule."""
import json
from run_qasper_demand_transfer import OUT, SOURCE

def main():
    frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    answers=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    scores=json.loads((OUT/'reader_official_rescore.json').read_text(encoding='utf-8'))
    evidence=json.loads((OUT/'evidence_results.json').read_text(encoding='utf-8'))
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    emap={(p['paper_id'],r['id']):r for p in evidence for r in p['records']}
    smap={r['id']:r for g in scores['groups'] for r in g}
    pmap={p['id']:p for p in frozen['papers']}
    rows=[]
    for p in answers['papers']:
        for r in p['records']:
            pid,qid=p['paper_id'],r['id']
            q=next(q for q in pmap[pid]['questions'] if q['id']==qid)
            s=smap[qid]['arms']; e=emap[(pid,qid)]['arms']
            choice=frozen['choices'][pid][qid]
            refs=next(x for x in source[pid]['qas'] if x['question_id']==qid)['answers']
            states={x['answer']['unanswerable'] for x in refs}
            booleans={x['answer']['yes_no'] for x in refs if x['answer']['yes_no'] is not None}
            rows.append({'paper_id':pid,'id':qid,'question':q['text'],'demand':q['demand'],
                'mixed_answerability':len(states)>1,'conflicting_boolean':len(booleans)>1,
                'changed_set':set(choice['bm25'])!=set(choice['global']),
                'changed_order':choice['bm25']!=choice['global'],
                'f1_difference':s['global']['prompt_aligned_f1']-s['bm25']['prompt_aligned_f1'],
                'recall_difference':None if e['global']['recall'] is None else e['global']['recall']-e['bm25']['recall'],
                'answers':{a:r['arms'][a].get('answer') for a in ['bm25','global']},
                'selected':choice})
    summary={}
    for label, subset in [('all',rows),('changed_set',[r for r in rows if r['changed_set']]),
        ('same_set',[r for r in rows if not r['changed_set']])]:
        ds=[r['f1_difference'] for r in subset]
        summary[label]={'n':len(ds),'mean_difference':sum(ds)/len(ds) if ds else None,
            'wins':sum(d>1e-9 for d in ds),'losses':sum(d < -1e-9 for d in ds),'ties':sum(abs(d)<=1e-9 for d in ds)}
    output={'label':'post hoc descriptive; no causal attribution or subgroup significance claim',
        'annotation_conflicts':{'mixed_answerability':sum(r['mixed_answerability'] for r in rows),
            'conflicting_boolean':sum(r['conflicting_boolean'] for r in rows)},
        'summary':summary,'largest_losses':sorted(rows,key=lambda r:r['f1_difference'])[:10],
        'largest_wins':sorted(rows,key=lambda r:r['f1_difference'],reverse=True)[:10],'rows':rows}
    (OUT/'diagnostics.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
