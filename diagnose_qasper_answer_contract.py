"""Descriptive typed errors and annotation conflicts for the frozen factorial pilot."""
import json
from run_qasper_answer_contract import OUT, SOURCE, ARMS

def main():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    r=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    src=json.loads(SOURCE.read_text(encoding='utf-8'))
    rows=[]
    for p in r['papers']:
        for q in p['records']:
            assert set(q['arms'])==set(ARMS)
            raw=next(x for x in src[p['paper_id']]['qas'] if x['question_id']==q['id'])
            states={a['answer']['unanswerable'] for a in raw['answers']}
            bools={a['answer']['yes_no'] for a in raw['answers'] if a['answer']['yes_no'] is not None}
            rows.append({'paper_id':p['paper_id'],'id':q['id'],'question':raw['question'],
                'type':f['choices'][p['paper_id']][q['id']]['contract'][0],
                'mixed_answerability':len(states)>1,'conflicting_boolean':len(bools)>1,
                'scores':{a:q['arms'][a]['official_f1'] for a in ARMS},
                'answers':{a:q['arms'][a]['answer'] for a in ARMS},
                'difference':q['arms']['combined']['official_f1']-q['arms']['base']['official_f1']})
    summary={}
    for typ in sorted({q['type'] for q in rows}):
        xs=[q for q in rows if q['type']==typ]
        summary[typ]={'n':len(xs),'f1':{a:sum(q['scores'][a] for q in xs)/len(xs) for a in ARMS}}
    out={'label':'post hoc descriptive subgroups, no subgroup significance or causal correctness inference',
        'mixed_answerability':sum(q['mixed_answerability'] for q in rows),
        'conflicting_boolean':sum(q['conflicting_boolean'] for q in rows),
        'by_type':summary,'largest_losses':sorted(rows,key=lambda q:q['difference'])[:8],
        'largest_wins':sorted(rows,key=lambda q:-q['difference'])[:8]}
    (OUT/'diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('largest_losses','largest_wins')},indent=2))

if __name__=='__main__':main()
