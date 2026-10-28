"""Descriptive full-versus-format answer differences; no subgroup gate."""
import json
from run_qasper_format_control import OUT,SOURCE,ARMS

def main():
    data=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    src=json.loads(SOURCE.read_text(encoding='utf-8'));rows=[]
    for p in data['papers']:
        for r in p['records']:
            assert set(r['arms'])==set(ARMS)
            q=next(q for q in src[p['paper_id']]['qas'] if q['question_id']==r['id'])
            rows.append({'paper_id':p['paper_id'],'id':r['id'],'question':q['question'],
                'boolean_reference':any(a['answer']['yes_no'] is not None for a in q['answers']),
                'answers':{a:r['arms'][a]['answer'] for a in ARMS},
                'scores':{a:r['arms'][a]['official_f1'] for a in ARMS},
                'difference':r['arms']['contract']['official_f1']-r['arms']['format']['official_f1']})
    subsets={}
    for label,xs in [('all',rows),('boolean',[r for r in rows if r['boolean_reference']]),('nonboolean',[r for r in rows if not r['boolean_reference']])]:
        subsets[label]={'n':len(xs),'f1':{a:sum(r['scores'][a] for r in xs)/len(xs) if xs else None for a in ARMS},
            'wins':sum(r['difference']>1e-9 for r in xs),'losses':sum(r['difference'] < -1e-9 for r in xs),'ties':sum(abs(r['difference'])<=1e-9 for r in xs)}
    out={'label':'post hoc descriptive; no subgroup significance or causal attribution',
        'subsets':subsets,'largest_wins':sorted(rows,key=lambda r:-r['difference'])[:8],
        'largest_losses':sorted(rows,key=lambda r:r['difference'])[:8]}
    (OUT/'diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(subsets,indent=2))

if __name__=='__main__':main()
