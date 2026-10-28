"""Post hoc all-arm one-line interface audit; original frozen gate preserved."""
import json
import random
from run_qasper_answer_contract import OUT, SOURCE, ARMS
from rescore_qasper_demand_transfer import references, f1, normalize

def main():
    data=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    groups=[]; recovered={a:0 for a in ARMS}
    for p in data['papers']:
        group=[]
        for row in p['records']:
            gold=references(next(q for q in source[p['paper_id']]['qas'] if q['question_id']==row['id']))
            rec={'id':row['id'],'arms':{}}
            for arm in ARMS:
                result=row['arms'][arm];answer=result.get('answer') or ''
                lines=[line.strip() for line in result.get('content','').splitlines() if line.strip()]
                did_recover=not answer and len(lines)==1 and len(lines[0])<=300
                if did_recover:answer=lines[0];recovered[arm]+=1
                aligned='Unanswerable' if normalize(answer)=='unknown' else answer
                rec['arms'][arm]={'answer':answer,'recovered':did_recover,'f1':max(f1(aligned,g) for g in gold)}
            group.append(rec)
        groups.append(group)
    rows=[r for g in groups for r in g]
    summary={a:sum(r['arms'][a]['f1'] for r in rows)/len(rows) for a in ARMS};paired={}
    for arm in ARMS[1:]:
        rng=random.Random(20261004);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]]
            boot.append(sum(r['arms'][arm]['f1']-r['arms']['base']['f1'] for r in sample)/len(sample))
        boot.sort();paired[arm]={'difference':summary[arm]-summary['base'],'paper_ci95':[boot[250],boot[9750]]}
    output={'label':'post hoc format sensitivity analysis; original answer gate remains failed',
        'rule':'For every arm: keep original parsed answer; if missing, accept only exactly one nonempty output line of <=300 characters. No answer labels used to select output.',
        'recovered':recovered,'f1':summary,'paired':paired,'groups':groups}
    (OUT/'format_audit.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
