"""Supplementary scalar coverage: unassessed is distinct from incorrect."""
import json
from run_qasper_numeric_fields import OUT,SOURCE
from audit_qasper_measurement import scalar
from rescore_qasper_demand_transfer import references

def main():
    src=json.loads(SOURCE.read_text(encoding='utf-8'));out={}
    for label in ('validation','evaluation'):
        path=OUT/(label+'_summary.json')
        if not path.exists():continue
        summary=json.loads(path.read_text(encoding='utf-8'));eligible=[]
        for r in summary['results']:
            q=next(q for q in src[r['paper_id']]['qas'] if q['question_id']==r['id']);gold=[g for g in references(q) if scalar(g) is not None]
            if not gold:continue
            eligible.append({'id':r['id'],'gold_scalar_references':gold,
                'arms':{a:{'answer':s,'assessed':scalar(s) is not None,
                    'match':None if scalar(s) is None else any(scalar(s)==scalar(g) for g in gold)} for a,s in r['answers'].items()}})
        arms={a:{'assessed':sum(r['arms'][a]['assessed'] for r in eligible),
            'matched':sum(r['arms'][a]['match'] is True for r in eligible),
            'mismatched':sum(r['arms'][a]['match'] is False for r in eligible),
            'unassessed':sum(r['arms'][a]['match'] is None for r in eligible)} for a in ('base','policy')}
        out[label]={'eligible_scalar_references':len(eligible),'arms':arms,'rows':eligible}
    (OUT/'scalar_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({label:{k:v for k,v in data.items() if k!='rows'} for label,data in out.items()},indent=2))

if __name__=='__main__':main()
