"""Narrow scalar fidelity and annotation completeness, preserving archived gates."""
import json
import re
import time
from decimal import Decimal
from pathlib import Path
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_fingerprint import prepare,evidence
from rescore_qasper_demand_transfer import f1,references

OUT=ROOT/'logs/qasper_measurement_audit'
WORDS=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(21)))
WORDS.update(dict(zip('thirty forty fifty sixty seventy eighty ninety hundred'.split(),[30,40,50,60,70,80,90,100])))

def scalar(text):
    s=text.strip().lower()
    # Require whole-answer shape; no stripping signs/decimal points or arbitrary text.
    m=re.fullmatch(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+)|\d{1,3}(?:,\d{3})+(?:\.\d+)?|[a-z]+)\s*(%|percent)?',s)
    if not m:return None
    value,unit=m.groups()
    if value in WORDS:n=Decimal(WORDS[value])
    elif re.fullmatch(r'[+-]?[\d.,]+',value):n=Decimal(value.replace(',',''))
    else:return None
    return n,('%' if unit else '')

def norm(s):return re.sub(r'\s+',' ',s).strip().lower()

def completeness(p,q,raw):
    selected=q['rank'][:8];texts=[norm(v['text']) for v in p['paragraphs']]
    ann=[]
    for ref in raw['answers']:
        a=ref['answer'];pieces=a.get('evidence') or []
        entries=[]
        for text in pieces:
            float_marker='FLOAT SELECTED' in text
            ids=[] if float_marker else [i for i,body in enumerate(texts) if norm(text) and norm(text) in body]
            entries.append({'text':text,'float_marker':float_marker,'paragraph_ids':ids,
                'retrieved':bool(set(ids).intersection(selected))})
        eligible=not a['unanswerable'] and bool(entries)
        kind='unanswerable' if a['unanswerable'] else 'boolean' if a['yes_no'] is not None else 'extractive' if a['extractive_spans'] else 'freeform'
        ann.append({'annotation_id':ref['annotation_id'],'type':kind,'eligible':eligible,
            'all_source_mapped':eligible and all(x['paragraph_ids'] for x in entries),
            'all_retrieved':eligible and all(x['retrieved'] for x in entries),'pieces':entries})
    eligible=[a for a in ann if a['eligible']]
    loose=bool(evidence(p,q,{p['id']:{'qas':[raw]}}).intersection(selected))
    return {'id':q['id'],'question':q['text'],'mixed_answerability':len({x['answer']['unanswerable'] for x in raw['answers']})>1,
        'has_answerable_annotation':any(not x['answer']['unanswerable'] for x in raw['answers']),
        'evidence_eligible':bool(eligible),'loose_any_hit':loose,
        'complete_source_support':any(a['all_source_mapped'] for a in eligible),
        'complete_retrieved_support':any(a['all_retrieved'] for a in eligible),'annotations':ann}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze={'protocol_sha256':sha((ROOT/'protocol_qasper_measurement_audit_20261002.md').read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
        'scope':'offline auxiliary audit, no reader/selection changes, not a deployed semantic verifier'}
    (OUT/'frozen.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    pairs=[('three','3',True),('3.0','3',True),('1,000','1000',True),('3 percent','3%',True),
        ('-3','3',False),('1.5','15',False),('3%','3',False),('3%','0.03',False),
        ('3','4',False),('0.5','0.05',False),('3 under condition A','3 under condition B',False),
        ('3-5','35',False),('3 kg','3 g',False),('at least 3','3',False)]
    cal=[]
    for p,g,same in pairs:
        sp,sg=scalar(p),scalar(g);assessed=sp is not None and sg is not None
        matched=assessed and sp==sg
        assert matched==same
        cal.append({'pred':p,'gold':g,'expected_equivalent':same,'official_f1':f1(p,g),'scalar_assessed':assessed,'scalar_equal':matched})
    (OUT/'calibration.json').write_text(json.dumps(cal,indent=2),encoding='utf-8')
    trainpath=ROOT/'data/qasper_official/qasper-train-v0.3.json';train=json.loads(trainpath.read_text(encoding='utf-8'))
    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    summary={'calibration_pairs':len(cal),'official_full_match_inequivalent_pairs':sum(not x['expected_equivalent'] and x['official_f1']==1 for x in cal),'splits':{}}
    for label in ('validation','heldout'):
        t=time.perf_counter();papers=[]
        for pid in split[label]:
            p=prepare(pid,train[pid]);rows=[]
            for q in p['questions']:
                raw=next(v for v in train[pid]['qas'] if v['question_id']==q['id']);rows.append(completeness(p,q,raw))
            papers.append({'paper_id':pid,'records':rows})
        rows=[r for p in papers for r in p['records']];eligible=[r for r in rows if r['evidence_eligible']]
        values={'papers':len(papers),'questions':len(rows),'any_answerable_annotation':sum(r['has_answerable_annotation'] for r in rows),
            'evidence_eligible':len(eligible),'complete_source_support':sum(r['complete_source_support'] for r in eligible),
            'complete_retrieved_support':sum(r['complete_retrieved_support'] for r in eligible),
            'loose_any_hit':sum(r['loose_any_hit'] for r in rows),
            'loose_hit_without_complete_retrieved':sum(r['loose_any_hit'] and not r['complete_retrieved_support'] for r in eligible),
            'float_dependent_no_complete_source':sum(not r['complete_source_support'] and any(e['float_marker'] for a in r['annotations'] if a['eligible'] for e in a['pieces']) for r in eligible),
            'mixed_answerability':sum(r['mixed_answerability'] for r in rows),'audit_sec':time.perf_counter()-t}
        summary['splits'][label]=values
        (OUT/(label+'_annotations.json')).write_text(json.dumps(papers,indent=2,ensure_ascii=False),encoding='utf-8')
        print(label,json.dumps(values),flush=True)
    # Existing answers audited post hoc; only exact whole-answer scalar forms assessed.
    path=ROOT/'logs/qasper_format_control/reader_results.json';reader=json.loads(path.read_text(encoding='utf-8'))
    src=json.loads((ROOT/'data/qasper_official/qasper-test-v0.3.json').read_text(encoding='utf-8'));aud=[]
    for p in reader['papers']:
        for r in p['records']:
            q=next(q for q in src[p['paper_id']]['qas'] if q['question_id']==r['id'])
            if not re.search(r'how (many|much)|\b(number|percentage|percent|rate)\b',q['question'],re.I):continue
            gold=[g for g in references(q) if scalar(g) is not None]
            if not gold:continue
            aud.append({'paper_id':p['paper_id'],'id':r['id'],'question':q['question'],'scalar_references':gold,
                'arms':{a:{'answer':v['answer'],'assessed':scalar(v['answer']) is not None,
                    'scalar_equal':scalar(v['answer']) is not None and any(scalar(v['answer'])==scalar(g) for g in gold),
                    'official_f1':v['official_f1']} for a,v in r['arms'].items()}})
    summary['posthoc_scalar_audit_questions']=len(aud)
    summary['posthoc_scalar_arms']={a:{'assessed':sum(r['arms'][a]['assessed'] for r in aud),'value_unit_matches':sum(r['arms'][a]['scalar_equal'] for r in aud)} for a in ('base','format','contract')}
    (OUT/'posthoc_scalar_answers.json').write_text(json.dumps(aud,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
