"""Narrow ordered-pair source compiler; no gold answers and no model calls."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'logs/qasper_paired_relations'
SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
LABEL=r'[A-Za-z][A-Za-z0-9_.-]*(?: [A-Za-z][A-Za-z0-9_.-]*){0,2}'
NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
PAT=re.compile(r'(?P<x>'+LABEL+r') and (?P<y>'+LABEL+r') (?P<verb>are|were|is|was|improve by|improves by|increase by|increases by|decrease by|decreases by|achieve|achieves|reach|reaches)\s+\$?(?P<a>'+NUMBER+r')\$?\s+and\s+\$?(?P<b>'+NUMBER+r')\$?(?:\s+(?P<unit>percentage points|points|percent|seconds|sentences|tokens|examples|documents|%))?\s*,?\s+respectively\b',re.I)

def sha(b):return hashlib.sha256(b).hexdigest()

def compile_pairs(text):
    rows=[]
    for m in PAT.finditer(text):
        # Prevent matching a suffix of a three-item list or an earlier clause.
        prefix=text[:m.start()].rstrip()
        if prefix and prefix[-1] not in '.!?;:\n':continue
        fields={}
        for name in ('x','y','a','b'):
            start,end=m.span(name)
            fields[name]={'text':text[start:end],'char_start':start,'char_length':end-start,
                'byte_start':len(text[:start].encode()),'byte_length':len(text[start:end].encode())}
        rows.append({'quote':m.group(),'char_start':m.start(),'char_length':m.end()-m.start(),
            'byte_start':len(text[:m.start()].encode()),'byte_length':len(m.group().encode()),
            'paragraph_sha256':sha(text.encode()),'relation':m['verb'],'unit':m['unit'] or '',
            'pairs':[{'label':fields['x'],'value':fields['a']},{'label':fields['y'],'value':fields['b']}],
            'condition':None,'comparator':None})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [SOURCE,ROOT/'logs/qasper_fingerprint/split.json',Path(__file__),ROOT/'protocol_qasper_paired_relations_20261002.md']}
    target=OUT/'frozen.json'
    if target.exists():raise RuntimeError('Refuse to overwrite frozen inventory')
    target.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    cases=[('Precision and Recall are 10.6 and 5.4 points, respectively.',1),
        ('Precision and Recall are -3 and +1.5 points, respectively.',1),
        ('Precision and Recall are 1,448 and 2,000 tokens, respectively.',1),
        ('Precision, Recall and F1 are 3 and 4 points, respectively.',0),
        ('Precision and Recall are 3, 4 and 5 points, respectively.',0),
        ('Precision and Recall are 3-5 and 4 points, respectively.',0),
        ('Precision and Recall are 3 and 4 points.',0),
        ('Precision and Recall are 3 and 4 points and improve, respectively.',0)]
    calibration=[{'text':t,'expected':n,'actual':len(compile_pairs(t))} for t,n in cases]
    (OUT/'calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    assert all(c['actual']==c['expected'] for c in calibration)
    raw=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    summary={}
    for cohort in ('fit','validation'):
        rows=[];paragraphs=0;markers=0;overlap=[]
        for pid in split[cohort]:
            p=raw[pid];texts=[p['abstract']]+[t for s in p['full_text'] for t in s['paragraphs']]
            paper_rows=[]
            for pi,t in enumerate(texts):
                paragraphs+=1;markers+=len(re.findall(r'\brespectively\b',t,re.I))
                for r in compile_pairs(t):
                    r.update(paper_id=pid,paragraph_id=pi);rows.append(r);paper_rows.append(r)
            for q in p['qas']:
                hits=[r for r in paper_rows if all(re.search(r'(?<!\w)'+re.escape(pair['label']['text'])+r'(?!\w)',q['question'],re.I) for pair in r['pairs'])]
                if hits:overlap.append({'paper_id':pid,'question_id':q['question_id'],'question':q['question'],'candidate_count':len(hits)})
        (OUT/(cohort+'_relations.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        (OUT/(cohort+'_question_overlap.json')).write_text(json.dumps(overlap,indent=2,ensure_ascii=False),encoding='utf-8')
        summary[cohort]={'papers':len(split[cohort]),'paragraphs':paragraphs,'respectively_markers':markers,'groups':len(rows),'papers_with_groups':len({r['paper_id'] for r in rows}),'questions_with_both_labels':len(overlap)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))

if __name__=='__main__':main()
