"""Extract explicit typed layer-count mentions without answering global depth."""
import json,re,hashlib
from pathlib import Path
from build_qasper_relation_ledger import paragraphs

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/qasper_layer_mentions';SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
WORDS={w:i for i,w in enumerate(('zero','one','two','three','four','five','six','seven','eight','nine','ten'))}
NUM=r'(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
TYPE=r'(?:BiGRU|BiLSTM|LSTM|GRU|RNN|recurrent|feed-forward|convolutional)'
PATTERNS=[re.compile(r'(?<![\w.+-])(?P<count>'+NUM+r')\s+(?P<type>'+TYPE+r')\s+layers?\b',re.I),re.compile(r'(?<![\w.+-])(?P<count>'+NUM+r')[ -]layers?[ -](?P<type>'+TYPE+r')\b',re.I),re.compile(r'(?<![\w.+-])(?P<count>'+NUM+r')\s+(?P<type>bi-directional gated recurrent unit)\s*\(BiGRU;[^)]*\)\s+layers?\b',re.I)]
AMBIG=re.compile(r'\b(?:not|no|without|up to|at most|at least|between|or)\b',re.I)

def sha(b):return hashlib.sha256(b).hexdigest()

def extract(text):
    boundaries=[0]+[m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])',text)]+[len(text)]
    rows=[];seen=set()
    for pat in PATTERNS:
        for m in pat.finditer(text):
            if m.span() in seen:continue
            seen.add(m.span());a,b=m.span();lo=max(x for x in boundaries if x<=a);hi=min(x for x in boundaries if x>=b);sentence=text[lo:hi]
            token=m['count'].lower();count=int(token) if token.isdigit() else WORDS[token]
            label=m['type'];family='feed_forward' if label.lower()=='feed-forward' else 'convolutional' if label.lower()=='convolutional' else 'recurrent'
            ambiguous=AMBIG.search(sentence)
            r={'count':count,'layer_type':label,'layer_family':family,'status':'unresolved_sentence_scope' if ambiguous else 'explicit_local_mention','ambiguity_signal':ambiguous.group() if ambiguous else None,'quote':m.group(),'char_start':a,'char_length':b-a,'byte_start':len(text[:a].encode()),'byte_length':len(m.group().encode()),'paragraph_sha256':sha(text.encode()),'context':sentence,'context_char_start':lo,'model_identity':None,'owner_role':None,'answer_ready':False}
            assert text[a:b]==r['quote'] and text.encode()[r['byte_start']:r['byte_start']+r['byte_length']].decode()==r['quote']
            rows.append(r)
    return sorted(rows,key=lambda r:r['char_start'])

def main():
    OUT.mkdir(parents=True,exist_ok=True);paths=[SOURCE,Path(__file__),ROOT/'build_qasper_relation_ledger.py',ROOT/'logs/qasper_fingerprint/split.json',ROOT/'protocol_qasper_layer_mentions_20261002.md']
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps({'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}},indent=2),encoding='utf-8')
    cases=[('We use two LSTM layers.',1,'recurrent'),('We use a 2-layer LSTM.',1,'recurrent'),('Two feed-forward layers of size 400.',1,'feed_forward'),('One bi-directional gated recurrent unit (BiGRU; BIBREF10 ) layer.',1,'recurrent'),('Hidden layer size is 150.',0,None),('The network has three components.',0,None),('We use 1.5 LSTM layers.',0,None),('We do not use two LSTM layers.',1,'recurrent'),('We use two or three LSTM layers.',1,'recurrent')]
    calibration=[]
    for text,n,family in cases:
        rows=extract(text);assert len(rows)==n
        if rows:assert rows[0]['layer_family']==family
        if 'not' in text or ' or ' in text:assert rows[0]['status']=='unresolved_sentence_scope'
        calibration.append({'text':text,'expected_mentions':n,'records':rows})
    (OUT/'calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    data=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));summary={}
    for cohort in ('fit','validation'):
        rows=[];n=0
        for pid in split[cohort]:
            for pi,p in enumerate(paragraphs(data[pid])):
                n+=1
                for r in extract(p['text']):r.update(paper_id=pid,paragraph_id=pi,heading=p['heading']);rows.append(r)
        (OUT/(cohort+'_records.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        summary[cohort]={'papers':len(split[cohort]),'paragraphs':n,'mentions':len(rows),'papers_with_mentions':len({r['paper_id'] for r in rows}),'explicit_local_mentions':sum(r['status']=='explicit_local_mention' for r in rows),'unresolved_sentence_scope':sum(r['status']!='explicit_local_mention' for r in rows),'answer_ready':0}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
