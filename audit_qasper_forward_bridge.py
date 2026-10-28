"""Post-reader full-annotation coverage; gold never affects routing."""
import json,re
from run_qasper_forward_bridge import OUT,SOURCE,ARMS

def norm(t):return re.sub(r'\s+',' ',t).strip().lower()

def main():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));source=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert len(d['rows'])==len(f['tasks']) and all(set(r['arms'])==set(ARMS) for r in d['rows'])
    rows=[];counts={a:0 for a in ARMS};eligible=0
    for t in f['tasks']:
        q=next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']);annotations=[]
        for ref in q['answers']:
            ans=ref['answer'];ev=ans.get('evidence',[])
            if ans['unanswerable'] or not ev:continue
            mapping=[[i for i,p in enumerate(t['paragraphs']) if norm(e) in norm(p['text'])] for e in ev]
            annotations.append({'mapping':mapping,'coverage':{a:all(any(i in t['choices'][a] for i in ids) for ids in mapping) for a in ARMS}})
        cover={a:any(x['coverage'][a] for x in annotations) for a in ARMS}
        if annotations:
            eligible+=1
            for a in ARMS:counts[a]+=cover[a]
        rows.append({'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'annotations':annotations,'coverage':cover,'choices':t['choices'],'triggers':t['triggers']})
    result={'posthoc_gold_only':True,'questions':len(rows),'answerable_evidence_eligible':eligible,'complete_annotation_hits':counts,'coverage_not_semantic_sufficiency':True,'normalization':'lowercase plus whitespace only; unmatched tables/formulas may be unavailable','rows':rows}
    (OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
