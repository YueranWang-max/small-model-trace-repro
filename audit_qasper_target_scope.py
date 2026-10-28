"""Posthoc diagnostics; does not claim a category-correctness judge."""
import json,re
from run_qasper_target_scope import OUT,ARMS
from rescore_qasper_demand_transfer import references
from run_qasper_forward_bridge import SOURCE

def main():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));source=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert len(d['rows'])==len(f['tasks']) and all(set(r['arms'])==set(ARMS) for r in d['rows'])
    cards=[];conflicts=[];boolean={a:{'eligible':0,'correct':0} for a in ARMS}
    for t in f['tasks']:
        r=next(r for r in d['rows'] if r['id']==t['id']);q=next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']);answers=q['answers'];states={x['answer']['unanswerable'] for x in answers};pol={x['answer']['yes_no'] for x in answers if x['answer']['yes_no'] is not None}
        refs=references(q);bare={x.strip() for x in refs if re.fullmatch(r'\s*[+-]?\d+(?:\.\d+)?\s*',x)}
        if len(bare)>1 or len(states)>1 or len(pol)>1:conflicts.append({'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'references':refs,'distinct_bare_numeric_references':sorted(bare),'mixed_answerability':len(states)>1,'mixed_boolean':len(pol)>1})
        if states=={False} and len(pol)==1:
            for a in ARMS:
                boolean[a]['eligible']+=1;m=re.match(r'^\s*(yes|no)(?:\s*[,.:;!]|\s*$)',r['arms'][a]['answer'],re.I);boolean[a]['correct']+=bool(m and (m.group(1).lower()=='yes')==next(iter(pol)))
        if len({r['arms'][a]['answer'].casefold() for a in ARMS})>1:
            cards.append({'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'qualifiers':t['qualifiers'],'references':references(q),'answers':{a:r['arms'][a]['answer'] for a in ARMS},'f1':{a:r['arms'][a]['official_f1'] for a in ARMS},'evidence':t['material']})
    out={'posthoc':True,'changed_text_questions':len(cards),'boolean':boolean,'reference_conflicts':conflicts,'category_semantics_not_automatically_judged':True,'cards':cards}
    (OUT/'diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='cards'},indent=2))

if __name__=='__main__':main()
