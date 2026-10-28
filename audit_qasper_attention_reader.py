"""After-selection coverage and exact source verification, without feedback."""
import json,re
from run_qasper_attention_reader import OUT,SOURCE,sha,ARMS

def norm(s):return re.sub(r'\s+',' ',s).strip().lower()

def main():
    sp=OUT/'selected.json';sel=json.loads(sp.read_text(encoding='utf-8'));fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'));rows=[];verified={a:0 for a in ARMS}
    for t in sel['tasks']:
        p=next(p for p in fr['papers'] if p['paper_id']==t['paper_id']);q=next(q for q in src[t['paper_id']]['qas'] if q['question_id']==t['id']);full=' '.join(u['text'] for u in p['units']);refs=[list({v.strip() for v in a['answer']['extractive_spans'] if v.strip()}) for a in q['answers']];refs=[ss for ss in refs if ss and all(norm(v) in norm(full) for v in ss)];ev={norm(e) for a in q['answers'] for e in a['answer']['evidence'] if e.strip()};mapped=[u for u in p['units'] if any(norm(u['text']) in e for e in ev)]
        row={'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'extractive_eligible':bool(refs),'evidence_sentences':len(mapped),'arms':{}}
        for a in ARMS:
            rs=t['arms'][a]['pointers'];keys={(r['pid'],r['sid']) for r in rs};assert {r['pid'] for r in rs}==set(range(len(p['paragraphs'])))
            for r in rs:
                body=p['paragraphs'][r['pid']]['text'];assert sha(body.encode())==r['paragraph_sha256'];assert body[r['start']:r['start']+r['length']]==r['text'];assert body.encode()[r['byte_start']:r['byte_start']+r['byte_length']].decode()==r['text'];verified[a]+=1
            text=' '.join(r['text'] for r in rs);row['arms'][a]={'complete_extractive_ref':any(all(norm(v) in norm(text) for v in rr) for rr in refs) if refs else None,'evidence_sentences_retained':sum((u['pid'],u['sid']) in keys for u in mapped),'before':t['arms'][a]['sentences_before_closure'],'after':t['arms'][a]['sentences_after_closure']}
        rows.append(row)
    out={'selected_sha256':sha(sp.read_bytes()),'post_selection_gold_only':True,'semantic_sufficiency_not_verified':True,'extractive_eligible':sum(r['extractive_eligible'] for r in rows),'arms':{},'rows':rows}
    for a in ARMS:out['arms'][a]={'verified_source_pointers':verified[a],'complete_extractive_refs':sum(r['arms'][a]['complete_extractive_ref'] is True for r in rows),'evidence_sentences_retained':sum(r['arms'][a]['evidence_sentences_retained'] for r in rows),'evidence_sentences_total':sum(r['evidence_sentences'] for r in rows),'sentences_before':sum(r['arms'][a]['before'] for r in rows),'sentences_after':sum(r['arms'][a]['after'] for r in rows)}
    (OUT/'source_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
