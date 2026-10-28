"""Post-freeze weak-label source coverage, never used for selecting retained text."""
import json,re
from run_qasper_paragraph_elements import OUT,SOURCE,sha
from run_qasper_fingerprint import demand

def norm(s):return re.sub(r'\s+',' ',s).strip().lower()

def main():
    frozen=OUT/'frozen.json';fr=json.loads(frozen.read_text(encoding='utf-8'));source=json.loads(SOURCE.read_text(encoding='utf-8'));rows=[]
    for t in fr['tasks']:
        q=next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']);ps=t['paragraphs'];full=' '.join(p['text'] for p in ps)
        refs=[sorted(set(s.strip() for s in r['answer']['extractive_spans'] if s.strip())) for r in q['answers']];refs=[ss for ss in refs if ss and all(norm(s) in norm(full) for s in ss)]
        evidence={norm(e) for r in q['answers'] for e in r['answer']['evidence'] if e.strip()}
        mapped=[s for s in t['units'] if any(norm(s['text']) in e for e in evidence)]
        row={'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'demand':demand(t['question']),'mappable_extractive_refs':len(refs),'evidence_sentences':len(mapped),'arms':{}}
        for arm,rs in t['pointers'].items():
            for r in rs:
                text=ps[r['pid']]['text'];assert sha(text.encode())==r['paragraph_sha256'];assert text[r['start']:r['start']+r['length']]==r['text'];assert text.encode()[r['byte_start']:r['byte_start']+r['byte_length']].decode()==r['text']
            retained=' '.join(r['text'] for r in rs);keys={(r['pid'],r['sid']) for r in rs}
            row['arms'][arm]={'any_complete_extractive_ref':any(all(norm(s) in norm(retained) for s in ss) for ss in refs) if refs else None,'retained_evidence_sentences':sum((s['pid'],s['sid']) in keys for s in mapped),'source_chars':t['stats'][arm]['retained_chars']}
        rows.append(row)
    out={'post_freeze_gold_only':True,'frozen_sha256':sha(frozen.read_bytes()),'semantic_sufficiency_not_verified':True,'weak_coverage_excludes_boolean_freeform':True,'extractive_eligible':sum(bool(r['mappable_extractive_refs']) for r in rows),'evidence_sentence_eligible':sum(bool(r['evidence_sentences']) for r in rows),'question_demands':{},'arms':{},'rows':rows}
    for r in rows:out['question_demands'][r['demand']]=out['question_demands'].get(r['demand'],0)+1
    for a in fr['tasks'][0]['pointers']:
        out['arms'][a]={'complete_extractive_ref_hits':sum(r['arms'][a]['any_complete_extractive_ref'] is True for r in rows),'evidence_sentences_retained':sum(r['arms'][a]['retained_evidence_sentences'] for r in rows),'evidence_sentences_total':sum(r['evidence_sentences'] for r in rows),'retention':sum(t['stats'][a]['retained_chars'] for t in fr['tasks'])/sum(t['stats'][a]['source_chars'] for t in fr['tasks']),'verified_source_spans':sum(len(t['pointers'][a]) for t in fr['tasks'])}
    (OUT/'source_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
