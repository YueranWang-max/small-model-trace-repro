"""Post-build source mapping and novel-question coverage; no selector feedback."""
import json,re
from run_qasper_semantic_slots import OUT,SOURCE,sha

def norm(s):return re.sub(r'\s+',' ',s).strip().lower()

def main():
    fp=OUT/'frozen.json';rp=OUT/'results.json';fr=json.loads(fp.read_text(encoding='utf-8'));d=json.loads(rp.read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'));rows=[];facts=[];verified=0
    assert len(d['papers'])==6 and all(len(p['rows'])==2 and len(r['arms'])==3 for p in d['papers'] for r in p['rows'])
    for t in fr['papers']:
        p=next(p for p in d['papers'] if p['paper_id']==t['paper_id']);sel=p['selection'];ps=t['paragraphs'];units=t['units'];bykey={(s['pid'],s['sid']):s for s in units}
        for r in sel['pointers']:
            body=ps[r['pid']]['text'];assert sha(body.encode())==r['paragraph_sha256'];assert body[r['start']:r['start']+r['length']]==r['text'];assert body.encode()[r['byte_start']:r['byte_start']+r['byte_length']].decode()==r['text'];verified+=1
        for f in sel['facts']:
            sentence=bykey[(f['p'],f['s'])]['text'];assert all(not f[k] or f[k] in sentence for k in ('subject','value','conditions'));facts.append({'paper_id':t['paper_id'],**f,'original_sentence':sentence,'semantic_entailment_independently_unverified':True})
        retained={'full':units,'elements':t['elements_pointers'],'semantic':sel['pointers'] if sel['accepted'] else units}
        for task in t['questions']:
            q=next(q for q in src[t['paper_id']]['qas'] if q['question_id']==task['id']);refs=[list({s.strip() for s in a['answer']['extractive_spans'] if s.strip()}) for a in q['answers']];full=' '.join(s['text'] for s in units);refs=[ss for ss in refs if ss and all(norm(s) in norm(full) for s in ss)]
            ev={norm(e) for a in q['answers'] for e in a['answer']['evidence'] if e.strip()};mapped=[s for s in units if any(norm(s['text']) in e for e in ev)];row={'paper_id':t['paper_id'],'id':q['question_id'],'question':q['question'],'extractive_eligible':bool(refs),'evidence_sentences':len(mapped),'arms':{}}
            for a,ss in retained.items():
                text=' '.join(s['text'] for s in ss);keys={(s['pid'],s['sid']) for s in ss};row['arms'][a]={'complete_extractive_ref':any(all(norm(s) in norm(text) for s in rr) for rr in refs) if refs else None,'evidence_sentences_retained':sum((s['pid'],s['sid']) in keys for s in mapped)}
            rows.append(row)
    out={'frozen_sha256':sha(fp.read_bytes()),'results_sha256':sha(rp.read_bytes()),'verified_semantic_source_spans':verified,'facts':facts,'fact_count':len(facts),'post_build_gold_only':True,'coverage_not_semantic_sufficiency':True,'extractive_eligible':sum(r['extractive_eligible'] for r in rows),'arms':{},'rows':rows}
    for a in ('full','elements','semantic'):out['arms'][a]={'complete_extractive_refs':sum(r['arms'][a]['complete_extractive_ref'] is True for r in rows),'retained_evidence_sentences':sum(r['arms'][a]['evidence_sentences_retained'] for r in rows),'total_evidence_sentences':sum(r['evidence_sentences'] for r in rows)}
    (OUT/'source_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','facts')},indent=2))

if __name__=='__main__':main()
