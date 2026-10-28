"""Source retention, exact guide provenance, and post-call annotation audit."""
import json,re
from run_qasper_retained_source import OUT,SOURCE,ARMS
from run_qasper_prefix_vs_bm25 import sha
from build_qasper_relation_ledger import paragraphs

def norm(t):return re.sub(r'\s+',' ',t).strip().lower()

def main():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));source=json.loads(SOURCE.read_text(encoding='utf-8'));assert len(d['rows'])==24 and all(set(r['arms'])==set(ARMS) for r in d['rows'])
    rows=[];verified=0;eligible=0;mappable=0;hits={'full':0,'core80':0};all_source_bytes=0;guide_bytes=0;material_bytes={a:0 for a in ARMS}
    for t in f['tasks']:
        ps=paragraphs(source[t['paper_id']]);assert t['full_ids']==list(range(len(ps)))
        for a in t['anchors']:
            text=ps[a['paragraph_id']]['text'];assert text[a['char_start']:a['char_start']+a['char_length']]==a['quote'];assert text.encode()[a['byte_start']:a['byte_start']+a['byte_length']]==a['quote'].encode();assert sha(text.encode())==a['paragraph_sha256'];verified+=1
        for r in t['guide_layer_records']:
            text=ps[r['paragraph_id']]['text'];assert text[r['char_start']:r['char_start']+r['char_length']]==r['quote'];assert sha(text.encode())==r['paragraph_sha256'];assert text[r['context_char_start']:r['context_char_start']+len(r['context'])]==r['context'];verified+=1
        actual=sum(len(ps[i]['text']) for i in t['core_ids']);assert actual==t['retained_chars'];assert actual/sum(len(p['text']) for p in ps)>=.8
        q=next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']);anns=[]
        for ref in q['answers']:
            a=ref['answer'];ev=a.get('evidence',[])
            if a['unanswerable'] or not ev:continue
            mapping=[[i for i,p in enumerate(ps) if norm(e) in norm(p['text'])] for e in ev]
            anns.append({'paragraph_matches':mapping,'all_source_mappable':all(mapping),'core_complete':all(any(i in t['core_ids'] for i in ms) for ms in mapping)})
        if anns:eligible+=1
        if any(a['all_source_mappable'] for a in anns):mappable+=1;hits['full']+=1
        if any(a['core_complete'] for a in anns):hits['core80']+=1
        row=next(r for r in d['rows'] if r['id']==t['id'])
        rows.append({'paper_id':t['paper_id'],'id':t['id'],'question':t['question'],'retention':t['retention'],'annotations':anns,'answers':{a:row['arms'][a]['answer'] for a in ARMS},'f1':{a:row['arms'][a]['official_f1'] for a in ARMS}})
        guide_bytes+=len(t['guide'].encode())
        for a in ARMS:material_bytes[a]+=len(t['material'][a].encode())
    for pid in f['paper_ids']:all_source_bytes+=len(json.dumps(paragraphs(source[pid]),ensure_ascii=False,separators=(',',':')).encode())
    out={'verified_guide_quotes':verified,'answerable_evidence_eligible':eligible,'questions_with_fully_source_mappable_annotation':mappable,'complete_annotation_hits':hits,'source_loss_cases':hits['full']-hits['core80'],'common_retained_source_projection_bytes':all_source_bytes,'query_guide_bytes_total':guide_bytes,'reader_material_bytes_total':material_bytes,'posthoc_gold_only':True,'coverage_not_semantic_sufficiency':True,'rows':rows}
    (OUT/'source_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
