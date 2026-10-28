"""Gold-only diagnostic source-pointer ledger, never used for answer selection."""
import json
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_fingerprint import prepare

def main():
    out=ROOT/'logs/qasper_measurement_audit';sourcepath=ROOT/'data/qasper_official/qasper-train-v0.3.json'
    raw=json.loads(sourcepath.read_text(encoding='utf-8'));ledgers=[]
    for label in ('validation','heldout'):
        data=json.loads((out/(label+'_annotations.json')).read_text(encoding='utf-8'));papers=[]
        for item in data:
            p=prepare(item['paper_id'],raw[item['paper_id']]);paragraphs=p['paragraphs']
            text='\n\n'.join(v['text'] for v in paragraphs);char_offset=byte_offset=0;ptrs=[]
            for i,para in enumerate(paragraphs):
                body=para['text'];encoded=body.encode('utf-8')
                ptrs.append({'paragraph_id':i,'heading':para['heading'],'char_start':char_offset,
                    'char_length':len(body),'utf8_byte_start':byte_offset,'utf8_byte_length':len(encoded),
                    'paragraph_sha256':sha(encoded)})
                assert text[char_offset:char_offset+len(body)]==body
                assert text.encode('utf-8')[byte_offset:byte_offset+len(encoded)]==encoded
                char_offset+=len(body)+2;byte_offset+=len(encoded)+2
            papers.append({'paper_id':item['paper_id'],'source_sha256':sha(text.encode('utf-8')),
                'serialization':'abstract then nonempty full_text paragraphs stripped, separated by two LF characters',
                'paragraph_pointers':ptrs,'gold_annotations':item['records']})
        path=out/(label+'_pointer_ledger.json');path.write_text(json.dumps(papers,indent=2,ensure_ascii=False),encoding='utf-8')
        ledgers.append({'split':label,'sha256':sha(path.read_bytes()),'papers':len(papers)})
    provenance={'label':'gold-only diagnostic, paragraph pointers are source matched, not semantic verification or deployment memory',
        'train_sha256':sha(sourcepath.read_bytes()),
        'test_sha256':sha((ROOT/'data/qasper_official/qasper-test-v0.3.json').read_bytes()),
        'ledgers':ledgers,'code_sha256':sha((ROOT/'build_qasper_evidence_audit_ledger.py').read_bytes())}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
