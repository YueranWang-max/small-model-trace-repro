"""Reconstruct privileged and adaptive diagnostic inputs and scores."""
import json,re
from pathlib import Path
from run_qasper_state_resume import ROOT,OUT,SOURCE,MODELS,sha,save,material,evidence,references,f1,normalize

def main():
    pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));ps=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))['papers'];src=json.loads(SOURCE.read_text(encoding='utf-8'));total=0
    for name,n in [('oracle',24),('ablation',48)]:
        dest=ROOT/f'logs/qasper_state_resume_{name}';fr=json.loads((dest/'frozen.json').read_text(encoding='utf-8'));data=json.loads((dest/'results.json').read_text(encoding='utf-8'));assert fr['script_sha256']==sha((ROOT/f'run_qasper_state_resume_{name}.py').read_bytes()) and fr['protocol_sha256']==sha((ROOT/f'protocol_qasper_state_resume_{name}_20261003.md').read_bytes())
        assert data['frozen_sha256']==sha((dest/'frozen.json').read_bytes());count=errors=0
        for row in data['records']:
            model=row['model'];p=next(p for p in ps if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id']);pr=next(r for r in pilot['models'][model]['records'] if r['id']==q['id']);goldq=next(q for q in src[p['id']]['qas'] if q['question_id']==row['id'])
            if name=='oracle':
                ids=sorted(evidence(p,q,src)) or list(range(len(p['paragraphs'])));assert ids==row['paragraph_ids']
                expected=pr['arms']['full']['request']['messages'][1]['content'].replace(material(p,list(range(len(p['paragraphs'])))),material(p,ids),1);pairs=[('oracle',row['result'],expected)]
            else:
                c=next(c for c in fr['cases'] if c['model']==model and c['id']==q['id']);assert set(row['arms'])=={'queries_only','state_extract'}
                pairs=[(a,r,c['prompts'][a]) for a,r in row['arms'].items()]
                expected_known=json.dumps(pr['producer']['parsed'],ensure_ascii=False);assert expected_known in c['prompts']['state_extract']
                assert material(p,q['rank'][:12]) in c['prompts']['queries_only'] and material(p,q['rank'][:12]) in c['prompts']['state_extract']
                assert json.dumps({'missing_queries':pr['producer']['parsed'].get('missing_queries',[])},ensure_ascii=False) in c['prompts']['queries_only']
            for a,r,expected in pairs:
                count+=1;errors+=('error' in r);assert r['request']['messages'][1]['content']==expected and r['request']['model']==model and r['http_attempts']==1 and r['request_sha256']==sha(json.dumps(r['request'],sort_keys=True).encode())
                if 'error' not in r:assert json.loads(r['raw']['message']['content'])==r['parsed']
                pred=r['parsed'].get('answer','');aligned='Unanswerable' if normalize(pred)=='unknown' else pred;assert abs(r['official_f1']-max(f1(aligned,g) for g in references(goldq)))<1e-12
        assert count==n;total+=count
        save(dest/'independent_audit.json',{'calls':count,'http_attempts':count,'errors':errors,'audit_pass':True,'frozen_sha256':sha((dest/'frozen.json').read_bytes()),'results_sha256':sha((dest/'results.json').read_bytes()),'scope':'privileged diagnostic' if name=='oracle' else 'adaptive reused-case diagnostic'})
    print('Audited72 diagnostic calls;240 formal calls with pilot;2 warmups separate.')

if __name__=='__main__':main()
