"""Verify saved method inputs, costs, state feedback and source integrity."""
import json
from run_qasper_state_resume import OUT,SOURCE,PROTOCOL,ARMS,MODELS,dependencies,sha,material,route,save,normalize

def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));data=json.loads((OUT/'scored_results.json').read_text(encoding='utf-8'))
    assert fr['dependencies']==dependencies() and fr['source_sha256']==sha(SOURCE.read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes())
    assert data['frozen_sha256']==sha((OUT/'frozen.json').read_bytes())
    calls=attempts=state_bindings=source_blocks=errors=0;diagnostics=[]
    for model in MODELS:
        rows=data['models'][model]['records'];assert len(rows)==12
        for row in rows:
            p=next(p for p in fr['papers'] if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id'])
            assert sha('\n\n'.join(x['text'] for x in p['paragraphs']).encode())==p['serialized_source_sha256']
            offset=0
            for para in p['paragraphs']:
                assert para['serialized_char_start']==offset and para['text_sha256']==sha(para['text'].encode());offset+=len(para['text'])+2
            ids,queries=route(p,q,row['producer']['parsed']);assert ids==row['adaptive_ids'] and queries==row['routing_queries']
            assert set(row['arms'])==set(ARMS)
            for arm,r in [('producer',row['producer']),*row['arms'].items()]:
                calls+=1;attempts+=r['http_attempts'];errors+=('error' in r);assert r['http_attempts']==1
                assert r['request_sha256']==sha(json.dumps(r['request'],sort_keys=True).encode())
                if 'error' not in r:assert json.loads(r['raw']['message']['content'])==r['parsed']
                assert r['request']['model']==model
                prompt=r['request']['messages'][1]['content']
                selected=q['rank'][:8] if arm=='producer' else r['paragraph_ids']
                assert material(p,selected) in prompt;source_blocks+=len(selected)
                if arm.startswith('state_'):
                    assert json.dumps(row['producer']['parsed'],ensure_ascii=False) in prompt;state_bindings+=1
            assert row['arms']['reread']['request']==row['arms']['wide']['request']
            diagnostics.append({'model':model,'paper_id':p['id'],'question_id':q['id'],'question':q['text'],'producer':row['producer']['parsed'],'queries':queries,'added_ids':[i for i in ids if i not in q['rank'][:8]],'arms':{a:{'answer':r['parsed'].get('answer',''),'error':r.get('error'),'f1':r['official_f1'],'evidence_all_hit':r['evidence_all_hit']} for a,r in row['arms'].items()}})
    assert calls==168 and attempts==168 and state_bindings==48
    out={'formal_calls':calls,'http_attempts':attempts,'warmup_calls':2,'state_bindings':state_bindings,'source_blocks_verified':source_blocks,'call_errors':errors,'audit_pass':True,'results_sha256':sha((OUT/'results.json').read_bytes()),'scored_results_sha256':sha((OUT/'scored_results.json').read_bytes())}
    save(OUT/'independent_audit.json',out);save(OUT/'diagnostics.json',diagnostics);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
