"""Independent reconstruction of every frozen replication input and score."""
import json
from collections import Counter
from run_qasper_interface_replication import OUT,ROOT,SOURCE,SPLIT,MODELS,ARMS,validate,prepare,reader_prompt,producer_prompt,sha,save,references,f1,normalize

def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));validate(fr);raw=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads(SPLIT.read_text(encoding='utf-8'));eligible=[pid for pid in split['heldout'] if len(raw[pid]['qas'])>=2];assert [p['id'] for p in fr['papers']]==eligible[30:42]
    old=json.loads((ROOT/'logs/qasper_state_resume/frozen.json').read_text(encoding='utf-8'));assert not {p['id'] for p in old['papers']}&{p['id'] for p in fr['papers']}
    data=json.loads((OUT/'scored_results.json').read_text(encoding='utf-8'));assert data['frozen_sha256']==sha((OUT/'frozen.json').read_bytes());count=errors=bindings=blocks=cap_hits=0
    for p in fr['papers']:
        rebuilt=prepare(p['id'],raw[p['id']]);assert [q['id'] for q in p['questions']]==[q['id'] for q in rebuilt['questions'][:2]] and p['title']==raw[p['id']]['title']
        assert [q['rank'] for q in p['questions']]==[q['rank'] for q in rebuilt['questions'][:2]]
        assert [q['text'] for q in p['questions']]==[q['text'] for q in rebuilt['questions'][:2]]
        assert len(p['paragraphs'])==len(rebuilt['paragraphs'])
        assert sha('\n\n'.join(x['text'] for x in p['paragraphs']).encode())==p['serialized_source_sha256'];offset=0
        for a,b in zip(p['paragraphs'],rebuilt['paragraphs']):
            assert a['text']==b['text'] and a['heading']==b['heading'] and a['serialized_char_start']==offset and a['text_sha256']==sha(a['text'].encode());offset+=len(a['text'])+2
    for model in MODELS:
        rows=data['models'][model]['records'];assert len(rows)==24
        assert len({r['id'] for r in rows})==24
        for pos in range(6):assert Counter(r['order'][pos] for r in rows)==Counter({a:4 for a in ARMS})
        for row in rows:
            p=next(p for p in fr['papers'] if p['id']==row['paper_id']);q=next(q for q in p['questions'] if q['id']==row['id']);gq=next(v for v in raw[p['id']]['qas'] if v['question_id']==q['id']);refs=references(gq)
            assert set(row['arms'])==set(ARMS)
            for arm,r in [('producer',row['producer']),*row['arms'].items()]:
                expected=producer_prompt(p,q) if arm=='producer' else reader_prompt(p,q,arm,row['producer']['parsed'])[0]
                req=r['request'];assert req['messages'][1]['content']==expected and req['model']==model and req['think'] is False and req['options']=={'temperature':0.0,'num_ctx':32768,'num_predict':384 if arm=='producer' else 768}
                assert r['http_attempts']==1 and r['request_sha256']==sha(json.dumps(req,sort_keys=True).encode());count+=1;errors+=('error' in r);cap_hits+=r.get('raw',{}).get('done_reason')=='length'
                if 'error' not in r:assert json.loads(r['raw']['message']['content'])==r['parsed']
                if arm!='producer':
                    _,ids=reader_prompt(p,q,arm,row['producer']['parsed']);assert ids==r['paragraph_ids'];blocks+=len(ids);pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred;assert abs(r['f1']-max(f1(aligned,g) for g in refs))<1e-12
                if arm=='state_bind':bindings+=1
            wide=row['arms']['wide_bind']['request']['messages'][1]['content'];state=row['arms']['state_bind']['request']['messages'][1]['content'];assert wide.split('\nFinal answer binding: ')[1]==state.split('\nFinal answer binding: ')[1]
    assert count==336 and bindings==48
    audit={'formal_calls':count,'http_attempts':count,'warmups':2,'errors':errors,'generation_cap_hits':cap_hits,'state_bindings_verified':bindings,'reader_source_blocks_verified':blocks,'audit_pass':True,'results_sha256':sha((OUT/'results.json').read_bytes()),'frozen_sha256':sha((OUT/'frozen.json').read_bytes()),'scope':'new strategy readout; source exposure not independent'}
    save(OUT/'independent_audit.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
