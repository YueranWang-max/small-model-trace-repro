import pathlib,json,hashlib,time
import run_bridge as base
ROOT=pathlib.Path(__file__).resolve().parent;ORIGINAL=ROOT/'bridge_run';OUT=ROOT/'bridge_continuation'
def main():
    assert not OUT.exists(),'Do not overwrite or restart an interrupted continuation.'
    frozen=json.loads((ORIGINAL/'frozen.json').read_text(encoding='utf-8'))
    assert base.sha(ROOT/'run_bridge.py')==frozen['runner_sha256']
    assert base.sha(ROOT/'bridge_protocol.md')==frozen['protocol_sha256']
    for ordinal in range(1,20):
        rec=json.loads((ORIGINAL/f'attempt_{ordinal:03d}.json').read_text(encoding='utf-8'))
        assert rec['job']==frozen['schedule'][ordinal-1] and rec['status']=='returned'
        assert rec['input_qualified'] and rec['resident_qualified']
    interrupted=json.loads((ORIGINAL/'attempt_020.json').read_text(encoding='utf-8'))
    assert interrupted['status']=='attempting_once' and 'response' not in interrupted
    assert not any((ORIGINAL/f'attempt_{i:03d}.json').exists() for i in range(21,65))
    tags=base.request('tags')
    for name,digest in frozen['digests'].items():assert next(x['digest'] for x in tags['models'] if x['name']==name)==digest
    OUT.mkdir();base.save(OUT/'frozen_continuation.json',dict(original_frozen_sha256=base.sha(ORIGINAL/'frozen.json'),script_sha256=base.sha(pathlib.Path(__file__)),protocol_sha256=base.sha(ROOT/'continuation_protocol.md'),ordinals=list(range(21,65)),unknown_original_attempts=[20],warmups=0))
    for ordinal in range(21,65):
        job=frozen['schedule'][ordinal-1];path=OUT/f'attempt_{ordinal:03d}.json'
        record=dict(ordinal=ordinal,job=job,status='not_yet_attempted',phase='continuation');base.save(path,record)
        payload=dict(model=job['model'],messages=[dict(role='system',content=base.SYSTEM),dict(role='user',content=job['user'])],think=True,stream=False,options=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,repeat_penalty=1,num_ctx=32768,num_predict=8192,seed=job['seed']))
        record.update(status='attempting_once',start_time=time.time());base.save(path,record)
        try:record.update(response=base.request('chat',payload,900),status='returned')
        except Exception as e:record.update(status='request_error',error=repr(e))
        record['end_time']=time.time();base.save(path,record)
        if 'response' not in record:print(ordinal,'request_error',flush=True);continue
        raw=record['response'];record['input_qualified']=raw.get('prompt_eval_count')==job['expected_prompt_tokens'];base.save(path,record)
        ps=base.request('ps');base.save(OUT/f'ps_{ordinal:03d}.json',ps)
        record['resident_qualified']=any(m.get('name')==job['model'] and m.get('digest')==frozen['digests'][job['model']] and m.get('context_length',0)>=32768 for m in ps.get('models',[]));base.save(path,record)
        print(ordinal,job['row_index'],job['model'],raw.get('done_reason'),raw.get('prompt_eval_count'),raw.get('eval_count'),flush=True)
        assert record['input_qualified'] and record['resident_qualified'],'Qualification failed; preserve ledger.'
    base.save(OUT/'completion.json',dict(completed_previously_unattempted_calls=44,original_primary_unknown_attempts=[20],original_primary_complete=False,finished_at=time.time()))
if __name__=='__main__':main()
