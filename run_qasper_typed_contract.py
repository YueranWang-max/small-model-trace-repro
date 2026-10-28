"""Deterministic checks on manually typed facts, plus fair answer-cache accounting."""
import json,hashlib,time,statistics
from pathlib import Path
from build_qasper_relation_ledger import paragraphs

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/qasper_typed_contract'
LEDGER=ROOT/'logs/qasper_relation_ledger/ledger.json';SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
KEYS=('paper_id','subject','relation','role','conditions')

def sha(b):return hashlib.sha256(b).hexdigest()
def packed(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
def key(record):return json.dumps({k:record[k] for k in KEYS},ensure_ascii=False,sort_keys=True,separators=(',',':'))

def compatibility(request,candidate,reference):
    for k in KEYS:
        if k not in request or k not in candidate or k not in reference:return {'status':'insufficient','reason':'missing_'+k}
    if 'value' not in candidate or 'value' not in reference:return {'status':'insufficient','reason':'missing_value'}
    if not all(isinstance(x.get('conditions'),dict) for x in (request,candidate,reference)):return {'status':'insufficient','reason':'conditions_schema'}
    for k in KEYS:
        if request[k]!=reference[k]:return {'status':'insufficient','reason':'reference_not_for_request_'+k}
    for k in KEYS[:-1]:
        if candidate[k]!=request[k]:return {'status':'reject','reason':'wrong_'+k}
    for name,value in request['conditions'].items():
        if name not in candidate['conditions']:return {'status':'insufficient','reason':'missing_condition_'+name}
        if candidate['conditions'][name]!=value:return {'status':'reject','reason':'wrong_condition_'+name}
    if candidate['conditions']!=reference['conditions']:return {'status':'insufficient','reason':'extra_unverified_condition'}
    if candidate['value']!=reference['value']:return {'status':'reject','reason':'value_mismatch_typed_reference'}
    return {'status':'support','reason':'exact_typed_record_match'}

def validate(fact,source):
    ps=paragraphs(source[fact['paper_id']])
    for e in fact['evidence']:
        text=ps[e['paragraph_id']]['text'];a=e['char_start'];b=a+e['char_length'];u=e['byte_start'];v=u+e['byte_length']
        if sha(text.encode())!=e['paragraph_sha256'] or text[a:b]!=e['quote'] or text.encode()[u:v]!=e['quote'].encode():return False
    return True

def main():
    OUT.mkdir(parents=True,exist_ok=True);ledger=json.loads(LEDGER.read_text(encoding='utf-8'));facts=ledger['facts'];source=json.loads(SOURCE.read_text(encoding='utf-8'))
    frozen={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [LEDGER,SOURCE,Path(__file__),ROOT/'build_qasper_relation_ledger.py',ROOT/'protocol_qasper_typed_contract_20261002.md']},'fact_ids':[f['id'] for f in facts],'no_model_calls':True}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    checks=[]
    for f in facts:
        ref={k:f[k] for k in KEYS+('value',)};request={k:f[k] for k in KEYS}
        checks.append({'fact_id':f['id'],'mutation':'unchanged','expected':'support','actual':compatibility(request,ref,ref)})
        for k in ('subject','relation','role','value'):
            mutant=dict(ref);mutant[k]='deliberate_wrong_'+k
            checks.append({'fact_id':f['id'],'mutation':'wrong_'+k,'expected':'reject','actual':compatibility(request,mutant,ref)})
        mutant=dict(ref);del mutant['role'];checks.append({'fact_id':f['id'],'mutation':'missing_role','expected':'insufficient','actual':compatibility(request,mutant,ref)})
        if f['conditions']:
            for k in f['conditions']:
                mutant=dict(ref);mutant['conditions']=dict(ref['conditions']);mutant['conditions'][k]='wrong_stage'
                checks.append({'fact_id':f['id'],'mutation':'wrong_condition_'+k,'expected':'reject','actual':compatibility(request,mutant,ref)})
                mutant=dict(ref);mutant['conditions']={};checks.append({'fact_id':f['id'],'mutation':'missing_condition_'+k,'expected':'insufficient','actual':compatibility(request,mutant,ref)})
        assert validate(f,source)
        broken=json.loads(json.dumps(f));broken['evidence'][0]['paragraph_sha256']='0'*64
        checks.append({'fact_id':f['id'],'mutation':'source_hash_tamper','expected':False,'actual':validate(broken,source)})
    assert all((x['actual']['status'] if isinstance(x['actual'],dict) else x['actual'])==x['expected'] for x in checks)
    (OUT/'calibration.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    cache={key(f):f['value'] for f in facts}
    typed={key(f):{k:v for k,v in f.items() if k not in ('evidence','semantic_label_author','independent_review')} for f in facts}
    pointer={key(f):{**typed[key(f)],'evidence':f['evidence']} for f in facts}
    pointer_only={key(f):{'evidence':[{k:v for k,v in e.items() if k not in ('quote','heading')} for e in f['evidence']]} for f in facts}
    storage={}
    for name,obj in [('direct_answer_cache',cache),('typed_value_cache',typed),('pointer_backed_typed_cache',pointer),('pointer_only_not_answer_cache',pointer_only)]:
        b=packed(obj);(OUT/(name+'.json')).write_bytes(b);storage[name]=len(b)
    common_source={pid:paragraphs(source[pid]) for pid in sorted({f['paper_id'] for f in facts})};common_bytes=len(packed(common_source))
    keys=[key(f) for f in facts];times={};outputs={}
    for mode in ('direct_answer_cache','typed_value_cache','pointer_backed_typed_cache'):
        runs=[];result=[]
        for rep in range(5):
            t=time.perf_counter();result=[]
            for _ in range(1000):
                for k in keys:
                    if mode=='direct_answer_cache':value=cache[k]
                    elif mode=='typed_value_cache':value=typed[k]['value']
                    else:
                        rec=pointer[k]
                        assert validate(rec,source)
                        value=rec['value']
                    result.append(value)
            runs.append(time.perf_counter()-t)
        outputs[mode]=sha(packed(result));times[mode]={'median_sec_10000_lookups':statistics.median(runs),'five_runs_sec':runs}
    assert len(set(outputs.values()))==1
    summary={'development_cases':len(checks),'calibration_passed':len(checks),'facts':len(facts),'storage_utf8_bytes':storage,'shared_audit_source_bytes_all_modes':common_bytes,'lookup_microbenchmark':times,'equal_output_hashes':outputs,'model_calls':0,'manual_build_and_interpretation_cost_unmeasured':True,'not_general_semantic_accuracy':True,'pointer_only_has_no_interpreted_answer':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
