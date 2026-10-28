"""Distinct frozen cohort with explicit per-paragraph sentence count quotas."""
import json,time,math,sys
import run_qasper_schema_core as parent
import run_qasper_semantic_slots as base
from run_qasper_paragraph_elements import ROOT,SOURCE,construct,sha
from build_qasper_relation_ledger import paragraphs

OUT=ROOT/'logs/qasper_bounded_core';PROTOCOL=ROOT/'protocol_qasper_bounded_core_20261002.md'
INSTRUCTION=parent.INSTRUCTION+'\nYou MUST respect maxItems for EACH paragraph. Select the best core facts within the quota. Do not output every sentence. If there are2sentences you maykeep2; if3sentences keepatmost2; if4sentences keepatmost3. The schema gives each exact upper limit. Keep different fact types rather than choosing sentences merely because they appear first.'

def schema(t):
    sc=parent.schema(t)
    for p,v in sc['properties']['keep']['properties'].items():v['maxItems']=max(1,math.ceil(.6*len(v['items']['enum'])))
    return sc

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();start=time.perf_counter();src=json.loads(SOURCE.read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_paragraph_elements/frozen.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));excluded={t['paper_id'] for t in old['tasks']}
    for name in ('qasper_semantic_slots','qasper_schema_core'):excluded.update(t['paper_id'] for t in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['papers'])
    papers=[]
    for pid in sorted(split['validation'],key=lambda p:sha(p.encode())):
        raw=src[pid];ps=paragraphs(raw);ns=sum(len(base.spans(p['text'])) for p in ps)
        if pid in excluded or len(raw['qas'])<4 or len(ps)>45 or ns>180 or sum(len(p['text']) for p in ps)>24000:continue
        c=construct(ps,[q['question'] for q in raw['qas'][:2]],old['priors']['weights']);qs=[{'id':q['question_id'],'question':q['question'],'kind':base.contract(q['question'])[0]} for q in raw['qas'][2:4]]
        t={'paper_id':pid,'title':raw['title'],'shared_question_texts':[q['question'] for q in raw['qas'][:2]],'questions':qs,'paragraphs':ps,'units':c['units'],'full':c['material']['full'],'elements':c['material']['elements'],'elements_pointers':c['pointers']['elements'],'elements_stats':c['stats']['elements']};t['schema']=schema(t);papers.append(t)
        if len(papers)==6:break
    assert len(papers)==6
    files=[SOURCE,PROTOCOL,ROOT/'run_qasper_bounded_core.py',ROOT/'run_qasper_schema_core.py',ROOT/'run_qasper_semantic_slots.py',ROOT/'logs/qasper_paragraph_elements/frozen.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/'logs'/n/'frozen.json' for n in ('qasper_semantic_slots','qasper_schema_core')]+[ROOT/n for n in ['run_qasper_paragraph_elements.py','build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','run_qasper_contract_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py','run_stageC.py']]
    fr={'papers':papers,'build_instruction':INSTRUCTION,'prep_sec':time.perf_counter()-start,'excluded_papers':sorted(excluded),'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}};fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'papers':6,'questions':12,'prep_sec':fr['prep_sec']})

def verify(content,t):
    try:o=json.loads(content)
    except Exception:return original_verify(content,t)
    if isinstance(o,dict) and isinstance(o.get('keep'),dict):
        for p,ids in o['keep'].items():
            spec=t['schema']['properties']['keep']['properties'].get(p)
            if spec and isinstance(ids,list) and len(ids)>spec['maxItems']:return {'accepted':False,'reason':'sentence_quota','material':t['full'],'retention':1.0,'facts':[],'pointers':[]}
    return original_verify(content,t)

original_verify=parent.verify
def run():
    parent.OUT=OUT;base.OUT=OUT;base.PROTOCOL=PROTOCOL;base.verify=verify;base.call=parent.dispatch;base.run()

def analyze():base.OUT=OUT;base.analyze()

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
