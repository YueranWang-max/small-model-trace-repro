"""Schema-constrained semantic ID selection; distinct cohort and frozen outputs."""
import json,re,time,sys,urllib.request
import run_qasper_semantic_slots as base
from run_qasper_paragraph_elements import ROOT,SOURCE,construct,sha
from build_qasper_relation_ledger import paragraphs
from repro_lib import OLLAMA_URL
from run_qasper_prefix_vs_bm25 import MODEL

OUT=ROOT/'logs/qasper_schema_core';PROTOCOL=ROOT/'protocol_qasper_schema_core_20261002.md'
INSTRUCTION='''Select original SENTENCE IDs for a reusable scientific-paper core. Preserve EVERY paragraph with at least one sentence. Aimabout60%original characters. Consider the full paper, not only example questions. Preserve these fact families: objects/definitions/components; methods/input/operation/output; data/provenance/settings; results/metrics/comparisons; supporting evidence/mechanisms; limitations/negation/conditions. Preserve object-relation-value-condition units, all needed list items and owners. Method/process sentences can contain core facts. Remove redundant exposition, not needed conditions or evidence. Return ONLY the schema object keep with one array of numeric sentence IDs for EACH paragraph P0..Pn. DO NOT output original sentence text, summaries or explanations. The paragraph and sentence IDs are local: sentence0 belongs separately to each paragraph. Never substitute paragraph IDs for sentence IDs. Example for TWO paragraphs only: {"keep":{"P0":[0,2],"P1":[1]}}. The real paper has a different count; all required keys must be present.'''

def schema(t):
    properties={f'P{i}':{'type':'array','items':{'type':'integer','enum':[s['sid'] for s in t['units'] if s['pid']==i]},'minItems':1,'uniqueItems':True} for i in range(len(t['paragraphs']))}
    return {'type':'object','properties':{'keep':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}},'required':['keep'],'additionalProperties':False}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();start=time.perf_counter();src=json.loads(SOURCE.read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_paragraph_elements/frozen.json').read_text(encoding='utf-8'));prev=json.loads((ROOT/'logs/qasper_semantic_slots/frozen.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));excluded={t['paper_id'] for t in old['tasks']}|{t['paper_id'] for t in prev['papers']};papers=[]
    for pid in sorted(split['validation'],key=lambda p:sha(p.encode())):
        raw=src[pid];ps=paragraphs(raw);ns=sum(len(base.spans(p['text'])) for p in ps)
        if pid in excluded or len(raw['qas'])<4 or len(ps)>45 or ns>180 or sum(len(p['text']) for p in ps)>24000:continue
        c=construct(ps,[q['question'] for q in raw['qas'][:2]],old['priors']['weights']);qs=[{'id':q['question_id'],'question':q['question'],'kind':base.contract(q['question'])[0]} for q in raw['qas'][2:4]]
        t={'paper_id':pid,'title':raw['title'],'shared_question_texts':[q['question'] for q in raw['qas'][:2]],'questions':qs,'paragraphs':ps,'units':c['units'],'full':c['material']['full'],'elements':c['material']['elements'],'elements_pointers':c['pointers']['elements'],'elements_stats':c['stats']['elements']};t['schema']=schema(t);papers.append(t)
        if len(papers)==6:break
    assert len(papers)==6
    files=[SOURCE,PROTOCOL,ROOT/'run_qasper_schema_core.py',ROOT/'run_qasper_semantic_slots.py',ROOT/'logs/qasper_paragraph_elements/frozen.json',ROOT/'logs/qasper_semantic_slots/frozen.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_paragraph_elements.py','build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','run_qasper_contract_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py','run_stageC.py']]
    fr={'papers':papers,'build_instruction':INSTRUCTION,'prep_sec':time.perf_counter()-start,'excluded_papers':sorted(excluded),'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}}
    fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'papers':6,'questions':12,'prep_sec':fr['prep_sec']})

def verify(content,t):
    def reject(reason):return {'accepted':False,'reason':reason,'material':t['full'],'retention':1.0,'facts':[],'pointers':[]}
    try:o=json.loads(content)
    except Exception:return reject('json_parse')
    n=len(t['paragraphs'])
    if not isinstance(o,dict) or set(o)!={'keep'} or not isinstance(o['keep'],dict) or set(o['keep'])!={f'P{i}' for i in range(n)}:return reject('schema')
    bypara=[[s for s in t['units'] if s['pid']==i] for i in range(n)];keys=set()
    for i in range(n):
        ids=o['keep'][f'P{i}']
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(type(j)is not int or not 0<=j<len(bypara[i]) for j in ids):return reject('sentence_id')
        keys.update((i,j) for j in ids)
    total=sum(s['length'] for s in t['units']);before=sum(s['length'] for s in t['units'] if (s['pid'],s['sid']) in keys);changed=True
    while changed:
        old=len(keys)
        for i,j in list(keys):
            s=bypara[i][j]
            if re.match(r'^(?:It|They|This|These|Those|Such|However|Therefore|Consequently|In contrast)\b',s['text']):
                if j:keys.add((i,j-1))
                elif i:keys.add((i-1,len(bypara[i-1])-1))
            if s['text'].rstrip().endswith(':') and j+1<len(bypara[i]):keys.add((i,j+1))
        changed=old!=len(keys)
    amount=sum(s['length'] for s in t['units'] if (s['pid'],s['sid']) in keys)
    if amount/total>.85:
        r=reject('closure_over_budget');r['proposed_retention_before_closure']=before/total;r['proposed_retention_after_closure']=amount/total;return r
    chunks=[];pointers=[]
    for i,ss in enumerate(bypara):
        parts=[];previous=None
        for s in ss:
            j=s['sid']
            if (i,j) not in keys:continue
            if (previous is None and j>0) or (previous is not None and j>previous+1):parts.append('[Sentences omitted]')
            parts.append(f'[S{j}] '+s['text']);previous=j;body=t['paragraphs'][i]['text'];a=s['start'];b=a+s['length'];rec={**s,'paragraph_sha256':sha(body.encode()),'byte_start':len(body[:a].encode()),'byte_length':len(body[a:b].encode())};assert body[a:b]==s['text'];assert body.encode()[rec['byte_start']:rec['byte_start']+rec['byte_length']].decode()==s['text'];pointers.append(rec)
        if previous<len(ss)-1:parts.append('[Sentences omitted]')
        chunks.append(f"[P{i}] Section: {t['paragraphs'][i]['heading']}\n"+'\n'.join(parts))
    return {'accepted':True,'reason':'schema_and_source_only_not_semantic_verification','material':'\n\n'.join(chunks),'retention':amount/total,'retention_before_closure':before/total,'facts':[],'pointers':pointers}

reader_call=base.call
def dispatch(prompt,inst,*,think,cap):
    if cap!=4096:return reader_call(prompt,inst,think=think,cap=cap)
    start=time.perf_counter();fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));t=next(t for t in fr['papers'] if t['full'] in prompt);sc=t['schema'];actual=prompt+'\nJSON schema:\n'+json.dumps(sc,separators=(',',':'))
    payload={'model':MODEL,'messages':[{'role':'system','content':'You select scientific source sentence IDs. Follow the schema. Output only numeric IDs, never source text.'},{'role':'user','content':actual}],'format':sc,'stream':False,'think':False,'options':{'temperature':0,'num_ctx':32768,'num_predict':1536,'top_p':.9}}
    try:
        req=urllib.request.Request(OLLAMA_URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=900) as response:raw=json.loads(response.read().decode())
        msg=raw.get('message') or {};return {'content':msg.get('content',''),'thinking':msg.get('thinking',''),'prompt_sha256':sha(actual.encode()),'request_sha256':sha(json.dumps(payload,sort_keys=True).encode()),'prompt_chars':len(actual),'prompt_tokens':raw.get('prompt_eval_count'),'generation_tokens':raw.get('eval_count'),'prefill_ns':raw.get('prompt_eval_duration'),'eval_ns':raw.get('eval_duration'),'api_total_ns':raw.get('total_duration'),'load_ns':raw.get('load_duration'),'wall_sec':round(time.perf_counter()-start,3),'response_sha256':sha(json.dumps(raw,sort_keys=True).encode())}
    except Exception as exc:return {'content':'','error':repr(exc),'wall_sec':round(time.perf_counter()-start,3)}

def run():
    base.OUT=OUT;base.PROTOCOL=PROTOCOL;base.verify=verify;base.call=dispatch;base.run()

def analyze():base.OUT=OUT;base.analyze()

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
