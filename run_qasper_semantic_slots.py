"""One shared semantic selection per paper, exact originals, novel question readout."""
import json,re,time,sys,random
from collections import Counter
from run_qasper_paragraph_elements import ROOT,SOURCE,spans,construct,sha
from build_qasper_relation_ledger import paragraphs
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from run_qasper_contract_replication import contract
from rescore_qasper_demand_transfer import references,f1,normalize

OUT=ROOT/'logs/qasper_semantic_slots';PROTOCOL=ROOT/'protocol_qasper_semantic_slots_20261002.md';ARMS=('full','elements','semantic')
RELATIONS=('definition','component','method','data','setting','result','comparison','evidence','limitation')
BUILD_INSTRUCTION='''Select an original-sentence reading core of the scientific paper that answers many ordinary questions, not just the two example questions. Every paragraph must retain at least one sentence. Aim for about60%original characters. Keep core entities, definitions, object-property-value facts, methods, data provenance, metrics, findings, comparisons, evidence, limitations and their conditions. A procedural sentence can contain an essential fact. Remove only redundant exposition. Keep all members needed for lists, owner definitions, negation, quantifiers and contrasting scopes. Do not rewrite sentences. Return ONLY JSON:
{"keep":[[sentence_ids_for_P0],[sentence_ids_for_P1],...],"facts":[{"p":0,"s":0,"subject":"exact original substring","relation":"definition","value":"exact original substring","conditions":"exact original substring or empty"},...]}
keep must have one nonempty list per paragraph in order, valid integer IDs. facts must have4to12 objects, all anchored to retained sentences. subject/value must be nonempty EXACT substrings of that original sentence; conditions can be empty or exact substring. relation is one of definition,component,method,data,setting,result,comparison,evidence,limitation. Facts are a proposed rationale, not a final answer. Do not invent unsupported binding. Include exact owning phrase/conditions when present.
'''

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();start=time.perf_counter()
    src=json.loads(SOURCE.read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_paragraph_elements/frozen.json').read_text(encoding='utf-8'));excluded={t['paper_id'] for t in old['tasks']};split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));papers=[]
    for pid in sorted(split['validation'],key=lambda p:sha(p.encode())):
        raw=src[pid];ps=paragraphs(raw);ns=sum(len(spans(p['text'])) for p in ps)
        if pid in excluded or len(raw['qas'])<4 or len(ps)>45 or ns>180 or sum(len(p['text']) for p in ps)>24000:continue
        c=construct(ps,[q['question'] for q in raw['qas'][:2]],old['priors']['weights']);qs=[]
        for q in raw['qas'][2:4]:qs.append({'id':q['question_id'],'question':q['question'],'kind':contract(q['question'])[0]})
        papers.append({'paper_id':pid,'title':raw['title'],'shared_question_texts':[q['question'] for q in raw['qas'][:2]],'questions':qs,'paragraphs':ps,'units':c['units'],'full':c['material']['full'],'elements':c['material']['elements'],'elements_pointers':c['pointers']['elements'],'elements_stats':c['stats']['elements']})
        if len(papers)==6:break
    assert len(papers)==6
    files=[SOURCE,PROTOCOL,ROOT/'run_qasper_semantic_slots.py',ROOT/'logs/qasper_paragraph_elements/frozen.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_paragraph_elements.py','build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','run_qasper_contract_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py','run_stageC.py']]
    fr={'papers':papers,'build_instruction':BUILD_INSTRUCTION,'prep_sec':time.perf_counter()-start,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}}
    fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'papers':6,'questions':12,'prep_sec':fr['prep_sec'],'source_chars':[sum(len(p['text']) for p in t['paragraphs']) for t in papers]})

def verify(content,t):
    def reject(reason):return {'accepted':False,'reason':reason,'material':t['full'],'retention':1.0,'facts':[],'pointers':[]}
    text=re.sub(r'\s*```$','',re.sub(r'^```(?:json)?\s*','',content.strip()))
    try:o=json.loads(text)
    except Exception:return reject('json_parse')
    ps=t['paragraphs'];units=t['units'];bypara=[[s for s in units if s['pid']==i] for i in range(len(ps))]
    if not isinstance(o,dict) or not isinstance(o.get('keep'),list) or len(o['keep'])!=len(ps) or not isinstance(o.get('facts'),list) or not 4<=len(o['facts'])<=12:return reject('schema_or_paragraph_count')
    keys=set()
    for i,ids in enumerate(o['keep']):
        if not isinstance(ids,list) or not ids or any(type(j)is not int or not 0<=j<len(bypara[i]) for j in ids):return reject('sentence_id')
        keys.update((i,j) for j in ids)
    facts=[]
    for r in o['facts']:
        if not isinstance(r,dict) or type(r.get('p'))is not int or type(r.get('s'))is not int or (r['p'],r['s']) not in keys or r.get('relation') not in RELATIONS:return reject('fact_anchor_or_relation')
        sent=bypara[r['p']][r['s']]['text']
        for k in ('subject','value','conditions'):
            if not isinstance(r.get(k),str) or (k!='conditions' and not r[k]) or (r[k] and r[k] not in sent):return reject('fact_substring')
        facts.append(r)
    total=sum(s['length'] for s in units);rawchars=sum(s['length'] for s in units if (s['pid'],s['sid']) in keys);changed=True
    while changed:
        previous=len(keys)
        for i,j in list(keys):
            s=bypara[i][j]
            if re.match(r'^(?:It|They|This|These|Those|Such|However|Therefore|Consequently|In contrast)\b',s['text']):
                if j:keys.add((i,j-1))
                elif i:keys.add((i-1,len(bypara[i-1])-1))
            if s['text'].rstrip().endswith(':') and j+1<len(bypara[i]):keys.add((i,j+1))
        changed=len(keys)!=previous
    chars=sum(s['length'] for s in units if (s['pid'],s['sid']) in keys)
    if chars/total>.85:
        v=reject('closure_over_budget');v['proposed_retention_before_closure']=rawchars/total;v['proposed_retention_after_closure']=chars/total;return v
    chunks=[];pointers=[]
    for i,ss in enumerate(bypara):
        parts=[];previous=None
        for s in ss:
            j=s['sid']
            if (i,j) not in keys:continue
            if (previous is None and j>0) or (previous is not None and j>previous+1):parts.append('[Sentences omitted]')
            parts.append(f'[S{j}] '+s['text']);previous=j;body=ps[i]['text'];a=s['start'];b=a+s['length']
            rec={**s,'paragraph_sha256':sha(body.encode()),'byte_start':len(body[:a].encode()),'byte_length':len(body[a:b].encode())};assert body[a:b]==s['text'];assert body.encode()[rec['byte_start']:rec['byte_start']+rec['byte_length']].decode()==s['text'];pointers.append(rec)
        if previous<len(ss)-1:parts.append('[Sentences omitted]')
        chunks.append(f"[P{i}] Section: {ps[i]['heading']}\n"+'\n'.join(parts))
    return {'accepted':True,'reason':'mechanical_only_not_entailment','material':'\n\n'.join(chunks),'retention':chars/total,'retention_before_closure':rawchars/total,'facts':facts,'pointers':pointers}

def tokens(r):return (r.get('prompt_tokens') or 0)+(r.get('generation_tokens') or 0)

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in fr['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    src=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'papers':[]};assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for pi,t in enumerate(fr['papers']):
        p=next((p for p in d['papers'] if p['paper_id']==t['paper_id']),None)
        if p is None:p={'paper_id':t['paper_id'],'rows':[]};d['papers'].append(p)
        if 'build' not in p:
            prompt=fr['build_instruction']+'\nTitle: '+t['title']+'\nOriginal paper:\n'+t['full']+'\nExample questions (not exhaustive):\n'+'\n'.join(t['shared_question_texts'])
            p['build']=call(prompt,{'answer':'','answer_aliases':[]},think=False,cap=4096);p['selection']=verify(p['build'].get('content',''),t);path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(pi+1,'build',p['selection']['accepted'],p['selection']['reason'],flush=True)
        for qi,q in enumerate(t['questions']):
            row=next((r for r in p['rows'] if r['id']==q['id']),None)
            if row is None:row={'id':q['id'],'arms':{}};p['rows'].append(row)
            gold=references(next(v for v in src[t['paper_id']]['qas'] if v['question_id']==q['id']));order=list(ARMS);off=(pi*2+qi)%3;order=order[off:]+order[:off]
            for arm in order:
                if arm in row['arms']:continue
                material=p['selection']['material'] if arm=='semantic' else t[arm]
                prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(q['kind'])+'\nEvidence:\n'+material+'\n\nQuestion: '+query_text(t['title'],q['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
                r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold);row['arms'][arm]=r;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(pi+1,qi+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));groups=[p['rows'] for p in d['papers']];rows=[r for g in groups for r in g];assert len(groups)==6 and len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows)
    build=sum(tokens(p['build']) for p in d['papers']);out={'papers':6,'questions':12,'build_tokens':build,'build_wall_sec':sum(p['build']['wall_sec'] for p in d['papers']),'build_errors':sum('error' in p['build'] for p in d['papers']),'accepted':sum(p['selection']['accepted'] for p in d['papers']),'rejections':dict(Counter(p['selection']['reason'] for p in d['papers'] if not p['selection']['accepted'])),'prep_sec':fr['prep_sec'],'arms':{},'paired':{}}
    for a in ARMS:
        rs=[r['arms'][a] for r in rows];out['arms'][a]={'f1':sum(r['official_f1'] for r in rs)/12,'read_tokens':sum(tokens(r) for r in rs),'read_wall_sec':sum(r['wall_sec'] for r in rs),'errors':sum('error' in r for r in rs)}
    out['arms']['semantic']['pipeline_tokens']=out['arms']['semantic']['read_tokens']+build
    for a,b in [('semantic','full'),('elements','full'),('semantic','elements')]:
        rng=random.Random(20261002);bs=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms'][a]['official_f1']-r['arms'][b]['official_f1'] for r in sample)/12)
        bs.sort();out['paired'][a+'_minus_'+b]={'diff':out['arms'][a]['f1']-out['arms'][b]['f1'],'ci95':[bs[250],bs[9750]]}
    out['read_only_reduction']=1-out['arms']['semantic']['read_tokens']/out['arms']['full']['read_tokens'];out['pipeline_reduction']=1-out['arms']['semantic']['pipeline_tokens']/out['arms']['full']['read_tokens'];out['gate_pass']=out['paired']['semantic_minus_full']['ci95'][0]>=-.02 and out['pipeline_reduction']>=.1 and out['build_errors']==0 and not any(r['errors'] for r in out['arms'].values())
    out['weighted_semantic_retention']=sum(p['selection']['retention']*sum(s['length'] for s in t['units']) for t,p in zip(fr['papers'],d['papers']))/sum(s['length'] for t in fr['papers'] for s in t['units'])
    saving=out['arms']['full']['read_tokens']-out['arms']['semantic']['read_tokens'];out['hypothetical_questions_per_paper_break_even']=2*build/saving if saving>0 else None;out['break_even_is_accounting_not_new_question_quality']=True
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
