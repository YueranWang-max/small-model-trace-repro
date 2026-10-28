"""Substantial original source plus exact extractive guide; 2x2 pilot."""
import json,re,time,sys,random,math
from pathlib import Path
from run_qasper_fingerprint import prepare
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_layer_mentions import extract
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from run_qasper_contract_replication import contract
from rescore_qasper_demand_transfer import references,f1,normalize
from build_qasper_relation_ledger import paragraphs

OUT=ROOT/'logs/qasper_retained_source';SOURCE=ROOT/'data/qasper_official/qasper-test-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_retained_source_20261002.md';ARMS=('full','core80','guide_full','guide_core80')

def bind(r):
    context=r['context'];prefix=context[:r['char_start']-r['context_char_start']]
    if r['status']!='explicit_local_mention' or re.search(r'\b(first|each|both|direction|directions)\b',context,re.I):return None
    m=re.search(r'\b(?:our |the )?(encoder|decoder)\s+(?:has|contains|consists of|is composed of|is also consist of)\s*$',prefix,re.I)
    return {'subject':m.group(1).lower(),'relation':'local_layer_count','count':r['count'],'layer_type':r['layer_type'],'source_subject_phrase':m.group()} if m else None

def material(ps,ids):
    chunks=[];previous=None
    for i in sorted(ids):
        if previous is not None and i>previous+1:chunks.append('[Original paragraphs omitted here]')
        chunks.append(f"[P{i}] Section: {ps[i]['heading']}\n"+ps[i]['text']);previous=i
    return '\n\n'.join(chunks)

def construct(p,q):
    ps=p['paragraphs'];n=len(ps);records=[];mandatory=set()
    for i,para in enumerate(ps):
        for r in extract(para['text']):
            r={**r,'paragraph_id':i,'heading':para['heading'],'binding':bind(r)};records.append(r);mandatory.add(i)
            if i and ps[i-1]['heading']==para['heading']:mandatory.add(i-1)
    total=sum(len(x['text']) for x in ps);selected=set(mandatory);chars=sum(len(ps[i]['text']) for i in selected)
    for i in q['rank']:
        if chars>=.8*total:break
        if i not in selected:selected.add(i);chars+=len(ps[i]['text'])
    anchors=[]
    for i in q['rank'][:4]:
        text=ps[i]['text'];end=next((m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])',text)),len(text));sentence=text[:end]
        if len(sentence)<=500:anchors.append({'paragraph_id':i,'heading':ps[i]['heading'],'quote':sentence,'char_start':0,'char_length':end,'byte_start':0,'byte_length':len(sentence.encode()),'paragraph_sha256':sha(text.encode())})
    rankpos={pid:j for j,pid in enumerate(q['rank'])};guide_records=sorted(records,key=lambda r:(rankpos[r['paragraph_id']],r['char_start']))[:6]
    lines=['Extractive reading guide. Check original evidence before answering. Local mentions are not total model depth.']
    for a in anchors:lines.append(f"[P{a['paragraph_id']}] {a['heading']}: "+a['quote'])
    for r in guide_records:
        b=r['binding'];label=(b['subject']+' local '+r['layer_type']+' count='+str(r['count'])) if b else 'Unbound local layer mention; scope unresolved'
        lines.append(f"[P{r['paragraph_id']}] {r['heading']} | {label} | exact context: "+r['context'])
    guide='\n'.join(lines);full=material(ps,range(n));core=material(ps,selected)
    return {'material':{'full':full,'core80':core,'guide_full':guide+'\n\nOriginal source:\n'+full,'guide_core80':guide+'\n\nRetained original source:\n'+core},'guide':guide,'anchors':anchors,'guide_layer_records':guide_records,'core_ids':sorted(selected),'full_ids':list(range(n)),'source_chars':total,'retained_chars':chars,'retention':chars/max(1,total)}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    calibration=[]
    for text,want in [('Our encoder has two LSTM layers.','encoder'),('The decoder contains 3 GRU layers.','decoder'),('The first two LSTM layers compress input.',None),('Each encoder has two LSTM layers.',None),('Our encoder has two LSTM layers for both directions.',None),('Our classifier has two LSTM layers.',None)]:
        rs=extract(text);actual=bind(rs[0]) if rs else None;assert (actual['subject'] if actual else None)==want
        calibration.append({'text':text,'expected_subject':want,'binding':actual})
    source=json.loads(SOURCE.read_text(encoding='utf-8'));excluded=set()
    for name in ('qasper_forward_bridge','qasper_target_scope'):
        excluded.update(t['paper_id'] for t in json.loads((ROOT/'logs'/name/'frozen.json').read_text(encoding='utf-8'))['tasks'])
    tasks=[];paperids=[]
    for pid in sorted(source,key=lambda pid:sha(pid.encode()))[70:]:
        if pid in excluded or len(source[pid]['qas'])<2:continue
        ps=paragraphs(source[pid]);length=sum(len(p['text']) for p in ps)
        if length>48000:continue
        p=prepare(pid,source[pid]);paperids.append(pid)
        for q in p['questions'][:2]:
            c=construct(p,q);assert c['retention']>=.8
            tasks.append({'paper_id':pid,'id':q['id'],'title':source[pid]['title'],'question':q['text'],'kind':contract(q['text'])[0],**c})
        if len(paperids)==12:break
    assert len(tasks)==24 and len(paperids)==12
    train=json.loads((ROOT/'data/qasper_official/qasper-train-v0.3.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));records=[]
    for pid in split['heldout']:
        for i,p in enumerate(paragraphs(train[pid])):
            for r in extract(p['text']):records.append({**r,'paper_id':pid,'paragraph_id':i,'heading':p['heading'],'binding':bind(r)})
    (OUT/'heldout_binding_inventory.json').write_text(json.dumps({'records':records,'papers':len(split['heldout']),'mentions':len(records),'bound_mentions':sum(bool(r['binding']) for r in records),'not_semantic_gold':True},indent=2,ensure_ascii=False),encoding='utf-8')
    files=[SOURCE,Path(__file__),PROTOCOL,ROOT/'data/qasper_official/qasper-train-v0.3.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_fingerprint.py','run_stageC.py','run_qasper_layer_mentions.py','build_qasper_relation_ledger.py','run_qasper_format_control.py','run_qasper_contract_replication.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]+[ROOT/'logs'/n/'frozen.json' for n in ['qasper_forward_bridge','qasper_target_scope']]
    f={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':tasks,'paper_ids':paperids,'excluded_papers':sorted(excluded),'prep_sec':time.perf_counter()-start,'binding_calibration':calibration}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(f,indent=2,ensure_ascii=False),encoding='utf-8');print({'questions':len(tasks),'papers':len(paperids),'heldout_mentions':len(records),'bound_mentions':sum(bool(r['binding']) for r in records),'prep_sec':f['prep_sec']})

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in f['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(f['tasks']):
        row=next((r for r in d['rows'] if r['id']==t['id']),None)
        if row is None:row={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(row)
        gold=references(next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']));order=list(ARMS);order=order[i%4:]+order[:i%4]
        for arm in order:
            if arm in row['arms']:continue
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\nEvidence:\n'+t['material'][arm]+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold)
            row['arms'][arm]=r;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==24 and all(set(r['arms'])==set(ARMS) for r in rows)
    out={'questions':len(rows),'papers':12,'prep_sec':f['prep_sec'],'warmup_sec':d['warmup']['wall_sec'],'retention_min':min(t['retention'] for t in f['tasks']),'retention_weighted':sum(t['retained_chars'] for t in f['tasks'])/sum(t['source_chars'] for t in f['tasks']),'arms':{},'paired':{}}
    for arm in ARMS:
        xs=[r['arms'][arm] for r in rows];out['arms'][arm]={'f1':sum(x['official_f1'] for x in xs)/len(xs),'tokens':sum((x.get('prompt_tokens') or 0)+(x.get('generation_tokens') or 0) for x in xs),'wall_sec':sum(x['wall_sec'] for x in xs),'errors':sum('error' in x for x in xs),'missing':sum(not x['answer'] for x in xs),'max_prompt_tokens':max(x.get('prompt_tokens') or 0 for x in xs)}
    groups=[[r for r in rows if r['paper_id']==pid] for pid in f['paper_ids']]
    for arm,ref in [('guide_core80','full'),('core80','full'),('guide_full','full'),('guide_core80','core80')]:
        rng=random.Random(20261002);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(r['arms'][arm]['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
        boot.sort();out['paired'][arm+'_minus_'+ref]={'diff':out['arms'][arm]['f1']-out['arms'][ref]['f1'],'ci95':[boot[250],boot[9750]],'wins':sum(r['arms'][arm]['official_f1']>r['arms'][ref]['official_f1'] for r in rows),'losses':sum(r['arms'][arm]['official_f1']<r['arms'][ref]['official_f1'] for r in rows)}
    out['guide_core_token_reduction']=1-out['arms']['guide_core80']['tokens']/out['arms']['full']['tokens'];out['gate_pass']=out['paired']['guide_core80_minus_full']['ci95'][0]>=-.02 and out['guide_core_token_reduction']>=.1 and not any(x['errors'] for x in out['arms'].values())
    cache={t['paper_id']+'::'+t['id']:next(r for r in rows if r['id']==t['id'])['arms']['full']['answer'] for t in f['tasks']};b=json.dumps(cache,ensure_ascii=False,separators=(',',':')).encode();(OUT/'full_generated_answer_cache.json').write_bytes(b);out['generated_cache_bytes']=len(b)
    out['repeat_scenarios']={str(n):{'full_reread_tokens':n*out['arms']['full']['tokens'],'guide_core_reread_tokens':n*out['arms']['guide_core80']['tokens'],'full_generated_answer_cache_tokens':out['arms']['full']['tokens'],'cached_f1_equals_full':True,'no_new_calls_simulated':True} for n in (1,5,20)}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
