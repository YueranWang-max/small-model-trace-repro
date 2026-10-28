"""Actual causal attention ranking of exact original sentence units."""
import json,re,time,sys,math,random
from pathlib import Path
from collections import Counter
from run_qasper_paragraph_elements import ROOT,SOURCE,sha,features
from run_stageC import BM25
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from rescore_qasper_demand_transfer import references,f1,normalize
from build_qasper_relation_ledger import paragraphs
from run_qasper_paragraph_elements import spans

OUT=ROOT/'logs/qasper_attention_reader';PROTOCOL=ROOT/'protocol_qasper_attention_reader_20261002.md'
ARMS=('full','elements','shared_attention','question_attention')
SCHEMA_QUERY='What are the objects, definitions and components? What are the inputs, methods and outputs? What are the datasets, sources and settings? What are the metrics, results and comparisons? What evidence and mechanisms support the claims? What limitations and conditions apply?'

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();old=json.loads((ROOT/'logs/qasper_bounded_core/frozen.json').read_text(encoding='utf-8'));pri=json.loads((ROOT/'logs/qasper_paragraph_elements/frozen.json').read_text(encoding='utf-8'));probe=json.loads((ROOT/'logs/qasper_attention_probe/frozen.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));cal=[]
    for pid in sorted(split['fit'],key=lambda p:sha(p.encode())):
        ss=[{'pid':i,'sid':j,**s} for i,p in enumerate(paragraphs(src[pid])) for j,s in enumerate(spans(p['text']))];index=BM25([s['text'] for s in ss])
        for q in src[pid]['qas']:
            values=[v.strip() for a in q['answers'] for v in a['answer']['extractive_spans'] if 3<=len(v.strip())<=100]
            matches=[(i,v) for v in values for i,s in enumerate(ss) if len(s['text'])<=700 and v in s['text']]
            if not matches:continue
            i,v=matches[0];j=next((j for j in index.search(q['question'],len(ss)) if j!=i and len(ss[j]['text'])<=700 and v not in ss[j]['text']),None)
            if j is None:continue
            picked=sorted((i,j));body='\n'.join(ss[x]['text'] for x in picked);start=body.find(v);assert start>=0
            cal.append({'paper_id':pid,'question_id':q['question_id'],'question':q['question'],'body':body,'target':v,'target_start':start,'source_units':[ss[x] for x in picked]})
            if len(cal)==16:break
        if len(cal)==16:break
    assert len(cal)==16
    papers=old['papers']
    for p in papers:p['shared_query']='\n'.join(p['shared_question_texts'])+'\n'+SCHEMA_QUERY
    files=[SOURCE,PROTOCOL,ROOT/'run_qasper_attention_reader.py',ROOT/'logs/qasper_bounded_core/frozen.json',ROOT/'logs/qasper_paragraph_elements/frozen.json',ROOT/'logs/qasper_attention_probe/frozen.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_paragraph_elements.py','build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','rescore_qasper_demand_transfer.py','run_stageC.py','repro_lib.py']]
    fr={'papers':papers,'calibration':cal,'weights':pri['priors']['weights'],'model_path':probe['model_path'],'checkpoint_hashes':probe['checkpoint_hashes'],'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}}
    fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'papers':6,'questions':12,'calibration_examples':16},flush=True)

def score():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in fr['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    model_path=Path(fr['model_path'])
    for name,h in fr['checkpoint_hashes'].items():assert sha((model_path/name).read_bytes())==h
    path=OUT/'scores.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'papers':[]};assert d['frozen_sha256']==sha(fp.read_bytes())
    start=time.perf_counter();import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,attn_implementation='eager',dtype=torch.float32).to('cpu').eval();load=time.perf_counter()-start
    d.setdefault('load_sessions',[]).append(load);d['environment']={'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','dtype':'float32'}
    def forward(body,query):
        prefix='Paper excerpt:\n';middle='\nQuestion: ';a=len(prefix);b=a+len(body);qa=b+len(middle);qb=qa+len(query);prompt=prefix+body+middle+query+'\nAnswer:';enc=tok(prompt,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False);offsets=enc.pop('offset_mapping')[0].tolist();n=enc['input_ids'].shape[1]
        if n>512:return {'error':'over512_no_truncation','input_tokens':n}
        srcids=[i for i,(lo,hi) in enumerate(offsets) if hi>a and lo<b];qids=[i for i,(lo,hi) in enumerate(offsets) if hi>qa and lo<qb];assert srcids and qids;start=time.perf_counter()
        with torch.inference_mode():result=model(**enc,output_attentions=True,use_cache=False)
        at=result.attentions;assert at and all(x is not None for x in at);arr=torch.stack([x[0,:,qids,:].mean(dim=1)[:,srcids] for x in at]);assert torch.isfinite(arr).all();arr=arr/(arr.sum(dim=-1,keepdim=True).clamp_min(1e-12));values=arr.cpu().numpy();elapsed=time.perf_counter()-start;del result,at,arr
        return {'weights':values,'offsets':[(offsets[i][0]-a,offsets[i][1]-a) for i in srcids],'input_tokens':n,'wall_sec':elapsed,'prompt_sha256':sha(prompt.encode())}
    if 'heads' not in d:
        totals=None;valid=[];skips=[];runtime=0;tokens=0
        for c in fr['calibration']:
            r=forward(c['body'],c['question'])
            if 'error' in r:skips.append({'id':c['question_id'],'error':r['error']});continue
            mask=[i for i,(a,b) in enumerate(r['offsets']) if b>c['target_start'] and a<c['target_start']+len(c['target'])];assert mask
            scores=r['weights'][:,:,mask].sum(axis=-1)-len(mask)/len(r['offsets']);totals=scores if totals is None else totals+scores;runtime+=r['wall_sec'];tokens+=r['input_tokens'];valid.append({'id':c['question_id'],'tokens':r['input_tokens'],'wall_sec':r['wall_sec'],'prompt_sha256':r['prompt_sha256']})
        assert len(valid)>=4
        mean=totals/len(valid);d['heads']=[list(x) for x in sorted(((l,h) for l in range(mean.shape[0]) for h in range(mean.shape[1])),key=lambda x:(-float(mean[x]),x))[:8]];d['calibration_result']={'valid':valid,'skipped':skips,'wall_sec':runtime,'input_tokens':tokens,'head_enrichment':mean.tolist(),'weak_labels_not_accuracy':True};path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print('calibrated',len(valid),d['heads'],flush=True)
    heads=d['heads']
    for pi,t in enumerate(fr['papers']):
        record=next((p for p in d['papers'] if p['paper_id']==t['paper_id']),None)
        if record is None:record={'paper_id':t['paper_id'],'paragraphs':[]};d['papers'].append(record)
        for pid,p in enumerate(t['paragraphs']):
            if any(r['pid']==pid for r in record['paragraphs']):continue
            units=[s for s in t['units'] if s['pid']==pid]
            if len(units)<=2:record['paragraphs'].append({'pid':pid,'automatic_full':True,'queries':{},'mandatory_sids':[s['sid'] for s in units]});continue
            chunks=[];pending=[];long=[]
            for s in units:
                if len(tok('Paper excerpt:\n'+s['text'],add_special_tokens=False)['input_ids'])>360:long.append(s['sid']);continue
                joined='\n'.join(x['text'] for x in pending+[s])
                if pending and len(tok('Paper excerpt:\n'+joined,add_special_tokens=False)['input_ids'])>360:chunks.append(pending);pending=[]
                pending.append(s)
            if pending:chunks.append(pending)
            queries={'shared':t['shared_query'],**{q['id']:q['question'] for q in t['questions']}};item={'pid':pid,'automatic_full':False,'mandatory_sids':long,'queries':{}}
            for qid,query in queries.items():
                qr={'sentence_scores':{},'windows':[],'wall_sec':0,'input_tokens':0,'errors':[]}
                for chunk in chunks:
                    body='\n'.join(s['text'] for s in chunk);r=forward(body,query)
                    if 'error' in r:qr['errors'].append({'reason':r['error'],'sids':[s['sid'] for s in chunk]});item['mandatory_sids'].extend(s['sid'] for s in chunk);continue
                    vec=sum(r['weights'][l,h,:] for l,h in heads)/len(heads);offset=0
                    for s in chunk:
                        ids=[i for i,(a,b) in enumerate(r['offsets']) if b>offset and a<offset+len(s['text'])];assert ids;qr['sentence_scores'][str(s['sid'])]=float(vec[ids].mean());offset+=len(s['text'])+1
                    qr['windows'].append({'sids':[s['sid'] for s in chunk],'body_sha256':sha(body.encode()),'offsets':r['offsets'],'normalized_attention':vec.tolist(),'input_tokens':r['input_tokens'],'wall_sec':r['wall_sec'],'prompt_sha256':r['prompt_sha256']});qr['wall_sec']+=r['wall_sec'];qr['input_tokens']+=r['input_tokens']
                item['queries'][qid]=qr
            item['mandatory_sids']=sorted(set(item['mandatory_sids']));record['paragraphs'].append(item);path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print('scored',pi+1,pid,flush=True)
        path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
    print('scores complete',flush=True)

def render(t,keys):
    bypara=[[s for s in t['units'] if s['pid']==i] for i in range(len(t['paragraphs']))];before=len(keys);changed=True
    while changed:
        old=len(keys)
        for i,j in list(keys):
            s=bypara[i][j]
            if re.match(r'^(?:It|They|This|These|Those|Such|However|Therefore|Consequently|In contrast)\b',s['text']):
                if j:keys.add((i,j-1))
                elif i and t['paragraphs'][i-1]['heading']==t['paragraphs'][i]['heading']:keys.add((i-1,len(bypara[i-1])-1))
            if s['text'].rstrip().endswith(':') and j+1<len(bypara[i]):keys.add((i,j+1))
        changed=old!=len(keys)
    chunks=[];pointers=[]
    for i,ss in enumerate(bypara):
        parts=[];previous=None
        for s in ss:
            j=s['sid']
            if (i,j) not in keys:continue
            if (previous is None and j>0) or (previous is not None and j>previous+1):parts.append('[Sentences omitted]')
            parts.append(f'[S{j}] '+s['text']);previous=j;body=t['paragraphs'][i]['text'];a=s['start'];b=a+s['length'];p={**s,'paragraph_sha256':sha(body.encode()),'byte_start':len(body[:a].encode()),'byte_length':len(body[a:b].encode())};assert body[a:b]==s['text'];assert body.encode()[p['byte_start']:p['byte_start']+p['byte_length']].decode()==s['text'];pointers.append(p)
        assert previous is not None
        if previous<len(ss)-1:parts.append('[Sentences omitted]')
        chunks.append(f"[P{i}] Section: {t['paragraphs'][i]['heading']}\n"+'\n'.join(parts))
    return {'material':'\n\n'.join(chunks),'pointers':pointers,'sentences_before_closure':before,'sentences_after_closure':len(keys),'retained_chars':sum(s['length'] for s in pointers),'source_chars':sum(s['length'] for s in t['units'])}

def select():
    fp=OUT/'frozen.json';sp=OUT/'scores.json';fr=json.loads(fp.read_text(encoding='utf-8'));d=json.loads(sp.read_text(encoding='utf-8'));assert d['frozen_sha256']==sha(fp.read_bytes());tasks=[]
    for t in fr['papers']:
        scores=next(p for p in d['papers'] if p['paper_id']==t['paper_id']);assert len(scores['paragraphs'])==len(t['paragraphs'])
        for q in t['questions']:
            arms={};costs={}
            for arm in ARMS:
                keys=set();wall=processed=errors=0
                for i,p in enumerate(t['paragraphs']):
                    ss=[s for s in t['units'] if s['pid']==i];sr=next(r for r in scores['paragraphs'] if r['pid']==i);quota=max(1,math.ceil(.6*len(ss)));mandatory=set(sr['mandatory_sids']);chosen=set(mandatory)
                    if arm=='full':chosen={s['sid'] for s in ss}
                    elif arm=='elements':chosen.update(s['sid'] for s in sorted(ss,key=lambda s:(-sum(fr['weights'][k] for k in s['features']),s['sid']))[:quota])
                    elif sr['automatic_full']:chosen.update(s['sid'] for s in ss)
                    else:
                        qr=sr['queries']['shared' if arm=='shared_attention' else q['id']];chosen.update(s['sid'] for s in sorted(ss,key=lambda s:(-qr['sentence_scores'].get(str(s['sid']),-1),s['sid']))[:quota]);wall+=qr['wall_sec'];processed+=qr['input_tokens'];errors+=len(qr['errors'])
                    keys.update((i,j) for j in chosen)
                arms[arm]=render(t,keys);costs[arm]={'score_wall_sec':wall,'processed_tokens':processed,'score_errors':errors}
            tasks.append({'paper_id':t['paper_id'],'title':t['title'],**q,'arms':arms,'score_costs':costs})
    out={'frozen_sha256':sha(fp.read_bytes()),'scores_sha256':sha(sp.read_bytes()),'tasks':tasks};path=OUT/'selected.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print('selected',len(tasks),flush=True)

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in fr['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    selected=json.loads((OUT/'selected.json').read_text(encoding='utf-8'));assert selected['frozen_sha256']==sha(fp.read_bytes()) and selected['scores_sha256']==sha((OUT/'scores.json').read_bytes());src=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'reader.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'selected_sha256':sha((OUT/'selected.json').read_bytes()),'rows':[]};assert d['selected_sha256']==sha((OUT/'selected.json').read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(selected['tasks']):
        row=next((r for r in d['rows'] if r['id']==t['id']),None)
        if row is None:row={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(row)
        gold=references(next(q for q in src[t['paper_id']]['qas'] if q['question_id']==t['id']));order=list(ARMS);off=i%4;order=order[off:]+order[:off]
        for arm in order:
            if arm in row['arms']:continue
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\nEvidence:\n'+t['arms'][arm]['material']+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold);row['arms'][arm]=r;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));sel=json.loads((OUT/'selected.json').read_text(encoding='utf-8'));scores=json.loads((OUT/'scores.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows);groups=[[r for r in rows if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rows})];out={'papers':6,'questions':12,'calibration_wall_sec':scores['calibration_result']['wall_sec'],'calibration_tokens':scores['calibration_result']['input_tokens'],'load_wall_sec':sum(scores['load_sessions']),'heads':scores['heads'],'arms':{},'paired':{}}
    for a in ARMS:
        rs=[r['arms'][a] for r in rows];costs=[t['score_costs'][a] for t in sel['tasks']];factor=.5 if a=='shared_attention' else 1;online=sum(c['score_wall_sec'] for c in costs)*factor;processed=sum(c['processed_tokens'] for c in costs)*factor;load=out['load_wall_sec'] if 'attention' in a else 0
        out['arms'][a]={'f1':sum(r['official_f1'] for r in rs)/12,'reader_tokens':sum((r.get('prompt_tokens') or 0)+(r.get('generation_tokens') or 0) for r in rs),'read_wall_sec':sum(r['wall_sec'] for r in rs),'score_wall_sec':online,'scorer_processed_tokens':processed,'pipeline_wall_sec':load+online+sum(r['wall_sec'] for r in rs),'reader_errors':sum('error' in r for r in rs),'score_errors':sum(c['score_errors'] for c in costs)*factor,'retention':sum(t['arms'][a]['retained_chars'] for t in sel['tasks'])/sum(t['arms'][a]['source_chars'] for t in sel['tasks'])}
    for a,b in [('question_attention','elements'),('shared_attention','elements'),('question_attention','full'),('shared_attention','full'),('elements','full')]:
        rng=random.Random(20261002);bs=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms'][a]['official_f1']-r['arms'][b]['official_f1'] for r in sample)/12)
        bs.sort();out['paired'][a+'_minus_'+b]={'diff':out['arms'][a]['f1']-out['arms'][b]['f1'],'ci95':[bs[250],bs[9750]]}
    for a in ('question_attention','shared_attention'):
        r=out['arms'][a];r['reader_token_reduction']=1-r['reader_tokens']/out['arms']['full']['reader_tokens'];r['attention_benefit_gate']=out['paired'][a+'_minus_elements']['ci95'][0]>0;r['deployment_gate']=out['paired'][a+'_minus_full']['ci95'][0]>=-.02 and r['reader_token_reduction']>=.1 and r['pipeline_wall_sec']<=out['arms']['full']['read_wall_sec'] and r['reader_errors']==0 and r['score_errors']==0
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'score':score,'select':select,'run':run,'analyze':analyze}[sys.argv[1]]()
