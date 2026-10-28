"""Source-only relation guards, novel reader cohort, native attention backbone."""
import json,re,time,sys,math,random
from pathlib import Path
import run_qasper_attention_reader as prior
from run_qasper_paragraph_elements import ROOT,SOURCE,sha,spans,features
from build_qasper_relation_ledger import paragraphs
from run_qasper_contract_replication import contract

OUT=ROOT/'logs/qasper_attention_guard';PROTOCOL=ROOT/'protocol_qasper_attention_guard_20261003.md'
ARMS=('full','elements','attention','guard','wide')
RULES={
 'baseline':r'\bcompar\w*\b|\bbaselines?\b|\bbenchmark\w*\b|\bagainst\b',
 'data':r'\bdatasets?\b|\bcorpus\b|\bcorpora\b|\blanguages?\b|\bcollect\w*\b|\bannotat\w*\b|\bsourc\w*\b',
 'quantity':r'\b(hours?|words?|tokens?|tweets?|reviews?|sentences?|documents?|examples?|samples?|instances?|layers?|dimensions?|parameters?|users?|subjects?|percent\w*|epochs?|iterations?)\b|%',
 'metric':r'\b(accuracy|precision|recall|f1|f-score|bleu|ribes|rouge|metric\w*|measure\w*|evaluat\w*)\b',
 'method':r'\b(?:we|our)\b.{0,100}\b(?:us\w*|train\w*|propos\w*|implement\w*|model\w*|method\w*)\b|\b(?:defined as|consists of|composed of|comprises|contains|refer to)\b',
 'negation':r'\b(not|no|never|without|cannot|only|except|unless)\b',
}

def family(q):
    q=q.lower()
    if re.search(r'baselines?|compar\w*|against',q):return 'baseline'
    if re.search(r'how (many|much)|\b(size|number|amount|percentage|dimension\w*|layers?)\b',q):return 'quantity'
    if re.search(r'dataset|corpus|corpora|language|where|collect|source|annotat',q):return 'data'
    if re.search(r'metric|measure|accuracy|precision|recall|quality',q):return 'metric'
    if re.match(r'\s*(is|are|do|does|did|can|was|were|has|have)\b',q):return 'boolean'
    return 'method'

def guards(t,q):
    fam=family(q);out=[];terms={w.lower() for w in re.findall(r'\b[A-Za-z][A-Za-z0-9-]+\b',q)}-set('is are do does did can was were has have the a an they their paper authors model method use used using this that pretrained any'.split())
    for s in t['units']:
        text=s['text'];reason=None
        if fam in ('baseline','data','metric','method') and re.search(RULES[fam],text,re.I):reason=fam+'_relation_candidate'
        if fam=='quantity' and re.search(r'\d|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b',text,re.I) and re.search(RULES['quantity'],text,re.I):reason='number_and_unit_candidate'
        if fam=='boolean' and (re.search(RULES['negation'],text,re.I) or (any(re.search(r'\b'+re.escape(w)+r'\b',text,re.I) for w in terms) and re.search(RULES['method'],text,re.I))):reason='boolean_property_or_scope_candidate'
        if reason:out.append({'pid':s['pid'],'sid':s['sid'],'reason':reason,'original':text})
    return {'family':fam,'records':out,'not_semantic_binding':True}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();start=time.perf_counter();src=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_attention_reader/frozen.json').read_text(encoding='utf-8'));sc=json.loads((ROOT/'logs/qasper_attention_reader/scores.json').read_text(encoding='utf-8'));seen=set();logfiles=[]
    def walk(v):
        if isinstance(v,dict):
            if isinstance(v.get('paper_id'),str):seen.add(v['paper_id'])
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    for p in (ROOT/'logs').glob('*/frozen.json'):
        if p==fp:continue
        try:walk(json.loads(p.read_text(encoding='utf-8')));logfiles.append(p)
        except (ValueError,OSError):pass
    papers=[]
    for pid in sorted(split['heldout'],key=lambda p:sha(p.encode())):
        raw=src[pid];ps=paragraphs(raw)
        if pid in seen or len(raw['qas'])<2 or sum(len(p['text']) for p in ps)>18000 or len(ps)>35:continue
        units=[{**s,'pid':i,'sid':j,'features':features(s['text'])} for i,p in enumerate(ps) for j,s in enumerate(spans(p['text']))];qs=[{'id':q['question_id'],'question':q['question'],'kind':contract(q['question'])[0]} for q in raw['qas'][:2]];t={'paper_id':pid,'title':raw['title'],'paragraphs':ps,'units':units,'questions':qs};t['guard_maps']={q['id']:guards(t,q['question']) for q in qs};papers.append(t)
        if len(papers)==6:break
    assert len(papers)==6
    calibration=[]
    examples=[('What are the baselines?','We compare our model with Alpha and Beta.',True),('How many tweets were used?','The corpus contains 12,000 tweets.',True),('Which datasets are used?','We evaluate on three datasets: Alpha, Beta and Gamma.',True),('What metric is used?','We report macro F1.',True),('Does the model use embeddings?','Our model does not use embeddings.',True),('How is the method implemented?','Our method uses attention.',True),('What are the baselines?','We thank the reviewers.',False),('How many layers are used?','A convenient layer representation is useful.',False)]
    for q,text,want in examples:
        actual=guards({'units':[{'pid':0,'sid':0,'text':text}]},q);assert bool(actual['records'])==want;calibration.append({'q':q,'text':text,'expected_match':want,'result':actual})
    fitting=[]
    for c in old['calibration']:
        units=[{'pid':i,'sid':0,'text':u['text']} for i,u in enumerate(c['source_units'])];g=guards({'units':units},c['question']);fitting.append({'question_id':c['question_id'],'family':g['family'],'target_string_covered':any(c['target'] in r['original'] for r in g['records']),'matched':len(g['records'])})
    files=[SOURCE,PROTOCOL,ROOT/'run_qasper_attention_guard.py',ROOT/'protocol_attention_core_forward_20261003.md',ROOT/'benchmark_attention_core_forward.py',ROOT/'logs/qasper_attention_reader/frozen.json',ROOT/'logs/qasper_attention_reader/scores.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_attention_reader.py','run_qasper_paragraph_elements.py','build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','run_qasper_contract_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]
    fr={'papers':papers,'rules':RULES,'model_path':old['model_path'],'checkpoint_hashes':old['checkpoint_hashes'],'heads':sc['heads'],'weights':old['weights'],'excluded_paper_ids':sorted(seen),'exclusion_files':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in logfiles},'constructed_calibration':calibration,'fitting_coverage':fitting,'prep_sec':time.perf_counter()-start,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}};fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'papers':[p['paper_id'] for p in papers],'questions':12,'fitting_guard_string_hits':sum(r['target_string_covered'] for r in fitting),'fitting_examples':len(fitting)},flush=True)

def score():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for p,h in fr['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    benchmark=json.loads((ROOT/'logs/attention_core_forward/results.json').read_text(encoding='utf-8'));assert benchmark['equivalence_pass'];modelpath=Path(fr['model_path'])
    for p,h in fr['checkpoint_hashes'].items():assert sha((modelpath/p).read_bytes())==h
    path=OUT/'scores.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'papers':[]};assert d['frozen_sha256']==sha(fp.read_bytes());start=time.perf_counter();import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);lm=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,attn_implementation='eager',dtype=torch.float32).to('cpu').eval();model=lm.model;d.setdefault('load_sessions',[]).append(time.perf_counter()-start);d['environment']={'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','forward':'native_backbone_no_vocab_logits'};d['benchmark_sha256']=sha((ROOT/'logs/attention_core_forward/results.json').read_bytes())
    for pi,t in enumerate(fr['papers']):
        pr=next((p for p in d['papers'] if p['paper_id']==t['paper_id']),None)
        if pr is None:pr={'paper_id':t['paper_id'],'paragraphs':[]};d['papers'].append(pr)
        for i,p in enumerate(t['paragraphs']):
            if any(r['pid']==i for r in pr['paragraphs']):continue
            ss=[s for s in t['units'] if s['pid']==i]
            if len(ss)<=2:pr['paragraphs'].append({'pid':i,'automatic_full':True,'mandatory_sids':[s['sid'] for s in ss],'queries':{}});continue
            chunks=[];pending=[];long=[]
            for s in ss:
                if len(tok('Paper excerpt:\n'+s['text'],add_special_tokens=False)['input_ids'])>360:long.append(s['sid']);continue
                body='\n'.join(x['text'] for x in pending+[s])
                if pending and len(tok('Paper excerpt:\n'+body,add_special_tokens=False)['input_ids'])>360:chunks.append(pending);pending=[]
                pending.append(s)
            if pending:chunks.append(pending)
            r={'pid':i,'automatic_full':False,'mandatory_sids':long,'queries':{}}
            for q in t['questions']:
                start=time.perf_counter();qr={'sentence_scores':{},'windows':[],'errors':[],'input_tokens':0}
                for chunk in chunks:
                    prefix='Paper excerpt:\n';body='\n'.join(s['text'] for s in chunk);a=len(prefix);b=a+len(body);qa=b+len('\nQuestion: ');qb=qa+len(q['question']);prompt=prefix+body+'\nQuestion: '+q['question']+'\nAnswer:';enc=tok(prompt,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False);offsets=enc.pop('offset_mapping')[0].tolist();n=enc['input_ids'].shape[1]
                    if n>512:qr['errors'].append({'error':'over512_no_truncation','sids':[s['sid'] for s in chunk]});r['mandatory_sids'].extend(s['sid'] for s in chunk);continue
                    srcids=[j for j,(lo,hi) in enumerate(offsets) if hi>a and lo<b];qids=[j for j,(lo,hi) in enumerate(offsets) if hi>qa and lo<qb];assert srcids and qids
                    with torch.inference_mode():result=model(**enc,output_attentions=True,use_cache=False)
                    values=[]
                    for layer,head in fr['heads']:
                        v=result.attentions[layer][0,head,qids,:].mean(dim=0)[srcids];assert torch.isfinite(v).all();values.append(v/v.sum().clamp_min(1e-12))
                    vec=torch.stack(values).mean(dim=0).cpu().numpy();relative=[(offsets[j][0]-a,offsets[j][1]-a) for j in srcids];off=0
                    for s in chunk:
                        ids=[j for j,(lo,hi) in enumerate(relative) if hi>off and lo<off+len(s['text'])];assert ids;qr['sentence_scores'][str(s['sid'])]=float(vec[ids].mean());off+=len(s['text'])+1
                    qr['windows'].append({'sids':[s['sid'] for s in chunk],'input_tokens':n,'offsets':relative,'normalized_attention':vec.tolist(),'prompt_sha256':sha(prompt.encode())});qr['input_tokens']+=n;del result,values
                qr['wall_sec']=time.perf_counter()-start;r['queries'][q['id']]=qr
            r['mandatory_sids']=sorted(set(r['mandatory_sids']));pr['paragraphs'].append(r);path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print('scored',pi+1,i,flush=True)
    path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print('scores complete',flush=True)

def select():
    fp=OUT/'frozen.json';sp=OUT/'scores.json';fr=json.loads(fp.read_text(encoding='utf-8'));scores=json.loads(sp.read_text(encoding='utf-8'));tasks=[];start=time.perf_counter()
    for t in fr['papers']:
        pr=next(p for p in scores['papers'] if p['paper_id']==t['paper_id']);assert len(pr['paragraphs'])==len(t['paragraphs'])
        for q in t['questions']:
            g=t['guard_maps'][q['id']];guardkeys={(r['pid'],r['sid']) for r in g['records']};sets={a:set() for a in ARMS};wall=tokens=errors=0
            for i,p in enumerate(t['paragraphs']):
                ss=[s for s in t['units'] if s['pid']==i];sr=next(r for r in pr['paragraphs'] if r['pid']==i);mandatory=set(sr['mandatory_sids']);k=max(1,math.ceil(.6*len(ss)))
                if sr['automatic_full']:rank=list(range(len(ss)))
                else:
                    qr=sr['queries'][q['id']];rank=[s['sid'] for s in sorted(ss,key=lambda s:(-qr['sentence_scores'].get(str(s['sid']),-1),s['sid']))];wall+=qr['wall_sec'];tokens+=qr['input_tokens'];errors+=len(qr['errors'])
                att=set(rank[:k])|mandatory;protected=att|{j for pi,j in guardkeys if pi==i};wide=set(rank[:len(protected)])|mandatory;el={s['sid'] for s in sorted(ss,key=lambda s:(-sum(fr['weights'][x] for x in s['features']),s['sid']))[:k]}|mandatory
                for a,ids in [('full',range(len(ss))),('elements',el),('attention',att),('guard',protected),('wide',wide)]:sets[a].update((i,j) for j in ids)
            arms={a:prior.render(t,keys) for a,keys in sets.items()};costs={a:{'score_wall_sec':wall if a in ('attention','guard','wide') else 0,'processed_tokens':tokens if a in ('attention','guard','wide') else 0,'score_errors':errors if a in ('attention','guard','wide') else 0} for a in ARMS};tasks.append({'paper_id':t['paper_id'],'title':t['title'],**q,'guard_map':g,'arms':arms,'score_costs':costs})
    path=OUT/'selected.json';assert not path.exists();path.write_text(json.dumps({'frozen_sha256':sha(fp.read_bytes()),'scores_sha256':sha(sp.read_bytes()),'selection_wall_sec':time.perf_counter()-start,'tasks':tasks},indent=2,ensure_ascii=False),encoding='utf-8');print('selected',len(tasks),flush=True)

def run():prior.OUT=OUT;prior.ARMS=ARMS;prior.run()

def analyze():
    d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));sel=json.loads((OUT/'selected.json').read_text(encoding='utf-8'));sc=json.loads((OUT/'scores.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows);groups=[[r for r in rows if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rows})];out={'papers':6,'questions':12,'load_wall_sec':sum(sc['load_sessions']),'selection_wall_sec':sel['selection_wall_sec'],'arms':{},'paired':{}}
    for a in ARMS:
        rs=[r['arms'][a] for r in rows];cost=[t['score_costs'][a] for t in sel['tasks']];sw=sum(c['score_wall_sec'] for c in cost);read=sum(r['wall_sec'] for r in rs);load=out['load_wall_sec'] if a in ('attention','guard','wide') else 0;out['arms'][a]={'f1':sum(r['official_f1'] for r in rs)/12,'reader_tokens':sum((r.get('prompt_tokens') or 0)+(r.get('generation_tokens') or 0) for r in rs),'read_wall_sec':read,'score_wall_sec':sw,'scorer_processed_tokens':sum(c['processed_tokens'] for c in cost),'load_score_read_wall_lower_bound_sec':load+sw+read,'reader_errors':sum('error' in r for r in rs),'score_errors':sum(c['score_errors'] for c in cost),'retention':sum(t['arms'][a]['retained_chars'] for t in sel['tasks'])/sum(t['arms'][a]['source_chars'] for t in sel['tasks'])}
    for a,b in [('guard','attention'),('guard','wide'),('guard','full'),('guard','elements'),('attention','elements')]:
        rng=random.Random(20261003);bs=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms'][a]['official_f1']-r['arms'][b]['official_f1'] for r in sample)/12)
        bs.sort();out['paired'][a+'_minus_'+b]={'diff':out['arms'][a]['f1']-out['arms'][b]['f1'],'ci95':[bs[250],bs[9750]]}
    g=out['arms']['guard'];g['reader_token_reduction']=1-g['reader_tokens']/out['arms']['full']['reader_tokens'];out['guard_benefit_gate']=out['paired']['guard_minus_attention']['ci95'][0]>0 and out['paired']['guard_minus_wide']['ci95'][0]>0 and g['reader_tokens']<=1.05*out['arms']['wide']['reader_tokens'];out['deployment_gate']=out['paired']['guard_minus_full']['ci95'][0]>=-.02 and g['reader_token_reduction']>=.1 and g['load_score_read_wall_lower_bound_sec']<=out['arms']['full']['read_wall_sec'] and g['reader_errors']==0 and g['score_errors']==0;(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'score':score,'select':select,'run':run,'analyze':analyze}[sys.argv[1]]()
