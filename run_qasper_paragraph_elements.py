"""Exact within-paragraph sentence retention with fit-only weak QA cue priors."""
import json,re,math,time,sys,random
from collections import Counter
from run_qasper_prefix_vs_bm25 import ROOT,sha
from build_qasper_relation_ledger import paragraphs
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from run_qasper_contract_replication import contract
from rescore_qasper_demand_transfer import references,f1,normalize
from run_stageC import BM25,_tokens

OUT=ROOT/'logs/qasper_paragraph_elements'
SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_paragraph_elements_20261002.md'
ARMS=('full','elements','multiq','combined')
CUES={
 'result':r'\b(result|find|found|show|shows|demonstrate|outperform|achiev|conclud|improv|accuracy|score|performance)',
 'definition':r'\b(is defined|defined as|refer to|consists of|composed of|contains|comprises|namely|denote|means)\b',
 'procedure':r'\b(train|training|annotat|optimiz|algorithm|implement|initialize|sample|sampling|step|preprocess|evaluate|evaluation|use|used)\b',
 'provenance':r'\b(dataset|corpus|corpora|collected|obtained|released|available|source|download|benchmark)\b',
 'scope':r'\b(not|no|only|except|unless|if|when|each|per|first|both|however|whereas|compared|than|without)\b',
 'limitation':r'\b(limitation|future|fail|cannot|unable|restricted|drawback)\b',
 'number':r'\d',
 'entity':r'\b(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Z][a-z]+[A-Z][A-Za-z]*)\b',
 'claim':r'\b(we propose|we present|our model|our method|our approach|we introduce)\b',
}

def spans(text):
    boundaries=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])',text)]+[len(text)]
    out=[]
    for a,b in zip(boundaries,boundaries[1:]):
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:out.append({'text':text[a:b],'start':a,'length':b-a})
    return out

def features(s):
    return [k for k,p in CUES.items() if re.search(p,s,0 if k=='entity' else re.I)]

def fit(source,ids):
    counts=Counter();positives=Counter();total=positive=0;papers_with_spans=0
    for pid in ids:
        raw=source[pid];answers={x.strip() for q in raw['qas'] for r in q['answers'] for x in r['answer']['extractive_spans'] if x.strip()}
        papers_with_spans+=bool(answers)
        for p in paragraphs(raw):
            for s in spans(p['text']):
                y=any(x in s['text'] for x in answers);fs=features(s['text']);total+=1;positive+=y
                counts.update(fs)
                if y:positives.update(fs)
    base=(positive+1)/(total+2)
    weights={k:1+max(-1,min(2,math.log(((positives[k]+1)/(counts[k]+2))/base))) for k in CUES}
    return {'weights':weights,'feature_counts':dict(counts),'positive_counts':dict(positives),'sentences':total,'weak_positive_sentences':positive,'papers_with_extractive_spans':papers_with_spans,'papers':len(ids),'weak_labels_not_entailment':True}

def qscores(units,questions):
    index=BM25([s['text'] for s in units]);scores=[0.0]*len(units)
    for q in questions:
        qt=_tokens(q);raw=[]
        for i in range(index.N):
            score=0.0
            for t in qt:
                count=index.tfs[i].get(t,0)
                if count:score+=index.idf.get(t,0)*count*2.2/(count+1.5*(1-.75+.75*index.dls[i]/max(1,index.avgdl)))
            raw.append(score)
        maximum=max(raw,default=0)
        if maximum:scores=[max(a,b/maximum) for a,b in zip(scores,raw)]
    return scores

def construct(ps,questions,weights):
    units=[];bypara=[]
    for i,p in enumerate(ps):
        ids=[]
        for j,s in enumerate(spans(p['text'])):
            ids.append(len(units));units.append({**s,'pid':i,'sid':j,'features':features(s['text'])})
        assert ids;bypara.append(ids)
    shared=qscores(units,questions[:2]);known=qscores(units,questions[:3])
    roles=[sum(weights[k] for k in s['features']) for s in units]
    selections={'full':list(range(len(units)))}
    for arm in ('elements','multiq','combined','known3_diagnostic'):
        scores=roles if arm=='elements' else shared if arm=='multiq' else [r+3*q for r,q in zip(roles,known if arm=='known3_diagnostic' else shared)]
        chosen=set()
        for pid,ids in enumerate(bypara):
            target=.5*len(ps[pid]['text']);amount=0
            for i in sorted(ids,key=lambda x:(-scores[x],x)):
                if amount>=target and any(x in chosen for x in ids):break
                chosen.add(i);amount+=units[i]['length']
        changed=True
        while changed:
            before=len(chosen)
            for i in list(chosen):
                s=units[i];ids=bypara[s['pid']];j=s['sid']
                if j and re.match(r'^(?:It|They|This|These|Those|Such|However|Therefore|Consequently|In contrast)\b',s['text']):chosen.add(ids[j-1])
                if s['text'].rstrip().endswith(':') and j+1<len(ids):chosen.add(ids[j+1])
            changed=len(chosen)!=before
        selections[arm]=sorted(chosen)
    materials={};stats={};pointers={}
    for arm,selected in selections.items():
        selected=set(selected);chunks=[];records=[]
        for pid,ids in enumerate(bypara):
            parts=[];previous=None
            for i in ids:
                if i not in selected:continue
                s=units[i]
                if (previous is None and s['sid']>0) or (previous is not None and s['sid']>previous+1):parts.append('[Sentences omitted]')
                parts.append(f"[S{s['sid']}] "+s['text']);previous=s['sid']
                text=ps[pid]['text'];a=s['start'];b=a+s['length'];assert text[a:b]==s['text']
                rec={**s,'paragraph_sha256':sha(text.encode()),'byte_start':len(text[:a].encode()),'byte_length':len(text[a:b].encode())}
                assert text.encode()[rec['byte_start']:rec['byte_start']+rec['byte_length']].decode()==s['text'];records.append(rec)
            if previous<len(ids)-1:parts.append('[Sentences omitted]')
            chunks.append(f"[P{pid}] Section: {ps[pid]['heading']}\n"+'\n'.join(parts))
        materials[arm]='\n\n'.join(chunks);pointers[arm]=records
        stats[arm]={'source_chars':sum(s['length'] for s in units),'retained_chars':sum(units[i]['length'] for i in selected),'sentences':len(selected),'all_paragraphs_retained':all(any(i in selected for i in ids) for ids in bypara)}
    return {'material':materials,'pointers':pointers,'stats':stats,'paragraphs':ps,'units':units}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();start=time.perf_counter()
    source=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));priors=fit(source,split['fit']);tasks=[]
    for pid in sorted(split['validation'],key=lambda x:sha(x.encode())):
        raw=source[pid];ps=paragraphs(raw)
        if len(raw['qas'])<3 or sum(len(p['text']) for p in ps)>48000:continue
        texts=[q['question'] for q in raw['qas'][:3]];q=raw['qas'][2];c=construct(ps,texts,priors['weights'])
        tasks.append({'paper_id':pid,'id':q['question_id'],'title':raw['title'],'question':q['question'],'shared_question_ids':[x['question_id'] for x in raw['qas'][:2]],'kind':contract(q['question'])[0],**c})
        if len(tasks)==12:break
    assert len(tasks)==12
    files=[SOURCE,PROTOCOL,ROOT/'logs/qasper_fingerprint/split.json',ROOT/'run_qasper_paragraph_elements.py']+[ROOT/n for n in ['build_qasper_relation_ledger.py','run_qasper_visible_trace_replication.py','run_qasper_format_control.py','run_qasper_contract_replication.py','rescore_qasper_demand_transfer.py','run_stageC.py','repro_lib.py']]
    frozen={'priors':priors,'tasks':tasks,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'prep_sec':time.perf_counter()-start}
    (OUT/'frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8');print({'questions':len(tasks),'prep_sec':frozen['prep_sec'],'fit':priors},flush=True)

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in fr['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(fr['tasks']):
        row=next((r for r in d['rows'] if r['id']==t['id']),None)
        if row is None:row={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(row)
        gold=references(next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']));order=list(ARMS);order=order[i%4:]+order[:i%4]
        for arm in order:
            if arm in row['arms']:continue
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\nEvidence:\n'+t['material'][arm]+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold)
            row['arms'][arm]=r;path.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows)
    out={'papers':12,'questions':12,'prep_sec':fr['prep_sec'],'arms':{},'paired':{}}
    for arm in ARMS:
        xs=[r['arms'][arm] for r in rows];st=[t['stats'][arm] for t in fr['tasks']]
        out['arms'][arm]={'f1':sum(x['official_f1'] for x in xs)/len(xs),'tokens':sum((x.get('prompt_tokens') or 0)+(x.get('generation_tokens') or 0) for x in xs),'wall_sec':sum(x['wall_sec'] for x in xs),'errors':sum('error' in x for x in xs),'source_retention':sum(x['retained_chars'] for x in st)/sum(x['source_chars'] for x in st),'all_paragraphs_retained':all(x['all_paragraphs_retained'] for x in st)}
    for arm,ref in [('elements','full'),('multiq','full'),('combined','full'),('combined','elements'),('combined','multiq')]:
        diffs=[r['arms'][arm]['official_f1']-r['arms'][ref]['official_f1'] for r in rows];rng=random.Random(20261002);bs=sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(10000));out['paired'][arm+'_minus_'+ref]={'diff':sum(diffs)/len(diffs),'ci95':[bs[250],bs[9750]]}
    out['token_reduction']=1-out['arms']['combined']['tokens']/out['arms']['full']['tokens'];out['gate_pass']=out['paired']['combined_minus_full']['ci95'][0]>=-.02 and out['token_reduction']>=.1 and not any(x['errors'] for x in out['arms'].values())
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
