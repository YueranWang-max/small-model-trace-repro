"""Model-produced brief + >=80% original, with construction charged."""
import json,re,time,sys,random
from pathlib import Path
import run_qasper_retained_source as base
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from rescore_qasper_demand_transfer import references,f1,normalize
from build_qasper_relation_ledger import paragraphs

OUT=ROOT/'logs/qasper_deep_retained';PROTOCOL=ROOT/'protocol_qasper_deep_retained_20261002.md';ARMS=('brief_full','brief_core80')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);f=json.loads((base.OUT/'frozen.json').read_text(encoding='utf-8'))
    files=[Path(__file__),PROTOCOL,base.SOURCE,base.OUT/'frozen.json',ROOT/'run_qasper_retained_source.py',ROOT/'repro_lib.py',ROOT/'run_qasper_visible_trace_replication.py',ROOT/'run_qasper_format_control.py',ROOT/'rescore_qasper_demand_transfer.py']
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps({'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':f['tasks'][:12]},indent=2,ensure_ascii=False),encoding='utf-8');print({'questions':12,'papers':len({t['paper_id'] for t in f['tasks'][:12]})})

def verify(content,ps):
    text=content.strip();text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    try:o=json.loads(text)
    except Exception:return {'accepted':False,'reason':'json_parse','guide':''}
    if not isinstance(o,dict) or not isinstance(o.get('summary'),str) or not isinstance(o.get('anchors'),list) or not 1<=len(o['anchors'])<=3 or len(o['summary'].split())>120 or not o['summary'].strip():return {'accepted':False,'reason':'schema_or_length','guide':''}
    anchors=[]
    for a in o['anchors']:
        if not isinstance(a,dict) or type(a.get('paragraph_id')) is not int or not isinstance(a.get('quote'),str):return {'accepted':False,'reason':'anchor_schema','guide':''}
        i=a['paragraph_id'];quote=a['quote']
        if not 0<=i<len(ps) or not 10<=len(quote)<=500:return {'accepted':False,'reason':'anchor_range','guide':''}
        body=ps[i]['text'];start=body.find(quote)
        if start<0:return {'accepted':False,'reason':'quote_not_exact','guide':''}
        anchors.append({'paragraph_id':i,'quote':quote,'char_start':start,'char_length':len(quote),'byte_start':len(body[:start].encode()),'byte_length':len(quote.encode()),'paragraph_sha256':sha(body.encode())})
    guide='Model-produced reading brief; summary entailment is unverified. Original source is authoritative.\n'+o['summary']+'\n'+'\n'.join(f"[P{a['paragraph_id']}] exact anchor: "+a['quote'] for a in anchors)
    return {'accepted':True,'reason':'schema_and_quotes_valid_not_semantic_verification','summary':o['summary'],'anchors':anchors,'guide':guide}

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in f['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(base.SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(f['tasks']):
        r=next((r for r in d['rows'] if r['id']==t['id']),None)
        if r is None:r={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(r)
        if 'build' not in r:
            prompt='Create a compact question-relevant reading map from the paper. Preserve entities, relations, conditions, units and contrasting categories. Do not give a final answer. Return only JSON with keys summary (at most120words) and anchors (1to3 objects with integer paragraph_id and exact quote10to500characters). No other keys required.\nPaper source:\n'+t['material']['full']+'\nQuestion: '+query_text(t['title'],t['question'])
            r['build']=call(prompt,{'answer':'','answer_aliases':[]},think=False,cap=768);r['verified']=verify(r['build'].get('content',''),paragraphs(source[t['paper_id']]))
            path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,'build',r['verified']['accepted'],flush=True)
        gold=references(next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']))
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            if arm in r['arms']:continue
            material=t['material']['full' if arm=='brief_full' else 'core80'];guide=r['verified']['guide'];evidence=(guide+'\n\nOriginal source:\n' if guide else '')+material
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\nEvidence:\n'+evidence+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            v=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(v);v['strict_answer']=v['answer'];v['answer']=a;v['recovered']=recovered;v['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold)
            r['arms'][arm]=v;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(v['official_f1'],3),flush=True)

def tokens(v):return (v.get('prompt_tokens') or 0)+(v.get('generation_tokens') or 0)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));old=json.loads((base.OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==12 and all(set(r['arms'])==set(ARMS) for r in rows)
    baserows=[next(r for r in old['rows'] if r['id']==t['id']) for t in f['tasks']];out={'questions':12,'papers':6,'accepted_briefs':sum(r['verified']['accepted'] for r in rows),'warmup_sec':d['warmup']['wall_sec'],'build_tokens':sum(tokens(r['build']) for r in rows),'build_wall_sec':sum(r['build']['wall_sec'] for r in rows),'build_errors':sum('error' in r['build'] for r in rows),'arms':{},'paired':{}}
    for arm in ('full','core80'):
        xs=[r['arms'][arm] for r in baserows];out['arms'][arm]={'f1':sum(v['official_f1'] for v in xs)/12,'read_tokens':sum(tokens(v) for v in xs),'read_wall_sec':sum(v['wall_sec'] for v in xs),'errors':sum('error' in v for v in xs)}
    for arm in ARMS:
        xs=[r['arms'][arm] for r in rows];out['arms'][arm]={'f1':sum(v['official_f1'] for v in xs)/12,'read_tokens':sum(tokens(v) for v in xs),'read_wall_sec':sum(v['wall_sec'] for v in xs),'pipeline_tokens':out['build_tokens']+sum(tokens(v) for v in xs),'pipeline_wall_sec':out['build_wall_sec']+sum(v['wall_sec'] for v in xs),'errors':sum('error' in v for v in xs)}
    pairs=[(t,next(r for r in rows if r['id']==t['id']),next(r for r in baserows if r['id']==t['id'])) for t in f['tasks']];groups=[[x for x in pairs if x[0]['paper_id']==pid] for pid in sorted({t['paper_id'] for t in f['tasks']})]
    for arm,ref in [('brief_core80','full'),('brief_core80','core80'),('brief_full','full')]:
        rng=random.Random(20261002);boot=[]
        for _ in range(10000):
            sample=[x for _ in groups for x in groups[rng.randrange(len(groups))]];boot.append(sum(r['arms'][arm]['official_f1']-b['arms'][ref]['official_f1'] for _,r,b in sample)/len(sample))
        boot.sort();out['paired'][arm+'_minus_'+ref]={'diff':out['arms'][arm]['f1']-out['arms'][ref]['f1'],'ci95':[boot[250],boot[9750]]}
    out['single_use_token_reduction']=1-out['arms']['brief_core80']['pipeline_tokens']/out['arms']['full']['read_tokens'];out['gate_pass']=out['paired']['brief_core80_minus_full']['ci95'][0]>=-.02 and out['single_use_token_reduction']>=.1 and out['build_errors']==0 and not any(x['errors'] for x in out['arms'].values())
    out['repeat_scenarios']={str(n):{'full_reread_tokens':n*out['arms']['full']['read_tokens'],'brief_core_tokens':out['build_tokens']+n*out['arms']['brief_core80']['read_tokens'],'full_answer_cache_tokens':out['arms']['full']['read_tokens'],'simulated_not_new_calls':True} for n in (1,5,20)}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
