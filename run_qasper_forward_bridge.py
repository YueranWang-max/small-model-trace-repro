"""Frozen explicit forward-reference routing, matched paragraph budgets."""
import json,re,time,sys,random
from pathlib import Path
from run_qasper_fingerprint import prepare
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from run_qasper_contract_replication import contract
from rescore_qasper_demand_transfer import references,f1,normalize

OUT=ROOT/'logs/qasper_forward_bridge';SOURCE=ROOT/'data/qasper_official/qasper-test-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_forward_bridge_20261002.md'
CUE=re.compile(r'\bas follows\b|\bfollowing details\b|\b(?:described|detailed) below\b|\bfollowing (?:subsections|paragraphs)\b|\bdescribed in the following\b',re.I)
ARMS=('bm25','neighbor','bridge')

def choose(p,q,mode):
    rank=q['rank'];seeds=rank[:4];extra=[];triggers=[]
    if mode!='bm25':
        for i in seeds:
            if mode=='bridge' and not CUE.search(p['paragraphs'][i]['text']):continue
            following=[j for j in range(i+1,min(i+3,len(p['paragraphs']))) if p['paragraphs'][j]['heading']==p['paragraphs'][i]['heading']]
            if mode=='bridge' and following:triggers.append({'seed':i,'successors':following,'cue':CUE.search(p['paragraphs'][i]['text']).group()})
            extra.extend(following)
    result=[]
    for i in seeds+extra+rank:
        if i not in result:result.append(i)
        if len(result)==8:break
    return result,triggers

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    train=json.loads((ROOT/'data/qasper_official/qasper-train-v0.3.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    fit_rows=[]
    for pid in split['fit']:
        raw=train[pid]
        for sec in raw['full_text']:
            for pi,text in enumerate(sec['paragraphs']):
                m=CUE.search(text)
                if m and pi+1<len(sec['paragraphs']):fit_rows.append({'paper_id':pid,'heading':sec['section_name'],'paragraph_in_section':pi,'cue':m.group(),'quote':text,'successors':sec['paragraphs'][pi+1:pi+3]})
    (OUT/'fit_source_diagnostic.json').write_text(json.dumps(fit_rows,indent=2,ensure_ascii=False),encoding='utf-8')
    source=json.loads(SOURCE.read_text(encoding='utf-8'));ids=sorted(source,key=lambda pid:sha(pid.encode()))[70:];tasks=[]
    for pid in ids:
        p=prepare(pid,source[pid])
        for q in p['questions']:
            selected,triggers=choose(p,q,'bridge')
            if not triggers or selected==q['rank'][:8]:continue
            choices={a:choose(p,q,a)[0] for a in ARMS}
            material={a:'\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in choices[a]) for a in ARMS}
            tasks.append({'paper_id':pid,'id':q['id'],'title':source[pid]['title'],'question':q['text'],'kind':contract(q['text'])[0],'choices':choices,'triggers':triggers,'material':material,'paragraphs':p['paragraphs']})
            if len(tasks)==24:break
        if len(tasks)==24:break
    files=[SOURCE,PROTOCOL,Path(__file__),ROOT/'data/qasper_official/qasper-train-v0.3.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_fingerprint.py','run_stageC.py','run_qasper_format_control.py','run_qasper_contract_replication.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]
    f={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':tasks,'fit_trigger_paragraphs':len(fit_rows),'prep_sec':time.perf_counter()-start,'eligible_reader_ids':ids}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(f,indent=2,ensure_ascii=False),encoding='utf-8');print({'tasks':len(tasks),'papers':len({t['paper_id'] for t in tasks}),'fit_triggers':len(fit_rows),'prep_sec':f['prep_sec']})

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in f['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json'
    d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(f['tasks']):
        row=next((r for r in d['rows'] if r['id']==t['id']),None)
        if row is None:row={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(row)
        gold=references(next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']))
        order=list(ARMS);order=order[i%3:]+order[:i%3]
        for arm in order:
            if arm in row['arms']:continue
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\nEvidence:\n'+t['material'][arm]+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold)
            row['arms'][arm]=r;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert rows and len(rows)==len(f['tasks']) and all(set(r['arms'])==set(ARMS) for r in rows)
    out={'questions':len(rows),'papers':len({r['paper_id'] for r in rows}),'fit_trigger_paragraphs':f['fit_trigger_paragraphs'],'prep_sec':f['prep_sec'],'warmup_sec':d['warmup']['wall_sec'],'arms':{},'paired':{}}
    for arm in ARMS:
        xs=[r['arms'][arm] for r in rows];out['arms'][arm]={'f1':sum(x['official_f1'] for x in xs)/len(xs),'tokens':sum((x.get('prompt_tokens') or 0)+(x.get('generation_tokens') or 0) for x in xs),'wall_sec':sum(x['wall_sec'] for x in xs),'errors':sum('error' in x for x in xs),'missing':sum(not x['answer'] for x in xs)}
    groups=[[r for r in rows if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rows})]
    for ref in ('bm25','neighbor'):
        rng=random.Random(20261002);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(r['arms']['bridge']['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
        boot.sort();out['paired']['bridge_minus_'+ref]={'diff':out['arms']['bridge']['f1']-out['arms'][ref]['f1'],'ci95':[boot[250],boot[9750]],'wins':sum(r['arms']['bridge']['official_f1']>r['arms'][ref]['official_f1'] for r in rows),'losses':sum(r['arms']['bridge']['official_f1']<r['arms'][ref]['official_f1'] for r in rows)}
    out['gate_pass']=all(x['ci95'][0]>0 for x in out['paired'].values()) and not any(x['errors'] for x in out['arms'].values())
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
