"""Literal question qualifiers against generic scope check; identical evidence."""
import json,re,time,sys,random
from pathlib import Path
import run_qasper_forward_bridge as base
from run_qasper_fingerprint import prepare
from run_qasper_contract_replication import contract
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from rescore_qasper_demand_transfer import references,f1,normalize

OUT=ROOT/'logs/qasper_target_scope';PROTOCOL=ROOT/'protocol_qasper_target_scope_20261002.md'
ARMS=('base','generic','target')
QUAL=re.compile(r'(?<!\w)(?:attention-based|semi-supervised|unsupervised|supervised|neural|statistical|monolingual|multilingual|pre-trained|pretrained|pre-training|pretraining|human|automatic|manually|synthetic|extractive|abstractive)(?!\w)',re.I)

def qualifiers(q):return list(dict.fromkeys(m.group() for m in QUAL.finditer(q)))

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    train=json.loads((ROOT/'data/qasper_official/qasper-train-v0.3.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    inventory=[{'paper_id':pid,'id':q['question_id'],'question':q['question'],'literal_qualifiers':qualifiers(q['question'])} for pid in split['fit'] for q in train[pid]['qas'] if qualifiers(q['question'])]
    (OUT/'fit_question_inventory.json').write_text(json.dumps({'total':len(inventory),'first20':inventory[:20]},indent=2,ensure_ascii=False),encoding='utf-8')
    prior=json.loads((base.OUT/'frozen.json').read_text(encoding='utf-8'));exclude={t['paper_id'] for t in prior['tasks']}
    source=json.loads(base.SOURCE.read_text(encoding='utf-8'));ids=[pid for pid in sorted(source,key=lambda pid:sha(pid.encode()))[70:] if pid not in exclude];tasks=[]
    for pid in ids:
        raw=source[pid]
        if not any(qualifiers(q['question']) for q in raw['qas']):continue
        p=prepare(pid,raw)
        for q in p['questions']:
            qual=qualifiers(q['text'])
            if not qual:continue
            selected=q['rank'][:8];material='\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in selected)
            tasks.append({'paper_id':pid,'id':q['id'],'title':raw['title'],'question':q['text'],'kind':contract(q['text'])[0],'qualifiers':qual,'selected':selected,'material':material,'paragraphs':p['paragraphs']})
            if len(tasks)==24:break
        if len(tasks)==24:break
    files=[base.SOURCE,PROTOCOL,Path(__file__),ROOT/'run_qasper_forward_bridge.py',ROOT/'logs/qasper_forward_bridge/frozen.json',ROOT/'data/qasper_official/qasper-train-v0.3.json',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_fingerprint.py','run_stageC.py','run_qasper_format_control.py','run_qasper_contract_replication.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]
    f={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':tasks,'fit_qualifier_questions':len(inventory),'excluded_papers':sorted(exclude),'prep_sec':time.perf_counter()-start}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(f,indent=2,ensure_ascii=False),encoding='utf-8');print({'questions':len(tasks),'papers':len({t['paper_id'] for t in tasks}),'fit_questions':len(inventory)})

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in f['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(base.SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json'
    d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert d['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,t in enumerate(f['tasks']):
        row=next((r for r in d['rows'] if r['id']==t['id']),None)
        if row is None:row={'id':t['id'],'paper_id':t['paper_id'],'arms':{}};d['rows'].append(row)
        gold=references(next(q for q in source[t['paper_id']]['qas'] if q['question_id']==t['id']));order=list(ARMS);order=order[i%3:]+order[:i%3]
        for arm in order:
            if arm in row['arms']:continue
            extra=''
            if arm=='generic':extra='Check the requested entities, conditions and scope before answering.\n'
            if arm=='target':extra='Question target terms: '+', '.join(t['qualifiers'])+'. Respect their role and scope in this question. Include an item only when the evidence supports the requested category; exclude items explicitly described as a different category. A term appearing elsewhere in the paragraph does not establish that an item belongs to that category.\n'
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(t['kind'])+'\n'+extra+'Evidence:\n'+t['material']+'\n\nQuestion: '+query_text(t['title'],t['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=a;r['recovered']=recovered;r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold)
            row['arms'][arm]=r;path.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert rows and len(rows)==len(f['tasks']) and all(set(r['arms'])==set(ARMS) for r in rows)
    out={'questions':len(rows),'papers':len({r['paper_id'] for r in rows}),'prep_sec':f['prep_sec'],'warmup_sec':d['warmup']['wall_sec'],'arms':{},'paired':{}}
    for a in ARMS:
        xs=[r['arms'][a] for r in rows];out['arms'][a]={'f1':sum(x['official_f1'] for x in xs)/len(xs),'tokens':sum((x.get('prompt_tokens') or 0)+(x.get('generation_tokens') or 0) for x in xs),'wall_sec':sum(x['wall_sec'] for x in xs),'errors':sum('error' in x for x in xs),'missing':sum(not x['answer'] for x in xs)}
    groups=[[r for r in rows if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rows})]
    for ref in ('base','generic'):
        rng=random.Random(20261002);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(r['arms']['target']['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
        boot.sort();out['paired']['target_minus_'+ref]={'diff':out['arms']['target']['f1']-out['arms'][ref]['f1'],'ci95':[boot[250],boot[9750]],'wins':sum(r['arms']['target']['official_f1']>r['arms'][ref]['official_f1'] for r in rows),'losses':sum(r['arms']['target']['official_f1']<r['arms'][ref]['official_f1'] for r in rows)}
    out['gate_pass']=all(x['ci95'][0]>0 for x in out['paired'].values()) and not any(x['errors'] for x in out['arms'].values());(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
