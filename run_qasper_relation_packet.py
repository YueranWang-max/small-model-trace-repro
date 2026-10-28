"""Source-substring relation windows versus full retrieval; development only."""
import json,re,sys,time,random
from pathlib import Path
from run_qasper_fingerprint import prepare
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_stageC import BM25
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_format_control import format_rule,parsed
from run_qasper_contract_replication import contract
from rescore_qasper_demand_transfer import references,f1,normalize

OUT=ROOT/'logs/qasper_relation_packet';SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_relation_packet_20261002.md'
SCOPE=r'how (many|much|large|big)|\b(number|percentage|percent|rate|size)\b'

def packet(text,question):
    cuts=[0]+[m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])',text)]+[len(text)]
    spans=[(a,b) for a,b in zip(cuts,cuts[1:]) if text[a:b].strip()]
    sentences=[text[a:b] for a,b in spans]
    keep=set(BM25(sentences).search(question,2)) if sentences else set()
    for i,s in enumerate(sentences):
        if re.search(r'\brespectively\b',s,re.I):keep.update(range(max(0,i-2),min(len(spans),i+2)))
    fragments=[]
    for i in sorted(keep):
        a,b=spans[i];t=text[a:b]
        fragments.append({'text':t,'char_start':a,'char_length':b-a,'byte_start':len(text[:a].encode()),'byte_length':len(t.encode()),'paragraph_sha256':sha(text.encode())})
    return fragments

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    source=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));tasks=[]
    for pid in split['validation']:
        p=prepare(pid,source[pid])
        for q in p['questions']:
            chosen=q['rank'][:8]
            if not re.search(SCOPE,q['text'],re.I):continue
            if not any(re.search(r'\brespectively\b',p['paragraphs'][i]['text'],re.I) and len(re.findall(r'\d+(?:\.\d+)?',p['paragraphs'][i]['text']))>=2 for i in chosen):continue
            material={};ledger=[]
            material['full']='\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in chosen)
            chunks=[]
            for i in chosen:
                parts=packet(p['paragraphs'][i]['text'],q['text']);ledger.append({'paragraph_id':i,'fragments':parts})
                chunks.append(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+'\n[omitted source between fragments]\n'.join(x['text'] for x in parts))
            material['packet']='\n\n'.join(chunks)
            tasks.append({'paper_id':pid,'id':q['id'],'question':q['text'],'title':source[pid]['title'],'kind':contract(q['text'])[0],'material':material,'ledger':ledger,'selected':chosen})
            if len(tasks)==24:break
        if len(tasks)==24:break
    files=[SOURCE,PROTOCOL,Path(__file__),ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_fingerprint.py','run_stageC.py','run_qasper_format_control.py','run_qasper_contract_replication.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]
    frozen={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':tasks,'prep_sec':time.perf_counter()-t}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(frozen,indent=2,ensure_ascii=False),encoding='utf-8');print({'tasks':len(tasks),'papers':len({x['paper_id'] for x in tasks}),'prep_sec':frozen['prep_sec']})

def run():
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in f['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    source=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'results.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'rows':[]}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in data:data['warmup']=call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    for i,task in enumerate(f['tasks']):
        row=next((r for r in data['rows'] if r['id']==task['id']),None)
        if row is None:row={'id':task['id'],'paper_id':task['paper_id'],'arms':{}};data['rows'].append(row)
        q=next(q for q in source[task['paper_id']]['qas'] if q['question_id']==task['id']);gold=references(q)
        for arm in (['full','packet'] if i%2==0 else ['packet','full']):
            if arm in row['arms']:continue
            prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: '+format_rule(task['kind'])+'\nEvidence:\n'+task['material'][arm]+'\n\nQuestion: '+query_text(task['title'],task['question'])+'\nOutput exactly one line: Final Answer: <short answer value>'
            r=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);answer,recovered=parsed(r);r['strict_answer']=r['answer'];r['answer']=answer;r['recovered']=recovered
            r['official_f1']=max(f1('Unanswerable' if normalize(answer)=='unknown' else answer,g) for g in gold)
            row['arms'][arm]=r;path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');print(i+1,arm,round(r['official_f1'],3),flush=True)

def analyze():
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=d['rows'];assert len(rows)==len(f['tasks']) and all(set(r['arms'])=={'full','packet'} for r in rows)
    out={'questions':len(rows),'papers':len({r['paper_id'] for r in rows}),'prep_sec':f['prep_sec'],'warmup_sec':d['warmup']['wall_sec'],'arms':{}}
    for arm in ('full','packet'):
        xs=[r['arms'][arm] for r in rows];out['arms'][arm]={'f1':sum(x['official_f1'] for x in xs)/len(xs),'tokens':sum((x.get('prompt_tokens') or 0)+(x.get('generation_tokens') or 0) for x in xs),'wall_sec':sum(x['wall_sec'] for x in xs),'errors':sum('error' in x for x in xs),'missing':sum(not x['answer'] for x in xs)}
    groups=[[r for r in rows if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rows})];rng=random.Random(20261002);boot=[]
    for _ in range(10000):
        sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(r['arms']['packet']['official_f1']-r['arms']['full']['official_f1'] for r in sample)/len(sample))
    boot.sort();out['diff']=out['arms']['packet']['f1']-out['arms']['full']['f1'];out['ci95']=[boot[250],boot[9750]];out['token_reduction']=1-out['arms']['packet']['tokens']/out['arms']['full']['tokens'];out['gate_pass']=out['ci95'][0]>=-.02 and out['token_reduction']>=.2 and not any(out['arms'][a]['errors'] for a in out['arms'])
    out['wins']=sum(r['arms']['packet']['official_f1']>r['arms']['full']['official_f1'] for r in rows);out['losses']=sum(r['arms']['packet']['official_f1']<r['arms']['full']['official_f1'] for r in rows)
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
