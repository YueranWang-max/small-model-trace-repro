"""Independent fixed-sample replication of question contracts with one shared parser."""
import json
import random
import re
import sys
import time
from pathlib import Path
from run_qasper_answer_contract import contract as old_contract
from run_qasper_fingerprint import prepare
from run_qasper_prefix_vs_bm25 import ROOT,sha
from run_qasper_visible_trace_replication import call,query_text
from rescore_qasper_demand_transfer import references,f1,normalize

OUT=ROOT/'logs/qasper_contract_replication'
SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_contract_replication_20261002.md'
ARMS=['base','contract']

def contract(text):
    kind,rule=old_contract(text)
    target=r'^(which|what)\s+(?:(?:are|were)\s+)?(?:the\s+)?(?:different\s+|specific\s+)?(datasets|models|methods|languages|tasks|metrics|authors|features|steps)\b'
    if kind=='list' and not re.search(target,text.lower().strip()):
        kind,rule='fact','Give only the specific fact requested, as a short phrase.'
    if kind=='boolean':rule='The answer VALUE after the Final Answer: prefix must be exactly Yes, No, or unknown. Verify the scope and any negation.'
    return kind,rule

def parsed(result):
    strict=result.get('answer') or ''
    if strict:return strict,False
    lines=[s.strip() for s in result.get('content','').splitlines() if s.strip()]
    return (lines[0],True) if len(lines)==1 and len(lines[0])<=300 else ('',False)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);source=json.loads(SOURCE.read_text(encoding='utf-8'))
    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    ids=split['heldout'][24:56]
    t=time.perf_counter();papers=[prepare(pid,source[pid]) for pid in ids];prep=time.perf_counter()-t
    t=time.perf_counter();choices={p['id']:{q['id']:{'selected':q['rank'][:8],'contract':contract(q['text'])} for q in p['questions']} for p in papers};selection=time.perf_counter()-t
    order=ARMS.copy();random.Random(20261002).shuffle(order)
    frozen={'source_sha256':sha(SOURCE.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),
        'script_sha256':sha(Path(__file__).read_bytes()),'papers':papers,'choices':choices,'arm_order':order,
        'prep_sec':prep,'selection_sec':selection,'dependency_sha256':{name:sha((ROOT/name).read_bytes()) for name in ['run_qasper_answer_contract.py','run_qasper_fingerprint.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py']}}
    path=OUT/'frozen.json';assert not path.exists();path.write_text(json.dumps(frozen,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'papers':len(papers),'questions':sum(len(p['questions']) for p in papers),'prep_sec':prep},indent=2))

def run():
    fp=OUT/'frozen.json';frozen=json.loads(fp.read_text(encoding='utf-8'))
    for path,key in [(SOURCE,'source_sha256'),(PROTOCOL,'protocol_sha256'),(Path(__file__),'script_sha256')]:assert sha(path.read_bytes())==frozen[key]
    for name,value in frozen['dependency_sha256'].items():assert sha((ROOT/name).read_bytes())==value
    source=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/'reader_results.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'papers':[]}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in data:
        data['warmup']=call('Reply with exactly: Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
        path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for pi,p in enumerate(frozen['papers']):
        pid=p['id'];order=frozen['arm_order'] if pi%2==0 else frozen['arm_order'][::-1]
        rp=next((x for x in data['papers'] if x['paper_id']==pid),None)
        if rp is None:rp={'paper_id':pid,'arm_order':order,'records':[]};data['papers'].append(rp)
        for arm in order:
            for q in p['questions']:
                row=next((r for r in rp['records'] if r['id']==q['id']),None)
                if row is None:row={'id':q['id'],'arms':{}};rp['records'].append(row)
                if arm in row['arms']:continue
                choice=frozen['choices'][pid][q['id']]
                material='\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in choice['selected'])
                prompt='Answer using only the evidence below. If insufficient, say unknown.\n'
                if arm=='contract':prompt+='Answer value requirement: '+choice['contract'][1]+'\nCheck the requested entities, conditions and scope. A table citation or caption without its values is insufficient for a numeric answer; do not infer missing values.\n'
                prompt+='Evidence:\n'+material+'\n\nQuestion: '+query_text(source[pid]['title'],q['text'])+'\nOutput exactly one line in this format: Final Answer: <short answer value>'
                rawq=next(x for x in source[pid]['qas'] if x['question_id']==q['id']);gold=references(rawq)
                result=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024)
                answer,recovered=parsed(result);result['strict_answer']=result.get('answer');result['answer']=answer;result['recovered']=recovered
                aligned='Unanswerable' if normalize(answer)=='unknown' else answer
                result['official_f1']=max(f1(aligned,g) for g in gold);result['literal_f1']=max(f1(answer,g) for g in gold)
                result['text_em']=float(any(normalize(aligned)==normalize(g) for g in gold))
                row['arms'][arm]=result;path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
                print(pi+1,pid,q['id'],arm,round(result['official_f1'],3),flush=True)

def analyze():
    frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));data=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'));source=json.loads(SOURCE.read_text(encoding='utf-8'))
    groups=[p['records'] for p in data['papers']];rows=[r for g in groups for r in g]
    assert len(groups)==32 and len(rows)==sum(len(p['questions']) for p in frozen['papers']) and all(set(r['arms'])==set(ARMS) for r in rows)
    out={'papers':32,'questions':len(rows),'arms':{},'prep_sec':frozen['prep_sec'],'selection_sec':frozen['selection_sec'],'warmup_sec':data['warmup']['wall_sec']}
    for a in ARMS:
        xs=[r['arms'][a] for r in rows]
        arm={k:sum(x[k] for x in xs)/len(xs) for k in ['official_f1','literal_f1','text_em']}
        arm.update({k:sum(x.get(k) or 0 for x in xs) for k in ['prompt_tokens','generation_tokens','wall_sec','load_ns']})
        arm.update({'errors':sum('error' in x for x in xs),'strict_prefix_missing':sum(not x.get('strict_answer') for x in xs),'recovered':sum(x['recovered'] for x in xs),'missing_answer':sum(not x['answer'] for x in xs)})
        counts={'answerable':0,'false_refusal':0,'unanswerable':0,'correct_refusal':0,'mixed':0,'boolean_n':0,'boolean_correct':0,'ambiguous_boolean':0}
        for p in data['papers']:
            for r in p['records']:
                q=next(q for q in source[p['paper_id']]['qas'] if q['question_id']==r['id']);refs=q['answers'];states={x['answer']['unanswerable'] for x in refs};answer=r['arms'][a]['answer'];refuse=normalize(answer) in ('unknown','unanswerable')
                if states=={False}:counts['answerable']+=1;counts['false_refusal']+=refuse
                elif states=={True}:counts['unanswerable']+=1;counts['correct_refusal']+=refuse
                else:counts['mixed']+=1
                bs={x['answer']['yes_no'] for x in refs if x['answer']['yes_no'] is not None}
                if len(bs)==1 and states=={False}:
                    counts['boolean_n']+=1
                    polarity=re.match(r'^\s*(yes|no)(?:\s*[,.:;!]|\s*$)',answer,re.I)
                    counts['boolean_correct']+=bool(polarity and (polarity.group(1).lower()=='yes')==next(iter(bs)))
                elif len(bs)>1:counts['ambiguous_boolean']+=1
        arm['diagnostics']=counts;out['arms'][a]=arm
    rng=random.Random(20261002);boot=[]
    for _ in range(10000):
        sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]]
        boot.append(sum(r['arms']['contract']['official_f1']-r['arms']['base']['official_f1'] for r in sample)/len(sample))
    boot.sort();out['paired']={'difference':out['arms']['contract']['official_f1']-out['arms']['base']['official_f1'],'paper_ci95':[boot[250],boot[9750]]};out['gate_pass']=boot[250]>0
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
