"""Frozen factorial pilot: question-only output contracts and adjacent evidence."""
import json
import random
import re
import sys
import time
from pathlib import Path
from run_qasper_prefix_vs_bm25 import ROOT, sha
from run_qasper_fingerprint import prepare, evidence
from run_qasper_visible_trace_replication import call, query_text
from rescore_qasper_demand_transfer import references, f1, normalize

OUT=ROOT/'logs/qasper_answer_contract'
SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
ARMS=['base','contract','expanded','combined']
PROTOCOL=ROOT/'protocol_qasper_answer_contract_20261004.md'

def contract(q):
    s=q.lower().strip()
    if re.match(r'(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b',s):
        return 'boolean','The final answer must be exactly Yes, No, or unknown. Verify the scope and any negation.'
    if re.search(r'how (many|much)|percentage|percent|\brate\b|\bnumber\b',s):
        return 'quantity','Give the requested number(s), units, and conditions. Do not substitute a nearby but different metric. If required values are absent, answer unknown.'
    if re.search(r'compar|baseline|better|worse',s):
        return 'comparison','Identify both compared methods/conditions and the requested outcome or metric. If a required side or comparison is absent, answer unknown.'
    if re.search(r'^(which|what)\b',s) and re.search(r'\b(datasets|models|methods|languages|tasks|metrics|authors|features|steps)\b',s):
        return 'list','Return all items requested within the stated scope, with a concise comma-separated list. Do not turn examples into a claim of completeness.'
    if re.search(r'^how\b|method|procedure|steps|approach',s):
        return 'procedure','Give the essential steps or mechanism in concise order, preserving relevant conditions. If the mechanism is not supported, answer unknown.'
    return 'fact','Give only the specific fact requested, as a short phrase.'

def expand(p,q,seeds,extra):
    base=q['rank'][:8]; candidates=[]
    for i in base[:seeds]:
        for j in (i-1,i+1):
            if 0<=j<len(p['paragraphs']) and j not in base and j not in candidates and p['paragraphs'][j]['heading']==p['paragraphs'][i]['heading']:
                candidates.append(j)
    return base+candidates[:extra]

def coverage(papers, source, seeds=None, extra=None):
    hits=eligible=chars=0; rec=[]
    for p in papers:
        for q in p['questions']:
            selected=q['rank'][:8] if seeds is None else expand(p,q,seeds,extra)
            chars+=sum(len(p['paragraphs'][i]['text']) for i in selected)
            e=evidence(p,q,source)
            if e:
                eligible+=1; hits+=e.issubset(selected); rec.append(len(e.intersection(selected))/len(e))
    return {'eligible':eligible,'all_hit':hits,'all_hit_rate':hits/eligible,'recall':sum(rec)/eligible,'chars':chars}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'))
    t=time.perf_counter(); val=[prepare(pid,source[pid]) for pid in split['validation']]
    base=coverage(val,source); configs=[]
    for seeds in (1,2,4):
        for extra in (2,4):
            v=coverage(val,source,seeds,extra)
            configs.append({'seeds':seeds,'extra':extra,**v})
    allowed=[c for c in configs if c['chars']<=base['chars']*1.5]
    selected=sorted(allowed,key=lambda c:(-c['all_hit_rate'],c['chars'],c['seeds'],c['extra']))[0]
    validation_sec=time.perf_counter()-t
    ids=split['heldout'][:24]
    t=time.perf_counter(); papers=[prepare(pid,source[pid]) for pid in ids]; prep=time.perf_counter()-t
    t=time.perf_counter(); choices={p['id']:{q['id']:{'base':q['rank'][:8],
        'expanded':expand(p,q,selected['seeds'],selected['extra']),'contract':contract(q['text'])} for q in p['questions']} for p in papers}
    selection=time.perf_counter()-t
    order=ARMS.copy();random.Random(20261004).shuffle(order)
    frozen={'source_sha256':sha(SOURCE.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),
        'script_sha256':sha(Path(__file__).read_bytes()),'selected':selected,'validation_base':base,
        'validation_configs':configs,'validation_selection_sec':validation_sec,'prepare_sec':prep,
        'selection_sec':selection,'papers':papers,'choices':choices,'arm_order_seed':order}
    path=OUT/'frozen.json'
    assert not path.exists(), 'Do not overwrite a frozen experiment'
    path.write_text(json.dumps(frozen,indent=2,ensure_ascii=False),encoding='utf-8')
    # Heldout annotations consulted only after source-only choices are saved.
    ev={'base':coverage(papers,source),'expanded':coverage(papers,source,selected['seeds'],selected['extra'])}
    (OUT/'evidence_summary.json').write_text(json.dumps(ev,indent=2),encoding='utf-8')
    print(json.dumps({'papers':len(papers),'questions':sum(len(p['questions']) for p in papers),'selected':selected,'evidence':ev},indent=2))

def run():
    fp=OUT/'frozen.json'; frozen=json.loads(fp.read_text(encoding='utf-8'))
    assert sha(SOURCE.read_bytes())==frozen['source_sha256']
    assert sha(PROTOCOL.read_bytes())==frozen['protocol_sha256']
    assert sha(Path(__file__).read_bytes())==frozen['script_sha256']
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    path=OUT/'reader_results.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'papers':[]}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in data:
        data['warmup']=call('Reply with exactly: Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
        path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for pi,p in enumerate(frozen['papers']):
        pid=p['id']; order=frozen['arm_order_seed']; order=order[pi%4:]+order[:pi%4]
        rp=next((x for x in data['papers'] if x['paper_id']==pid),None)
        if rp is None:
            rp={'paper_id':pid,'arm_order':order,'records':[]};data['papers'].append(rp)
        for arm in order:
            for q in p['questions']:
                row=next((r for r in rp['records'] if r['id']==q['id']),None)
                if row is None:
                    row={'id':q['id'],'arms':{}};rp['records'].append(row)
                if arm in row['arms']:continue
                choice=frozen['choices'][pid][q['id']]
                selection=choice['expanded' if arm in ('expanded','combined') else 'base']
                material='\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in selection)
                prompt='Answer using only the evidence below. If insufficient, say unknown.\n'
                if arm in ('contract','combined'):
                    prompt+='Answer requirement: '+choice['contract'][1]+'\nCheck that evidence supports the requested entities, conditions and scope. A table citation or caption without its values is insufficient for a numeric answer. Do not infer missing values.\n'
                prompt+='Evidence:\n'+material+'\n\nQuestion: '+query_text(source[pid]['title'],q['text'])+'\nEnd with exactly one line: Final Answer: <short answer>'
                rawq=next(x for x in source[pid]['qas'] if x['question_id']==q['id'])
                refs=references(rawq)
                result=call(prompt,{'answer':refs[0],'answer_aliases':refs[1:]},think=False,cap=1024)
                pred=result.get('answer') or ''; aligned='Unanswerable' if normalize(pred)=='unknown' else pred
                result['official_f1']=max(f1(aligned,g) for g in refs)
                result['literal_f1']=max(f1(pred,g) for g in refs)
                result['text_em']=float(any(normalize(aligned)==normalize(g) for g in refs))
                row['arms'][arm]=result
                path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
                print(pi+1,pid,q['id'],arm,round(result['official_f1'],3),flush=True)

def analyze():
    frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    data=json.loads((OUT/'reader_results.json').read_text(encoding='utf-8'))
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    groups=[p['records'] for p in data['papers']];rows=[r for g in groups for r in g]
    assert len(groups)==24 and len(rows)==sum(len(p['questions']) for p in frozen['papers']) and all(set(r['arms'])==set(ARMS) for r in rows)
    summary={'papers':24,'questions':len(rows),'arms':{},'paired':{},'warmup_wall_sec':data['warmup']['wall_sec'],
        'validation_selection_sec':frozen['validation_selection_sec'],'prepare_sec':frozen['prepare_sec'],'selection_sec':frozen['selection_sec']}
    for arm in ARMS:
        xs=[r['arms'][arm] for r in rows]
        summary['arms'][arm]={k:sum(x[k] for x in xs)/len(xs) for k in ('official_f1','literal_f1','text_em')}
        summary['arms'][arm].update({k:sum(x.get(k) or 0 for x in xs) for k in ('prompt_tokens','generation_tokens','wall_sec','load_ns')})
        summary['arms'][arm]['errors']=sum('error' in x for x in xs)
        counts={'unanimous_unanswerable':0,'correct_refusal':0,'unanimous_answerable':0,'false_refusal':0,'mixed':0}
        for p in data['papers']:
            for r in p['records']:
                q=next(q for q in source[p['paper_id']]['qas'] if q['question_id']==r['id'])
                states={ref['answer']['unanswerable'] for ref in q['answers']}
                refuse=normalize(r['arms'][arm].get('answer') or '') in ('unknown','unanswerable')
                if states=={True}:counts['unanimous_unanswerable']+=1;counts['correct_refusal']+=refuse
                elif states=={False}:counts['unanimous_answerable']+=1;counts['false_refusal']+=refuse
                else:counts['mixed']+=1
        summary['arms'][arm]['refusal']=counts
    for arm,ref in [('combined','base'),('contract','base'),('expanded','base'),('combined','contract'),('combined','expanded')]:
        rng=random.Random(20261004);boot=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]]
            boot.append(sum(r['arms'][arm]['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
        boot.sort();diff=summary['arms'][arm]['official_f1']-summary['arms'][ref]['official_f1']
        summary['paired'][arm+'_minus_'+ref]={'difference':diff,'paper_ci95':[boot[250],boot[9750]]}
    summary['answer_gate_pass']=summary['paired']['combined_minus_base']['paper_ci95'][0]>0
    (OUT/'reader_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
