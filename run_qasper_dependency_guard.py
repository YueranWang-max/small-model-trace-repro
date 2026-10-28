"""Conservative source guards on a separate reader cohort."""
import json,re,time,sys
from pathlib import Path
import run_qasper_relation_packet as base
from run_qasper_fingerprint import prepare
from run_qasper_contract_replication import contract
from run_qasper_prefix_vs_bm25 import ROOT,sha

OUT=ROOT/'logs/qasper_dependency_guard'
PROTOCOL=ROOT/'protocol_qasper_dependency_guard_20261002.md'
STOP=set('how many much large big is are was were the a an of for in on to and what which do does did they their it used use get can'.split())

def guarded(text,question):
    multi=bool(re.search(r'\btype\b',question,re.I) and re.search(r'\bsize\b',question,re.I))
    terms=set(re.findall(r'[a-z]+',question.lower()))-STOP
    numeric=bool(re.search(r'\d',text))
    protect=multi or (numeric and bool(terms.intersection(re.findall(r'[a-z]+',text.lower()))))
    parts=base.packet(text,question)
    spans=[(0,len(text))] if protect else [(0,next((m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])',text)),len(text)))]+[(r['char_start'],r['char_start']+r['char_length']) for r in parts]
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return [{'text':text[a:b],'char_start':a,'char_length':b-a,'byte_start':len(text[:a].encode()),'byte_length':len(text[a:b].encode()),'paragraph_sha256':sha(text.encode()),'whole_paragraph_guard':protect} for a,b in merged]

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    source=json.loads(base.SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));tasks=[]
    for pid in split['heldout'][56:86]:
        p=prepare(pid,source[pid])
        for q in p['questions']:
            if not re.search(base.SCOPE,q['text'],re.I):continue
            selected=q['rank'][:8];chunks=[];ledger=[]
            full='\n\n'.join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+p['paragraphs'][i]['text'] for i in selected)
            for i in selected:
                text=p['paragraphs'][i]['text'];parts=guarded(text,q['text']);ledger.append({'paragraph_id':i,'fragments':parts})
                for r in parts:
                    assert text[r['char_start']:r['char_start']+r['char_length']]==r['text']
                    assert text.encode()[r['byte_start']:r['byte_start']+r['byte_length']].decode()==r['text']
                chunks.append(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n"+'\n[omitted source between fragments]\n'.join(r['text'] for r in parts))
            tasks.append({'paper_id':pid,'id':q['id'],'title':source[pid]['title'],'question':q['text'],'kind':contract(q['text'])[0],'material':{'full':full,'packet':'\n\n'.join(chunks)},'ledger':ledger,'selected':selected})
            if len(tasks)==24:break
        if len(tasks)==24:break
    previous=json.loads((ROOT/'logs/qasper_relation_packet/frozen.json').read_text(encoding='utf-8'))
    assert not {t['id'] for t in tasks}.intersection(t['id'] for t in previous['tasks'])
    files=[base.SOURCE,PROTOCOL,Path(__file__),ROOT/'run_qasper_relation_packet.py',ROOT/'logs/qasper_fingerprint/split.json']+[ROOT/n for n in ['run_qasper_fingerprint.py','run_stageC.py','run_qasper_format_control.py','run_qasper_contract_replication.py','run_qasper_visible_trace_replication.py','rescore_qasper_demand_transfer.py','repro_lib.py']]
    f={'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files},'tasks':tasks,'prep_sec':time.perf_counter()-start,'cohort_paper_ids':split['heldout'][56:86]}
    fp=OUT/'frozen.json';assert not fp.exists();fp.write_text(json.dumps(f,indent=2,ensure_ascii=False),encoding='utf-8');print({'questions':len(tasks),'papers':len({t['paper_id'] for t in tasks}),'prep_sec':f['prep_sec']})

if __name__=='__main__':
    base.OUT=OUT
    {'freeze':freeze,'run':base.run,'analyze':base.analyze}[sys.argv[1]]()
