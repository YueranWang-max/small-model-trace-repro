"""Own visible trace on installed 1.7B, fixed matched development papers."""
import json,time,sys,random,urllib.request
import run_qasper_visible_trace_replication as old
from rescore_qasper_demand_transfer import references,f1,normalize
ROOT=old.ROOT;OUT=ROOT/'logs/qasper_small_trace_boundary';MODEL='qwen3:1.7b';ARMS=('reread','trace_pre','trace_post','question_pre')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'frozen.json').exists();src=json.loads(old.DEV.read_text(encoding='utf-8'));base=ROOT/'logs/qasper_visible_trace_replication/frozen.json';fr=json.loads(base.read_text(encoding='utf-8'));tags=json.load(urllib.request.urlopen('http://localhost:11434/api/tags'));identity=next(m for m in tags['models'] if m['name']==MODEL);papers=[]
    for p in fr['papers']:
        text='\n\n'.join(old.build_paper(p['paper_id'],src[p['paper_id']])['paragraphs']);assert old.sha(text.encode())==p['source_sha256'];papers.append({**p,'source':text,'qas':[q for q in src[p['paper_id']]['qas'] if q['question_id'] in p['question_ids']]})
    files=[old.DEV,base,ROOT/'run_qasper_small_trace_boundary.py',ROOT/'protocol_qasper_small_trace_boundary_20261003.md',ROOT/'run_qasper_visible_trace_replication.py',ROOT/'run_qasper_outline_route.py',ROOT/'repro_lib.py',ROOT/'rescore_qasper_demand_transfer.py'];d={'model':MODEL,'model_identity':identity,'matched_development_reuse':True,'papers':papers,'hashes':{str(p.relative_to(ROOT)):old.sha(p.read_bytes()) for p in files}};(OUT/'frozen.json').write_text(json.dumps(d,indent=2),encoding='utf-8');print('frozen',len(papers),sum(len(p['qas']) for p in papers),flush=True)

def scored(prompt,q):
    gold=references(q);r=old.call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);a=r['answer'];r['official_f1']=max(f1('Unanswerable' if normalize(a)=='unknown' else a,g) for g in gold);return r

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in fr['hashes'].items():assert old.sha((ROOT/name).read_bytes())==h
    old.MODEL=MODEL;path=OUT/'reader.json';d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':old.sha(fp.read_bytes()),'rows':[]};assert d['frozen_sha256']==old.sha(fp.read_bytes())
    if 'warmup' not in d:d['warmup']=old.call('Reply exactly Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
    def save():path.write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in fr['papers']:
        for q in p['qas']:
            row=next((r for r in d['rows'] if r['id']==q['question_id']),None)
            if row is None:row={'id':q['question_id'],'paper_id':p['paper_id'],'arms':{}};d['rows'].append(row)
            if 'first' not in row:
                query=old.query_text(p['title'],q['question']);prompt='Original paper:\n'+p['source']+'\nQuestion: '+query+'\nRespond with exactly three short lines. Trace: <what fact or relation is needed and where to look>; Evidence Quote: <one exact original-source sentence>; Final Answer: <short answer>. Keep the trace under 80 words. Do not add other lines.';r=scored(prompt,q);trace,trunc=old.extract_trace(r['content']);quote,pid=old.extract_quote(r['content'],old.build_paper(p['paper_id'],json.loads(old.DEV.read_text(encoding='utf-8'))[p['paper_id']])['paragraphs']);row['first']=r;row['feedback']={'trace':trace,'truncated':trunc,'quote':quote};save();print('first',len(d['rows']),bool(trace),flush=True)
        for q in p['qas']:
            row=next(r for r in d['rows'] if r['id']==q['question_id']);index=next(i for i,r in enumerate(d['rows']) if r['id']==row['id']);order=list(ARMS);off=index%4;order=order[off:]+order[:off]
            for arm in order:
                if arm in row['arms']:continue
                query=old.query_text(p['title'],q['question']);prompt=old.second_prompt('reread' if arm=='question_pre' else arm,p['source'],query,row['feedback']['trace'],row['feedback']['quote'])
                if arm=='question_pre':prompt='Question: '+query+'\n'+prompt
                row['arms'][arm]=scored(prompt,q);save();print(index+1,arm,round(row['arms'][arm]['official_f1'],3),flush=True)

def analyze():
    d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));rs=d['rows'];assert len(rs)==18 and all(set(r['arms'])==set(ARMS) for r in rs);groups=[[r for r in rs if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in rs})];out={'questions':18,'papers':6,'model':MODEL,'nonempty_traces':sum(bool(r['feedback']['trace']) for r in rs),'truncated_traces':sum(r['feedback']['truncated'] for r in rs),'exact_quotes':sum(bool(r['feedback']['quote']) for r in rs),'arms':{},'paired':{}}
    for arm in ('first',)+ARMS:
        vs=[r['first'] if arm=='first' else r['arms'][arm] for r in rs];out['arms'][arm]={'f1':sum(v['official_f1'] for v in vs)/18,'tokens':sum((v.get('prompt_tokens') or 0)+(v.get('generation_tokens') or 0) for v in vs),'wall_sec':sum(v['wall_sec'] for v in vs),'errors':sum('error' in v for v in vs),'empty_answers':sum(not v['answer'] for v in vs)}
    for b in ('trace_post','reread','question_pre'):
        rng=random.Random(20261003);bs=[]
        for _ in range(10000):
            sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms']['trace_pre']['official_f1']-r['arms'][b]['official_f1'] for r in sample)/len(sample))
        bs.sort();out['paired']['pre_minus_'+b]={'diff':out['arms']['trace_pre']['f1']-out['arms'][b]['f1'],'ci95':[bs[250],bs[9750]]}
    out['placement_gate']=out['paired']['pre_minus_trace_post']['ci95'][0]>0;out['usefulness_gate']=all(out['paired']['pre_minus_'+b]['ci95'][0]>0 for b in ('reread','question_pre'));out['two_pass_pre_tokens']=out['arms']['first']['tokens']+out['arms']['trace_pre']['tokens'];out['two_pass_pre_wall_sec']=out['arms']['first']['wall_sec']+out['arms']['trace_pre']['wall_sec'];(OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run,'analyze':analyze}[sys.argv[1]]()
