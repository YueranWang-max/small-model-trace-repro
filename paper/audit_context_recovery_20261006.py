"""Independent raw-response audit of the single-case capacity diagnostic."""
from pathlib import Path
import json,hashlib,csv,sys,random
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_confirmatory_20261006 import gold,rescore
OUT=ROOT/'logs/context_recovery_20261006'
REPORT=ROOT/'paper/evidence_context_recovery_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    fr=json.loads((OUT/'frozen.json').read_bytes())
    for name,h in fr['dependencies'].items():assert sha(ROOT/name.replace(chr(92),"/"))==h
    answer=gold(fr['case']);assert answer==set(fr['case']['answer_nodes'])
    old=ROOT/'recovery_bridge_20261003/hybrid_five_trace_replication'
    bundle=json.loads((old/'bundle_00.json').read_bytes())
    assert hashlib.sha256(bundle['bundle'].encode()).hexdigest()==fr['bundle_sha256']
    thinking=[];source_tokens=0
    for entry in fr['original_sources']:
        path=ROOT/entry['path'].replace(chr(92),'/');assert sha(path)==entry['sha256']
        record=json.loads(path.read_bytes());assert record['completion']=='returned'
        assert all(record[k] for k in ['input_qualified','resident_qualified','response_qualified'])
        raw=record['raw'];assert raw['prompt_eval_count']==entry['input_tokens'] and raw['eval_count']==entry['output_tokens']
        source_tokens+=entry['input_tokens']+entry['output_tokens']
        thinking.append(raw['message'].get('thinking','') or '')
    assert len(thinking)==5
    warning='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
    components=[warning,'<first_run_reasoning_traces>']
    for i,t in enumerate(thinking,1):components.append(f'[Trace {i}] '+(t[:50000]+'...' if len(t)>50000 else t).strip())
    assert bundle['bundle']=='\n'.join(components+['</first_run_reasoning_traces>'])
    for i,t in enumerate(thinking,1):
        clipped=t[:50000]+'...' if len(t)>50000 else t
        assert f'[Trace {i}] '+clipped.strip() in bundle['bundle']
    whole=fr['case']['prompt']+'\nReturn exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
    head,tail=whole.rsplit('Operation:',1);question='Operation:'+tail
    prompts={'fresh':whole,'question_first':question+'\n'+whole,'re2':whole+'\n'+whole,
             'prefix':bundle['bundle']+'\n'+whole,'append':head+bundle['bundle']+'\n'+question,
             'probe':bundle['bundle']+'\n'+whole}
    tokenizer=Tokenizer.from_file(str(ROOT/'recovery_bridge_20261003/hybrid_candidate_qualification_20261004/tokenizer.json'))
    system='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
    rng=random.Random(202610198);expected_order=['full_input_probe']
    for repeat in range(5):
        jobs=[j for j in bundle['reader_jobs'] if j['repeat']==repeat];rng.shuffle(jobs)
        expected_order.extend(f'r{repeat}_{j["arm"]}' for j in jobs)
    assert [j['key'] for j in fr['jobs']]==expected_order
    for job in fr['jobs']:
        req=job['request'];cap=1 if job['arm']=='probe' else 32768
        assert req['model']==fr['model'] and req['stream'] is False and req['think'] is True and req['keep_alive']=='10m'
        assert req['messages']==[{'role':'system','content':system},{'role':'user','content':prompts[job['arm']]}]
        assert req['options']=={'temperature':1.,'top_p':.95,'top_k':20,'min_p':0,'presence_penalty':1.5,'frequency_penalty':0,'repeat_penalty':1,'mirostat':0,'num_ctx':196608,'num_predict':cap,'seed':202610199 if job['arm']=='probe' else 202610130+job['repeat']}
        rendered='<|im_start|>system\n'+system.strip()+'<|im_end|>\n<|im_start|>user\n'+prompts[job['arm']].strip()+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
        count=len(tokenizer.encode(rendered,add_special_tokens=False).ids)
        assert count==job['expected_prompt_tokens'] and count+32768+64<=196608
    rows=[];manifest=[];unknown=[];observed_usage=[];unknown_attempted_usage=[];observed_wall=[]
    for job in fr['jobs']:
        path=OUT/(job['key']+'.json')
        if not path.exists():unknown.append(job['key']);continue
        entry=json.loads(path.read_bytes());manifest.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        assert entry['job']==job
        req=job['request'];assert req['messages'][1]['content']==prompts[job['arm']]
        assert req['options']['num_ctx']==196608
        assert req['options']['num_predict']==(1 if job['arm']=='probe' else 32768)
        assert entry['request_sha256']==hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
        # Account for transmitted calls even if their identity/context check fails.
        rawpath=OUT/(job['key']+'.response.json');raw=entry.get('raw',{})
        if rawpath.exists():
            assert json.loads(rawpath.read_bytes())==raw and sha(rawpath)==entry['raw_response_sha256']
            manifest.append({'path':str(rawpath.relative_to(ROOT)),'sha256':sha(rawpath)})
        errorpath=OUT/(job['key']+'.http_error.txt')
        if errorpath.exists():
            assert sha(errorpath)==entry['raw_response_sha256']
            manifest.append({'path':str(errorpath.relative_to(ROOT)),'sha256':sha(errorpath)})
        if 'wall_sec' in entry:observed_wall.append(entry['wall_sec'])
        if type(raw.get('prompt_eval_count')) is int and type(raw.get('eval_count')) is int:
            observed_usage.append({'key':job['key'],'input_tokens':raw['prompt_eval_count'],'output_tokens':raw['eval_count'],'qualified':bool(entry.get('qualified'))})
        elif entry.get('transmission')=='attempting_once':unknown_attempted_usage.append(job['key'])
        if entry['completion']!='returned' or entry.get('error') or not entry.get('qualified'):
            unknown.append(job['key']);continue
        assert raw['model']==fr['model'] and raw['prompt_eval_count']==job['expected_prompt_tokens'] and raw.get('done') is True
        assert type(raw['eval_count']) is int and 0<=raw['eval_count']<=req['options']['num_predict']
        assert raw['prompt_eval_count']+raw['eval_count']<=196608
        assert any(m.get('digest')==fr['digest'] and m.get('context_length')==196608 for m in entry['runtime_ps']['models'])
        values=rescore(raw,answer) if job['arm']!='probe' else {'f1':None,'em':None,'valid':None}
        if job['arm']!='probe':
            original=next(r for r in json.loads((OUT/'results.json').read_bytes()) if r['job']['key']==job['key'])
            for metric in ['f1','em','valid']:assert original['score'][metric]==values[metric]
        rows.append({'key':job['key'],'arm':job['arm'],'repeat':job['repeat'],**values,
                     'input_tokens':raw['prompt_eval_count'],'output_tokens':raw['eval_count'],'wall_sec':entry['wall_sec']})
    complete=len(rows)==26 and not unknown
    summary={'complete':complete,'qualified_records':len(rows),'unknown':unknown,
             'case_row_index':fr['case']['row_index'],'context':196608,
             'capacity_preflight':fr['capacity_preflight'],'source_preparation_tokens':source_tokens,
             'new_research_tokens_known':sum(r['input_tokens']+r['output_tokens'] for r in observed_usage),
             'usage_records':observed_usage,'unknown_attempted_usage':unknown_attempted_usage,
             'observed_request_wall_seconds':sum(observed_wall),
             'scope':'One deliberately selected prior overflow case; no population accuracy inference',
             'planned_payload_order_and_tokenization_checks':'all 26 passed independently before scoring'}
    arms=[]
    if complete:
        for arm in ['prefix','append','fresh','question_first','re2']:
            v=[r for r in rows if r['arm']==arm];assert len(v)==5
            mean=lambda k:sum(r[k] for r in v)/5
            arms.append({'arm':arm,'f1':mean('f1'),'em':mean('em'),'valid_count':sum(r['valid'] for r in v),
                         'mean_reader_tokens':mean('input_tokens')+mean('output_tokens'),
                         'deployment_tokens':(source_tokens if arm in ['prefix','append'] else 0)+mean('input_tokens')+mean('output_tokens')})
        summary['arms']=arms
    for name,data in [('audited_calls.csv',rows),('arm_summary.csv',arms)]:
        if data:
            with (REPORT/name).open('w',encoding='utf-8',newline='') as s:
                writer=csv.DictWriter(s,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
