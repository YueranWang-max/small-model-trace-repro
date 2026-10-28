"""Audit primary original plus previously unattempted continuation; never impute unknown 020."""
import pathlib,json,hashlib,re,csv,statistics,collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parent;ORIG=ROOT/'bridge_run';CONT=ROOT/'bridge_continuation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((ORIG/'frozen.json').read_text(encoding='utf-8'))
continuation=json.loads((CONT/'frozen_continuation.json').read_text(encoding='utf-8'))
assert (CONT/'completion.json').exists(),'Inference must finish before audit.'
assert sha(ORIG/'frozen.json')==continuation['original_frozen_sha256']
assert sha(ROOT/'run_bridge.py')==frozen['runner_sha256']
assert sha(ROOT/'bridge_protocol.md')==frozen['protocol_sha256']
assert sha(ROOT/'continue_bridge.py')==continuation['script_sha256']
assert sha(ROOT/'continuation_protocol.md')==continuation['protocol_sha256']
for name,expected in [('graphwalks_128k_and_shorter.parquet','54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'),('tokenizer.json','aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4')]:assert sha(ROOT/name)==expected
blocks=[c['prompt'].split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0] for c in frozen['cases']]
assert len(blocks)==len(set(blocks))==16
historical_path=ROOT/'historical_manifest_exclusion_check_20261004.json'
historical=json.loads(historical_path.read_text(encoding='utf-8'))
assert historical['all_checks_pass'] and historical['bridge_frozen_sha256']==sha(ORIG/'frozen.json')
for archive in historical['archives']:assert sha(pathlib.Path(archive['path']))==archive['sha256']
rows=pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist()
assert len(rows)==750
def edge_block(text):return text.split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0]
def independent_answer(case):
    edges=re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$',edge_block(case['prompt']),re.M)
    assert edges
    operation=case['prompt'].rsplit('Operation:',1)[1]
    if case['problem_type']=='parents':
        target=re.search(r'Find the parents of node ([0-9a-f]{10})',operation)[1]
        return {u for u,v in edges if v==target and u!=target}
    target,depth=re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',operation).groups()
    visited={target};frontier={target}
    for _ in range(int(depth)):
        frontier={v for u,v in edges if u in frontier and v not in visited}
        visited.update(frontier)
    return frontier-{target}
excluded={edge_block(rows[i]['prompt']) for i in historical['historical_rows']}
selected_indices=[]
for lo,hi in [(2000,3500),(14000,16000)]:
    for typ in ['parents','bfs']:
        matches=[i for i,r in enumerate(rows) if lo<=r['prompt_chars']<hi and r['problem_type']==typ and edge_block(r['prompt']) not in excluded]
        selected_indices.extend(matches[:4])
assert selected_indices==[c['row_index'] for c in frozen['cases']]
for case in frozen['cases']:
    assert {k:v for k,v in case.items() if k not in ['row_index','bin']}==rows[case['row_index']]
    assert independent_answer(case)==set(case['answer_nodes'])
source_character_metadata=[dict(row_index=c['row_index'],catalogue_prompt_chars=c['prompt_chars'],actual_source_characters=len(c['prompt'])) for c in frozen['cases']]
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
tok=Tokenizer.from_file(str(ROOT/'tokenizer.json'));expected_schedule=[]
models=['qwen3:1.7b','qwen3:8b']
for n,case in enumerate(frozen['cases']):
    user=case['prompt']+'\n'+FORMAT
    rendered='<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\n'+user+' /think<|im_end|>\n<|im_start|>assistant\n'
    count=len(tok.encode(rendered,add_special_tokens=False).ids);assert count+8192+64<=32768
    for model in (models if n%2==0 else models[::-1]):
        for seed in [202610090,202610091]:expected_schedule.append(dict(row_index=case['row_index'],model=model,seed=seed,user=user,expected_prompt_tokens=count,gold=case['answer_nodes']))
assert expected_schedule==frozen['schedule']
records=[]
for ordinal,job in enumerate(frozen['schedule'],1):
    path=(ORIG if ordinal<=20 else CONT)/f'attempt_{ordinal:03d}.json'
    rec=json.loads(path.read_text(encoding='utf-8'));assert rec['job']==job and rec['ordinal']==ordinal
    out=dict(ordinal=ordinal,row_index=job['row_index'],model=job['model'],seed=job['seed'],status=rec['status'],phase='original' if ordinal<=20 else 'continuation',response_known='response' in rec,error=rec.get('error'))
    if rec['status'] in ['attempting_once','not_yet_attempted']:
        out.update(outcome_known=False,score_determined=False,usage_known=False);records.append(out);continue
    assert rec['status'] in ['returned','request_error'] and rec['end_time']>=rec['start_time']
    assert ('response' in rec)==(rec['status']=='returned')
    raw=rec.get('response',{})
    if raw:assert raw.get('model')==job['model'] and raw.get('done') is True
    content=raw.get('message',{}).get('content','').strip();line=content.splitlines()[-1] if content else ''
    match=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',line.strip());valid=bool(match and raw.get('done_reason')=='stop' and not raw.get('error'))
    prediction={n.strip() for n in match.group(1).split(',') if n.strip()} if valid else set();gold=set(job['gold'])
    f1=(2*len(prediction&gold)/(len(prediction)+len(gold)) if prediction or gold else 1) if valid else 0
    known=isinstance(raw.get('prompt_eval_count'),int) and isinstance(raw.get('eval_count'),int)
    qualified=raw.get('prompt_eval_count')==job['expected_prompt_tokens'] and bool(rec.get('input_qualified'))
    if raw:
        ps=json.loads(((ORIG if ordinal<=20 else CONT)/f'ps_{ordinal:03d}.json').read_text(encoding='utf-8'))
        resident=any(m.get('name')==job['model'] and m.get('digest')==frozen['digests'][job['model']] and m.get('context_length',0)>=32768 for m in ps.get('models',[]))
        assert resident==bool(rec.get('resident_qualified'))
    else:resident=False
    out.update(outcome_known=True,score_determined=True,usage_known=known,normal_parse=valid,f1=f1,em=int(valid and prediction==gold),prediction=sorted(prediction),prompt_tokens=raw.get('prompt_eval_count'),output_tokens=raw.get('eval_count'),done_reason=raw.get('done_reason'),qualified=qualified and resident)
    records.append(out)
assert [r['ordinal'] for r in records if not r['outcome_known']]==[20]
summary={}
for model in frozen['digests']:
    rs=[r for r in records if r['model']==model];known=[r for r in rs if r['outcome_known']];missing=len(rs)-len(known)
    s=dict(planned_calls=32,known_outcomes=len(known),unknown_outcomes=missing,normal_parse=sum(r['normal_parse'] for r in known),known_usage_calls=sum(r['usage_known'] for r in rs),known_token_lower_bound=sum(r['prompt_tokens']+r['output_tokens'] for r in known if r['usage_known']),all_known_responses_qualified=all(r['qualified'] for r in known),available_call_mean_f1=statistics.mean(r['f1'] for r in known),available_call_mean_em=statistics.mean(r['em'] for r in known))
    for metric in ['f1','em']:
        value=sum(r[metric] for r in known);s[f'primary_{metric}_missingness_bounds']=[value/32,(value+missing)/32]
    s['frozen_global_gate_passed']=missing==0 and s['normal_parse']>=31 and s['primary_f1_missingness_bounds'][0]>=.70 and s['primary_em_missingness_bounds'][0]>=.50 and s['all_known_responses_qualified']
    s['returned_responses']=sum(r['response_known'] for r in rs)
    s['all_returned_responses_qualified']=all(r['qualified'] for r in known if r['response_known'])
    s['qualification_field_note']='Legacy all_known_responses_qualified includes known client failures; all_returned_responses_qualified covers actual returned responses only.'
    s['client_request_errors']=sum(r['status']=='request_error' for r in rs)
    s['unknown_usage_calls']=sum(not r['usage_known'] for r in rs)
    s['length_terminations']=sum(r.get('done_reason')=='length' for r in rs)
    summary[model]=s
result=dict(primary_complete=False,unknown_original_attempts=[20],planned_calls=64,known_outcomes=63,known_outcome_definition='Persisted client outcomes, including request errors scored zero by the frozen deployment rule; not 63 returned model answers.',summary=summary,records=records,main_manuscript_updated=False,historical_manifest_check_pending=False,historical_manifest_check_timing='Post-freeze against ten restored official-study archives; original frozen pending field retained.',historical_manifest_check_sha256=sha(historical_path),selected_labels_independently_recomputed=16,selection_order_reconstructed=True,input_token_schedule_reconstructed=True,source_character_metadata=source_character_metadata,character_metadata_note='Selection uses the official catalogue prompt_chars, which differs from actual full source length; both are retained. Runtime qualification uses tokenization of the actual input.',request_payload_evidence='Frozen script hashes and job records reconstruct intended payload; full wire payload was not separately persisted by this runner.')
(ROOT/'continued_bridge_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'continued_bridge_cases.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['row_index','type','bin_lower','model','known_repeats','mean_f1_available','mean_em_available','mean_tokens_available'])
    for case in frozen['cases']:
        for model in frozen['digests']:
            rs=[r for r in records if r['row_index']==case['row_index'] and r['model']==model and r['outcome_known']]
            w.writerow([case['row_index'],case['problem_type'],case['bin'][0],model,len(rs),statistics.mean(r['f1'] for r in rs),statistics.mean(r['em'] for r in rs),statistics.mean(r['prompt_tokens']+r['output_tokens'] for r in rs) if all(r['usage_known'] for r in rs) else ''])
print(json.dumps(summary,indent=2))
with (ROOT/'continued_bridge_cost_lower_bounds.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['model','known_token_lower_bound','unknown_usage_calls','returned_responses','client_request_errors'])
    for model,s in summary.items():w.writerow([model,s['known_token_lower_bound'],s['unknown_usage_calls'],s['returned_responses'],s['client_request_errors']])
