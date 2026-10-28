"""Independent archived audit; no inference, runner imports, or label reuse.

Run only after the native process is terminal. Missing outcomes remain bounded.
"""
import pathlib, json, hashlib, re, collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'hybrid_five_trace_replication'
META = ROOT / 'hybrid_candidate_qualification_20261004'
MODEL = 'qwen3.5:4b'
DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
SYSTEM = 'You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT = 'Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
PD = 'Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
ARMS = ['fresh', 'question_first', 're2', 'prefix', 'append']
OPTIONS = dict(temperature=1.0, top_p=.95, top_k=20, min_p=0, presence_penalty=1.5,
               frequency_penalty=0, repeat_penalty=1, mirostat=0, num_ctx=131072, num_predict=32768)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def edges(text):
    block = text.split('Here is the graph to operate on:', 1)[1].split('Operation:', 1)[0]
    parsed = re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$', block, re.M)
    assert parsed
    return parsed

def signature(text):
    return hashlib.sha256(json.dumps(sorted(edges(text)), separators=(',', ':')).encode()).hexdigest()

def gold(case):
    parsed, question = edges(case['prompt']), case['prompt'].rsplit('Operation:', 1)[1]
    if case['problem_type'] == 'parents':
        node = re.search(r'Find the parents of node ([0-9a-f]{10})', question)[1]
        return {u for u, v in parsed if v == node and u != node}
    start, depth = re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)', question).groups()
    adjacency = collections.defaultdict(set)
    for u, v in parsed:
        adjacency[u].add(v)
    distances, queue = {start: 0}, collections.deque([start])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)
    return {v for v, d in distances.items() if d == int(depth) and v != start}

def fail(message):
    raise ValueError(message)

def main():
    assert not (OUT / 'independent_audit.json').exists(), 'Preserve audit evidence.'
    completion = json.loads((OUT / 'completion.json').read_bytes())
    frozen = json.loads((OUT / 'frozen.json').read_bytes())
    assert all(sha(pathlib.Path(p)) == h for p, h in frozen['dependencies'].items())
    assert frozen['model'] == MODEL and frozen['digest'] == DIGEST and frozen['runtime']['version'] == '0.35.1'
    assert frozen['ctx'] == 131072 and frozen['cap'] == 32768 and frozen['margin'] == 64
    assert frozen['pd'] == PD and frozen['arm_order'] == ARMS and frozen['trace_char_cap'] == 50000
    assert frozen['analysis']['status'] == 'frozen_before_first_call'
    assert sha(ROOT / 'graphwalks_128k_and_shorter.parquet') == '54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    rows = pq.read_table(ROOT / 'graphwalks_128k_and_shorter.parquet').to_pylist()
    historical = set()
    for path in sorted((ROOT.parent / 'logs').glob('official_graph_*/frozen.json')):
        archive = json.loads(path.read_bytes())
        for record in archive.get('selected', archive.get('cases', [])):
            historical.add(signature(record.get('paper_row', record)['prompt']))
    assert len(historical) == 58
    bridge = json.loads((ROOT / 'bridge_run/frozen.json').read_bytes())['cases']
    qualification = json.loads((ROOT / 'hybrid_source_qualification/frozen.json').read_bytes())['cases']
    excluded = historical | {signature(c['prompt']) for c in bridge + qualification}
    assert len(excluded) == 82
    selected = []
    for lo, hi in [(14000, 16000)]:
        for kind in ['bfs']:
            count = 0
            for index, row in enumerate(rows):
                if row['problem_type'] != kind or not lo <= row['prompt_chars'] < hi:
                    continue
                sig = signature(row['prompt'])
                if sig in excluded:
                    continue
                selected.append(index)
                excluded.add(sig)
                count += 1
                if count == 16:
                    break
            assert count == 16
    cases = frozen['cases']
    assert selected == [c['row_index'] for c in cases]
    assert len({signature(c['prompt']) for c in cases}) == 16
    for c in cases:
        assert {k: v for k, v in c.items() if k not in ['row_index', 'length_bin']} == rows[c['row_index']]
        assert gold(c) == set(c['answer_nodes'])
    plan = [dict(ordinal=1, case_index=None, row_index=None, phase='input_calibration', arm='calibration', repeat=0, seed=202610119)]
    for index, case in enumerate(cases):
        for repeat in range(5):
            plan.append(dict(ordinal=len(plan)+1, case_index=index, row_index=case['row_index'], phase='source', arm='source', repeat=repeat, seed=202610120+repeat))
        for repeat in range(5):
            order = ARMS[(index+repeat) % 5:] + ARMS[:(index+repeat) % 5]
            for arm in order:
                plan.append(dict(ordinal=len(plan)+1, case_index=index, row_index=case['row_index'], phase='reader', arm=arm, repeat=repeat, seed=202610130+repeat))
    assert plan == frozen['plan'] and len(plan) == 481
    env = Environment(undefined=StrictUndefined, extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = fail
    template = env.from_string(json.loads((META / 'tokenizer_config.json').read_bytes())['chat_template'])
    tokenizer = Tokenizer.from_file(str(META / 'tokenizer.json'))

    def reconstruct(p, user, cap=32768):
        request = dict(model=MODEL, messages=[dict(role='system', content=SYSTEM), dict(role='user', content=user)],
                       think=True, stream=False, options={**OPTIONS, 'seed': p['seed'], 'num_predict': cap})
        rendered = template.render(messages=request['messages'], tools=None, add_generation_prompt=True, enable_thinking=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        return dict(**p, request=request, expected_prompt_tokens=count,
                    request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                    input_plus_output_reserve=count+cap+64)

    expected_jobs = {}
    calibration = reconstruct(plan[0], 'A directed graph has the nodes A and B and no edges. Find the parents of node B.\n'+FORMAT, 1)
    assert calibration == frozen['calibration_job']
    expected_jobs[1] = calibration
    source_jobs = []
    for p in plan:
        if p['phase'] == 'source':
            j = reconstruct(p, cases[p['case_index']]['prompt']+'\n'+FORMAT)
            assert j['input_plus_output_reserve'] <= 131072
            expected_jobs[p['ordinal']] = j
            source_jobs.append(j)
    assert source_jobs == frozen['source_jobs']
    joint_fits = {}
    bundle_hashes = []
    for index, case in enumerate(cases):
        path = OUT / f'bundle_{index:02d}.json'
        if not path.exists():
            continue
        bundle_info = json.loads(path.read_bytes())
        sources = [p for p in plan if p['phase'] == 'source' and p['case_index'] == index]
        traces = []
        for p in sources:
            record = json.loads((OUT / f"attempt_{p['ordinal']:03d}.json").read_bytes())
            assert record['completion'] == 'returned'
            traces.append((record['raw'].get('message') or {}).get('thinking', '') or '')
        lines, clips = [PD, '<first_run_reasoning_traces>'], []
        for source_index, text in enumerate(traces, 1):
            retained = text[:50000]+'...' if len(text) > 50000 else text
            lines.append(f'[Trace {source_index}] '+retained.strip())
            clips.append(dict(source_index=source_index, full_characters=len(text), retained_characters_before_strip=len(retained), clipped=len(text) > 50000))
        bundle = '\n'.join(lines+['</first_run_reasoning_traces>'])
        bundle_hash = hashlib.sha256(bundle.encode()).hexdigest()
        assert bundle_info['bundle'] == bundle and bundle_info['bundle_sha256'] == bundle_hash
        assert bundle_info['clips'] == clips and bundle_info['source_ordinals'] == [p['ordinal'] for p in sources]
        assert bundle_info['row_index'] == case['row_index']
        whole = case['prompt']+'\n'+FORMAT
        head, suffix = whole.rsplit('Operation:', 1)
        q = 'Operation:'+suffix
        assert head+q == whole
        users = dict(fresh=whole, question_first=q+'\n'+whole, re2=whole+'\n'+whole,
                     prefix=bundle+'\n'+whole, append=head+bundle+'\n'+q)
        assert all(user.endswith(q) for user in users.values())
        readers = [reconstruct(p, users[p['arm']]) for p in plan if p['phase'] == 'reader' and p['case_index'] == index]
        assert readers == bundle_info['reader_jobs']
        pair_ok = all(j['input_plus_output_reserve'] <= 131072 for j in readers if j['arm'] in ['prefix', 'append'])
        assert pair_ok == bundle_info['trace_pair_fits']
        joint_fits[index] = pair_ok
        assert bundle_info['bundle_and_all_reader_job_preparation_seconds']>=0
        bundle_hashes.append(dict(case_index=index, sha256=bundle_hash, clips=clips, pair_fits=pair_ok,
                                 preparation_seconds=bundle_info['bundle_and_all_reader_job_preparation_seconds']))
        for j in readers:
            expected_jobs[j['ordinal']] = j
    results = []
    for p in plan:
        r = dict(**p, attempted=False, response_known=False, score_determined=False,
                 qualified=False, usage_known=False, completion='not_reached')
        path = OUT / f"attempt_{p['ordinal']:03d}.json"
        if not path.exists():
            results.append(r)
            continue
        record = json.loads(path.read_bytes())
        assert p['ordinal'] in expected_jobs, 'Reader cannot precede a validated source bundle.'
        j = expected_jobs[p['ordinal']]
        assert record['job'] == j
        r.update(completion=record['completion'], attempted=record['transmission']=='attempting_once',
                 ledger_sha256=sha(path), client_seconds=record.get('elapsed_seconds'),
                 qualification_metadata_seconds=record.get('qualification_metadata_seconds'))
        decline = p['phase']=='reader' and ((p['arm'] in ['prefix', 'append'] and not joint_fits[p['case_index']]) or j['input_plus_output_reserve'] > 131072)
        if record['completion'] == 'budget_declined':
            assert decline and record['transmission'] == 'not_sent'
            r.update(score_determined=True, normal_parse=False, prediction=[], f1=0., em=0,
                     usage_known=True, prompt_tokens=0, output_tokens=0, client_seconds=0.)
            results.append(r)
            continue
        assert not decline and r['attempted']
        if record['completion'] == 'client_error':
            r.update(score_determined=True, normal_parse=False, prediction=[], f1=0., em=0, error=record['error'])
            results.append(r)
            continue
        if record['completion'] != 'returned':
            results.append(r)
            continue
        raw = record['raw']
        ps = record.get('runtime_ps', {}).get('models', [])
        input_ok = raw.get('prompt_eval_count') == j['expected_prompt_tokens']
        resident = any(m.get('digest')==DIGEST and m.get('context_length', 0)>=131072 for m in ps)
        response_ok = raw.get('done') is True and raw.get('model') == MODEL and not raw.get('error')
        if 'input_qualified' in record:
            assert input_ok == record['input_qualified'] and resident == record['resident_qualified'] and response_ok == record['response_qualified']
        qualified = input_ok and resident and response_ok
        usage = all(isinstance(raw.get(k), int) and raw[k]>=0 for k in ['prompt_eval_count', 'eval_count'])
        if usage:
            assert raw['eval_count'] <= j['request']['options']['num_predict']
        line = ((raw.get('message') or {}).get('content', '') or '').rstrip().split('\n')[-1].strip()
        match = re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]', line)
        normal = bool(match) and raw.get('done_reason') == 'stop' and not raw.get('error')
        prediction = {s.strip() for s in match[1].split(',') if s.strip()} if normal else set()
        answer = gold(cases[p['case_index']]) if p['case_index'] is not None else set()
        f1 = (2*len(prediction & answer)/(len(prediction)+len(answer)) if prediction or answer else 1.) if normal else 0.
        r.update(response_known=True, score_determined=qualified, qualified=qualified, usage_known=usage,
                 normal_parse=normal, prediction=sorted(prediction), f1=f1, em=int(normal and prediction==answer),
                 done_reason=raw.get('done_reason'), prompt_tokens=raw.get('prompt_eval_count'), output_tokens=raw.get('eval_count'))
        for key in ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']:
            r[key+'_seconds'] = raw[key]/1e9 if isinstance(raw.get(key), int) else None
        results.append(r)
    # A stopped run must be a prefix, never a collection of selective outcomes.
    present = [r['ordinal'] for r in results if r['completion'] != 'not_reached']
    assert present == list(range(1, len(present)+1))
    if completion['primary_complete']:
        assert len(present) == 481
    else:
        assert completion['last_ordinal'] == max(present)
    scored = [r for r in results if r['phase'] != 'input_calibration']
    cost = [r for r in results if r['usage_known']]
    output = dict(audit_passed=True, primary_complete=completion['primary_complete'],
                  planned_scored_slots=480, planned_calibration_slots=1,
                  known_scored_outcomes=sum(r['score_determined'] for r in scored),
                  unknown_scored_outcomes=sum(not r['score_determined'] for r in scored),
                  actual_scored_transmissions=sum(r['attempted'] for r in scored),
                  budget_declined_slots=sum(r['completion']=='budget_declined' for r in scored),
                  known_research_tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in cost),
                  unknown_attempted_usage_calls=sum(r['attempted'] and not r['usage_known'] for r in results),
                  preparation_seconds=frozen['preparation_before_freeze_serialization_seconds'],
                  bundle_and_all_reader_job_preparation_seconds=sum(b['preparation_seconds'] for b in bundle_hashes),
                  full_research_workflow_seconds=completion.get('full_research_workflow_seconds'),
                  bundles=bundle_hashes, records=results,
                  limitations=['Budget declines are known deployment failures, not transmitted unknown answers.',
                               'No source answer is serialized into native trace bundles.',
                               'Fresh source-only sampling is distinct from full-context Re2.',
                               'Sixteen graph clusters, not 480 independent graph cases.',
                               'Single pilot cannot establish important replicable progress.'])
    (OUT / 'independent_audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['records', 'bundles']}, indent=2))

if __name__ == '__main__':
    main()
