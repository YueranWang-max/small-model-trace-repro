"""One-shot native five-trace study. Preparation only until protocol is frozen.

Never launch while another inference process is active. No retries or resumes.
The source-performance gate is recorded but is not a launch prerequisite.
"""
import pathlib, json, hashlib, time
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import run_hybrid_source_qualification as base

ROOT = pathlib.Path(__file__).resolve().parent
META = ROOT / 'hybrid_candidate_qualification_20261004'
OUT = ROOT / 'hybrid_five_trace_replication'
PROTOCOL = ROOT / 'hybrid_five_trace_replication_protocol.md'
ANALYSIS = ROOT / 'hybrid_trace_analysis_spec.json'
PD = 'Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
ARMS = ['fresh', 'question_first', 're2', 'prefix', 'append']
CTX, CAP, MARGIN = 131072, 32768, 64
OPTIONS = {**base.OPTIONS, 'num_ctx': CTX, 'num_predict': CAP}

def digest_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def serialize(traces):
    lines = [PD, '<first_run_reasoning_traces>']
    metadata = []
    for index, trace in enumerate(traces, 1):
        retained = trace[:50000] + '...' if len(trace) > 50000 else trace
        lines.append(f'[Trace {index}] ' + retained.strip())
        metadata.append(dict(source_index=index, full_characters=len(trace),
                             retained_characters_before_strip=len(retained),
                             clipped=len(trace) > 50000))
    bundle = '\n'.join(lines + ['</first_run_reasoning_traces>'])
    return bundle, metadata

def prompts(case, bundle):
    whole = case['prompt'] + '\n' + base.FORMAT
    head, tail = whole.rsplit('Operation:', 1)
    question = 'Operation:' + tail
    assert head + question == whole
    # Separators are frozen. Every arm ends with this identical question.
    return dict(fresh=whole, question_first=question + '\n' + whole,
                re2=whole + '\n' + whole, prefix=bundle + '\n' + whole,
                append=head + bundle + '\n' + question)

def payload(user, seed, cap=CAP):
    return dict(model=base.MODEL,
                messages=[dict(role='system', content=base.SYSTEM),
                          dict(role='user', content=user)],
                think=True, stream=False,
                options={**OPTIONS, 'seed': seed, 'num_predict': cap})

def job(plan, user, tokenizer, cap=CAP):
    request = payload(user, plan['seed'], cap)
    count = len(tokenizer.encode(base.rendered(user), add_special_tokens=False).ids)
    return dict(**plan, request=request, expected_prompt_tokens=count,
                request_sha256=digest_object(request),
                input_plus_output_reserve=count + cap + MARGIN)

def execute(request_job):
    path = OUT / f"attempt_{request_job['ordinal']:03d}.json"
    assert not path.exists(), 'No primary call may be retried or overwritten.'
    record = dict(job=request_job, completion='pending', transmission='not_yet_attempted')
    base.save(path, record)
    record.update(transmission='attempting_once', started_at=time.time(),
                  monotonic_started=time.monotonic())
    base.save(path, record)
    try:
        raw = base.request('chat', request_job['request'], timeout=7200)
    except Exception as error:
        record.update(completion='client_error', error=repr(error), ended_at=time.time(),
                      elapsed_seconds=time.monotonic() - record['monotonic_started'])
        base.save(path, record)
        return record, False
    record.update(completion='returned', raw=raw, ended_at=time.time(),
                  elapsed_seconds=time.monotonic() - record['monotonic_started'])
    base.save(path, record)  # Persist the actual answer before fetching metadata.
    qualification_started = time.monotonic()
    try:
        ps = base.request('ps')
    except Exception as error:
        record['qualification_metadata_error'] = repr(error)
        base.save(path, record)
        return record, False
    record.update(runtime_ps=ps, qualification_metadata_seconds=time.monotonic()-qualification_started,
                  input_qualified=raw.get('prompt_eval_count') == request_job['expected_prompt_tokens'],
                  resident_qualified=any(m.get('digest') == base.DIGEST and m.get('context_length', 0) >= CTX
                                        for m in ps.get('models', [])),
                  response_qualified=raw.get('done') is True and raw.get('model') == base.MODEL and not raw.get('error'))
    base.save(path, record)
    print(request_job['ordinal'], request_job['row_index'], request_job['arm'],
          request_job['repeat'], raw.get('done_reason'), raw.get('prompt_eval_count'),
          raw.get('eval_count'), flush=True)
    return record, all(record[k] for k in ['input_qualified', 'resident_qualified', 'response_qualified'])

def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Never resume or replace the primary study.'
    assert PROTOCOL.exists(), 'Preparation draft is not a frozen launch protocol.'
    source_completion = json.loads((ROOT / 'hybrid_source_qualification/completion.json').read_bytes())
    source_audit = json.loads((ROOT / 'hybrid_source_qualification/independent_audit.json').read_bytes())
    assert 'primary_complete' in source_completion and source_audit['audit_passed']
    # A terminal, fully audited transport failure is not a source-performance
    # veto on a separate experiment. Keep its missingness and cost as reported.
    assert all(r['qualified'] for r in source_audit['records'] if r['response_known'])
    assert source_audit['records'][0]['qualified'] and source_audit['records'][0]['usage_known']
    preflight_path = ROOT / 'hybrid_trace_offline_preflight.json'
    preflight = json.loads(preflight_path.read_bytes())
    assert preflight['offline_preflight_passed'] and preflight['scored_inference_calls'] == 0
    assert all(base.sha(ROOT / name) == expected for name, expected in preflight['dependency_sha256'].items())
    analysis = json.loads(ANALYSIS.read_bytes())
    assert analysis['status'] == 'frozen_before_first_call'
    assert analysis['graph_clusters'] == 16 and analysis['reader_repeats'] == 5
    rows = pq.read_table(ROOT / 'graphwalks_128k_and_shorter.parquet').to_pylist()
    assert len(rows) == 750
    assert base.sha(ROOT / 'graphwalks_128k_and_shorter.parquet') == '54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    historical = json.loads((ROOT / 'historical_manifest_exclusion_check_20261004.json').read_bytes())
    bridge = json.loads((ROOT / 'bridge_run/frozen.json').read_bytes())
    qualification = json.loads((ROOT / 'hybrid_source_qualification/frozen.json').read_bytes())
    excluded = {base.signature(rows[i]['prompt']) for i in historical['historical_rows']}
    excluded |= {base.signature(c['prompt']) for c in bridge['cases'] + qualification['cases']}
    assert len(excluded) == 82
    cases = []
    for lo, hi in [(14000, 16000)]:
        for kind in ['bfs']:
            selected = []
            for index, row in enumerate(rows):
                if row['problem_type'] != kind or not lo <= row['prompt_chars'] < hi:
                    continue
                signature = base.signature(row['prompt'])
                if signature in excluded:
                    continue
                selected.append(dict(row_index=index, length_bin=[lo, hi], **row))
                excluded.add(signature)
                if len(selected) == 16:
                    break
            assert len(selected) == 16
            cases.extend(selected)
    assert len({base.signature(c['prompt']) for c in cases}) == 16
    assert [c['row_index'] for c in cases] == preflight['selected_rows']
    for case in cases:
        assert base.oracle(case) == set(case['answer_nodes'])
    inventory, version = base.request('tags'), base.request('version')
    assert version['version'] == '0.35.1'
    assert next(m['digest'] for m in inventory['models'] if m['name'] == base.MODEL) == base.DIGEST
    tokenizer = Tokenizer.from_file(str(META / 'tokenizer.json'))
    plan = [dict(ordinal=1, case_index=None, row_index=None, phase='input_calibration',
                 arm='calibration', repeat=0, seed=202610119)]
    for index, case in enumerate(cases):
        for repeat in range(5):
            plan.append(dict(ordinal=len(plan)+1, case_index=index, row_index=case['row_index'],
                             phase='source', arm='source', repeat=repeat, seed=202610120+repeat))
        for repeat in range(5):
            shift = (index + repeat) % 5
            for arm in ARMS[shift:] + ARMS[:shift]:
                plan.append(dict(ordinal=len(plan)+1, case_index=index, row_index=case['row_index'],
                                 phase='reader', arm=arm, repeat=repeat, seed=202610130+repeat))
    assert len(plan) == 481
    calibration_user = 'A directed graph has the nodes A and B and no edges. Find the parents of node B.\n' + base.FORMAT
    calibration = job(plan[0], calibration_user, tokenizer, cap=1)
    base_jobs = [job(p, cases[p['case_index']]['prompt'] + '\n' + base.FORMAT, tokenizer)
                 for p in plan if p['phase'] == 'source']
    assert all(j['input_plus_output_reserve'] <= CTX for j in base_jobs)
    deps = [pathlib.Path(__file__), pathlib.Path(base.__file__), PROTOCOL, ANALYSIS,
            ROOT / 'analyze_hybrid_five_trace_replication.py',
            ROOT / 'audit_hybrid_five_trace_replication.py',
            ROOT / 'historical_manifest_exclusion_check_20261004.json',
            ROOT / 'hybrid_source_qualification/frozen.json',
            ROOT / 'hybrid_source_qualification/independent_audit.json',
            ROOT / 'bridge_run/frozen.json', META / 'tokenizer.json', META / 'tokenizer_config.json',
            META / 'official_README.md', META / 'offline_render_checks.json',
            META / 'padding_difference_checks.json', META / 'runtime_qwen35_renderer.go',
            preflight_path, ROOT / 'offline_hybrid_trace_preflight.py']
    hashes = {str(p): base.sha(p) for p in deps}  # All implementations must exist before freeze.
    OUT.mkdir()
    base.save(OUT / 'frozen.json', dict(dependencies=hashes, cases=cases, plan=plan,
              source_jobs=base_jobs, calibration_job=calibration, model=base.MODEL,
              digest=base.DIGEST, runtime=version, inventory=inventory, analysis=analysis,
              ctx=CTX, cap=CAP, margin=MARGIN, pd=PD, arm_order=ARMS,
              trace_char_cap=50000, source_performance_gate_passed=source_audit['frozen_gate_passed'],
              preparation_before_freeze_serialization_seconds=time.monotonic()-started))
    _, qualified = execute(calibration)
    if not qualified:
        base.save(OUT / 'completion.json', dict(primary_complete=False, reason='calibration_failure', last_ordinal=1))
        return
    for index, case in enumerate(cases):
        source_records = []
        for source_job in [j for j in base_jobs if j['case_index'] == index]:
            record, qualified = execute(source_job)
            if not qualified:
                base.save(OUT / 'completion.json', dict(primary_complete=False, reason='source_execution_or_qualification_failure', last_ordinal=source_job['ordinal']))
                return
            source_records.append(record)
        traces = [(r['raw'].get('message') or {}).get('thinking', '') or '' for r in source_records]
        assert all(isinstance(t, str) for t in traces)
        bundle_preparation_started = time.monotonic()
        bundle, clips = serialize(traces)
        users = prompts(case, bundle)
        reader_jobs = [job(p, users[p['arm']], tokenizer) for p in plan
                       if p['phase'] == 'reader' and p['case_index'] == index]
        pair_fits = all(j['input_plus_output_reserve'] <= CTX for j in reader_jobs
                        if j['arm'] in ['prefix', 'append'])
        bundle_info = dict(row_index=case['row_index'], source_ordinals=[r['job']['ordinal'] for r in source_records],
                           bundle=bundle, bundle_sha256=hashlib.sha256(bundle.encode()).hexdigest(),
                           clips=clips, trace_pair_fits=pair_fits, reader_jobs=reader_jobs,
                           bundle_and_all_reader_job_preparation_seconds=time.monotonic()-bundle_preparation_started)
        base.save(OUT / f'bundle_{index:02d}.json', bundle_info)
        for reader_job in reader_jobs:
            declined = ((reader_job['arm'] in ['prefix', 'append'] and not pair_fits)
                        or reader_job['input_plus_output_reserve'] > CTX)
            if declined:
                base.save(OUT / f"attempt_{reader_job['ordinal']:03d}.json",
                          dict(job=reader_job, completion='budget_declined', transmission='not_sent',
                               reason='joint_trace_pair_or_individual_input_plus_output_reserve_exceeds_frozen_context'))
                print(reader_job['ordinal'], case['row_index'], reader_job['arm'], 'budget_declined', flush=True)
                continue
            _, qualified = execute(reader_job)
            if not qualified:
                base.save(OUT / 'completion.json', dict(primary_complete=False, reason='reader_execution_or_qualification_failure', last_ordinal=reader_job['ordinal']))
                return
    base.save(OUT / 'completion.json', dict(primary_complete=True, planned_scored_slots=480,
                                          planned_calibration_slots=1, finished_at=time.time(),
                                          full_research_workflow_seconds=time.monotonic()-started))

if __name__ == '__main__':
    main()
