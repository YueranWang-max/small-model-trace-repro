"""Prospective private-endpoint study; prepare first, then one nonresumable run."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import random
import time
import urllib.request
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined
from budgeted_partial_state_arithmetic_draft import IncurredUsage, allocate_reader

ROOT = pathlib.Path(__file__).resolve().parent
META = ROOT/'hybrid_candidate_qualification_20261004'
PREP = ROOT/'partial_state_prepared_20261004.json'
OUT = ROOT/'partial_state_study_20261004'
ENDPOINT = 'http://127.0.0.1:11435/api/'
MODEL = 'qwen3.5:4b'
DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
SYSTEM = ('You solve directed-graph algorithm problems. Use only the graph and operation in the user message. '
          'Return exactly one visible line in this format: Final Answer: [node1, node2]. '
          'Use [] for the empty set. Do not include any text before or after that line.')
FORMAT = 'Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
CUE = 'Track shortest-path distance; keep visited nodes and the requested frontier.'
STATE_PREFIX = ('Below is an early, possibly incomplete reasoning state from an independent attempt on the same graph. '
                'It may contain mistakes. Verify it against the complete graph.\n<partial_native_state>\n')
STATE_SUFFIX = '\n</partial_native_state>\n'
ARMS = ['direct','question_first','re2','task_cue','prefix','append','neutral_prefix']
OPTIONS = dict(temperature=1.0,top_p=.95,top_k=20,min_p=0,presence_penalty=1.5,
               frequency_penalty=0,repeat_penalty=1,mirostat=0,num_ctx=65536)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def request(endpoint,payload=None,timeout=30,base=ENDPOINT):
    req = urllib.request.Request(base+endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=timeout) as response:
        return json.load(response)


def external_snapshot():
    try:
        return dict(ps=request('ps',base='http://127.0.0.1:11434/api/'),recorded_at=time.time())
    except Exception as error:
        return dict(error=repr(error),recorded_at=time.time())


class Counter:
    def __init__(self):
        config = json.loads((META/'tokenizer_config.json').read_text(encoding='utf-8'))
        env = Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
        self.template = env.from_string(config['chat_template'])
        self.tokenizer = Tokenizer.from_file(str(META/'tokenizer.json'))

    def count(self,user):
        text = self.template.render(messages=[dict(role='system',content=SYSTEM),dict(role='user',content=user)],
                                    tools=None,add_generation_prompt=True,enable_thinking=True)
        return len(self.tokenizer.encode(text,add_special_tokens=False).ids)


def whole(case):
    return case['prompt']+'\n'+FORMAT


def prompt(case,arm,state=''):
    text = whole(case)
    head,tail = text.rsplit('Operation:',1)
    question = 'Operation:'+tail
    assert head+question == text and 'Here is the graph to operate on:' in head
    block = STATE_PREFIX+state+STATE_SUFFIX
    if arm == 'direct':
        return text
    if arm == 'question_first':
        return question+'\n'+text
    if arm == 're2':
        return text+'\n'+text
    if arm == 'task_cue':
        return CUE+'\n\n'+text
    if arm in ['prefix','neutral_prefix']:
        return block+text
    if arm == 'append':
        return head+block+question
    raise ValueError(arm)


def neutral(case,target,counter):
    # Count matching uses no labels, graph algorithm or scored answer.
    def user(n):
        return prompt(case,'neutral_prefix',' .' * n)
    low, high = 0, max(64,target-counter.count(user(0)))*4+32
    while low < high:
        middle = (low+high)//2
        if counter.count(user(middle)) < target:
            low = middle+1
        else:
            high = middle
    candidates = range(max(0,low-4),low+5)
    n = min(candidates,key=lambda n:(abs(counter.count(user(n))-target),n))
    count = counter.count(user(n))
    assert abs(count-target) <= 2, 'Neutral count matching failed; stop rather than silently change control.'
    return user(n),dict(repetitions=n,target_input_tokens=target,actual_input_tokens=count,
                        mismatch_tokens=count-target,algorithm='repeated space-period; closest count, shorter tie')


def payload(user,seed,cap):
    return dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=user)],
                think=True,stream=False,keep_alive='30m',options={**OPTIONS,'seed':seed,'num_predict':cap})


def prepare():
    assert not PREP.exists() and not OUT.exists()
    draft = json.loads((ROOT/'partial_state_feasibility_draft_v2_20261004.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT/'partial_state_feasibility_draft_v2_audit_20261004.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'offline_selection_and_four_control_renderings_passed'
    assert audit['preparation_sha256'] == sha(ROOT/'partial_state_feasibility_draft_v2_20261004.json')
    dataset = ROOT/'graphwalks_128k_and_shorter.parquet'
    assert sha(dataset) == '54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    rows = pq.read_table(dataset).to_pylist()
    cases = [dict(row_index=i,**rows[i]) for i in draft['selected_rows']]
    count = Counter()
    schedule = [dict(ordinal=1,phase='calibration',row_index=None,repeat=None,seed=202610198,arm='calibration')]
    rng = random.Random(202610199)
    for index,case in enumerate(cases):
        case['direct_input_tokens'] = count.count(whole(case))
        case['total_allowance'] = case['direct_input_tokens']+32768
        assert case['direct_input_tokens'] > 8000
        assert case['total_allowance']+64 <= 65536
        for repeat in range(2):
            source_ordinal = len(schedule)+1
            schedule.append(dict(ordinal=source_ordinal,phase='source',row_index=case['row_index'],
                                 repeat=repeat,seed=202610200+index*2+repeat,arm='partial_source'))
            order = ARMS.copy()
            rng.shuffle(order)
            for arm in order:
                schedule.append(dict(ordinal=len(schedule)+1,phase='reader',row_index=case['row_index'],repeat=repeat,
                                     seed=202610300+index*2+repeat,arm=arm,source_ordinal=source_ordinal))
    assert len(schedule) == 257
    save(PREP,dict(status='prepared_not_executed',cases=cases,schedule=schedule,
                   source_output_cap=1024,ctx=65536,margin=64,maximum_output_cap=32768,
                   model=MODEL,digest=DIGEST,endpoint=ENDPOINT,options=OPTIONS,
                   arms=ARMS,seed_order=202610199,selection_draft_sha256=sha(ROOT/'partial_state_feasibility_draft_v2_20261004.json'),
                   selection_audit_sha256=sha(ROOT/'partial_state_feasibility_draft_v2_audit_20261004.json'),
                   runner_sha256=sha(pathlib.Path(__file__)),
                   environment=dict(python=platform.python_version(),platform=platform.platform(),
                                    packages={name:importlib.metadata.version(name) for name in ['tokenizers','jinja2','pyarrow','numpy']})))
    print('Prepared',len(cases),'graphs',len(schedule),'slots; no inference.',flush=True)


def execute(plan,user,cap,counter,metadata):
    protocol_start = time.monotonic()
    path = OUT/f"attempt_{plan['ordinal']:03d}.json"
    assert not path.exists(), 'No retries or overwrite.'
    prep_start = time.monotonic()
    req = payload(user,plan['seed'],cap)
    expected = counter.count(user)
    record = dict(plan=plan,request=req,request_sha256=digest(req),expected_prompt_tokens=expected,
                  completion='pending',transmission='not_yet_attempted',**metadata)
    assert expected+cap+64 <= 65536
    record['request_preparation_seconds'] = time.monotonic()-prep_start
    record['external_before'] = external_snapshot()
    save(path,record)
    record.update(transmission='attempting_once',started_at=time.time(),monotonic_started=time.monotonic())
    save(path,record)
    try:
        raw = request('chat',req,timeout=7200)
    except Exception as error:
        record.update(completion='client_error',error=repr(error),ended_at=time.time(),
                      client_elapsed_seconds=time.monotonic()-record['monotonic_started'])
        record['request_protocol_seconds_before_final_persist'] = time.monotonic()-protocol_start
        save(path,record)
        return record,False
    record.update(completion='returned',raw=raw,ended_at=time.time(),
                  client_elapsed_seconds=time.monotonic()-record['monotonic_started'])
    save(path,record)  # Save response and actual usage before metadata requests.
    try:
        record['own_runtime_ps'] = request('ps')
    except Exception as error:
        record['metadata_error'] = repr(error)
        record['request_protocol_seconds_before_final_persist'] = time.monotonic()-protocol_start
        save(path,record)
        return record,False
    record['external_after'] = external_snapshot()
    record['input_qualified'] = raw.get('prompt_eval_count') == expected
    record['response_qualified'] = (raw.get('done') is True and raw.get('model') == MODEL and not raw.get('error')
                                    and raw.get('done_reason') in ['stop','length']
                                    and isinstance(raw.get('message'),dict)
                                    and raw['message'].get('role') == 'assistant'
                                    and isinstance(raw['message'].get('thinking',''),str)
                                    and isinstance(raw['message'].get('content',''),str))
    record['usage_qualified'] = (type(raw.get('prompt_eval_count')) is int and raw['prompt_eval_count'] >= 0
                               and type(raw.get('eval_count')) is int and 0 <= raw['eval_count'] <= cap)
    record['resident_qualified'] = any(m.get('digest') == DIGEST and m.get('context_length',0) >= 65536
                                      for m in record['own_runtime_ps'].get('models',[]))
    record['qualified'] = all(record[name] for name in
                              ['input_qualified','response_qualified','usage_qualified','resident_qualified'])
    record['request_protocol_seconds_before_final_persist'] = time.monotonic()-protocol_start
    save(path,record)
    print(plan['ordinal'],plan['phase'],plan['row_index'],plan['arm'],raw.get('done_reason'),
          expected,raw.get('eval_count'),'qualified',record['qualified'],flush=True)
    return record,record['qualified']


def run():
    workflow_start = time.monotonic()
    assert not OUT.exists(), 'No resume, replay or overwrite.'
    prepared = json.loads(PREP.read_text(encoding='utf-8'))
    assert prepared['runner_sha256'] == sha(pathlib.Path(__file__))
    preflight_path = ROOT/'partial_state_study_preflight_20261004.json'
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    assert preflight['all_checks_passed'] and preflight['prepared_sha256'] == sha(PREP)
    assert preflight['runner_sha256'] == sha(pathlib.Path(__file__))
    version,tags = request('version'),request('tags')
    assert version['version'] == '0.35.1'
    assert next(m['digest'] for m in tags['models'] if m['name'] == MODEL) == DIGEST
    assert external_snapshot().get('ps',{}).get('models') == [], 'No competing resident model at launch.'
    dependencies = [pathlib.Path(__file__),ROOT/'budgeted_partial_state_arithmetic_draft.py',PREP,preflight_path,
                    ROOT/'partial_state_protocol_20261004.md',ROOT/'offline_partial_state_study_preflight_20261004.py',
                    ROOT/'analyze_partial_state_20261004.py',ROOT/'audit_partial_state_study_20261004.py',
                    META/'tokenizer.json',META/'tokenizer_config.json',ROOT/'owned_ollama_11435_start_20261004.json',
                    ROOT/'partial_state_feasibility_draft_v2_20261004.json',
                    ROOT/'partial_state_feasibility_draft_v2_audit_20261004.json']
    OUT.mkdir()
    frozen = {**prepared,'status':'frozen_before_first_calibration','frozen_at':time.time(),'runtime':version,
              'inventory':tags,'dependencies':{str(p):sha(p) for p in dependencies},
              'preparation_before_freeze_seconds':time.monotonic()-workflow_start}
    save(OUT/'frozen.json',frozen)
    counter = Counter()
    cases = {c['row_index']:c for c in prepared['cases']}
    records = {}
    for plan in prepared['schedule']:
        metadata = {}
        prep_start = time.monotonic()
        if plan['phase'] == 'calibration':
            user = 'A directed graph has the nodes A and B and no edges. Find the parents of node B.\n'+FORMAT
            cap = 1
        elif plan['phase'] == 'source':
            user,cap = whole(cases[plan['row_index']]),1024
        else:
            case = cases[plan['row_index']]
            source = records[plan['source_ordinal']]
            state = source['raw'].get('message',{}).get('thinking','')
            assert isinstance(state,str)
            if plan['arm'] == 'neutral_prefix':
                target = counter.count(prompt(case,'prefix',state))
                try:
                    user,match = neutral(case,target,counter)
                except Exception as error:
                    save(OUT/'completion.json',dict(primary_complete=False,reason='neutral_construction_failed',
                                                   next_ordinal=plan['ordinal'],error=repr(error),
                                                   actual_workflow_seconds=time.monotonic()-workflow_start))
                    return
                metadata['neutral_match'] = match
            else:
                user = prompt(case,plan['arm'],state)
            state_arm = plan['arm'] in ['prefix','append','neutral_prefix']
            incurred = [IncurredUsage(source['raw']['prompt_eval_count'],source['raw']['eval_count'],True)] if state_arm else []
            decision = allocate_reader(total_allowance=case['total_allowance'],incurred=incurred,
                                       reader_input_tokens=counter.count(user),maximum_output_cap=32768,
                                       context_limit=65536,context_margin=64)
            metadata.update(budget_decision=dataclasses.asdict(decision),
                            charged_source_ordinal=plan['source_ordinal'] if state_arm else None,
                            source_thinking_characters=len(state) if state_arm else None,
                            source_thinking_standalone_tokens=len(counter.tokenizer.encode(state,add_special_tokens=False).ids) if state_arm else None)
            if not decision.send_reader:
                path = OUT/f"attempt_{plan['ordinal']:03d}.json"
                assert not path.exists()
                record = dict(plan=plan,completion='known_budget_decline',transmission='not_transmitted',
                              expected_prompt_tokens=decision.intended_reader_input,
                              prompt_construction_seconds=time.monotonic()-prep_start,
                              client_elapsed_seconds=0.,request_protocol_seconds_before_final_persist=0.,**metadata)
                save(path,record)
                records[plan['ordinal']] = record
                print(plan['ordinal'],'budget_decline',flush=True)
                continue
            cap = decision.reader_output_cap
        metadata['prompt_construction_seconds'] = time.monotonic()-prep_start
        record,qualified = execute(plan,user,cap,counter,metadata)
        records[plan['ordinal']] = record
        if not qualified:
            save(OUT/'completion.json',dict(primary_complete=False,reason='execution_or_qualification_failure',
                                           last_ordinal=plan['ordinal'],actual_workflow_seconds=time.monotonic()-workflow_start))
            return
    save(OUT/'completion.json',dict(primary_complete=True,planned_formal_calls=256,calibration_calls=1,
                                   last_ordinal=257,finished_at=time.time(),
                                   actual_workflow_seconds=time.monotonic()-workflow_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except Exception as error:
            if OUT.exists() and not (OUT/'completion.json').exists():
                save(OUT/'completion.json',dict(primary_complete=False,reason='local_error',
                                               error=repr(error),finished_at=time.time()))
            raise
