"""Independent completed/interrupted study audit; no runner or arithmetic imports."""
import collections
import hashlib
import json
import pathlib
import random
import re
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'partial_state_study_20261004'
MODEL = 'qwen3.5:4b'
DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
SYSTEM = ('You solve directed-graph algorithm problems. Use only the graph and operation in the user message. '
          'Return exactly one visible line in this format: Final Answer: [node1, node2]. '
          'Use [] for the empty set. Do not include any text before or after that line.')
FORMAT = 'Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
START = ('Below is an early, possibly incomplete reasoning state from an independent attempt on the same graph. '
         'It may contain mistakes. Verify it against the complete graph.\n<partial_native_state>\n')
END = '\n</partial_native_state>\n'
ARMS = ['direct','question_first','re2','task_cue','prefix','append','neutral_prefix']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def edges(prompt):
    text = prompt.split('Here is the graph to operate on:',1)[1].rsplit('Operation:',1)[0]
    result = []
    for line in text.splitlines():
        if ' -> ' in line:
            u,v = line.strip().split(' -> ')
            assert re.fullmatch('[0-9a-f]{10}',u) and re.fullmatch('[0-9a-f]{10}',v)
            result.append((u,v))
    assert result
    return result


def gold(case):
    question = case['prompt'][case['prompt'].rfind('Operation:'):]
    target,depth = re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',question).groups()
    adjacency = collections.defaultdict(set)
    for u,v in edges(case['prompt']):
        adjacency[u].add(v)
    queue,distances = collections.deque([target]),{target:0}
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in distances:
                distances[v] = distances[u]+1
                queue.append(v)
    return {v for v,d in distances.items() if d == int(depth) and v != target}


def score(raw,target):
    visible = raw.get('message',{}).get('content','')
    valid = False
    predicted = set()
    if raw.get('done_reason') == 'stop' and isinstance(visible,str) and len(visible.strip().splitlines()) == 1:
        match = re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',visible.strip())
        if match:
            tokens = [t.strip() for t in match[1].split(',')] if match[1].strip() else []
            if all(re.fullmatch(r'[A-Za-z0-9_.-]+',t) for t in tokens):
                valid,predicted = True,set(tokens)
    f1 = (2*len(predicted & target)/(len(predicted)+len(target)) if predicted or target else 1.0) if valid else 0.0
    return dict(f1=f1,em=float(valid and predicted == target),normal_valid=valid,
                predicted=sorted(predicted),visible_only=True)


def component_time(rec):
    values = [rec.get('prompt_construction_seconds',0.),rec.get('request_protocol_seconds_before_final_persist',0.)]
    assert all(type(value) in [int,float] and value >= 0 for value in values)
    return sum(values)


def main():
    result_path = OUT/'independent_audit.json'
    assert not result_path.exists(), 'Preserve audit evidence.'
    frozen,completion = read(OUT/'frozen.json'),read(OUT/'completion.json')
    for path,expected in frozen['dependencies'].items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected
    assert frozen['model'] == MODEL and frozen['digest'] == DIGEST
    assert frozen['runtime']['version'] == '0.35.1' and frozen['endpoint'] == 'http://127.0.0.1:11435/api/'
    assert frozen['ctx'] == 65536 and frozen['margin'] == 64 and frozen['source_output_cap'] == 1024
    assert frozen['status'] == 'frozen_before_first_calibration'
    options = dict(temperature=1.0,top_p=.95,top_k=20,min_p=0,presence_penalty=1.5,
                   frequency_penalty=0,repeat_penalty=1,mirostat=0,num_ctx=65536)
    assert frozen['options'] == options
    dataset = ROOT/'graphwalks_128k_and_shorter.parquet'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == '54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    rows = pq.read_table(dataset).to_pylist()
    exclusion = read(ROOT/'partial_state_feasibility_draft_v2_20261004.json')
    prior = set()
    for archive in exclusion['excluded_archives']:
        path = pathlib.Path(archive['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == archive['sha256']
        data = read(path)
        for c in data.get('selected',data.get('cases',[])):
            prior.add(tuple(sorted(edges(c.get('paper_row',c)['prompt']))))
    assert len(prior) == 98
    chosen = []
    for i,c in enumerate(rows):
        signature = tuple(sorted(edges(c['prompt'])))
        if c['problem_type'] == 'bfs' and 14000 <= c['prompt_chars'] < 16000 and signature not in prior:
            chosen.append(i)
            prior.add(signature)
            if len(chosen) == 16:
                break
    assert chosen == [c['row_index'] for c in frozen['cases']] == list(range(522,538))
    cases = {c['row_index']:c for c in frozen['cases']}
    for case in cases.values():
        original = rows[case['row_index']]
        assert all(case[key] == value for key,value in original.items())
        assert gold(case) == set(case['answer_nodes'])
    schedule = [dict(ordinal=1,phase='calibration',row_index=None,repeat=None,seed=202610198,arm='calibration')]
    rng = random.Random(202610199)
    for i,case in enumerate(frozen['cases']):
        for repeat in range(2):
            source_ordinal = len(schedule)+1
            schedule.append(dict(ordinal=source_ordinal,phase='source',row_index=case['row_index'],repeat=repeat,
                                 seed=202610200+2*i+repeat,arm='partial_source'))
            order = ARMS.copy()
            rng.shuffle(order)
            for arm in order:
                schedule.append(dict(ordinal=len(schedule)+1,phase='reader',row_index=case['row_index'],repeat=repeat,
                                     seed=202610300+2*i+repeat,arm=arm,source_ordinal=source_ordinal))
    assert schedule == frozen['schedule'] and len(schedule) == 257
    meta = ROOT/'hybrid_candidate_qualification_20261004'
    env = Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
    template = env.from_string(read(meta/'tokenizer_config.json')['chat_template'])
    tok = Tokenizer.from_file(str(meta/'tokenizer.json'))

    def count(user):
        rendered = template.render(messages=[dict(role='system',content=SYSTEM),dict(role='user',content=user)],
                                   tools=None,add_generation_prompt=True,enable_thinking=True)
        return len(tok.encode(rendered,add_special_tokens=False).ids)

    records = {}
    summaries = []
    known_tokens = 0
    attempted_unknown = 0
    raw_qualified = 0
    external_resident_snapshots = 0
    saw_missing = False
    for plan in schedule:
        file = OUT/f"attempt_{plan['ordinal']:03d}.json"
        if not file.exists():
            saw_missing = True
            if plan['phase'] == 'reader':
                source_raw = records.get(plan['source_ordinal'],{}).get('raw',{})
                known_prior = 0
                if plan['arm'] in ['prefix','append','neutral_prefix'] and all(
                    type(source_raw.get(k)) is int and source_raw[k] >= 0 for k in ['prompt_eval_count','eval_count']):
                    known_prior = source_raw['prompt_eval_count']+source_raw['eval_count']
                summaries.append(dict(**plan,known=False,f1=None,em=None,status='not_reached',tokens_lower=known_prior,
                                      tokens_complete=False,prior_cost=known_prior,reader_cost=0,
                                      deployment_component_seconds_lower=component_time(records.get(plan['source_ordinal'],{}))
                                      if plan['arm'] in ['prefix','append','neutral_prefix'] else 0.))
            continue
        assert not saw_missing, 'Records must be a continuous prefix, no silent skips.'
        rec = read(file)
        records[plan['ordinal']] = rec
        assert rec['plan'] == plan
        state_arm = plan['arm'] in ['prefix','append','neutral_prefix']
        prior_cost = 0
        if plan['phase'] == 'calibration':
            user = 'A directed graph has the nodes A and B and no edges. Find the parents of node B.\n'+FORMAT
            cap = 1
        else:
            case = cases[plan['row_index']]
            whole = case['prompt']+'\n'+FORMAT
            assert case['direct_input_tokens'] == count(whole)
            assert case['total_allowance'] == count(whole)+32768
            head,tail = whole.rsplit('Operation:',1)
            question = 'Operation:'+tail
            if plan['phase'] == 'source':
                user,cap = whole,1024
            else:
                source = records[plan['source_ordinal']]
                assert source['qualified'] is True
                state = source['raw'].get('message',{}).get('thinking','')
                assert isinstance(state,str)
                block = START+state+END
                if plan['arm'] == 'direct':
                    user = whole
                elif plan['arm'] == 'question_first':
                    user = question+'\n'+whole
                elif plan['arm'] == 're2':
                    user = whole+'\n'+whole
                elif plan['arm'] == 'task_cue':
                    user = 'Track shortest-path distance; keep visited nodes and the requested frontier.\n\n'+whole
                elif plan['arm'] == 'prefix':
                    user = block+whole
                elif plan['arm'] == 'append':
                    user = head+block+question
                    assert user[:len(head)]+user[len(head)+len(block):] == whole
                else:
                    neutral = rec['neutral_match']
                    assert type(neutral['repetitions']) is int and neutral['repetitions'] >= 0
                    user = START+' .' * neutral['repetitions']+END+whole
                    target = count(block+whole)
                    assert neutral['target_input_tokens'] == target
                    assert neutral['actual_input_tokens'] == count(user)
                    assert neutral['mismatch_tokens'] == count(user)-target
                    assert abs(count(user)-target) <= 2
                prior_cost = source['raw']['prompt_eval_count']+source['raw']['eval_count'] if state_arm else 0
                expected = count(user)
                remaining = case['total_allowance']-prior_cost-expected
                cap = max(0,min(32768,remaining))
                decision = rec['budget_decision']
                assert decision['total_allowance'] == case['total_allowance']
                assert decision['prior_usage'] == prior_cost
                assert decision['intended_reader_input'] == expected
                if rec['completion'] == 'known_budget_decline':
                    assert remaining <= 0 or expected+cap+64 > 65536
                    assert rec['transmission'] == 'not_transmitted'
                    summaries.append(dict(**plan,known=True,f1=0.,em=0.,normal_valid=False,status='budget_decline',
                                          tokens_lower=prior_cost,tokens_complete=True,prior_cost=prior_cost,reader_cost=0,
                                          deployment_component_seconds_lower=component_time(rec)+(component_time(source) if state_arm else 0.)))
                    continue
                assert cap > 0 and expected+cap+64 <= 65536
                assert decision['reader_output_cap'] == cap and decision['send_reader'] is True
                assert decision['planned_total_upper_bound'] == prior_cost+expected+cap <= case['total_allowance']
                assert rec['charged_source_ordinal'] == (plan['source_ordinal'] if state_arm else None)
                if state_arm:
                    assert rec['source_thinking_characters'] == len(state)
                    assert rec['source_thinking_standalone_tokens'] == len(tok.encode(state,add_special_tokens=False).ids)
        expected = count(user)
        req = dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=user)],
                   think=True,stream=False,keep_alive='30m',options={**options,'seed':plan['seed'],'num_predict':cap})
        assert rec['request'] == req and rec['expected_prompt_tokens'] == expected
        assert rec['request_sha256'] == hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()
        assert expected+cap+64 <= 65536
        for name in ['external_before','external_after']:
            if rec.get(name,{}).get('ps',{}).get('models'):
                external_resident_snapshots += 1
        raw = rec.get('raw',{})
        usage_known = (type(raw.get('prompt_eval_count')) is int and raw['prompt_eval_count'] >= 0
                       and type(raw.get('eval_count')) is int and raw['eval_count'] >= 0)
        reader_cost = raw['prompt_eval_count']+raw['eval_count'] if usage_known else 0
        known_tokens += reader_cost
        if not usage_known and rec.get('transmission') == 'attempting_once':
            attempted_unknown += 1
        qualified = (rec.get('completion') == 'returned' and raw.get('done') is True and raw.get('model') == MODEL
                     and not raw.get('error') and raw.get('done_reason') in ['stop','length']
                     and isinstance(raw.get('message'),dict) and raw['message'].get('role') == 'assistant'
                     and isinstance(raw['message'].get('thinking',''),str) and isinstance(raw['message'].get('content',''),str)
                     and usage_known and raw['prompt_eval_count'] == expected and raw['eval_count'] <= cap
                     and any(m.get('digest') == DIGEST and m.get('context_length',0) >= 65536
                             for m in rec.get('own_runtime_ps',{}).get('models',[])))
        if 'qualified' in rec:
            assert rec['qualified'] == qualified
        raw_qualified += int(qualified)
        if plan['phase'] == 'reader':
            scored = score(raw,gold(cases[plan['row_index']])) if qualified else dict(f1=None,em=None,normal_valid=False)
            if usage_known:
                assert prior_cost+reader_cost <= cases[plan['row_index']]['total_allowance']
            summaries.append(dict(**plan,known=qualified,status='qualified_response' if qualified else 'unknown_or_unqualified',
                                  tokens_lower=prior_cost+reader_cost,tokens_complete=usage_known,
                                  prior_cost=prior_cost,reader_cost=reader_cost,
                                  deployment_component_seconds_lower=component_time(rec)+(component_time(source) if state_arm else 0.),
                                  **scored))
    assert len(summaries) == 224
    if any(rec['plan']['phase'] != 'calibration' for rec in records.values()):
        assert records[1]['qualified'] is True
    if completion['primary_complete']:
        assert len(records) == 257 and all(s['known'] for s in summaries)
    result = dict(audit_passed=True,primary_complete=completion['primary_complete'],
                  audit_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  frozen_sha256=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest(),
                  planned_reader_slots=224,known_reader_slots=sum(s['known'] for s in summaries),
                  saved_records=len(records),qualified_raw_responses=raw_qualified,
                  known_research_tokens_lower_bound=known_tokens,attempted_unknown_usage_calls=attempted_unknown,
                  external_resident_snapshot_count=external_resident_snapshots,reader_results=summaries,
                  limitations=['Own endpoint qualification does not prove exclusive hardware or continuous resource residency.',
                               'Two seeds are not independent graph units; public-case exclusion is not pretraining exclusion.',
                               'Research source sharing does not amortize single-prediction deployment costs.'])
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='reader_results'},indent=2))


if __name__ == '__main__':
    main()
