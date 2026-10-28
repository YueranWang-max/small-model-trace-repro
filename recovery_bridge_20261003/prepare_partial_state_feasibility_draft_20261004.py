"""Offline draft only: new-case selection and exact rendered fixture counts.

No inference, answer scoring, runtime mutation, or experiment freeze.
"""
import hashlib
import json
import pathlib
import re
import collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
META = ROOT / 'hybrid_candidate_qualification_20261004'
OUTPUT = ROOT / 'partial_state_feasibility_draft_v2_20261004.json'
SYSTEM = ('You solve directed-graph algorithm problems. Use only the graph and operation in the user message. '
          'Return exactly one visible line in this format: Final Answer: [node1, node2]. '
          'Use [] for the empty set. Do not include any text before or after that line.')
FORMAT = 'Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(prompt):
    block = prompt.split('Here is the graph to operate on:', 1)[1].split('Operation:', 1)[0]
    edges = re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$', block, re.M)
    assert edges
    return hashlib.sha256(json.dumps(sorted(edges), separators=(',', ':')).encode()).hexdigest()


def fail(message):
    raise ValueError(message)


def main():
    assert not OUTPUT.exists(), 'Preserve prior preparation evidence.'
    dataset = ROOT / 'graphwalks_128k_and_shorter.parquet'
    assert sha(dataset) == '54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    rows = pq.read_table(dataset).to_pylist()
    archives = sorted((REPO / 'logs').glob('official_graph_*/frozen.json'))
    archives += [ROOT / name / 'frozen.json' for name in
                 ['bridge_run', 'hybrid_source_qualification', 'hybrid_five_trace_replication']]
    excluded = set()
    provenance = []
    for path in archives:
        archive = json.loads(path.read_text(encoding='utf-8'))
        records = archive.get('selected', archive.get('cases', []))
        for record in records:
            excluded.add(signature(record.get('paper_row', record)['prompt']))
        provenance.append(dict(path=str(path), sha256=sha(path), selected_count=len(records)))
    exclusion_count = len(excluded)
    assert exclusion_count == 98
    selected = []
    for index, row in enumerate(rows):
        if row['problem_type'] != 'bfs' or not 14000 <= row['prompt_chars'] < 16000:
            continue
        sig = signature(row['prompt'])
        if sig in excluded:
            continue
        selected.append(dict(row_index=index, signature=sig, **row))
        excluded.add(sig)
        if len(selected) == 16:
            break
    assert len(selected) == 16
    config = json.loads((META / 'tokenizer_config.json').read_text(encoding='utf-8'))
    env = Environment(undefined=StrictUndefined, extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = fail
    template = env.from_string(config['chat_template'])
    tokenizer = Tokenizer.from_file(str(META / 'tokenizer.json'))

    def count(user):
        rendered = template.render(messages=[dict(role='system', content=SYSTEM),
                                             dict(role='user', content=user)],
                                   tools=None, add_generation_prompt=True, enable_thinking=True)
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    records = []
    for case in selected:
        direct = case['prompt'] + '\n' + FORMAT
        x, tail = direct.rsplit('Operation:', 1)
        q = 'Operation:' + tail
        assert x + q == direct
        assert 'Here is the graph to operate on:' in x
        assert signature(direct) == case['signature']
        inputs = dict(direct=count(direct), question_first=count(q+'\n'+direct),
                      re2=count(direct+'\n'+direct),
                      task_cue=count('Track shortest-path distance; keep visited nodes and the requested frontier.\n\n'+direct))
        budget = inputs['direct'] + 32768
        assert inputs['direct'] > 8000, 'Long source cannot be reduced to instruction-only prefix.'
        controls = {arm: dict(input_tokens=n, output_allowance=budget-n,
                             context_reserve=n+max(0,budget-n)+64,
                             fits_context=n+max(0,budget-n)+64 <= 131072)
                    for arm, n in inputs.items()}
        fixtures = []
        for source_cap in [256, 1024, 4096]:
            # Deliberately content-free ASCII fixture, never a simulated answer/trace.
            fixture = ' scratchpad' * source_cap
            j = count('Partial scratchpad; verify it against the graph:\n'+fixture+'\n\n'+direct)
            remaining = budget - inputs['direct'] - source_cap - j
            fixtures.append(dict(source_output_cap=source_cap,
                                 fixture_standalone_tokens=len(tokenizer.encode(fixture,add_special_tokens=False).ids),
                                 reader_input_tokens=j, hypothetical_reader_output_allowance=max(0,remaining),
                                 hypothetical_decline=remaining<=0))
        records.append(dict(row_index=case['row_index'], graph_signature=case['signature'],
                            catalogue_prompt_chars=case['prompt_chars'], total_allowance=budget,
                            controls=controls, synthetic_state_fixtures=fixtures))
    result = dict(status='unfrozen_offline_feasibility_draft', inference_calls=0,
                  correction='v1 mistakenly split at the example Operation marker; v1 counts invalid. v2 retains the complete original prompt and final question.',
                  answer_scores_read_or_used=False, independent_confirmation=False,
                  selection_rule='first 16 BFS catalogue [14000,16000), excluding all prior frozen graph identities',
                  excluded_graphs=exclusion_count, excluded_archives=provenance,
                  selected_rows=[c['row_index'] for c in selected], records=records,
                  source_sha256=sha(pathlib.Path(__file__)), tokenizer_sha256=sha(META/'tokenizer.json'),
                  limitations=['Synthetic state fixture counts are not bounds on actual generated traces.',
                               'Actual reader allowance must subtract every incurred deployment call and complete rendered reader input.',
                               'No runtime isolation, new calibration, independent selection audit, or frozen success criteria yet.',
                               'No performance, latency, cache, or novelty result follows from this preparation.'])
    OUTPUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],selected_rows=result['selected_rows'],
                         excluded_graphs=exclusion_count,inference_calls=0),indent=2))


if __name__ == '__main__':
    main()
