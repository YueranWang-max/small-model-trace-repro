"""Independent offline reconstruction, no imports of preparation/runner modules."""
import hashlib
import json
import pathlib
import re
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'partial_state_feasibility_draft_v2_audit_20261004.json'


def graph(prompt):
    text = prompt.split('Here is the graph to operate on:', 1)[1].rsplit('Operation:', 1)[0]
    lines = text.strip().splitlines()
    edges = []
    for line in lines:
        if ' -> ' not in line:
            continue
        pair = line.strip().split(' -> ')
        assert len(pair) == 2 and all(re.fullmatch('[0-9a-f]{10}', node) for node in pair)
        edges.append(tuple(pair))
    assert edges
    return tuple(sorted(edges))


def main():
    assert not OUT.exists()
    path = ROOT / 'partial_state_feasibility_draft_v2_20261004.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    rows = pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist()
    old = set()
    for archive in d['excluded_archives']:
        file = pathlib.Path(archive['path'])
        assert hashlib.sha256(file.read_bytes()).hexdigest() == archive['sha256']
        data = json.loads(file.read_text(encoding='utf-8'))
        for c in data.get('selected',data.get('cases',[])):
            old.add(graph(c.get('paper_row',c)['prompt']))
    assert len(old) == d['excluded_graphs'] == 98
    expected = []
    seen = set(old)
    for i,c in enumerate(rows):
        if c['problem_type'] == 'bfs' and 14000 <= c['prompt_chars'] < 16000 and graph(c['prompt']) not in seen:
            expected.append(i)
            seen.add(graph(c['prompt']))
            if len(expected) == 16:
                break
    assert expected == d['selected_rows'] == list(range(522,538))
    meta = ROOT/'hybrid_candidate_qualification_20261004'
    env = Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
    template = env.from_string(json.loads((meta/'tokenizer_config.json').read_text(encoding='utf-8'))['chat_template'])
    tok = Tokenizer.from_file(str(meta/'tokenizer.json'))
    system = ('You solve directed-graph algorithm problems. Use only the graph and operation in the user message. '
              'Return exactly one visible line in this format: Final Answer: [node1, node2]. '
              'Use [] for the empty set. Do not include any text before or after that line.')
    suffix = 'Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
    checked = 0
    for record in d['records']:
        original = rows[record['row_index']]['prompt']
        whole = original+'\n'+suffix
        operation_position = whole.rfind('Operation:')
        assert operation_position > whole.find('Here is the graph to operate on:')
        question = whole[operation_position:]
        users = dict(direct=whole,question_first=question+'\n'+whole,re2=whole+'\n'+whole,
                     task_cue='Track shortest-path distance; keep visited nodes and the requested frontier.\n\n'+whole)
        for arm,user in users.items():
            assert original in user
            rendered = template.render(messages=[{'role':'system','content':system},{'role':'user','content':user}],
                                       tools=None,add_generation_prompt=True,enable_thinking=True)
            count = len(tok.encode(rendered,add_special_tokens=False).ids)
            control = record['controls'][arm]
            assert count == control['input_tokens']
            assert control['output_allowance'] == record['total_allowance']-count > 0
            assert control['context_reserve'] == record['total_allowance']+64 <= 131072
            checked += 1
    result = dict(status='offline_selection_and_four_control_renderings_passed', inference_calls=0,
                  source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  preparation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  excluded_unique_graphs=98,selected_rows=expected,complete_original_control_requests_checked=checked,
                  limitations=['Not a frozen primary experiment or runtime qualification.',
                               'Synthetic partial-state fixtures are not audited performance or generated-state upper bounds.',
                               'Does not prove public graphs absent from model pretraining.'])
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
