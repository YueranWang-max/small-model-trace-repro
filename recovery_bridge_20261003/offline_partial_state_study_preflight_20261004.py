"""Preparation checks only. Never call the model or scored runner main."""
import ast
import hashlib
import importlib.util
import json
import pathlib
import random
from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer
ROOT = pathlib.Path(__file__).resolve().parent


def main():
    out = ROOT/'partial_state_study_preflight_20261004.json'
    assert not out.exists()
    prepared = ROOT/'partial_state_prepared_20261004.json'
    data = json.loads(prepared.read_text(encoding='utf-8'))
    path = ROOT/'run_partial_state_20261004.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == data['runner_sha256']
    for name in ['run_partial_state_20261004.py','audit_partial_state_study_20261004.py',
                 'analyze_partial_state_20261004.py','offline_partial_state_study_preflight_20261004.py']:
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    assert len(data['cases']) == 16 and len(data['schedule']) == 257
    assert [c['row_index'] for c in data['cases']] == list(range(522,538))
    assert data['source_output_cap'] == 1024 and data['ctx'] == 65536
    spec = importlib.util.spec_from_file_location('partial_runner_for_offline_checks',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __main__ is not executed.
    counter = module.Counter()
    cfg = ROOT/'hybrid_candidate_qualification_20261004/tokenizer_config.json'
    env = Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
    template = env.from_string(json.loads(cfg.read_text(encoding='utf-8'))['chat_template'])
    tok = Tokenizer.from_file(str(ROOT/'hybrid_candidate_qualification_20261004/tokenizer.json'))
    check_count = 0
    for case in data['cases']:
        original = case['prompt']+'\n'+module.FORMAT
        assert case['direct_input_tokens'] == counter.count(original) > 8000
        assert case['total_allowance'] == case['direct_input_tokens']+32768
        for state in ['', 'A provisional frontier; it may be wrong.', '\n中文提示\n', ' scratchpad'*1024]:
            real_prefix = module.prompt(case,'prefix',state)
            neutral,match = module.neutral(case,counter.count(real_prefix),counter)
            assert abs(match['mismatch_tokens']) <= 2
            for arm in module.ARMS:
                user = neutral if arm == 'neutral_prefix' else module.prompt(case,arm,state)
                if arm == 'append':
                    head,tail = original.rsplit('Operation:',1)
                    inserted = module.STATE_PREFIX+state+module.STATE_SUFFIX
                    assert user[:len(head)] == head
                    assert user[len(head):len(head)+len(inserted)] == inserted
                    assert user[:len(head)]+user[len(head)+len(inserted):] == original
                else:
                    assert original in user
                reference = template.render(messages=[dict(role='system',content=module.SYSTEM),dict(role='user',content=user)],
                                            tools=None,add_generation_prompt=True,enable_thinking=True)
                count = len(tok.encode(reference,add_special_tokens=False).ids)
                assert count == counter.count(user)
                native = ('<|im_start|>system\n'+module.SYSTEM.strip()+'<|im_end|>\n<|im_start|>user\n'
                          +user.strip()+'<|im_end|>\n<|im_start|>assistant\n<think>\n')
                assert native == reference
                check_count += 1
    rng = random.Random(202610199)
    cursor = 1
    for index,case in enumerate(data['cases']):
        for repeat in range(2):
            source = data['schedule'][cursor]
            assert source['phase'] == 'source' and source['seed'] == 202610200+2*index+repeat
            source_ordinal = cursor+1
            assert source['ordinal'] == source_ordinal
            cursor += 1
            arms = module.ARMS.copy()
            rng.shuffle(arms)
            for arm in arms:
                plan = data['schedule'][cursor]
                assert plan['ordinal'] == cursor+1 and plan['arm'] == arm
                assert plan['seed'] == 202610300+2*index+repeat
                assert plan['source_ordinal'] == source_ordinal and plan['row_index'] == case['row_index']
                cursor += 1
    assert cursor == 257
    result = dict(all_checks_passed=True,inference_calls=0,fixture_render_checks=check_count,
                  complete_original_reconstruction=True,randomized_schedule_reconstructed=True,
                  prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  preflight_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  limitations=['Actual generated states and input counts require postresponse independent audit.',
                               'Runtime calibration still pending; no performance evidence.'])
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
