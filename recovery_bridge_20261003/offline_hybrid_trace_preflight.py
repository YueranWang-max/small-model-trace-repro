"""Offline qualification before freeze; never imports/executes scored runner main."""
import pathlib, json, hashlib, ast
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment, StrictUndefined
import run_hybrid_five_trace_replication as runner
import audit_hybrid_five_trace_replication as independent

ROOT=pathlib.Path(__file__).resolve().parent
META=ROOT/'hybrid_candidate_qualification_20261004'

def main():
    assert (ROOT/'hybrid_source_qualification/completion.json').exists()
    assert (ROOT/'hybrid_source_qualification/independent_audit.json').exists()
    assert not runner.OUT.exists(), 'No preparation should modify a primary archive.'
    assert not (ROOT/'hybrid_trace_offline_preflight.json').exists()
    names=['run_hybrid_five_trace_replication.py','audit_hybrid_five_trace_replication.py',
           'analyze_hybrid_five_trace_replication.py','offline_hybrid_trace_preflight.py']
    for name in names:
        ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name)
    rows=pq.read_table(ROOT/'graphwalks_128k_and_shorter.parquet').to_pylist()
    historical=json.loads((ROOT/'historical_manifest_exclusion_check_20261004.json').read_bytes())
    bridge=json.loads((ROOT/'bridge_run/frozen.json').read_bytes())['cases']
    qualification=json.loads((ROOT/'hybrid_source_qualification/frozen.json').read_bytes())['cases']
    excluded={independent.signature(rows[i]['prompt']) for i in historical['historical_rows']}
    excluded|={independent.signature(c['prompt']) for c in bridge+qualification}
    assert len(excluded)==82
    cases=[]
    for lower,upper in [(14000,16000)]:
        for kind in ['bfs']:
            count=0
            for index,row in enumerate(rows):
                if row['problem_type']!=kind or not lower<=row['prompt_chars']<upper:
                    continue
                signature=independent.signature(row['prompt'])
                if signature in excluded:
                    continue
                cases.append(dict(row_index=index,length_bin=[lower,upper],**row))
                excluded.add(signature);count+=1
                if count==16:break
            assert count==16
    assert len({independent.signature(c['prompt']) for c in cases})==16
    for case in cases:
        assert independent.gold(case)==runner.base.oracle(case)==set(case['answer_nodes'])
    env=Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception']=independent.fail
    template=env.from_string(json.loads((META/'tokenizer_config.json').read_bytes())['chat_template'])
    tok=Tokenizer.from_file(str(META/'tokenizer.json'))
    traces=['leading and trailing whitespace ', '\n中文 cue\n', '', 'A'*50001, 'Final Answer appears inside thinking but is not a separate source answer.']
    bundle,clips=runner.serialize(traces)
    assert clips[3]['clipped'] and clips[3]['retained_characters_before_strip']==50003
    assert '[Trace 4] '+'A'*50000+'...' in bundle
    fixture_checks=[]
    for case in cases:
        users=runner.prompts(case,bundle)
        question='Operation:'+case['prompt'].rsplit('Operation:',1)[1]+'\n'+runner.base.FORMAT
        for arm,user in users.items():
            assert user.endswith(question)
            request=runner.payload(user,202610130)
            reference=template.render(messages=request['messages'],tools=None,add_generation_prompt=True,enable_thinking=True)
            assert reference==runner.base.rendered(user)
            count=len(tok.encode(reference,add_special_tokens=False).ids)
            fixture_checks.append(dict(row_index=case['row_index'],arm=arm,prompt_tokens=count,
                                       input_plus_output_reserve=count+32768+64,
                                       fits=count+32768+64<=131072))
    result=dict(offline_preflight_passed=True,scored_inference_calls=0,
                selected_rows=[c['row_index'] for c in cases],all16_independent_labels_match=True,
                rendering_fixture_count=len(fixture_checks),fixtures=fixture_checks,
                serializer_boundary_fixture_passed=True,
                dependency_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
                limitations=['Synthetic trace fixtures validate formatting, not actual future trace-budget feasibility.',
                             'Native131072context qualification remains pending logged calibration.',
                             'The final frozen archive must independently reconstruct the same selection and all481slots.'])
    (ROOT/'hybrid_trace_offline_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='fixtures'},indent=2))

if __name__=='__main__':main()
