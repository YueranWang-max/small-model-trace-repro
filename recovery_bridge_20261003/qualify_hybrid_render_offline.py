"""Compare narrow text-only two-message renders, without invoking inference."""
import pathlib,json,hashlib
from jinja2 import Environment,StrictUndefined
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parent
ARCHIVE=ROOT/'hybrid_candidate_qualification_20261004'
SYSTEM='You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.'
FORMAT='Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
def digest(value):return hashlib.sha256(value.encode()).hexdigest()
def raise_exception(message):raise ValueError(message)
def runtime_text_only_render(messages,think):
    # Narrow translation of archived Ollama v0.35.1 Qwen35Renderer.Render:
    # exactly one leading system plus one user; no tools/images/history/prefill.
    assert [m['role'] for m in messages]==['system','user']
    s=''.join('<|im_start|>'+m['role']+'\n'+m['content'].strip()+'<|im_end|>\n' for m in messages)
    return s+'<|im_start|>assistant\n'+('<think>\n' if think else '<think>\n\n</think>\n\n')
def main():
    out=ARCHIVE/'offline_render_checks.json';assert not out.exists()
    conf=json.loads((ARCHIVE/'tokenizer_config.json').read_bytes())
    env=Environment(undefined=StrictUndefined,extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception']=raise_exception
    template=env.from_string(conf['chat_template'])
    tokenizer=Tokenizer.from_file(str(ARCHIVE/'tokenizer.json'))
    cases=json.loads((ROOT/'bridge_run/frozen.json').read_bytes())['cases']
    fixtures=[{'case_id':str(c['row_index']),'user':c['prompt']+'\n'+FORMAT} for c in cases]
    fixtures += [{'case_id':'whitespace_unicode_fixture','user':' \n测试 graph α -> β\n\t'}]
    rows=[]
    for case in fixtures:
        for think in [True,False]:
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':case['user']}]
            official=template.render(messages=messages,tools=None,add_generation_prompt=True,enable_thinking=think)
            runtime=runtime_text_only_render(messages,think)
            rows.append({'case_id':case['case_id'],'thinking':think,'byte_identical':official==runtime,
                'official_sha256':digest(official),'runtime_translation_sha256':digest(runtime),
                'official_tokens':len(tokenizer.encode(official,add_special_tokens=False).ids),
                'runtime_translation_tokens':len(tokenizer.encode(runtime,add_special_tokens=False).ids)})
    result={'rows':rows,'all_34_offline_renders_equal':all(x['byte_identical'] for x in rows),
        'inference_calls':0,'server_actual_prompt_eval_count_verified':False,
        'scope':'Text-only leading system/user, no tools/images/history; bridge inputs used as metadata fixtures, not new scored study cases.',
        'runtime_renderer_source_sha256':hashlib.sha256((ARCHIVE/'runtime_qwen35_renderer.go').read_bytes()).hexdigest(),
        'limitations':['Python translation equality is not execution of the installed Go renderer.',
           'This does not prove runtime prompt count, active model identity or context fitting during generation.',
           'New scored cases and actual token counts require a separately frozen generation protocol.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    print('token_range',min(x['official_tokens'] for x in rows),max(x['official_tokens'] for x in rows))
    assert result['all_34_offline_renders_equal']
if __name__=='__main__':main()
