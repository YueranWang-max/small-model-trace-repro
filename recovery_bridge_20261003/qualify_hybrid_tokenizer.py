"""Metadata-only tokenizer and runtime-source archive; zero generations."""
import json,pathlib,urllib.request,hashlib
ROOT=pathlib.Path(__file__).resolve().parent/'hybrid_candidate_qualification_20261004'
def get(url):
    with urllib.request.urlopen(url,timeout=90) as response:return response.read()
def main():
    out=ROOT/'tokenizer_runtime_checks.json';assert not out.exists()
    qualification=json.loads((ROOT/'qualification.json').read_bytes())
    revision=qualification['revision']
    path=ROOT/'tokenizer.json'
    if not path.exists():path.write_bytes(get('https://huggingface.co/Qwen/Qwen3.5-4B/resolve/'+revision+'/tokenizer.json'))
    raw=path.read_bytes();official=json.loads(raw)
    local=json.loads((ROOT/'local_qwen3_5_4b_show.json').read_bytes())['model_info']
    vocab=dict(official['model']['vocab'])
    for item in official.get('added_tokens',[]):
        assert item['content'] not in vocab or vocab[item['content']]==item['id']
        vocab[item['content']]=item['id']
    checks={'all_official_token_ids_match_local':all(local['tokenizer.ggml.tokens'][i]==s for s,i in vocab.items()),
       'vocabulary_count_matches':len(vocab)==len(local['tokenizer.ggml.tokens'])}
    merges=[' '.join(m) if isinstance(m,list) else m for m in official['model']['merges']]
    checks['all_merges_match']=merges==local['tokenizer.ggml.merges']
    runtime={}
    for name in ['version','tags']:
        dest=ROOT/('runtime_'+name+'.json');assert not dest.exists()
        body=get('http://127.0.0.1:11434/api/'+name);dest.write_bytes(body);runtime[name]=json.loads(body)
    version=runtime['version']['version']
    source_url='https://raw.githubusercontent.com/ollama/ollama/v'+version+'/model/renderers/qwen35.go'
    renderer_source_status={}
    try:
        source=get(source_url);(ROOT/'runtime_qwen35_renderer.go').write_bytes(source)
        renderer_source_status={'url':source_url,'sha256':hashlib.sha256(source).hexdigest(),'retrieved':True}
    except Exception as e:
        renderer_source_status={'url':source_url,'retrieved':False,'error':repr(e)}
    result={'checks':checks,'all_tokenizer_checks_pass':all(checks.values()),
        'tokenizer_sha256':hashlib.sha256(raw).hexdigest(),'official_token_count':len(vocab),
        'runtime_version':version,'model_inventory_entries':[m for m in runtime['tags']['models'] if m['name']=='qwen3.5:4b'],
        'runtime_renderer_source':renderer_source_status,'inference_calls':0,
        'actual_rendered_prompt_count_verified':False,
        'limitations':['Vocabulary/merges equality is not a verification of active chat rendering.',
          'Runtime source tag is an implementation reference, not a binary reproducible-build attestation.',
          'Model input/output qualification is still required before scored experiments.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    assert all(checks.values())
if __name__=='__main__':main()
