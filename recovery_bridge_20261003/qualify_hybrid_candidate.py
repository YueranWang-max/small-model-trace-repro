"""Archive candidate architecture metadata; no inference or weight downloads."""
import hashlib,json,pathlib,urllib.request,collections
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'hybrid_candidate_qualification_20261004'
def fetch(url):
    with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'Preserve existing qualification archive.'
    OUT.mkdir(exist_ok=True)
    base='https://huggingface.co/'
    repo='Qwen/Qwen3.5-4B'
    meta=fetch(base+'api/models/'+repo)
    (OUT/'repository.json').write_bytes(meta)
    revision=json.loads(meta)['sha']
    records={}
    for name in ['config.json','tokenizer_config.json']:
        url=base+repo+'/resolve/'+revision+'/'+name
        raw=fetch(url);(OUT/name).write_bytes(raw)
        records[name]={'source_url':url,'sha256':hashlib.sha256(raw).hexdigest()}
    config=json.loads((OUT/'config.json').read_bytes())['text_config']
    result={'revision':revision,'sources':records,'inference_calls':0,
        'layer_counts':dict(collections.Counter(config['layer_types'])),
        'full_attention_interval':config['full_attention_interval'],
        'hidden_layers':config['num_hidden_layers'],'vocab_size':config['vocab_size'],
        'max_position_embeddings':config['max_position_embeddings'],
        'hypothesis':'Hybrid state processing motivates within-model prefix/postfix trace controls.',
        'limitations':['Architecture metadata alone establishes no performance or memory benefit.',
          'Cross-family model comparisons cannot isolate architecture causally.',
          'Local tokenizer, model architecture, digest and rendered input counts require qualification before inference.',
          'This does not change the frozen active Qwen3 bridge protocol.']}
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
