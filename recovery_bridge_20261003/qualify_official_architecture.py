"""Metadata-only post-freeze qualification; no inference and no weight downloads."""
import pathlib,json,urllib.request,hashlib,math
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'architecture_qualification_20261004'
def fetch(url):
    with urllib.request.urlopen(url,timeout=90) as response:return response.read()
def main():
    assert not OUT.exists(),'Do not overwrite archived source metadata.'
    OUT.mkdir();records={}
    specs=[('qwen3:1.7b','Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','bridge_metadata_qwen3_1_7b.json'),('qwen3:8b','Qwen/Qwen3-8B',None,'bridge_metadata_qwen3_8b.json')]
    fields={'hidden_size':'qwen3.embedding_length','intermediate_size':'qwen3.feed_forward_length','num_hidden_layers':'qwen3.block_count','num_attention_heads':'qwen3.attention.head_count','num_key_value_heads':'qwen3.attention.head_count_kv','head_dim':'qwen3.attention.key_length','rms_norm_eps':'qwen3.attention.layer_norm_rms_epsilon','max_position_embeddings':'qwen3.context_length','rope_theta':'qwen3.rope.freq_base'}
    for model,repo,revision,localfile in specs:
        if revision is None:
            meta=fetch('https://huggingface.co/api/models/'+repo);(OUT/'qwen3_8b_repo_metadata.json').write_bytes(meta);revision=json.loads(meta)['sha']
        url='https://huggingface.co/'+repo+'/resolve/'+revision+'/config.json';raw=fetch(url);config=json.loads(raw)
        path=OUT/(model.replace(':','_').replace('.','_')+'_official_config.json');path.write_bytes(raw)
        local=json.loads((ROOT/localfile).read_text(encoding='utf-8'));info=local['model_info'];checks={key:config[key]==info[value] for key,value in fields.items()}
        checks['vocab_size']=config['vocab_size']==len(info['tokenizer.ggml.tokens'])
        tensors=local['tensors'];stored_count=sum(math.prod(t['shape']) for t in tensors)
        assert stored_count==info['general.parameter_count']
        embedding_elements=config['vocab_size']*config['hidden_size']
        record=dict(source_url=url,revision=revision,config_sha256=hashlib.sha256(raw).hexdigest(),architecture_checks=checks,all_architecture_fields_match=all(checks.values()),official_tie_word_embeddings=config['tie_word_embeddings'],gguf_stored_tensor_elements=stored_count,one_embedding_matrix_elements=embedding_elements,stored_count_minus_one_embedding=stored_count-embedding_elements,embedding_output_tensors=[t for t in tensors if t['name'] in ['token_embd.weight','output.weight']],qualification_timing='Post-freeze metadata check, not pre-call qualification',weight_identity_verified=False)
        records[model]=record
    result=dict(model_inference_calls=0,records=records,limitations=['Matching architecture and tokenizer do not prove weight provenance.','Stored GGUF tensor elements are not necessarily independent trained parameters.','Mixed quantization can distinguish tensors originating from shared weights; no equality of quantized matrices is assumed.','Do not reinterpret performance contrasts as size-only causal effects.'])
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
