"""Inspect archived metadata only; generation qualification remains pending."""
import json,pathlib,hashlib,math,re
ROOT=pathlib.Path(__file__).resolve().parent/'hybrid_candidate_qualification_20261004'
def main():
    out=ROOT/'local_architecture_checks.json'
    assert not out.exists()
    raw=(ROOT/'local_qwen3_5_4b_show.json').read_bytes();local=json.loads(raw)
    official=json.loads((ROOT/'config.json').read_bytes())['text_config']
    tok=json.loads((ROOT/'tokenizer_config.json').read_bytes());info=local['model_info']
    mapping={'hidden_size':'embedding_length','intermediate_size':'feed_forward_length',
        'num_hidden_layers':'block_count','num_attention_heads':'attention.head_count',
        'head_dim':'attention.key_length','rms_norm_eps':'attention.layer_norm_rms_epsilon',
        'max_position_embeddings':'context_length','full_attention_interval':'full_attention_interval'}
    checks={k:official[k]==info['qwen35.'+v] for k,v in mapping.items()}
    tokens=info['tokenizer.ggml.tokens'];checks['vocab_size']=official['vocab_size']==len(tokens)
    checks['chat_eos_matches_official_tokenizer']=tokens[info['tokenizer.ggml.eos_token_id']]==tok['eos_token']
    tensor_count=sum(math.prod(t['shape']) for t in local['tensors'])
    checks['stored_tensor_count']=tensor_count==info['general.parameter_count']
    modelfile=local['modelfile']
    result={'local_metadata_sha256':hashlib.sha256(raw).hexdigest(),'checks':checks,
        'all_inspected_fields_match':all(checks.values()),
        'renderer':re.search(r'^RENDERER (.+)$',modelfile,re.M).group(1),
        'parser':re.search(r'^PARSER (.+)$',modelfile,re.M).group(1),
        'stored_tensor_elements':tensor_count,'nominal_model_tag':'qwen3.5:4b',
        'official_tie_word_embeddings':official['tie_word_embeddings'],
        'default_parameters':local['parameters'],
        'inference_calls':0,'tokenizer_full_vocab_and_merges_verified':False,
        'actual_rendered_prompt_tokens_verified':False,'weight_provenance_verified':False,
        'notes':['The {{ .Prompt }} template is accompanied by a qwen3.5 renderer/parser; it does not by itself imply missing chat formatting.',
          'Model config EOS endoftext and tokenizer chat EOS im_end are distinct roles; compare local chat EOS with tokenizer configuration.',
          'Stored tensor count includes separately stored tensors and is not automatically independent trained parameter count.',
          'Do not inherit Qwen3 sampling parameters or model defaults without checking official Qwen3.5 guidance and explicitly freezing all options.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
