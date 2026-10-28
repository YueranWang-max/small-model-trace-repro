"""Metadata only: preserve a useful bridge qualification while project disk is absent."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
models=['qwen3:1.7b','qwen3:4b','qwen3:8b'];shows={};result={'model_inference_calls':0,'project_disk_unavailable':True,'metadata':{}}
for model in models:
    try:
        req=urllib.request.Request('http://127.0.0.1:11434/api/show',data=json.dumps({'model':model,'verbose':True}).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=90) as response:show=json.load(response)
        shows[model]=show;path=ROOT/('bridge_metadata_'+model.replace(':','_').replace('.','_')+'.json');path.write_text(json.dumps(show,ensure_ascii=False),encoding='utf-8');mi=show.get('model_info',{})
        result['metadata'][model]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'capabilities':show.get('capabilities'),'architecture':mi.get('general.architecture'),'name':mi.get('general.name'),'context_fields':{k:v for k,v in mi.items() if k.endswith('.context_length')},'token_count':len(mi.get('tokenizer.ggml.tokens',[]))}
    except Exception as e:result['metadata'][model]={'error':repr(e)}
if 'qwen3:1.7b' in shows and 'qwen3:8b' in shows:
    a,b=shows['qwen3:1.7b'],shows['qwen3:8b'];result['same_1_7_and_8_template']=a['template']==b['template'];result['same_all_1_7_and_8_token_ids']=a['model_info']['tokenizer.ggml.tokens']==b['model_info']['tokenizer.ggml.tokens'];result['same_all_1_7_and_8_merges']=a['model_info'].get('tokenizer.ggml.merges')==b['model_info'].get('tokenizer.ggml.merges')
path=ROOT/'model_bridge_metadata_preflight_20261003.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,indent=2))
