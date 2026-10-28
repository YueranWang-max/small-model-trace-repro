"""Offline real causal attention extraction on fitting-source paragraphs."""
import json,time,sys,platform
from pathlib import Path
from run_qasper_paragraph_elements import ROOT,SOURCE,spans,sha
from build_qasper_relation_ledger import paragraphs

OUT=ROOT/'logs/qasper_attention_probe';PROTOCOL=ROOT/'protocol_qasper_attention_probe_20261002.md'
MODEL_DIR=Path('C:/Users/dragoncat/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);fp=OUT/'frozen.json';assert not fp.exists();src=json.loads(SOURCE.read_text(encoding='utf-8'));split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));chosen=None
    for pid in sorted(split['fit'],key=lambda p:sha(p.encode())):
        ps=paragraphs(src[pid]);eligible=[(i,p) for i,p in enumerate(ps) if len(p['text'])<=1200 and 2<=len(spans(p['text']))<=6]
        if len(eligible)>=2 and len(src[pid]['qas'])>=2:chosen=(pid,eligible[:2]);break
    assert chosen
    snapshots=[p for p in MODEL_DIR.iterdir() if (p/'model.safetensors').exists() and (p/'model.safetensors').stat().st_size>1000000];assert snapshots;model_path=sorted(snapshots)[0]
    pid,ps=chosen;fr={'paper_id':pid,'paragraphs':[{'id':i,**p} for i,p in ps],'questions':[q['question'] for q in src[pid]['qas'][:2]],'model_path':str(model_path),'checkpoint_hashes':{p.name:sha(p.read_bytes()) for p in model_path.iterdir() if p.name in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors')},'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [SOURCE,PROTOCOL,ROOT/'probe_qasper_attention.py',ROOT/'run_qasper_paragraph_elements.py',ROOT/'build_qasper_relation_ledger.py',ROOT/'logs/qasper_fingerprint/split.json']}}
    fp.write_text(json.dumps(fr,indent=2,ensure_ascii=False),encoding='utf-8');print({'paper_id':pid,'paragraphs':2,'passes':4,'checkpoint':str(model_path)},flush=True)

def run():
    fp=OUT/'frozen.json';fr=json.loads(fp.read_text(encoding='utf-8'))
    for name,h in fr['hashes'].items():assert sha((ROOT/name).read_bytes())==h
    model_path=Path(fr['model_path'])
    for name,h in fr['checkpoint_hashes'].items():assert sha((model_path/name).read_bytes())==h
    start=time.perf_counter();import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,attn_implementation='eager',torch_dtype=torch.float32).to('cpu').eval();load_sec=time.perf_counter()-start
    out={'frozen_sha256':sha(fp.read_bytes()),'model':'Qwen2.5-0.5B-Instruct','device':'cpu','dtype':'float32','torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'load_sec':load_sec,'rows':[],'not_qa_efficacy':True}
    for p in fr['paragraphs']:
        for query in fr['questions']:
            prefix='Paper excerpt:\n';body=p['text'];prompt=prefix+body+'\nQuestion: '+query+'\nAnswer:';enc=tokenizer(prompt,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False);offsets=enc.pop('offset_mapping')[0].tolist();n=enc['input_ids'].shape[1]
            row={'paragraph_id':p['id'],'question':query,'input_tokens':n,'prompt_sha256':sha(prompt.encode())}
            if n>512:row['skipped']='over512_no_truncation';out['rows'].append(row);continue
            start=time.perf_counter()
            with torch.inference_mode():g=model(**enc,output_attentions=True,use_cache=False)
            at=g.attentions;assert at and all(a is not None for a in at);assert all(torch.isfinite(a).all() for a in at);score=torch.stack([a[0,:,-1,:].max(dim=0).values for a in at[-4:]]).mean(dim=0);assert float(score.sum())>0
            row.update({'forward_sec':time.perf_counter()-start,'attention_layers':len(at),'attention_shape':list(at[-1].shape),'query_source_attention_mass_mean_heads_last_layer':float(at[-1][0,:,-1,:len(offsets)].mean(dim=0)[[i for i,(a,b) in enumerate(offsets) if b>len(prefix) and a<len(prefix)+len(body)]].sum()),'sentences':[]})
            for j,s in enumerate(spans(body)):
                lo=len(prefix)+s['start'];hi=lo+s['length'];ids=[i for i,(a,b) in enumerate(offsets) if b>lo and a<hi];assert ids
                rec={'sentence_id':j,'text':s['text'],'char_start':s['start'],'char_length':s['length'],'paragraph_sha256':sha(body.encode()),'byte_start':len(body[:s['start']].encode()),'byte_length':len(s['text'].encode()),'source_tokens':len(ids),'attention_mean':float(score[ids].mean()),'attention_sum':float(score[ids].sum())};assert body[rec['char_start']:rec['char_start']+rec['char_length']]==s['text'];row['sentences'].append(rec)
            row['rank']=sorted(range(len(row['sentences'])),key=lambda j:-row['sentences'][j]['attention_mean']);out['rows'].append(row);(OUT/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(p['id'],query,round(row['forward_sec'],3),row['rank'],flush=True);del g,at,score
    out['completed_passes']=sum('sentences' in r for r in out['rows']);out['forward_sec']=sum(r.get('forward_sec',0) for r in out['rows']);out['input_tokens']=sum(r['input_tokens'] for r in out['rows'] if 'sentences' in r);(OUT/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
