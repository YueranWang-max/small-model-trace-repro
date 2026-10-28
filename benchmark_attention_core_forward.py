import json,time
from pathlib import Path
from run_qasper_attention_reader import ROOT,sha

OUT=ROOT/'logs/attention_core_forward';PROTOCOL=ROOT/'protocol_attention_core_forward_20261003.md'

def main():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json';assert not path.exists();fr=json.loads((ROOT/'logs/qasper_attention_reader/frozen.json').read_text(encoding='utf-8'));examples=fr['calibration'][:4];start=time.perf_counter();import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);modelpath=Path(fr['model_path'])
    for n,h in fr['checkpoint_hashes'].items():assert sha((modelpath/n).read_bytes())==h
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,attn_implementation='eager',dtype=torch.float32).to('cpu').eval();load=time.perf_counter()-start
    inputs=[tok('Paper excerpt:\n'+c['body']+'\nQuestion: '+c['question']+'\nAnswer:',return_tensors='pt',add_special_tokens=False) for c in examples]
    assert all(x['input_ids'].shape[1]<=512 for x in inputs)
    out={'date':'2026-10-03','torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','load_sec':load,'hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [PROTOCOL,ROOT/'benchmark_attention_core_forward.py',ROOT/'logs/qasper_attention_reader/frozen.json']},'checkpoint_hashes':fr['checkpoint_hashes'],'rows':[],'equivalence':[]}
    with torch.inference_mode():
        for i,x in enumerate(inputs):
            a=model(**x,output_attentions=True,use_cache=False);b=model.model(**x,output_attentions=True,use_cache=False);assert len(a.attentions)==len(b.attentions);diff=max(float((u-v).abs().max()) for u,v in zip(a.attentions,b.attentions));equal=all(torch.equal(u,v) for u,v in zip(a.attentions,b.attentions));out['equivalence'].append({'id':examples[i]['question_id'],'max_abs_diff':diff,'bitwise_equal':equal});del a,b
        for fn in (model,model.model):y=fn(**inputs[0],output_attentions=True,use_cache=False);del y
        for rep in range(3):
            for i,x in enumerate(inputs):
                order=('full','core') if (rep+i)%2==0 else ('core','full');row={'repeat':rep,'input':i,'tokens':x['input_ids'].shape[1]}
                for name in order:
                    fn=model if name=='full' else model.model;start=time.perf_counter();y=fn(**x,output_attentions=True,use_cache=False);row[name+'_sec']=time.perf_counter()-start;del y
                out['rows'].append(row)
    out['full_sec']=sum(r['full_sec'] for r in out['rows']);out['core_sec']=sum(r['core_sec'] for r in out['rows']);out['speedup']=out['full_sec']/out['core_sec'];out['equivalence_pass']=all(r['bitwise_equal'] for r in out['equivalence']);out['not_qa_or_end_to_end_speed']=True;path.write_text(json.dumps(out,indent=2),encoding='utf-8');print({k:v for k,v in out.items() if k not in ('rows','hashes','checkpoint_hashes')},flush=True)

if __name__=='__main__':main()
