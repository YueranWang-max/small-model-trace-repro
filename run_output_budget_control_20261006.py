"""Fixed same-question output-budget diagnostic, with new durable raw calls."""
from pathlib import Path
import json,time,sys
import run_official_graph_paired4b_20261006 as client
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'logs/output_budget_control_20261006'
PROTOCOL=ROOT/'protocol_output_budget_control_20261006.md'
BASE=ROOT/'logs/official_graph_paired4b_20261006'
CAP=28672

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    names=[Path(__file__).name,PROTOCOL.name,'run_official_graph_paired4b_20261006.py',
           'run_official_graph_native_probe.py','run_official_graph_long_boundary.py',
           'data/qwen3_tokenizer_official/tokenizer.json','logs/official_graph_paired4b_20261006/frozen.json']
    deps={n:client.sha((ROOT/n).read_bytes()) for n in names}
    if (OUT/'frozen.json').exists():
        fr=json.loads((OUT/'frozen.json').read_bytes());assert fr['dependencies']==deps;return fr
    old=json.loads((BASE/'frozen.json').read_bytes());assert CAP==5*old['source_cap']+old['reader_cap']
    jobs=[]
    for i,row in enumerate(old['selected']):
        user=row['prompt']+'\n'+client.FORMAT;n=client.input_count(client.SYSTEM,user)
        assert n+CAP+64<=32768 and set(client.oracle(row))==set(row['answer_nodes'])
        for j,seed in enumerate(old['reader_seeds']):
            jobs.append({'key':f'g{i:02d}_directlong_{j}','graph_index':i,'row_index':row['row_index'],
                         'repeat':j,'seed':seed,'user':user,'expected_input_tokens':n,
                         'old_direct_key':f'g{i:02d}_direct_{j}'})
    inventory=client.get('tags');assert next(m['digest'] for m in inventory['models'] if m['name']==client.MODEL)==client.DIGEST
    version=client.get('version');assert version['version']=='0.35.1'
    fr={'dependencies':deps,'selected':old['selected'],'jobs':jobs,'digest':client.DIGEST,
        'model':client.MODEL,'server_version':version,'inventory':inventory,'context':32768,
        'output_cap':CAP,'old_direct_cap':8192,'planned':60,'bootstrap_seed':202610067,
        'rationale':'Diagnostic selected after paired4b interim truncation; same known questions, not fresh confirmation',
        'budget_scope':'Equal maximum generated-token allowances, not equal total logical tokens or compute'}
    save(OUT/'frozen.json',fr);return fr

def main():
    fr=freeze()
    if '--prepare' in sys.argv:print('Frozen 60 new direct-long calls; all full-input budgets fit',flush=True);return
    assert json.loads((BASE/'status.json').read_bytes())['state']=='completed'
    assert not (OUT/'results.json').exists() and not list(OUT.glob('g*.json')),'No silent replay or overwrite'
    client.OUT=OUT;records=[]
    try:
        for job in fr['jobs']:
            r=client.call(job['key'],job['user'],CAP,job['seed'])
            assert not r.get('budget_declined') and r['raw']['eval_count']<=CAP
            r={**r,'key':job['key'],'row_index':job['row_index'],'arm':'direct_long','repeat':job['repeat']}
            row=fr['selected'][job['graph_index']];r['score']=client.score(r['raw'],row['answer_nodes'],r['error'])
            records.append(r);save(OUT/'results.json',records)
            save(OUT/'status.json',{'state':'running','recorded':len(records),'planned':60,'latest':job['key'],'updated_unix':time.time()})
            print(job['key'],r['raw'].get('done_reason'),r['score'],flush=True)
            if job['repeat']==4:time.sleep(60)
        save(OUT/'status.json',{'state':'completed','recorded':60,'planned':60,'updated_unix':time.time()})
    except Exception as exc:
        save(OUT/'status.json',{'state':'stopped_on_exception','recorded':len(records),'planned':60,'error':repr(exc),'updated_unix':time.time()});raise

if __name__=='__main__':main()
