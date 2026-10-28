"""Independent replay of frozen maps, source text and model-only feedback."""
import csv,hashlib,json,random,re
from pathlib import Path
import run_lookup_semantics as dev
import run_lookup_followup as follow

ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
audit=[];failures=[];figure=[]
for study,expected in [('lookup_semantics',160),('lookup_confirmation',192),('lookup_chain',208),('lookup_chain_replication',224)]:
    folder=ROOT/'logs'/study;fp=folder/'frozen.json';f=json.loads(fp.read_text());d=json.loads((folder/'reader.json').read_text());assert d['frozen_sha256']==sha(fp.read_bytes())
    for p,h in f['hashes'].items():assert sha((ROOT/p).read_bytes())==h
    tasks={t['id']:t for t in f['tasks']};count=feedback=valid=0
    for t in tasks.values():
        rng=random.Random(t['seed']);maps=[];orders=[]
        for _ in range(6):
            vals=list(range(8));rng.shuffle(vals);maps.append(dict(zip(map(str,range(8)),vals)));order=list(range(8));rng.shuffle(order);orders.append(order)
        assert maps==t['maps'] and orders==t['row_orders'];assert rng.randrange(8)==t['input']
    for row in d['rows']:
        t=tasks[row['id']];calls=row.get('calls',[row])
        for i,v in enumerate(calls):
            current=v.get('input',t['input']);step=v.get('step',t['step']);fmt='arrow' if row['arm'].startswith('legacy') else f.get('selection',{}).get(row['model'],'arrow')
            if study=='lookup_semantics':
                arm=row['arm'];fmt='arrow' if arm=='legacy_local' else arm.split('_')[0];p=dev.prompt(t,arm)
            else:
                p=follow.prompt(t,row['arm'],fmt,step,current)
                assert current==(t['input'] if i==0 else calls[i-1]['output']);feedback+=int(i>0)
            assert sha(p.encode())==v['prompt_sha256']
            blocks=re.findall(r'Step (\d+):\n(.*?)(?=\n\nStep |\nOperation:|$)',p,re.S)
            full=row['arm']=='continuous' or row['arm'].endswith('_all') or row['arm'].endswith('_noise')
            assert [int(j) for j,_ in blocks]==(list(range(1,7)) if full else [step])
            for provided_step,text in blocks:
                if fmt=='arrow':pairs=re.findall(r'(S\d+)->(S\d+)',text)
                elif fmt=='json':pairs=json.loads(text).items()
                else:pairs=re.findall(r'input is (S\d+), the output is (S\d+)',text)
                parsed={int(a[1:]):int(b[1:]) for a,b in pairs};assert parsed=={int(k):val for k,val in t['maps'][int(provided_step)-1].items()}
            raw=v.get('raw',{})
            if raw and raw['schema_valid']:
                obj=json.loads(raw['native_schema_content']);assert set(obj)=={'state'};assert obj['state'] in dev.interface.SCHEMA['properties']['state']['enum'];assert int(obj['state'][1:])==v['output'];valid+=1
            if row['arm']!='continuous':
                mapping=t['maps'][step-1];truth=mapping[str(current)];output=v['output']
                if output!=truth:
                    failures.append(dict(study=study,model=row['model'],task=t['id'],arm=row['arm'],step=step,input=current,truth=truth,output=output,input_copy=output==current,two_applications=output==mapping[str(truth)],inverse_lookup=output==next(int(k) for k,val in mapping.items() if val==current)))
            count+=1
    assert count==expected
    out=dict(study=study,calls=count,native_json_valid=valid,model_feedback_checks=feedback,seed_maps_reconstructed=True,source_rows_verified=True,frozen_code_hashes_verified=True);audit.append(out);(folder/'independent_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    summary=json.loads((folder/'summary.json').read_text())
    if study!='lookup_semantics':
        for model,arms in summary['models'].items():
            for arm,a in arms.items():
                if arm=='paired_disagreements':continue
                figure.append(dict(study=study,model=model,arm=arm,correct=a['correct'],n=a['n'],perfect_chains=a.get('perfect_chains'),tokens=a['tokens'],wall_sec=a['wall_sec']))
(ROOT/'logs/lookup_audit.json').write_text(json.dumps(dict(studies=audit,total_calls=sum(a['calls'] for a in audit)),indent=2),encoding='utf-8')
(ROOT/'logs/lookup_failure_diagnostics.json').write_text(json.dumps(dict(posthoc=True,nonexclusive_output_matches_not_mechanism=True,rows=failures),indent=2),encoding='utf-8')
outdir=ROOT/'paper/figures/data';outdir.mkdir(parents=True,exist_ok=True)
with (outdir/'lookup_progress.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(figure[0]));writer.writeheader();writer.writerows(figure)
print(json.dumps(audit,indent=2));print('total',sum(a['calls'] for a in audit))
folder=ROOT/'logs/lookup_confirmation'
f=json.loads((folder/'frozen.json').read_text());d=json.loads((folder/'reader.json').read_text())
indexed={(r['model'],r['id'],r['arm']):r['calls'][0]['output'] for r in d['rows']}
paired={}
for model in f['models']:
    wins=losses=0
    for t in f['tasks']:
        left=indexed[(model,t['id'],'selected_all')]==t['answer']
        right=indexed[(model,t['id'],'selected_noise')]==t['answer']
        wins+=left and not right;losses+=right and not left
    paired[model]=dict(all_only_correct=wins,noise_only_correct=losses)
(folder/'additional_paired_diagnostics.json').write_text(
    json.dumps(dict(supplementary_descriptive=True,all_vs_noise=paired),indent=2),encoding='utf-8')
