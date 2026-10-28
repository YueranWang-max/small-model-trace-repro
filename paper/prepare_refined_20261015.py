from pathlib import Path
import shutil,json,hashlib,runpy
P=Path(__file__).resolve().parent
R=P/'revision_evidence_20261015_refined'; R.mkdir(exist_ok=True)
builder=runpy.run_path(str(P/'build_submission_bundle_20261014_origin.py'))
files,_,_=builder['collect']()
for f in files+[P/'main.pdf']:
 dst=R/'before'/f.relative_to(P);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(f,dst)
out=P/'figures/refined_r_20261015';out.mkdir(exist_ok=True)
records=[]
for f in (P/'figures/distinctive_r_20261015').rglob('*.csv'):
 dst=out/f.relative_to(P/'figures/distinctive_r_20261015');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
 records.append({'source':str(f),'copy':str(dst),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
extra={'user_benefit/graph_means.csv':P/'evidence_user_benefit_20261009/graph_means.csv','output_budget_20261006/old_graphs.csv':P/'evidence_paired4b_20261006/graph_summary.csv'}
for dest,f in extra.items():
 dst=out/dest;shutil.copy2(f,dst);records.append({'source':str(f),'copy':str(dst),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(R/'INPUTS.json').write_text(json.dumps(records,indent=2))
print('Archived current manuscript; copied',len(records),'data files')
