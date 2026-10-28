"""Copy audited inputs for two R figure prototypes; no visual work in Python."""
from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent;out=P/'figures/visual_pilot_20261015'
inputs={
 'method':{'dictionary.csv':P/'figures/editaplot_informed_20261013/real_case_dictionary.csv','edges.csv':P/'figures/editaplot_informed_20261013/real_case_edges.csv','inputs.csv':P/'figures/editaplot_informed_20261013/real_case_layout.csv','components.csv':P/'figures/editaplot_informed_20261013/prefix_token_components.csv'},
 'models':{'graphs_17.csv':P/'evidence_verification_20261006/graph_summary.csv','graphs_4.csv':P/'evidence_paired4b_20261006/graph_summary.csv','arms_17.csv':P/'evidence_verification_20261006/arm_summary.csv','arms_4.csv':P/'evidence_paired4b_20261006/arm_summary.csv','contrasts_17.csv':P/'evidence_verification_20261006/contrasts.csv','contrasts_4.csv':P/'evidence_paired4b_20261006/contrasts.csv'}}
manifest=[]
for n,files in inputs.items():
 d=out/n;d.mkdir(parents=True,exist_ok=True)
 for target,src in files.items():
  shutil.copy2(src,d/target);assert src.read_bytes()==(d/target).read_bytes()
  manifest.append({'figure':n,'file':target,'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(out/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Preserved',len(manifest),'audited source files')
