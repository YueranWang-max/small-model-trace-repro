"""Re-run closed study audits in a separate root and compare their outputs."""
from pathlib import Path
import subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
STAGE=ROOT/'paper/reproducibility_staging_20261006'
checks=[]
for script,folder,names in [
 ('audit_output_budget_control_20261006.py','evidence_output_budget_control_20261006',['summary.json','manifest.json','audited_calls.csv','old_direct_calls.csv','graph_summary.csv']),
 ('audit_context_recovery_20261006.py','evidence_context_recovery_20261006',['summary.json','manifest.json','audited_calls.csv','arm_summary.csv'])]:
 assert json.loads((ROOT/'paper'/folder/'summary.json').read_bytes())['complete']
 result=subprocess.run([sys.executable,str(STAGE/'paper'/script)],cwd=STAGE,capture_output=True,text=True)
 assert result.returncode==0,(script,result.stderr,result.stdout[-3000:])
 assert json.loads((STAGE/'paper'/folder/'summary.json').read_bytes())['complete']
 for name in names:
  a=ROOT/'paper'/folder/name;b=STAGE/'paper'/folder/name
  assert a.read_bytes()==b.read_bytes(),str(a)
  checks.append({'path':str(a.relative_to(ROOT)),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'relocated_byte_identical':True})
report={'scope':'Fresh read-only CPU audits of saved records on the same host in a separate root; not independent inference replication or external execution attestation','audits':2,'files':checks,'all_passed':True}
(ROOT/'paper/revision_evidence_20261006/relocated_final_diagnostic_audits.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Both relocated diagnostic audits complete; all nine output files byte-identical.')
