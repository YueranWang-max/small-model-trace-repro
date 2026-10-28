"""Await independent confirmation audit, then write report and R figures only."""
from pathlib import Path
import json,time,subprocess,sys
root=Path(__file__).resolve().parent;out=root/'logs/trace_benefit_confirmation_20261010'
def save(state,**kw):
 p=out/'artifact_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'updated_unix':time.time(),**kw},indent=2),encoding='utf-8');tmp.replace(p)
if (out/'artifact_status.json').exists() and json.loads((out/'artifact_status.json').read_bytes())['state']=='complete':raise SystemExit(0)
save('waiting_for_independent_audit')
while True:
 p=out/'postprocess_status.json'
 if p.exists():
  s=json.loads(p.read_bytes())
  if s['state']=='audit_complete_report_and_paper_pending':break
  if s['state'] in ['failed','blocked_by_runner_error']:
   save('blocked',audit_status=s);raise SystemExit(1)
 time.sleep(30)
try:
 for cmd in [[sys.executable,'paper/write_trace_benefit_report_20261010.py'],[str(root.parent/'.local-r-4.6.1/bin/Rscript.exe'),'paper/plot_trace_benefit_confirmation_20261010.R']]:
  save('processing',command=cmd);r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
  with (out/'artifact_postprocess.log').open('a',encoding='utf-8') as f:f.write(str(cmd)+'\n'+r.stdout+'\n'+r.stderr+'\n')
  if r.returncode:raise RuntimeError('Artifact generation failed; inspect preserved log')
 save('complete',report='paper/TRACE_BENEFIT_CONFIRMATION_RESULTS_20261010.md',paper_revision='pending')
except Exception as exc:save('failed',error=repr(exc));raise
