"""Monitor the existing study and run audit/report/R plotting once on completion.

Never launches inference or retries ambiguous requests. May run independently
of the interactive chat, preserving a durable completion/error report.
"""
from pathlib import Path
import time,json,subprocess,sys
root=Path(__file__).resolve().parent;out=root/'logs/user_benefit_20261009'
def state(o):
 p=out/'postprocess_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({**o,'updated_unix':time.time()},indent=2),encoding='utf-8');tmp.replace(p)
state({'state':'waiting_for_existing_runner'})
while True:
 path=out/'status.json'
 if path.exists():
  s=json.loads(path.read_bytes())
  if s['state']=='completed':break
  if s['state']=='stopped_on_exception':state({'state':'blocked_by_runner_error','error':s['error']});raise SystemExit(1)
 time.sleep(30)
try:
 commands=[[sys.executable,'paper/audit_user_benefit_20261009.py'],[sys.executable,'paper/write_user_benefit_report_20261009.py'],[str(root.parent/'.local-r-4.6.1/bin/Rscript.exe'),'paper/plot_user_benefit_20261009.R']]
 for cmd in commands:
  state({'state':'processing','command':cmd})
  result=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
  with (out/'postprocess.log').open('a',encoding='utf-8') as f:f.write(str(cmd)+'\n'+result.stdout+'\n'+result.stderr+'\n')
  if result.returncode:raise RuntimeError(f'Postprocess failed: {cmd}; inspect preserved log')
 state({'state':'complete','report':'paper/USER_BENEFIT_RESULTS_20261009.md','independent_audit':'paper/evidence_user_benefit_20261009/summary.json'})
except Exception as exc:state({'state':'failed','error':repr(exc)});raise
