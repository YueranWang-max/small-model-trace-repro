"""One finite serial queue for two already-frozen experiments; no scheduler/retry."""
from pathlib import Path
import json,subprocess,sys,urllib.request,time
ROOT=Path(__file__).resolve().parent
LOG=ROOT/'paper/revision_evidence_20261006/finite_queue_status.json'
def save(state,**kw):
 LOG.write_text(json.dumps({'state':state,'updated_unix':time.time(),**kw},indent=2),encoding='utf-8')
def api(route,payload=None):
 req=urllib.request.Request('http://127.0.0.1:11434/api/'+route,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
def run(name):
 save('running',phase=name);print('QUEUE_PHASE',name,flush=True)
 subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,check=True)
def complete(directory):
 d=json.loads((ROOT/directory/'summary.json').read_bytes());assert d['complete'],directory

def main():
 assert json.loads((ROOT/'logs/official_graph_paired4b_20261006/status.json').read_bytes())['state']=='completed'
 assert api('version')['version']=='0.35.1'
 tags={m['name']:m['digest'] for m in api('tags')['models']}
 budget=json.loads((ROOT/'logs/output_budget_control_20261006/frozen.json').read_bytes())
 assert tags[budget['model']]==budget['digest']
 run('paper/audit_paired4b_20261006.py');complete('paper/evidence_paired4b_20261006')
 run('run_output_budget_control_20261006.py')
 run('paper/audit_output_budget_control_20261006.py');complete('paper/evidence_output_budget_control_20261006')
 save('releasing_own_4b_model');api('generate',{'model':budget['model'],'keep_alive':0})
 fr=json.loads((ROOT/'logs/context_recovery_20261006/frozen.json').read_bytes())
 tags={m['name']:m['digest'] for m in api('tags')['models']}
 assert tags[fr['model']]==fr['digest'],'Do not substitute the updated public tag'
 assert api('version')['version']=='0.35.1'
 run('run_context_recovery_20261006.py')
 run('paper/audit_context_recovery_20261006.py');complete('paper/evidence_context_recovery_20261006')
 api('generate',{'model':fr['model'],'keep_alive':0})
 save('completed',studies=['output_budget_control_20261006','context_recovery_20261006'])
 print('FINITE_QUEUE_COMPLETED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:save('stopped_on_exception',error=repr(exc));raise
