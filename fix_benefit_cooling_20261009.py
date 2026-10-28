"""Version a thermal-only execution correction after a clean between-call stop."""
from pathlib import Path
import json,hashlib,shutil,time
root=Path(__file__).resolve().parent;out=root/'logs/user_benefit_20261009'
s=json.loads((out/'status.json').read_bytes());assert s['state']=='stopped_on_exception' and 'STOP between calls' in s['error']
snap=out/'execution_correction_cooling';assert not snap.exists();snap.mkdir()
runner=root/'run_user_benefit_20261009.py';frozen=out/'frozen.json'
shutil.copy2(runner,snap/runner.name);shutil.copy2(frozen,snap/'frozen_before.json');shutil.copy2(out/'status.json',snap/'status_before.json')
text=runner.read_text(encoding='utf-8')
old=" while gpu()['temperature_c']>=82:\n  save(OUT/'cooling.json',{'updated_unix':time.time(),'telemetry':gpu()});time.sleep(30)"
new=" if gpu()['temperature_c']>=82:\n  while gpu()['temperature_c']>=76:\n   save(OUT/'cooling.json',{'updated_unix':time.time(),'telemetry':gpu()});time.sleep(30)"
assert old in text;runner.write_text(text.replace(old,new),encoding='utf-8')
fr=json.loads(frozen.read_bytes());before=fr['dependencies'][runner.name];after=hashlib.sha256(runner.read_bytes()).hexdigest()
amendment={'scope':'Execution-only thermal correction; no change to selection, seeds, model, prompts, source/read budgets, scoring or planned sample.','reason':'Implement the existing protocol below76C cooldown after82C trigger. Original loop resumed below82C.','recorded_calls_before_change':s['recorded'],'old_runner_sha256':before,'new_runner_sha256':after,'old_runner_snapshot':'execution_correction_cooling/'+runner.name,'old_frozen_snapshot':'execution_correction_cooling/frozen_before.json','updated_unix':time.time()}
fr['dependencies'][runner.name]=after;fr['execution_amendments']=[amendment]
oldfr=json.loads((snap/'frozen_before.json').read_bytes())
assert {k:v for k,v in fr.items() if k not in ['dependencies','execution_amendments']}=={k:v for k,v in oldfr.items() if k!='dependencies'}
tmp=frozen.with_suffix('.tmp');tmp.write_text(json.dumps(fr,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(frozen)
(snap/'AMENDMENT.json').write_text(json.dumps(amendment,indent=2),encoding='utf-8')
(out/'STOP').unlink()
print(json.dumps(amendment,indent=2))
