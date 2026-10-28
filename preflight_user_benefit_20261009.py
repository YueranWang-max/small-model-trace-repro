"""Read-only resume checks. Never sends inference or restarts a device."""
from pathlib import Path
import json,hashlib,re,subprocess,sys
root=Path(__file__).resolve().parent;out=root/'logs/user_benefit_20261009'
fr=json.loads((out/'frozen.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
for name,h in fr['dependencies'].items():assert sha((root/name).read_bytes())==h,name
planned=set()
for i,orders in enumerate(fr['orders']):
 planned.add(f'g{i:02d}_source')
 for j,arms in enumerate(orders):
  for arm in arms:planned.add(f'g{i:02d}_{arm}_{j}')
assert len(planned)==168
known=[];ambiguous=[]
for path in sorted(out.glob('g*.json')):
 if path.name.endswith('.response.json'):continue
 key=path.stem;assert key in planned,'Unexpected ledger '+key
 r=json.loads(path.read_bytes());response=out/(key+'.response.json')
 if r.get('completion')!='returned' or not response.exists():ambiguous.append(key);continue
 result=r['result'];raw=json.loads(response.read_bytes())
 assert result['raw']==raw and result['raw_response_sha256']==sha(response.read_bytes())
 assert r['request_sha256']==sha(json.dumps(r['request'],sort_keys=True).encode())
 assert result['qualified'] and result['error'] is None
 known.append(key)
for path in out.glob('g*.response.json'):
 key=path.name.removesuffix('.response.json');assert (out/(key+'.json')).exists(),'Orphan raw response '+key
report={'known_completed':len(known),'planned':168,'remaining':168-len(known),'ambiguous_transmissions':ambiguous,'known_raw_and_payload_hashes_verified':True,'inference_allowed':False,'mode':'offline_ledger_check','resume_rule':'Read completed ledgers without retransmission; only send remaining planned keys after GPU health and runtime identity pass.'}
if '--check-gpu' in sys.argv:
 cmd="Get-CimInstance Win32_VideoController | Select-Object Name,Status,ConfigManagerErrorCode | ConvertTo-Json -Compress"
 result=subprocess.run(['pwsh','-NoProfile','-Command',cmd],text=True,capture_output=True)
 if result.returncode:report['device_query_error']=result.stderr.strip()
 else:
  devices=json.loads(result.stdout);devices=devices if isinstance(devices,list) else [devices]
  report['devices']=devices;nvidia=[d for d in devices if 'NVIDIA' in d['Name']]
  healthy=bool(nvidia) and all(d['Status']=='OK' and d['ConfigManagerErrorCode']==0 for d in nvidia)
  if healthy:
   probe=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
   report['telemetry_available']=probe.returncode==0
   if probe.returncode==0:report['telemetry']=probe.stdout.strip()
   else:report['telemetry_error']=(probe.stdout+probe.stderr).strip()
   report['inference_allowed']=probe.returncode==0 and not ambiguous
  else:report['device_blocked']=True
 report['mode']='device_and_ledger_check'
(out/'resume_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if ambiguous:raise SystemExit('Ambiguous transmissions: no silent replay')
