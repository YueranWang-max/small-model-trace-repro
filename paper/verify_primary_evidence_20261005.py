"""Independent read-only recheck of archived evidence, not a new model run.

No imports from solver, original scorer or prior audit implementations.
Local archives support consistency checks, not external attestation of execution.
"""
from pathlib import Path, PureWindowsPath
import collections, csv, hashlib, json, re, statistics
import pyarrow.parquet as pq
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/evidence_verification_20261005';OUT.mkdir(exist_ok=True)
BR=ROOT/'recovery_bridge_20261003'
def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edges(prompt):
 block=prompt.split('Here is the graph to operate on:',1)[1].split('Operation:',1)[0]
 return re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$',block,re.M)
def gold(case):
 ed=edges(case['prompt']);assert ed
 op=case['prompt'].rsplit('Operation:',1)[1]
 if case['problem_type']=='parents':
  target=re.search(r'Find the parents of node ([0-9a-f]{10})',op)[1]
  return {u for u,v in ed if v==target and u!=target}
 target,depth=re.search(r'Perform a BFS from node ([0-9a-f]{10}).*?depth (\d+)',op).groups()
 adj=collections.defaultdict(set)
 for u,v in ed:adj[u].add(v)
 queue=collections.deque([target]);distance={target:0}
 while queue:
  u=queue.popleft()
  for v in adj[u]:
   if v not in distance:distance[v]=distance[u]+1;queue.append(v)
 return {v for v,d in distance.items() if d==int(depth) and v!=target}
def parse(raw,strict=False,error=None):
 text=(raw.get('message') or {}).get('content','') or ''
 m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',text.strip() if strict else text.rstrip().split('\n')[-1].strip())
 valid=raw.get('done_reason')=='stop' and not error and not raw.get('error') and bool(m)
 nodes=[v.strip() for v in m[1].split(',') if v.strip()] if m else []
 if strict:valid=valid and all(re.fullmatch('[A-Za-z0-9_.-]+',v) for v in nodes)
 return valid,nodes
def score(valid,nodes,target):
 p=set(nodes);return ((2*len(p&target)/(len(p)+len(target)) if p or target else 1.) if valid else 0.,float(valid and p==target))
def check_score(f,e,s):
 assert abs(f-float(s['f1']))<1e-12 and abs(e-float(s['em']))<1e-12
def archive_path(name):
 # Map preserved original Windows archive paths to this checkout, without
 # editing frozen manifests or consulting a different copy of the evidence.
 win=PureWindowsPath(name)
 if win.drive:
  relative=win.relative_to(PureWindowsPath('D:/project/tangjie/trace_as_state_repro'))
  path=ROOT.joinpath(*relative.parts)
 else:
  path=Path(name)
  if not path.is_absolute():path=ROOT.joinpath(*win.parts)
 assert path.resolve().is_relative_to(ROOT.resolve()),name
 return path

def deps(f):
 checks=[]
 for path,expected in f.get('dependencies',{}).items():
  p=archive_path(path)
  actual=sha(p);assert actual==expected,(str(p),'dependency changed')
  checks.append({'path':str(p.relative_to(ROOT)),'sha256':actual})
 return checks
def requestcheck(req,digest,expected,rec,raw,ps):
 assert hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()==digest
 if not raw:return False
 assert raw.get('done') is True and raw['model']==req['model']
 assert type(raw['prompt_eval_count']) is int and type(raw['eval_count']) is int
 assert raw['eval_count']<=req['options']['num_predict']
 # The interrupted native long trial retains a returned unqualified prompt count.
 countok=raw['prompt_eval_count']==expected
 resident=[m for m in ps.get('models',[]) if m.get('name',m.get('model'))==req['model'] or m.get('model')==req['model']]
 # Absence is an archived qualification failure, not evidence to discard.
 return countok

dataset=BR/'graphwalks_128k_and_shorter.parquet'
assert sha(dataset)=='54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
rows=pq.read_table(dataset).to_pylist();assert len(rows)==750
assert all(gold(c)==set(c['answer_nodes']) for c in rows)
studies=[];call_manifest=[];graphs=[]
def casescheck(f):
 cs=f.get('cases',f.get('selected',[]));cs=[c.get('paper_row',c) for c in cs]
 for c in cs:
  assert c['prompt']==rows[c['row_index']]['prompt']
  assert set(c['answer_nodes'])==gold(c)==set(rows[c['row_index']]['answer_nodes'])
  graphs.append({'row_index':c['row_index'],'edge_signature':hashlib.sha256(json.dumps(sorted(edges(c['prompt']))).encode()).hexdigest()})
 return {c['row_index']:c for c in cs}
def recordpath(p):call_manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})

oldnames=['official_graph_five_trace','official_graph_identity_recode','official_graph_reasoning_mode','official_graph_indexed_source','official_graph_long_boundary','official_graph_sampling_control']
for name in oldnames:
 folder=ROOT/'logs'/name;f=load(folder/'frozen.json');cs=casescheck(f);dependency=deps(f)
 results=load(folder/'results.json');groups=collections.defaultdict(list);tokens=0
 for r in results:
  ledger=archive_path(r['attempt_ledger']);rec=load(ledger);recordpath(ledger)
  assert rec['result']['raw']==r['raw'] and rec['request']==r['request']
  assert rec['result']['request']==r['request'] and r['attempts']==1
  ok=requestcheck(r['request'],r['request_sha256'],r['expected_input_tokens'],rec,r['raw'],r['runtime_ps']);assert ok
  assert any(m.get('digest')==f['identity']['digest'] and m.get('context_length',0)>=f['ctx'] for m in r['runtime_ps']['models'])
  c=cs[r['row_index']];valid,pred=parse(r['raw'],error=r['error']);arm=r.get('arm',r.get('regime'))
  if 'inverse' in c:
   enc=arm if name=='official_graph_identity_recode' else ('original' if arm.startswith('original') else 'compact')
   inv=c['inverse'][enc]
   if name=='official_graph_identity_recode':
    valid=valid and all(v in inv for v in pred);pred=[inv[v] for v in pred if v in inv]
   else:pred=[inv.get(v,'<unknown-node:'+v+'>') for v in pred]
  a,b=score(valid,pred,gold(c));check_score(a,b,r['score'])
  cost=r['raw']['prompt_eval_count']+r['raw']['eval_count'];tokens+=cost
  groups[arm].append({'row':r['row_index'],'f1':a,'em':b,'valid':valid,'tokens':cost})
 arms={}
 for arm,rs in groups.items():
  bygraph=collections.defaultdict(list)
  for r in rs:bygraph[r['row']].append(r)
  arms[arm]={'saved':len(rs),'graphs':len(bygraph),'f1':statistics.mean(statistics.mean(v['f1'] for v in rr) for rr in bygraph.values()),'em':statistics.mean(statistics.mean(v['em'] for v in rr) for rr in bygraph.values()),'normal_valid':sum(r['valid'] for r in rs),'tokens_sum':sum(r['tokens'] for r in rs)}
 intervals={}
 if name in ['official_graph_five_trace','official_graph_identity_recode']:
  order=list(cs);n=len(order);seed=202610040 if n==4 else 202610064
  draws=np.random.default_rng(seed).integers(0,n,size=(20000,n))
  left='pre' if n==4 else 'compact';controls=['reread','post'] if n==4 else ['original','long_alias']
  means=lambda arm:np.array([statistics.mean(r['f1'] for r in groups[arm] if r['row']==i) for i in order])
  for control in controls:
   delta=means(left)-means(control);ci=np.quantile(delta[draws].mean(axis=1),[.025,.975])
   intervals[left+'-'+control]={'mean':float(delta.mean()),'percentile95':ci.tolist(),'graphs':n,'draws':20000,'seed':seed}
 if name=='official_graph_five_trace':
  arms['pre']['full_deployment_sum']=arms['source']['tokens_sum']+arms['pre']['tokens_sum']/5
  arms['reread']['single_deployment_sum']=arms['reread']['tokens_sum']/5
  assert abs(arms['pre']['full_deployment_sum']/arms['reread']['single_deployment_sum']-8.066638369)<1e-5
 # Stored aggregate is a second consistency check, not the source of scores.
 summary=load(folder/'summary.json') if (folder/'summary.json').exists() else {}
 for a in summary.get('summary',[]):
  key=a.get('arm',a.get('regime'))
  if key in arms:
   assert abs(arms[key]['f1']-a['f1'])<1e-12 and abs(arms[key]['em']-a['em'])<1e-12
 saved_ledgers=len(results)
 if name=='official_graph_indexed_source':
  p=folder/'attempt_095.json';r=load(p);recordpath(p)
  assert r['completion']=='pending' and r['transmission']=='attempting_once' and 'result' not in r
  assert hashlib.sha256(json.dumps(r['request'],sort_keys=True).encode()).hexdigest()==r['request_sha256']
  assert not (folder/'attempt_096.json').exists();saved_ledgers+=1
  table=list(csv.DictReader((folder/'interrupted_figure_summary.csv').open(encoding='utf-8-sig')))
  for t in table:
   rs=groups[t['arm']];unknown=16-len(rs);lo=sum(r['f1'] for r in rs)/16;hi=lo+unknown/16
   assert abs(lo-float(t['f1_lower']))<1e-12 and abs(hi-float(t['f1_upper']))<1e-12
   assert sum(r['tokens'] for r in rs)==int(t['tokens_lower'])
 studies.append({'name':name,'frozen_sha256':sha(folder/'frozen.json'),'dependencies':dependency,'saved_response_rescoring':'pass','calls':saved_ledgers,'returned':len(results),'known_tokens':tokens,'arms':arms,'independently_recomputed_paired_intervals':intervals})

for name in ['hybrid_source_qualification','hybrid_five_trace_replication','partial_state_study_20261004']:
 folder=BR/name;f=load(folder/'frozen.json');cs=casescheck(f);dependency=deps(f);au=load(folder/'independent_audit.json')
 prior={r['ordinal']:r for r in au.get('records',au.get('reader_results',[]))};calls=list(sorted(folder.glob('attempt_*.json')))
 tokens=qualified=returned=0;groups=collections.defaultdict(list);exceptions=[]
 for path in calls:
  rec=load(path);recordpath(path);job=rec.get('job',rec.get('plan'));ordinal=job['ordinal'];raw=rec.get('raw',{})
  req=rec.get('request',job.get('request'));digest=rec.get('request_sha256',job.get('request_sha256'));expected=rec.get('expected_prompt_tokens',job.get('expected_prompt_tokens'))
  assert hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()==digest
  if not raw:
   exceptions.append({'ordinal':ordinal,'completion':rec['completion'],'usage':'unknown' if rec.get('transmission')=='attempting_once' else 'not sent'});continue
  returned+=1;tokens+=raw['prompt_eval_count']+raw['eval_count']
  ps=rec.get('runtime_ps',rec.get('own_runtime_ps',{}));ok=requestcheck(req,digest,expected,rec,raw,ps)
  resident=any(m.get('digest')==f['digest'] and m.get('context_length',0)>=f.get('ctx',65536) for m in ps['models'])
  good=ok and resident;qualified+=good
  if ordinal in prior and 'qualified' in prior[ordinal]:assert good==prior[ordinal]['qualified']
  if not good:exceptions.append({'ordinal':ordinal,'completion':'returned unqualified','actual_input':raw['prompt_eval_count'],'expected_input':expected})
  if job['row_index'] is None or not good:continue
  valid,pred=parse(raw,strict=name=='partial_state_study_20261004');a,b=score(valid,pred,gold(cs[job['row_index']]))
  if ordinal in prior and prior[ordinal].get('f1') is not None:check_score(a,b,prior[ordinal])
  groups[job.get('arm',job['phase'])].append({'ordinal':ordinal,'row':job['row_index'],'repeat':job.get('repeat'),'f1':a,'em':b,'valid':valid})
 expectedtokens=au.get('known_research_tokens',au.get('known_research_tokens_lower_bound'));assert tokens==expectedtokens
 if name=='partial_state_study_20261004':
  assert returned==qualified==149 and len(calls)==150
  strictreader=groups;details=load(BR/'FORMAT_SENSITIVITY_20261005.json')['details'];saveddiag={r['ordinal']:r for r in details}
  keys=set.intersection(*[{(r['row'],r['repeat']) for r in groups[arm]} for arm in ['direct','question_first','prefix']]);assert len(keys)==18
  paired={}
  for arm,rs in groups.items():
   if arm in ['source','partial_source']:continue
   select=[r for r in rs if (r['row'],r['repeat']) in keys];correct=0
   for r in select:
    raw=load(folder/f"attempt_{r['ordinal']:03d}.json")['raw'];text=raw['message']['content'];matches=re.findall(r'^\s*Final Answer:\s*\[([^\]\n]*)\]\s*$',text,re.M)
    normal=raw['done_reason']=='stop' and len(matches)==1
    nodes=[v.strip() for v in matches[0].split(',')] if normal and matches[0].strip() else []
    normal=normal and all(re.fullmatch('[A-Za-z0-9_.-]+',v) for v in nodes)
    exact=int(normal and set(nodes)==gold(cs[r['row']]));assert exact==saveddiag[r['ordinal']]['sensitive_em'];correct+=exact
   paired[arm]={'matched_outputs':len(select),'visible_only_exact':correct,'strict_exact':sum(r['em'] for r in select)}
  assert [paired[a]['visible_only_exact'] for a in ['prefix','direct','question_first','re2']]==[13,13,13,15]
 else:paired=None
 studies.append({'name':name,'frozen_sha256':sha(folder/'frozen.json'),'dependencies':dependency,'calls':len(calls),'returned':returned,'qualified':qualified,'known_tokens':tokens,'exceptions':exceptions,'paired_format_sensitivity':paired,'rescored_groups':{k:{'n':len(v),'f1_sum':sum(r['f1'] for r in v),'em_sum':sum(r['em'] for r in v),'normal_valid':sum(r['valid'] for r in v)} for k,v in groups.items()}})

# Bridge payloads were not separately retained; do not fabricate stronger provenance.
bridge=load(BR/'bridge_run/frozen.json');casescheck(bridge)
assert sha(BR/'run_bridge.py')==bridge['runner_sha256'] and sha(BR/'bridge_protocol.md')==bridge['protocol_sha256']
bridgeaudit=load(BR/'continued_bridge_audit.json')
cs={c['row_index']:c for c in bridge['cases']};records={r['ordinal']:r for r in bridgeaudit['records']};bridgegroups=collections.defaultdict(list)
for ordinal in range(1,65):
 path=BR/('bridge_run' if ordinal<=20 else 'bridge_continuation')/f'attempt_{ordinal:03d}.json'
 rec=load(path);recordpath(path);r=records[ordinal];job=rec['job'];raw=rec.get('response',{})
 if raw:
  assert raw['model']==job['model'] and raw['prompt_eval_count']==job['expected_prompt_tokens']
  assert raw['done'] is True and raw['eval_count']<=8192
  valid,pred=parse(raw);a,b=score(valid,pred,gold(cs[job['row_index']]));check_score(a,b,r)
 elif r['score_determined']:
  assert rec['status']=='request_error';a=b=0.;check_score(a,b,r)
 else:
  assert ordinal==20 and not r['usage_known'];a=b=None
 bridgegroups[job['model']].append({'ordinal':ordinal,'f1':a,'em':b,'tokens':raw.get('prompt_eval_count',0)+raw.get('eval_count',0)})
brsummary={}
for model,rr in bridgegroups.items():
 unknown=sum(r['f1'] is None for r in rr);bounds={m:[sum(r[m] for r in rr if r[m] is not None)/32,(sum(r[m] for r in rr if r[m] is not None)+unknown)/32] for m in ['f1','em']}
 for m in bounds:assert np.allclose(bounds[m],bridgeaudit['summary'][model]['primary_'+m+'_missingness_bounds'],atol=1e-12)
 assert sum(r['tokens'] for r in rr)==bridgeaudit['summary'][model]['known_token_lower_bound']
 brsummary[model]={'bounds':bounds,'known_token_lower_bound':sum(r['tokens'] for r in rr)}
studies.append({'name':'model_bridge_original_and_continuation','calls':64,'frozen_sha256':sha(BR/'bridge_run/frozen.json'),'summary':brsummary,'payload_provenance':'intended requests reconstructed; no separately captured complete transmitted payload'})
cert=load(BR/'hybrid_five_trace_single_prediction_cost_bound_20261004.json')
assert cert['frozen_sha256']==sha(BR/'hybrid_five_trace_replication/frozen.json')
assert len(cert['case_bounds'])==16
for c in cert['case_bounds']:
 i=c['source_input_tokens'];assert i>8192 and c['five_source_input_token_lower_bound']==5*i and c['fresh_single_call_token_upper_bound']==i+32768
output={'scope':'Independent recomputation from local records, not external attestation','dataset':{'sha256':sha(dataset),'rows':750,'all_gold_answers_independently_recomputed':True},'studies':studies,'call_manifest':call_manifest,'graph_manifest':graphs,'five_source_cost_certificate':'16 rows agree with frozen identity and conditional arithmetic','limitations':['No signed third-party execution records','No proof of continuous exclusive hardware','Unknown interrupted responses/usage not reconstructed','Legacy synthetic and document archives not included in this primary-response rescore','Author identity and affiliations still unverified']}
target=OUT/'primary_reverification.json';target.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps({'status':'passed','archived_call_records':len(call_manifest),'studies':len(studies),'dataset_gold':750,'study_calls':{s['name']:s['calls'] for s in studies}},indent=2))
