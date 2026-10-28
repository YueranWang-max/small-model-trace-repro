"""Independently verify new-case payloads, source round trips, raw answers and costs.

No runner, scoring, oracle or recoding implementation is imported. --interim
reports incomplete collection without promoting it to a confirmed result.
"""
from pathlib import Path
from collections import deque,Counter
import json,re,hashlib,csv,sys,random
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/trace_benefit_confirmation_20261010'
REPORT=ROOT/'paper/evidence_trace_benefit_confirmation_20261010';REPORT.mkdir(exist_ok=True)
digest=lambda b:hashlib.sha256(b).hexdigest()
fr=json.loads((OUT/'frozen.json').read_bytes());rows=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist()
tok=Tokenizer.from_file(str(ROOT/'data/qwen3_tokenizer_official/tokenizer.json'))
for n,h in fr['dependencies'].items():assert digest((ROOT/n).read_bytes())==h,n
for amendment in fr.get('execution_amendments',[]):
 assert digest((OUT/amendment['old_runner_snapshot']).read_bytes())==amendment['old_runner_sha256']
 oldfr=json.loads((OUT/amendment['old_frozen_snapshot']).read_bytes())
 assert {k:v for k,v in oldfr.items() if k!='dependencies'}=={k:v for k,v in fr.items() if k not in ['dependencies','execution_amendments']}
assert digest((ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').read_bytes())==fr['data_sha256']
used=set()
for path,h in fr['exclusion_archive_hashes'].items():
 assert digest((ROOT/path).read_bytes())==h,path
 old=json.loads((ROOT/path).read_text(encoding='utf-8-sig'));items=old.get('selected',[])
 for item in items if isinstance(items,list) else []:
  if isinstance(item,dict) and isinstance(item.get('prompt'),str) and 'Operation:' in item['prompt']:
   used.add(digest(item['prompt'].rsplit('Operation:',1)[0].encode()))
rng=random.Random(2026101010);expected_indices=[]
for typ in ['parents','bfs']:
 eligible=[];seen_heads=set(used)
 for i,row in enumerate(rows):
  h=digest(row['prompt'].rsplit('Operation:',1)[0].encode())
  if row['problem_type']==typ and 3500<=row['prompt_chars']<6500 and h not in seen_heads:
   eligible.append(i);seen_heads.add(h)
 assert eligible==fr['candidate_rows'][typ]
 expected_indices+=rng.sample(eligible,12)
rng.shuffle(expected_indices)
assert expected_indices==[r['row_index'] for r in fr['selected']]
ARMS=['original','original_long','compact','prefix','append']
for orders in fr['orders']:
 for actual in orders:
  expected=ARMS.copy();rng.shuffle(expected);assert expected==actual

def answer(row):
 head,q=row['prompt'].rsplit('Operation:',1)
 edges=re.findall(r'^([A-Za-z0-9]+) -> ([A-Za-z0-9]+)\s*$',head,re.M)
 if row['problem_type']=='parents':
  v=re.search(r'Find the parents of node (\w+)\.',q)[1]
  return {u for u,w in edges if w==v and u!=v}
 v=re.search(r'BFS from node (\w+)',q)[1];depth=int(re.search(r'exactly depth (\d+)',q)[1]);adj={}
 for u,w in edges:adj.setdefault(u,[]).append(w)
 queue=deque([v]);distance={v:0}
 while queue:
  u=queue.popleft()
  if distance[u]>=depth:continue
  for w in adj.get(u,[]):
   if w not in distance:distance[w]=distance[u]+1;queue.append(w)
 return {u for u,d in distance.items() if d==depth}
def count(user):
 s='<|im_start|>system\n'+fr['system']+'<|im_end|>\n<|im_start|>user\n'+user+' /think<|im_end|>\n<|im_start|>assistant\n'
 return len(tok.encode(s,add_special_tokens=False).ids)
def score(raw,gold,inverse):
 content=(raw.get('message') or {}).get('content','') or ''
 m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',content.rstrip().split('\n')[-1].strip())
 valid=bool(m) and raw.get('done_reason')=='stop'
 pred={x.strip() for x in m[1].split(',') if x.strip()} if m else set()
 restored={inverse[x] if x in inverse else '<unknown-node:'+x+'>' for x in pred}
 f=2*len(restored&gold)/(len(restored)+len(gold)) if restored or gold else 1.
 return {'valid':valid,'f1':f if valid else 0.,'em':float(valid and restored==gold),'restored_prediction':sorted(restored)}
PD='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
records=[];unknown=[];graph_checks=[];seen=set()
for i,row in enumerate(fr['selected']):
 original=rows[row['row_index']];assert row['prompt']==original['prompt'] and row['answer_nodes']==original['answer_nodes']
 h=digest(row['prompt'].rsplit('Operation:',1)[0].encode());assert h not in used and h not in seen;seen.add(h)
 gold=answer(original);assert gold==set(original['answer_nodes'])
 x=row['prompts']['original'];assert x==row['prompt']+'\nReturn exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.'
 inv=row['inverse']['compact'];mp=row['maps']['compact'];assert len(set(mp.values()))==len(mp)
 assert inv=={v:k for k,v in mp.items()}
 patt=re.compile(r'\b(?:'+'|'.join(sorted(map(re.escape,inv),key=len,reverse=True))+r')\b')
 recovered=patt.sub(lambda m:inv[m[0]],row['prompts']['compact']);assert recovered==x,'Source round-trip failure'
 assert all(len(tok.encode(v,add_special_tokens=False).ids)==1 for v in inv)
 old_n=count(x);new_n=count(row['prompts']['compact']);ctx=32768
 assert old_n+16384+64<=ctx
 graph_checks.append({'graph_index':i,'row_index':row['row_index'],'stratum':row['stratum'],'problem_type':row['problem_type'],'gold_count':len(gold),'original_input_tokens':old_n,'compact_input_tokens':new_n,'source_round_trip':True,'graph_head_sha256':h,'dictionary_bytes':row['dictionary_json_bytes'],'cpu_preprocess_seconds':row['preprocess_wall_sec']})
 src=None
 jobs=[('source',None,fr['source_seed'],ctx,1024)]
 for j,seed in enumerate(fr['reader_seeds']):
  for arm in fr['orders'][i][j]:
   cap=16384 if arm=='original_long' else 7168 if arm in ['prefix','append'] else 8192
   jobs.append((arm,j,seed,ctx,cap))
 for arm,j,seed,context,cap in jobs:
  key=f'g{i:02d}_source' if arm=='source' else f'g{i:02d}_{arm}_{j}'
  ledger=OUT/(key+'.json');response=OUT/(key+'.response.json')
  if not ledger.exists() or not response.exists():unknown.append(key);continue
  rec=json.loads(ledger.read_bytes())
  if rec.get('completion')!='returned':unknown.append(key);continue
  req=rec['request'];raw=json.loads(response.read_bytes());result=rec['result']
  assert result['raw']==raw and result['raw_response_sha256']==digest(response.read_bytes())
  assert rec['request_sha256']==digest(json.dumps(req,sort_keys=True).encode())
  enc='original' if arm.startswith('original') else 'compact';user=row['prompts'][enc]
  if arm in ['prefix','append']:
   assert src is not None,'Prefix without archived source'
   thinking=(src.get('message') or {}).get('thinking','') or ''
   if len(thinking)>50000:thinking=thinking[:50000]+'...'
   T=PD+'\n<first_run_reasoning_traces>\n[Trace 1] '+thinking.strip()+'\n</first_run_reasoning_traces>'
   if arm=='prefix':user=T+'\n'+user
   else:
    head,operation=user.rsplit('Operation:',1);user=head+T+'\nOperation:'+operation
  assert req=={'model':fr['model'],'messages':[{'role':'system','content':fr['system']},{'role':'user','content':user}],'stream':False,'think':True,'keep_alive':'10m','options':{'num_ctx':context,'num_predict':cap,'seed':seed,'temperature':.6,'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}},key
  n=count(user);assert n==rec['expected_input_tokens']==raw['prompt_eval_count'] and n+cap+64<=context
  assert raw.get('eval_count',cap+1)<=cap and raw['model']==fr['model']
  assert any(m.get('digest')==fr['digest'] and m.get('context_length')==context for m in result['runtime_ps']['models'])
  assert result['qualified'] and result['error'] is None
  s=score(raw,gold,row['inverse'][enc]);r={'key':key,'graph_index':i,'row_index':row['row_index'],'stratum':row['stratum'],'problem_type':row['problem_type'],'arm':arm,'repeat':j,'seed':seed,**s,'input_tokens':n,'output_tokens':raw['eval_count'],'total_tokens':n+raw['eval_count'],'wall_seconds':result['wall_sec'],'termination':raw['done_reason'],'context':context,'gpu_after':result['gpu_after'],'resident_size_vram':next(m.get('size_vram') for m in result['runtime_ps']['models'] if m.get('digest')==fr['digest'])}
  records.append(r)
  if arm=='source':src=raw
complete=len(records)==fr['planned_calls'] and not unknown
summary={'complete':complete,'planned_model_calls':fr['planned_calls'],'known_model_calls':len(records),'unknown_keys':unknown,'independent_graph_units':24,'new_runtime_version':fr['server_version'],'source_round_trips_verified':len(graph_checks),'all_known_payloads_and_raw_scores_independently_verified':True,'research_tokens_known':sum(r['total_tokens'] for r in records),'research_request_wall_seconds_known':sum(r['wall_seconds'] for r in records),'scope':'New questions on a public graph benchmark; no cross-domain guarantee, external execution attestation, or energy/money inference.'}
summary['execution_amendments']=fr.get('execution_amendments',[])
if complete:
 matrix=json.loads((OUT/'results.json').read_bytes());assert len(matrix)==264
 for r in records:
  actual=next(v for v in matrix if v['key']==r['key'])
  assert actual['score']['f1']==r['f1'] and actual['score']['em']==r['em'] and actual['score']['valid']==r['valid']
 summaries=[];graph_means=[];contrasts=[]
 for g in graph_checks:
  source=next(r for r in records if r['graph_index']==g['graph_index'] and r['arm']=='source')
  for arm in ARMS:
   rs=[r for r in records if r['graph_index']==g['graph_index'] and r['arm']==arm];assert len(rs)==2
   reuse=arm in ['prefix','append']
   graph_means.append({**g,'arm':arm,'f1':float(np.mean([r['f1'] for r in rs])),'em':float(np.mean([r['em'] for r in rs])),'deployment_tokens':float(np.mean([r['total_tokens'] for r in rs]))+(source['total_tokens'] if reuse else 0),'deployment_wall_seconds':float(np.mean([r['wall_seconds'] for r in rs]))+(source['wall_seconds'] if reuse else 0),'both_seeds_exact':all(r['em']==1 for r in rs)})
 for arm in ARMS:
  gs=[g for g in graph_means if g['arm']==arm];rs=[r for r in records if r['arm']==arm]
  summaries.append({'arm':arm,'calls':len(rs),'graph_units':len(gs),'f1':float(np.mean([g['f1'] for g in gs])),'em':float(np.mean([g['em'] for g in gs])),'mean_deployment_tokens':float(np.mean([g['deployment_tokens'] for g in gs])),'mean_deployment_wall_seconds':float(np.mean([g['deployment_wall_seconds'] for g in gs])),'valid_count':sum(r['valid'] for r in rs),'length_terminated':sum(r['termination']=='length' for r in rs),'observed_resident_vram_bytes':sorted(set(r['resident_size_vram'] for r in rs))})
 index={(g['graph_index'],g['arm']):g for g in graph_means}
 for left,right in [('prefix','compact'),('prefix','append'),('prefix','original'),('prefix','original_long'),('append','compact'),('compact','original')]:
  blocks=[]
  for typ in ['parents','bfs']:
   ids=[g['graph_index'] for g in graph_checks if g['problem_type']==typ];assert len(ids)==12
   blocks.append(np.array([[index[i,left]['f1']-index[i,right]['f1'],index[i,left]['deployment_tokens'],index[i,right]['deployment_tokens'],index[i,left]['deployment_wall_seconds'],index[i,right]['deployment_wall_seconds']] for i in ids]))
  rng=np.random.default_rng(2026101014)
  draws=np.mean([v[rng.integers(0,len(v),(20000,len(v)))].mean(axis=1) for v in blocks],axis=0)
  observed=np.mean([v.mean(axis=0) for v in blocks],axis=0)
  fci=np.quantile(draws[:,0],[.025,.975]).tolist();tci=np.quantile(draws[:,1]/draws[:,2],[.025,.975]).tolist()
  diffs=np.concatenate([v[:,0] for v in blocks]);rng=np.random.default_rng(2026101015)
  null=(rng.choice([-1,1],(100000,len(diffs)))*diffs).mean(axis=1)
  p=float((np.count_nonzero(np.abs(null)>=abs(observed[0])-1e-15)+1)/100001)
  primary=(left,right)==('prefix','compact')
  contrasts.append({'contrast':left+'-'+right,'primary':primary,'f1_difference':float(observed[0]),'descriptive_interval':fci,'two_sided_paired_sign_flip_p':p,'full_token_ratio':float(observed[1]/observed[2]),'full_token_ratio_interval':tci,'full_wall_ratio':float(observed[3]/observed[4]),'full_wall_ratio_interval':np.quantile(draws[:,3]/draws[:,4],[.025,.975]).tolist(),'secondary_joint_quality_token_gate':bool(not primary and fci[0]>=0 and tci[1]<1)})
 primary=contrasts[0]
 summary.update(arms=summaries,contrasts=contrasts,primary_accuracy_benefit_supported=bool(primary['f1_difference']>0 and primary['descriptive_interval'][0]>0 and primary['two_sided_paired_sign_flip_p']<=.05),inference_limits='24 graph units, two seeds averaged within unit; primary fixed before responses; descriptive intervals and sign-flip exchangeability assumption; secondary comparisons unadjusted for multiplicity; complete source preparation charged to each reused answer; no energy or monetary cost claim.')
 with (REPORT/'graph_means.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(graph_means[0]));w.writeheader();w.writerows(graph_means)
(REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(REPORT/'graph_round_trip_checks.json').write_text(json.dumps(graph_checks,indent=2),encoding='utf-8')
(REPORT/'rescored_calls.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
if not complete and '--interim' not in sys.argv:raise SystemExit('Incomplete: no completed user-benefit claim.')
