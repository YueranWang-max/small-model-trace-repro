"""Frozen fresh-case study of low-resource graph encoding and one-trace adaptation."""
import json,random,re,time,sys,urllib.request,subprocess
from pathlib import Path
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,OPENER,sha,score,oracle
from run_official_graph_long_boundary import TOKENIZER,input_count
from run_official_graph_identity_recode import recode
from run_official_graph_five_trace import FORMAT,serialize
OUT=ROOT/'logs/user_benefit_20261009';PROTOCOL=ROOT/'protocol_user_benefit_20261009.md'
MODEL='qwen3:1.7b';DIGEST='8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7'
def save(p,o):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def get(endpoint):
 with OPENER.open('http://127.0.0.1:11434/api/'+endpoint,timeout=30) as r:return json.load(r)
def gpu():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True)
 a=s.strip().split(',');return {'memory_mib':int(a[0]),'utilization_percent':int(a[1]),'temperature_c':int(a[2])}
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 names=[Path(__file__).name,PROTOCOL.name,'run_official_graph_native_probe.py','run_official_graph_long_boundary.py','run_official_graph_identity_recode.py','run_official_graph_five_trace.py','run_official_graph_native_order.py','data/qwen3_tokenizer_official/tokenizer.json','report/benefit_population_20261009.json']
 deps={n:sha((ROOT/n).read_bytes()) for n in names}
 if (OUT/'frozen.json').exists():
  fr=json.loads((OUT/'frozen.json').read_bytes());assert fr['dependencies']==deps
  assert fr['server_version']==get('version');return fr
 pop=json.loads((ROOT/'report/benefit_population_20261009.json').read_bytes())
 for path,h in pop['archive_hashes'].items():assert sha((ROOT/path).read_bytes())==h
 rows=pq.read_table(DATA).to_pylist();rng=random.Random(2026100910);selected=[];orders=[]
 for stratum in ['capacity','short']:
  block=[]
  for typ in ['parents','bfs']:
   eligible=pop['pools'][stratum][typ]
   if stratum=='capacity':eligible=[v for v in eligible if v['compact_input_tokens']<=3500]
   assert len(eligible)>=6,(stratum,typ,len(eligible))
   for chosen in rng.sample(eligible,6):
    i=chosen['row_index'] if isinstance(chosen,dict) else chosen;row=rows[i]
    x=row['prompt']+'\n'+FORMAT;t=time.perf_counter();coded=recode(x,i,TOKENIZER)
    assert set(oracle(row))==set(row['answer_nodes'])
    for arm in ['original','compact']:
     assert {coded['inverse'][arm][v] for v in oracle({'prompt':coded['prompts'][arm],'problem_type':typ})}==set(row['answer_nodes'])
    caps={'original':8192,'compact':4096 if stratum=='capacity' else 8192}
    ctxs={'original':32768,'compact':8192 if stratum=='capacity' else 32768}
    for arm in caps:assert input_count(SYSTEM,coded['prompts'][arm])+caps[arm]+64<=ctxs[arm]
    block.append({'row_index':i,'stratum':stratum,'graph_head_sha256':sha(row['prompt'].rsplit('Operation:',1)[0].encode()),**row,'prompts':{k:coded['prompts'][k] for k in caps},'inverse':{k:coded['inverse'][k] for k in caps},'maps':{k:coded['maps'][k] for k in caps},'preprocess_wall_sec':time.perf_counter()-t,'dictionary_json_bytes':coded['compact_dictionary_json_bytes'],'input_tokens':{k:input_count(SYSTEM,coded['prompts'][k]) for k in caps}})
  rng.shuffle(block);selected+=block
 assert len({r['graph_head_sha256'] for r in selected})==24
 for row in selected:
  repeats=[]
  for _ in range(2):
   arms=['original','compact','prefix'];rng.shuffle(arms);repeats.append(arms)
  orders.append(repeats)
 inv=get('tags');identity=next(m for m in inv['models'] if m['name']==MODEL);assert identity['digest']==DIGEST
 fr={'dependencies':deps,'data_sha256':sha(DATA.read_bytes()),'selection_seed':2026100910,'population_report_sha256':sha((ROOT/'report/benefit_population_20261009.json').read_bytes()),'selected':selected,'orders':orders,'source_seed':2026100913,'reader_seeds':[2026100911,2026100912],'identity':identity,'server_version':get('version'),'planned_calls':168,'system':SYSTEM,'model':MODEL,'digest':DIGEST}
 assert fr['server_version']['version']=='0.40.0'
 save(OUT/'frozen.json',fr);return fr
def restored(raw,row,encoding):
 parsed=score(raw,[],None);inv=row['inverse'][encoding]
 mapped=[inv[v] if v in inv else '<unknown-node:'+v+'>' for v in parsed['prediction']]
 a,b=set(mapped),set(row['answer_nodes']);valid=parsed['valid'];f=2*len(a&b)/(len(a)+len(b)) if a or b else 1
 return {'prediction':parsed['prediction'],'restored_prediction':mapped,'valid':valid,'em':float(valid and a==b),'f1':f if valid else 0.}
def call(key,user,cap,ctx,seed):
 path=OUT/(key+'.json')
 if path.exists():
  record=json.loads(path.read_bytes());assert record['completion']=='returned' and record['result']['qualified'],'Ambiguous/nonqualified ledger; do not replay '+key
  return record['result']
 if (OUT/'STOP').exists():raise RuntimeError('User STOP between calls')
 if gpu()['temperature_c']>=82:
  while gpu()['temperature_c']>=76:
   save(OUT/'cooling.json',{'updated_unix':time.time(),'telemetry':gpu()});time.sleep(30)
 req={'model':MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}],'stream':False,'think':True,'keep_alive':'10m','options':{'num_ctx':ctx,'num_predict':cap,'seed':seed,'temperature':.6,'top_p':.95,'top_k':20,'min_p':0.,'repeat_penalty':1.}}
 n=input_count(SYSTEM,user);assert n+cap+64<=ctx,(key,n,cap,ctx)
 record={'request':req,'request_sha256':sha(json.dumps(req,sort_keys=True).encode()),'expected_input_tokens':n,'completion':'pending','transmission':'not_yet_attempted','gpu_before':gpu()}
 save(path,record);record['transmission']='attempting_once';save(path,record)
 started=time.perf_counter()
 try:
  rq=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
  with OPENER.open(rq,timeout=900) as response:body=response.read()
  (OUT/(key+'.response.json')).write_bytes(body);raw=json.loads(body);ps=get('ps')
  resident=next((m for m in ps['models'] if m.get('digest')==DIGEST),None)
  qualified=bool(resident and resident.get('context_length')==ctx and raw.get('model')==MODEL and raw.get('prompt_eval_count')==n and raw.get('eval_count',cap+1)<=cap)
  result={'raw':raw,'wall_sec':time.perf_counter()-started,'runtime_ps':ps,'gpu_after':gpu(),'raw_response_sha256':sha(body),'expected_input_tokens':n,'qualified':qualified,'error':None}
  record.update(completion='returned',result=result);save(path,record)
  assert qualified,('Interface qualification failure preserved',key,n,raw.get('prompt_eval_count'),ps)
 except Exception as exc:
  if record['completion']=='pending':record.update(completion='transport_unknown',error=repr(exc));save(path,record)
  raise
 time.sleep(3);return result
def main():
 fr=freeze()
 if '--prepare' in sys.argv:print('Frozen24 unused graph units,168 planned calls, two seeds; no inference yet.',flush=True);return
 records=[]
 try:
  for i,row in enumerate(fr['selected']):
   small=row['stratum']=='capacity';ctx=8192 if small else 32768
   key=f'g{i:02d}_source';r=call(key,row['prompts']['compact'],1024,ctx,fr['source_seed'])
   r={**r,'key':key,'row_index':row['row_index'],'stratum':row['stratum'],'arm':'source','score':restored(r['raw'],row,'compact')};records.append(r);save(OUT/'results.json',records)
   T=serialize([(r['raw'].get('message') or {}).get('thinking','') or ''])
   source_key=key
   for j,seed in enumerate(fr['reader_seeds']):
    for arm in fr['orders'][i][j]:
     key=f'g{i:02d}_{arm}_{j}';enc='original' if arm=='original' else 'compact'
     user=row['prompts'][enc];cap=8192 if arm=='original' or not small else 4096;ac=32768 if arm=='original' else ctx
     if arm=='prefix':user=T+'\n'+user;cap-=1024
     r=call(key,user,cap,ac,seed)
     r={**r,'key':key,'row_index':row['row_index'],'stratum':row['stratum'],'arm':arm,'repeat':j,'source_key':source_key if arm=='prefix' else None,'trace_sha256':sha(T.encode()) if arm=='prefix' else None,'score':restored(r['raw'],row,enc)}
     records.append(r);save(OUT/'results.json',records)
     save(OUT/'status.json',{'state':'running','recorded':len(records),'planned':168,'latest':key,'updated_unix':time.time()})
     print(key,row['stratum'],r['raw'].get('done_reason'),r['score']['f1'],r['score']['em'],r['raw']['prompt_eval_count']+r['raw']['eval_count'],flush=True)
   time.sleep(30)
  save(OUT/'status.json',{'state':'completed','recorded':len(records),'planned':168,'updated_unix':time.time()})
 except Exception as exc:
  save(OUT/'status.json',{'state':'stopped_on_exception','recorded':len(records),'planned':168,'error':repr(exc),'updated_unix':time.time()});raise
if __name__=='__main__':main()
