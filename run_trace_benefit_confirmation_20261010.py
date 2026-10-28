"""Independent new-question confirmation; reuses only durable HTTP plumbing."""
import json,random,time,sys
from pathlib import Path
import pyarrow.parquet as pq
import run_user_benefit_20261009 as c
ROOT=c.ROOT;OUT=ROOT/'logs/trace_benefit_confirmation_20261010';PROTOCOL=ROOT/'protocol_trace_benefit_confirmation_20261010.md'
ARMS=['original','original_long','compact','prefix','append']
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 names=[Path(__file__).name,PROTOCOL.name,'run_user_benefit_20261009.py','run_official_graph_native_probe.py','run_official_graph_long_boundary.py','run_official_graph_identity_recode.py','run_official_graph_five_trace.py','run_official_graph_native_order.py','data/qwen3_tokenizer_official/tokenizer.json','paper/evidence_user_benefit_20261009/summary.json']
 deps={n:c.sha((ROOT/n).read_bytes()) for n in names}
 if (OUT/'frozen.json').exists():
  fr=json.loads((OUT/'frozen.json').read_bytes());assert fr['dependencies']==deps and fr['server_version']==c.get('version');return fr
 assert json.loads((ROOT/names[-1]).read_bytes())['complete']
 used=set();archives={}
 for path in sorted((ROOT/'logs').rglob('frozen.json')):
  if path.parent==OUT:continue
  old=json.loads(path.read_text(encoding='utf-8-sig'));items=old.get('selected',[])
  for row in items if isinstance(items,list) else []:
   if isinstance(row,dict) and isinstance(row.get('prompt'),str) and 'Operation:' in row['prompt']:used.add(c.sha(row['prompt'].rsplit('Operation:',1)[0].encode()))
  archives[str(path.relative_to(ROOT))]=c.sha(path.read_bytes())
 rows=pq.read_table(c.DATA).to_pylist();rng=random.Random(2026101010);selected=[];pools={}
 for typ in ['parents','bfs']:
  eligible=[];seen=set(used)
  for i,row in enumerate(rows):
   h=c.sha(row['prompt'].rsplit('Operation:',1)[0].encode())
   if row['problem_type']==typ and 3500<=row['prompt_chars']<6500 and h not in seen:eligible.append(i);seen.add(h)
  pools[typ]=eligible;assert len(eligible)>=12,(typ,len(eligible))
  for i in rng.sample(eligible,12):
   row=rows[i];x=row['prompt']+'\n'+c.FORMAT;coded=c.recode(x,i,c.TOKENIZER)
   assert set(c.oracle(row))==set(row['answer_nodes'])
   for enc in ['original','compact']:assert {coded['inverse'][enc][v] for v in c.oracle({'prompt':coded['prompts'][enc],'problem_type':typ})}==set(row['answer_nodes'])
   assert c.input_count(c.SYSTEM,x)+16384+64<=32768
   selected.append({'row_index':i,'stratum':'confirmation','graph_head_sha256':c.sha(row['prompt'].rsplit('Operation:',1)[0].encode()),**row,'prompts':{k:coded['prompts'][k] for k in ['original','compact']},'maps':{k:coded['maps'][k] for k in ['original','compact']},'inverse':{k:coded['inverse'][k] for k in ['original','compact']},'dictionary_json_bytes':coded['compact_dictionary_json_bytes'],'preprocess_wall_sec':coded['preprocess_wall_sec']})
 rng.shuffle(selected);orders=[]
 for row in selected:
  repeats=[]
  for _ in range(2):a=ARMS.copy();rng.shuffle(a);repeats.append(a)
  orders.append(repeats)
 identity=next(m for m in c.get('tags')['models'] if m['name']==c.MODEL);assert identity['digest']==c.DIGEST
 fr={'dependencies':deps,'data_sha256':c.sha(c.DATA.read_bytes()),'exclusion_archive_hashes':archives,'candidate_rows':pools,'selected':selected,'orders':orders,'source_seed':2026101011,'reader_seeds':[2026101012,2026101013],'server_version':c.get('version'),'model':c.MODEL,'digest':c.DIGEST,'system':c.SYSTEM,'identity':identity,'planned_calls':264,'selection_seed':2026101010,'primary':'prefix-compact graph-mean F1; positive descriptive interval and paired sign-flip support','discovery_cases_excluded':True}
 assert fr['server_version']['version']=='0.40.0';c.save(OUT/'frozen.json',fr);return fr
def main():
 fr=freeze()
 if '--prepare' in sys.argv:print('Frozen24 fresh graph units,264 calls, matched append and stronger direct baseline; no inference yet.',flush=True);return
 c.OUT=OUT;records=[]
 try:
  for i,row in enumerate(fr['selected']):
   key=f'g{i:02d}_source';r=c.call(key,row['prompts']['compact'],1024,32768,fr['source_seed'])
   records.append({**r,'key':key,'row_index':row['row_index'],'arm':'source','score':c.restored(r['raw'],row,'compact')});c.save(OUT/'results.json',records)
   T=c.serialize([(r['raw'].get('message') or {}).get('thinking','') or ''])
   head,operation=row['prompts']['compact'].rsplit('Operation:',1)
   prompts={'original':row['prompts']['original'],'original_long':row['prompts']['original'],'compact':row['prompts']['compact'],'prefix':T+'\n'+row['prompts']['compact'],'append':head+T+'\nOperation:'+operation}
   for j,seed in enumerate(fr['reader_seeds']):
    for arm in fr['orders'][i][j]:
     cap=16384 if arm=='original_long' else 7168 if arm in ['prefix','append'] else 8192
     key=f'g{i:02d}_{arm}_{j}';r=c.call(key,prompts[arm],cap,32768,seed);enc='original' if arm.startswith('original') else 'compact'
     records.append({**r,'key':key,'row_index':row['row_index'],'arm':arm,'repeat':j,'source_key':f'g{i:02d}_source' if arm in ['prefix','append'] else None,'trace_sha256':c.sha(T.encode()) if arm in ['prefix','append'] else None,'score':c.restored(r['raw'],row,enc)})
     c.save(OUT/'results.json',records);c.save(OUT/'status.json',{'state':'running','recorded':len(records),'planned':264,'latest':key,'updated_unix':time.time()})
     print(key,r['raw']['done_reason'],records[-1]['score']['f1'],records[-1]['score']['em'],flush=True)
   time.sleep(30)
  c.save(OUT/'status.json',{'state':'completed','recorded':len(records),'planned':264,'updated_unix':time.time()})
 except Exception as exc:c.save(OUT/'status.json',{'state':'stopped_on_exception','recorded':len(records),'planned':264,'error':repr(exc),'updated_unix':time.time()});raise
if __name__=='__main__':main()
