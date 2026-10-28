"""Inspect source-only eligibility before freezing any new inference."""
from pathlib import Path
import json,collections
import pyarrow.parquet as pq
from run_official_graph_native_probe import ROOT,DATA,SYSTEM,sha
from run_official_graph_long_boundary import TOKENIZER,input_count
from run_official_graph_identity_recode import recode
from run_official_graph_five_trace import FORMAT
used=set();archives={}
for path in sorted((ROOT/'logs').rglob('frozen.json')):
 try:fr=json.loads(path.read_text(encoding='utf-8-sig'))
 except (ValueError,UnicodeError):continue
 selected=fr.get('selected',[])
 for row in selected if isinstance(selected,list) else []:
  if isinstance(row,dict) and isinstance(row.get('prompt'),str) and 'Operation:' in row['prompt']:
   used.add(sha(row['prompt'].rsplit('Operation:',1)[0].encode()))
 archives[str(path.relative_to(ROOT))]=sha(path.read_bytes())
rows=pq.read_table(DATA).to_pylist();pools={'short':collections.defaultdict(list),'capacity':collections.defaultdict(list)}
seen=set(used)
for i,r in enumerate(rows):
 h=sha(r['prompt'].rsplit('Operation:',1)[0].encode())
 if h in seen:continue
 seen.add(h)
 x=r['prompt']+'\n'+FORMAT
 if 3500<=r['prompt_chars']<6500:pools['short'][r['problem_type']].append(i)
 if 9000<=r['prompt_chars']<35000:
  coded=recode(x,i,TOKENIZER);n=input_count(SYSTEM,x);nc=input_count(SYSTEM,coded['prompts']['compact'])
  if n>8192 and nc+4096+64<=8192:
   pools['capacity'][r['problem_type']].append({'row_index':i,'original_input_tokens':n,'compact_input_tokens':nc})
report={'excluded_graph_heads':len(used),'archive_hashes':archives,'pools':{k:dict(v) for k,v in pools.items()},'selection_uses':'Source length, operation type, graph identity and input token counts only; no gold labels or model scores.'}
(ROOT/'report/benefit_population_20261009.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'excluded_graph_heads':len(used),'counts':{k:{t:len(v) for t,v in d.items()} for k,d in pools.items()}},indent=2))
