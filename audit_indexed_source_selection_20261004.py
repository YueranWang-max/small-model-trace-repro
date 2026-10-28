"""Independent indexed-study selection reconstruction; no inference or imports from runner."""
import pathlib,json,hashlib,re
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'logs/official_graph_indexed_source'
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def head_id(prompt): return sha_bytes(prompt.rsplit('Operation:',1)[0].encode())
def graph_id(prompt):
 block=prompt.split('Here is the graph to operate on:',1)[1].rsplit('Operation:',1)[0]
 edges=re.findall(r'^([0-9a-f]{10}) -> ([0-9a-f]{10})\s*$',block,re.M)
 assert edges
 return sha_bytes(json.dumps(sorted(edges),separators=(',',':')).encode())
def main():
 target=OUT/'selection_reconstruction_audit_20261004.json'
 assert not target.exists(),'Preserve audit evidence'
 frozen=json.loads((OUT/'frozen.json').read_bytes())
 prior=json.loads((OUT/'interrupted_independent_audit.json').read_bytes())
 assert prior['partial_audit_passed'] and prior['selection_reconstruction_pending']
 data_path=ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet'
 assert sha_bytes(data_path.read_bytes())==frozen['data_sha256']
 rows=pq.read_table(data_path).to_pylist()
 names=['official_graph_native_probe','official_graph_budget_extension','official_graph_order_replication','official_graph_long_boundary','official_graph_sampling_control','official_graph_five_trace','official_graph_identity_recode','official_graph_reasoning_mode']
 history=[];used=set();prior_graphs=set()
 for name in names:
  p=ROOT/'logs'/name/'frozen.json';d=json.loads(p.read_bytes())
  history.append(dict(path=str(p.relative_to(ROOT)),sha256=sha_bytes(p.read_bytes()),selected_rows=[c['row_index'] for c in d['selected']]))
  used.update(head_id(c['prompt']) for c in d['selected'])
  prior_graphs.update(graph_id(c['prompt']) for c in d['selected'])
 initial_excluded=len(used);selected=[];groups=[]
 for lo,hi in ((2000,3500),(7000,9000)):
  for kind in ('parents','bfs'):
   eligible=[]
   for i,row in enumerate(rows):
    if row['problem_type']==kind and lo<=row['prompt_chars']<hi and head_id(row['prompt']) not in used:
     eligible.append(i);used.add(head_id(row['prompt']))
     if len(eligible)==2:break
   assert len(eligible)==2
   groups.append(dict(length_bin=[lo,hi],type=kind,selected_rows=eligible))
   selected.extend(eligible)
 expected=[selected[i] for i in (0,2,4,6,1,3,5,7)]
 assert expected==[c['row_index'] for c in frozen['selected']]
 identities=[]
 for c in frozen['selected']:
  row=rows[c['row_index']]
  assert c['prompt']==row['prompt'] and c['problem_type']==row['problem_type'] and c['prompt_chars']==row['prompt_chars']
  assert c['graph_head_sha256']==head_id(row['prompt'])
  assert c['length_bin'][0]<=row['prompt_chars']<c['length_bin'][1]
  identities.append(graph_id(row['prompt']))
 assert len(set(identities))==8 and not(set(identities)&prior_graphs)
 result=dict(selection_reconstruction_passed=True,prior_response_audit_preserved=True,inference_calls=0,primary_complete=False,selected_rows=expected,initial_excluded_prompt_heads=initial_excluded,prior_canonical_graphs=len(prior_graphs),selected_graphs_unique_and_excluded=True,groups=groups,history=history,input_sha256={p.name:sha_bytes(p.read_bytes()) for p in (OUT/'frozen.json',OUT/'interrupted_independent_audit.json',data_path)},limits=['Separate selection audit completes the pending selection requirement; it does not recover calls095/096 or establish pretraining non-overlap.'])
 target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('history','input_sha256','groups')},indent=2))
if __name__=='__main__':main()
