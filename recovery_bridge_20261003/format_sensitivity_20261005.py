"""Post hoc visible-only parser sensitivity, never replacement primary scores."""
import collections,json,pathlib,re
root=pathlib.Path(__file__).resolve().parent
out=root/'partial_state_study_20261004'
audit=json.loads((out/'independent_audit.json').read_text())
cases={c['row_index']:c for c in json.loads((out/'frozen.json').read_text())['cases']}
summaries=collections.defaultdict(lambda:dict(known=0,strict_exact=0,sensitive_exact=0,recovered_valid=0,ambiguous_or_invalid=0))
details=[]
for r in audit['reader_results']:
 if not r['known']:continue
 a=json.loads((out/f"attempt_{r['ordinal']:03d}.json").read_text());raw=a.get('raw',{})
 s=summaries[r['arm']];s['known']+=1;s['strict_exact']+=int(r['em'])
 visible=raw.get('message',{}).get('content','')
 matches=re.findall(r'^\s*Final Answer:\s*\[([^\]\n]*)\]\s*$',visible,re.M)
 valid=raw.get('done_reason')=='stop' and len(matches)==1
 nodes=[]
 if valid:
  nodes=[v.strip() for v in matches[0].split(',')] if matches[0].strip() else []
  valid=all(re.fullmatch(r'[A-Za-z0-9_.-]+',v) for v in nodes)
 exact=int(valid and set(nodes)==set(cases[r['row_index']]['answer_nodes']))
 s['sensitive_exact']+=exact
 s['recovered_valid']+=int(valid and not r['normal_valid'])
 s['ambiguous_or_invalid']+=int(not valid)
 details.append(dict(ordinal=r['ordinal'],arm=r['arm'],strict_em=r['em'],sensitive_em=exact,valid=valid))
result=dict(scope='Post hoc diagnostic only; original strict scores and gate unchanged. Single unambiguous visible Final Answer line may have surrounding prose; length outputs remain invalid, no thinking extraction.',summaries=dict(summaries),details=details)
with (root/'FORMAT_SENSITIVITY_20261005.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
