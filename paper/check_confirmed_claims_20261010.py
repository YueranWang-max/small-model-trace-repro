"""Check claims against independently audited results and retain historical objects."""
from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent;O=P/'revision_evidence_20261010'
body=(P/'migration_primary_body.tex').read_text(encoding='utf-8');main=(P/'main.tex').read_text(encoding='utf-8');checks=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
s=json.loads((P/'evidence_trace_benefit_confirmation_20261010/summary.json').read_bytes());d=json.loads((P/'evidence_user_benefit_20261009/summary.json').read_bytes())
assert s['complete'] and d['complete'] and s['known_model_calls']==264 and d['known_model_calls']==168
src=[v for v in json.loads((P/'evidence_trace_benefit_confirmation_20261010/rescored_calls.json').read_bytes()) if v['arm']=='source']
assert len(src)==24 and sum(v['termination']=='length' for v in src)==22 and sum(v['valid'] for v in src)==2 and sum(v['em'] for v in src)==0
assert not s['unknown_keys'] and not d['unknown_keys'] and s['primary_accuracy_benefit_supported']
def require(label,v,text=body):
 assert v in text,(label,v);checks.append({'claim':label,'literal':v})
arms={v['arm']:v for v in s['arms']};labels={'original':'Original direct','original_long':'Original direct, 16K output','compact':'Compact direct','prefix':'Compact trace prefix','append':'Compact trace append'}
for name,r in arms.items():
 require(name+' table',labels[name]+' & '+f"{r['f1']:.4f} & {r['em']:.4f} & {r['mean_deployment_tokens']:,.2f} & {r['valid_count']} & {r['length_terminated']}")
c={v['contrast']:v for v in s['contrasts']};primary=c['prefix-compact']
for text in [body,main]:
 for k in ['compact','prefix']:require('primary scores',f"{arms[k]['f1']:.4f}",text)
 for v in [primary['f1_difference'],*primary['descriptive_interval']]:require('primary change/interval',f'{v:.4f}',text)
 require('primary two-sided p',f"p={primary['two_sided_paired_sign_flip_p']:.5f}",text)
 require('primary token increase',f"{100*(primary['full_token_ratio']-1):.1f}\\%",text)
for name,v in c.items():
 require(name+' difference',f"{v['f1_difference']:.4f}")
 require(name+' interval',f"[{v['descriptive_interval'][0]:.4f},{v['descriptive_interval'][1]:.4f}]")
 if name not in ['prefix-append','append-compact']:
  require(name+' token ratio',f"{v['full_token_ratio']:.3f}")
  require(name+' token interval',f"[{v['full_token_ratio_interval'][0]:.3f},{v['full_token_ratio_interval'][1]:.3f}]")
require('wall interval',f"[{primary['full_wall_ratio_interval'][0]:.3f},{primary['full_wall_ratio_interval'][1]:.3f}]")
assert not d['capacity_8k']['bounded_observed_capacity_benefit']
require('negative capacity','no nonempty gold graph was answered exactly in both compact-direct repeats')
v=next(v for v in d['contrasts'] if v['stratum']=='short' and v['contrast']=='prefix-compact')
require('exploratory difference',f"{v['f1_difference']:.4f}");require('exploratory interval',f"[{v['descriptive_interval'][0]:.4f},{v['descriptive_interval'][1]:.4f}]")
require('multiplicity','secondary and unadjusted for multiplicity');require('no cost isolation','combined improvement cannot be assigned entirely to traces')
# Old result blocks remain literal, before the new extension subsection.
old=(O/'before_integration/migration_primary_body.tex').read_text(encoding='utf-8')
old_results=old[old.index(r'\section{Results}'):old.index(r'\section{Discussion}')]
assert old_results in body,'Historical Results were silently rewritten'
old_contract=json.loads((P/'revision_evidence_20261009_scoped/reviewed_content_contract_cost_framing.json').read_bytes())
new_contract=json.loads((O/'reviewed_content_contract_confirmed_extension_v3.json').read_bytes())
for key in ['figures','tables']:
 new={v['label']:v for v in new_contract[key]}
 for v in old_contract[key]:
  w=new[v['label']];assert v['caption_sha256']==w['caption_sha256']
  if key=='figures':assert v['asset_sha256']==w['asset_sha256']
  else:assert v['tabular_sha256']==w['tabular_sha256']
assert old_contract['display_math']==new_contract['display_math']
assert old_contract['bibliography_sha256']==new_contract['bibliography_sha256']
assert 'Yueran Wang' in main and 'wangyueran@mail.bnu.edu.cn' in main
r={'passed':True,'numeric_and_scope_checks':checks,'historical_result_block_byte_preserved':True,'historical_figures_preserved':len(old_contract['figures']),'historical_tables_preserved':len(old_contract['tables']),'display_formulae_and_bibliography_preserved':True,'evidence_sha256':{n:sha((P/n).read_bytes()) for n in ['evidence_user_benefit_20261009/summary.json','evidence_trace_benefit_confirmation_20261010/summary.json']},'scope':'Exact claim strings and archived object preservation, not external execution attestation or comprehensive semantic equivalence.'}
(O/'confirmed_claims_and_historical_preservation.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(f'{len(checks)} new numeric/scope checks; historical nine figures and seven tables unchanged.')
