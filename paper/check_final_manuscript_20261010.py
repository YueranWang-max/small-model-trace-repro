"""Read-only checks of complete-study numeric clauses and source dependencies."""
from pathlib import Path
import json,hashlib,re,importlib.util
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
body=(ROOT/'migration_primary_body.tex').read_text(encoding='utf-8')
main=(ROOT/'main.tex').read_text(encoding='utf-8')
inputs=[];checks=[]
def load(name):
 p=ROOT/name/'summary.json';d=json.loads(p.read_bytes());assert d['complete'],name
 inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 return d
small=load('evidence_verification_20261006');large=load('evidence_paired4b_20261006')
budget=load('evidence_output_budget_control_20261006');context=load('evidence_context_recovery_20261006')
def require(label,clause,text=body):
 assert clause in text,(label,clause)
 checks.append({'claim':label,'clause':clause,'matched':True})
require('budget F1 and EM',f"mean F1 was {budget['f1']:.4f}, EM {budget['em']:.4f}")
require('budget normal-valid count',f"{budget['valid_count']}/60 responses were normal-valid")
require('budget length terminations',f"{budget['length_terminated']} length-terminated outputs")
d=budget['long_minus_old_direct']
require('budget paired F1 change',f"{100*d['mean_difference']:.2f} percentage points")
require('budget descriptive interval',f"[{100*d['lower']:.2f},{100*d['upper']:.2f}]")
require('budget deployment tokens',f"{budget['mean_deployment_tokens']:,.2f}")
require('budget new research tokens',f"{budget['known_new_research_tokens']:,} model tokens")
require('budget request wall',f"{budget['observed_new_request_wall_seconds']:,.2f} seconds")
assert budget['f1']==1 and 'larger-output direct 4B diagnostic reaches the trace accuracy ceiling' in main
a={v['arm']:v for v in context['arms']}
for name,label in [('prefix','prefix F1/EM were'),('append','append'),('fresh','fresh direct reading')]:
 require('context '+name+' scores',f"{label} {a[name]['f1']:.4f}/{a[name]['em']:.4f}")
require('context source cost',f"{context['source_preparation_tokens']:,} tokens")
require('context new usage',f"{context['new_research_tokens_known']:,} model tokens")
require('context request wall',f"{context['observed_request_wall_seconds']:,.2f} seconds")
for name,data in [('paired',None),('budget',None),('context',None)]:
 p=ROOT/'final_numeric_tables_20261006'/f'{name}.tex';assert p.exists();inputs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
pairtext=(ROOT/'final_numeric_tables_20261006/paired.tex').read_text()
for name,s in [('1.7B',small),('4B',large)]:
 for v in s['arms']:
  require('paired table '+name+' '+v['arm'],name+' '+{'pre':'prefix','post':'append','direct':'direct'}[v['arm']]+' & '+f"{v['f1']:.4f} & {v['em']:.4f} & {v['valid_count']}/60 & {v['mean_deployment_tokens']:,.0f}",pairtext)
budgettext=(ROOT/'final_numeric_tables_20261006/budget.tex').read_text()
old=next(v for v in large['arms'] if v['arm']=='direct')
for cap,v in [(8192,old),(28672,budget)]:
 require('budget table '+str(cap),str(cap)+' & '+f"{v['f1']:.4f} & {v['em']:.4f} & {v['valid_count']}/60 & {v['mean_deployment_tokens']:,.0f}",budgettext)
contexttext=(ROOT/'final_numeric_tables_20261006/context.tex').read_text()
for v in context['arms']:
 require('context table '+v['arm'],v['arm'].replace('_',' ')+' & '+f"{v['f1']:.4f} & {v['em']:.4f} & {v['valid_count']}/5 & {v['deployment_tokens']:,.0f}",contexttext)
# Follow only actual TeX inputs. Fail on missing dependencies, references or duplicate labels.
tex=[];seen=set();figures=[]
def visit(p):
 p=p.resolve();assert p.is_relative_to(ROOT) and p.exists()
 if p in seen:return
 seen.add(p);tex.append(p);s=p.read_text(encoding='utf-8')
 for name in re.findall(r'\\input\{([^}]+)\}',s):visit(ROOT/(name if name.endswith('.tex') else name+'.tex'))
 for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):
  q=(ROOT/name).resolve();assert q.is_relative_to(ROOT) and q.exists();figures.append(q)
visit(ROOT/'main.tex')
text='\n'.join(p.read_text(encoding='utf-8') for p in tex)
labels=re.findall(r'\\label\{([^}]+)\}',text);assert len(labels)==len(set(labels))
refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',text);assert not set(refs)-set(labels)
keys={k.strip() for g in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',text) for k in g.split(',')}
bib=(ROOT/'references.bib').read_text(encoding='utf-8');bibkeys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
assert len(bibkeys)==len(set(bibkeys));assert not keys-set(bibkeys)
assert len(figures)==11 and len(set(figures))==11
# Verify the exact component case and portable filename characters on Windows.
for p in tex+figures+[ROOT/'references.bib']:
 parent=ROOT
 for part in p.relative_to(ROOT).parts:
  assert re.fullmatch(r'[A-Za-z0-9_+.,=\-]+',part),part
  assert part in [entry.name for entry in parent.iterdir()],str(p)
  parent=parent/part
report={'complete_current_studies':True,'scope':'Explicit current numeric clauses and TeX dependency graph; historical studies have separate audits','checks':checks,'inputs':inputs,'tex_files':[str(p.relative_to(ROOT)) for p in tex],'figure_pdfs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in figures],'citation_keys':sorted(keys),'reference_count':len(refs),'label_count':len(labels),'missing_references':[],'source_hashes':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in tex]}
(ROOT/'revision_evidence_20261010/final_numeric_and_dependency_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'{len(checks)} final numeric clauses; {len(tex)} TeX files, eleven figures, {len(keys)} cited references checked.')
