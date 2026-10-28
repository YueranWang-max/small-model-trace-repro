"""Check explicitly mapped numerical manuscript clauses against full audit summaries."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
body=(ROOT/'migration_primary_body.tex').read_text(encoding='utf-8');main=(ROOT/'main.tex').read_text(encoding='utf-8')
def load(name):
 p=ROOT/name/'summary.json';d=json.loads(p.read_bytes());assert d['complete'];return d,p
small,sp=load('evidence_verification_20261006');large,lp=load('evidence_paired4b_20261006')
a={r['arm']:r for r in small['arms']};b={r['arm']:r for r in large['arms']}
checks=[]
def require(label,clause,source,document=body):
 assert clause in document,(label,clause)
 checks.append({'claim':label,'required_exact_clause':clause,'audit_source':str(source.relative_to(ROOT)),'matched':True})
require('1.7B F1 by placement',f"Mean F1 was {a['pre']['f1']:.4f} for prefix, {a['post']['f1']:.4f} for append and {a['direct']['f1']:.4f} for direct reading",sp)
require('4B direct F1 and EM',f"Direct reading achieved F1 {b['direct']['f1']:.4f} and EM {b['direct']['em']:.4f}.",lp)
require('4B preparation-inclusive tokens',f"{b['pre']['mean_deployment_tokens']:,.2f} for prefix, {b['post']['mean_deployment_tokens']:,.2f} for append and {b['direct']['mean_deployment_tokens']:,.2f} for direct reading",lp)
require('4B logical research tokens',f"{large['research_tokens_known']:,} model tokens",lp)
require('4B request wall total',f"{large['research_observed_wall_seconds']:,.2f} seconds",lp)
ct=next(c for c in large['contrasts'] if c['contrast']=='pre-direct')
require('4B secondary F1 difference',f"{100*ct['mean_difference']:.2f} percentage points",lp)
require('4B secondary descriptive interval',f"[{100*ct['lower']:.2f},{100*ct['upper']:.2f}]",lp)
require('4B deployment ratio',f"${b['pre']['mean_deployment_tokens']/b['direct']['mean_deployment_tokens']:.2f}$ times",lp)
assert b['pre']['f1']==b['post']['f1']==b['pre']['em']==b['post']['em']==1
assert large['source_valid_count']==0 and b['direct']['valid_count']==42
require('4B ceiling scores','mean F1 and EM were both 1.0000',lp)
require('Abstract 1.7B placement scores',f"prefix F1 was {a['pre']['f1']:.4f} versus {a['post']['f1']:.4f} for append",sp,main)
report={'scope':'Mapped current completed paired-study clauses only; not blanket certification of every historical sentence','checks':checks,'source_hashes':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [sp,lp,ROOT/'migration_primary_body.tex',ROOT/'main.tex']]}
(ROOT/'revision_evidence_20261006/paired_numeric_text_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'{len(checks)} explicitly mapped numeric clauses matched complete independent audit summaries.')
