"""Build numerical TeX tables only from complete independently audited studies."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_numeric_tables_20261006';OUT.mkdir(exist_ok=True)
def read(name):
 p=ROOT/name/'summary.json';d=json.loads(p.read_bytes());assert d['complete'],name
 return d,{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def table(name,header,lines,caption,label):
 n=len(header)
 text='\\begin{table}[htbp]\n\\centering\\small\n\\begin{tabular}{'+'l'+'r'*(n-1)+'}\n\\toprule\n'+' & '.join(header)+'\\\\\n\\midrule\n'
 text+='\n'.join(' & '.join(line)+'\\\\' for line in lines)
 text+='\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
 (OUT/(name+'.tex')).write_text(text,encoding='utf-8')
def main():
 small,ms=read('evidence_verification_20261006');large,ml=read('evidence_paired4b_20261006');budget,mb=read('evidence_output_budget_control_20261006');context,mc=read('evidence_context_recovery_20261006')
 lines=[]
 for name,d in [('1.7B',small),('4B',large)]:
  for a in d['arms']:
   lines.append([name+' '+{'pre':'prefix','post':'append','direct':'direct'}[a['arm']],f"{a['f1']:.4f}",f"{a['em']:.4f}",str(a['valid_count'])+'/60',f"{a['mean_deployment_tokens']:,.0f}"])
 table('paired',['Configuration','F1','EM','Normal-valid','Tokens/answer'],lines,'Completed Qwen3 comparisons on the same twelve graphs, five reader repeats each. Trace costs charge all five sources and one reader. Normal-valid refers to termination and visible-answer format, not correctness. The same questions are not twenty-four independent graph units.','tab:paired-final')
 old=next(a for a in large['arms'] if a['arm']=='direct')
 lines=[['8192',f"{old['f1']:.4f}",f"{old['em']:.4f}",str(old['valid_count'])+'/60',f"{old['mean_deployment_tokens']:,.0f}"],['28672',f"{budget['f1']:.4f}",f"{budget['em']:.4f}",str(budget['valid_count'])+'/60',f"{budget['mean_deployment_tokens']:,.0f}"]]
 table('budget',['Output cap','F1','EM','Normal-valid','Tokens/answer'],lines,'Sequential Qwen3-4B diagnostic on the same known questions and seeds. The larger cap equals the five-source pipeline maximum generated-token allowance, not its total logical-token or compute budget. Original 8192-cap responses remain unchanged.','tab:budget-final')
 lines=[]
 for a in context['arms']:
  lines.append([a['arm'].replace('_',' '),f"{a['f1']:.4f}",f"{a['em']:.4f}",str(a['valid_count'])+'/5',f"{a['deployment_tokens']:,.0f}"])
 table('context',['Reader','F1','EM','Normal-valid','Tokens/answer'],lines,'One previously selected overflow case, evaluated at a configured 196608-token context with the original trace bundle. Five repeats per condition describe this case only; no population interval is inferred. Prefix and append include the complete original source preparation cost.','tab:context-final')
 (OUT/'manifest.json').write_text(json.dumps({'inputs':[ms,ml,mb,mc],'outputs':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.glob('*.tex'))]},indent=2),encoding='utf-8')
 print('Three numerical tables generated from four complete audit summaries.')
if __name__=='__main__':main()
