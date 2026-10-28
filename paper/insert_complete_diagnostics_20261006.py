"""Insert complete diagnostic evidence; fail closed on missing/partial audit outputs."""
from pathlib import Path
import json,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parent

def load(name):
 p=ROOT/name/'summary.json';d=json.loads(p.read_bytes());assert d['complete'],name;return d

def insert(s,label,block,before):
 start='% BEGIN '+label;end='% END '+label
 if start in s:
  a=s.index(start);b=s.index(end,a)+len(end);return s[:a]+start+'\n'+block+end+s[b:]
 assert before in s;return s.replace(before,start+'\n'+block+end+'\n\n'+before,1)

def main():
 small=load('evidence_verification_20261006');large=load('evidence_paired4b_20261006');budget=load('evidence_output_budget_control_20261006');context=load('evidence_context_recovery_20261006')
 subprocess.run([sys.executable,str(ROOT/'build_final_results_20261006.py')],check=True,cwd=ROOT.parent)
 path=ROOT/'migration_primary_body.tex';old=path.read_bytes();body=path.read_text(encoding='utf-8')
 method=r'''\paragraph{Output-budget diagnostic.} Interim 4B output truncation motivated a separately frozen follow-up on the same twelve known questions and five matched reader seeds. Sixty new direct requests used an output cap of 28,672 tokens, equal to $5\times4096+8192$, with all other recorded inputs, model and sampling settings retained. Original 8,192-cap direct responses remained the historical comparator. This matched maximum generated-token allowances, not total logical tokens, memory, time or compute. Conditions ran sequentially; the diagnostic hypothesis was chosen after interim observations. All cases were retained. The descriptive long-minus-original-direct interval used 20,000 stratified graph-bootstrap draws, seed 202610067.

\paragraph{Configured-context diagnostic.} A separate frozen follow-up retained the original five-trace bundle for the previously declined row-506 case. With the original Qwen3.5-4B digest and sampling profile, it changed the configured context from 131,072 to 196,608 tokens. One full-prefix input probe with a one-token output cap checked accounting and residency without scoring an answer. Five paired seeds supplied prefix, append, fresh, question-first and Re2 readers, each with a 32,768-token output cap: 26 new calls in total. Five-second call gaps and sixty-second repeat-block rests were retained. Full original source preparation was charged to trace conditions; new-run usage was separately reported. This deliberately selected failure case diagnosed execution and case-specific outcomes, not a new population sample or completion of the old sixteen-graph trial.

'''
 body=insert(body,'COMPLETE_DIAGNOSTIC_METHODS',method,'\\paragraph{Trace construction and controls.}')
 diff=budget['long_minus_old_direct'];oldarm=next(a for a in large['arms'] if a['arm']=='direct')
 result=(r'\subsection{A larger output allowance was tested on the same questions}'+'\n'+r'\label{sec:output-budget}'+'\n'+
  f"All sixty new direct-long calls completed and passed independent request and scoring checks. With a 28,672-token output cap, mean F1 was {budget['f1']:.4f}, EM {budget['em']:.4f}, and {budget['valid_count']}/60 responses were normal-valid. There were {budget['length_terminated']} length-terminated outputs, compared with eighteen under the original cap. The paired F1 change from the original direct condition was {100*diff['mean_difference']:.2f} percentage points, with a descriptive interval of $[{100*diff['lower']:.2f},{100*diff['upper']:.2f}]$ points. These results quantify this sequential diagnostic on the known questions, rather than a fresh-case or equal-compute comparison.\n\n"+
  f"Mean logical tokens for one direct-long answer were {budget['mean_deployment_tokens']:,.2f}, compared with {oldarm['mean_deployment_tokens']:,.2f} in the original direct condition. The new diagnostic incurred {budget['known_new_research_tokens']:,} model tokens and {budget['observed_new_request_wall_seconds']:,.2f} seconds of observed request time, excluding planned rests. The original trace scores and costs were not overwritten. Prefix preparation cost was {next(a for a in large['arms'] if a['arm']=='pre')['mean_deployment_tokens']/budget['mean_deployment_tokens']:.2f} times direct-long tokens, with the same observed F1 and EM ceiling.\n\n"+
  f"Among the eighteen previously length-terminated responses, {budget['old_length_thinking_prefix_match_count']}/18 new thinking strings began with the old thinking text with trailing whitespace removed. The new runs were therefore not byte-identical continuations of the old runs despite matching seeds. This descriptive check further limits attribution of every changed answer to the output cap alone.\n\n"+
  r'\input{final_numeric_tables_20261006/budget}'+'\n\n'+
  r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/output_budget_20261006.pdf}
\caption{Output-budget diagnostic on the same twelve Qwen3-4B questions, rendered in R from complete independent audits. (a) Direct F1 averaged over five seeds per graph, with parents and BFS displayed separately. Old length-terminated responses retain zero scores. (b) Mean logical tokens for one answer; trace conditions charge all five sources. New direct answers and both trace placements reached the observed F1/EM ceiling. The sequential known-question diagnostic matches maximum generated allowances, not total compute, and does not establish population equivalence or isolated causality.}
\label{fig:output-budget}
\end{figure}

''')
 body=insert(body,'COMPLETE_OUTPUT_BUDGET_RESULTS',result,'\\subsection{Native source qualification succeeded on the selected graphs}')
 arms={a['arm']:a for a in context['arms']};p=arms['prefix'];a=arms['append'];f=arms['fresh']
 result=(r'\subsection{A larger configured context enabled the retained-case diagnostic}'+'\n'+r'\label{sec:context-diagnostic}'+'\n'+
  'All twenty-six new calls passed independent request, input-count and resident-model checks at a configured 196,608-token context. The probe verified the full 144,677-token prefix input; it was not a scored answer. This established execution at the changed allocation for this case, rather than reasoning capacity throughout the context window.\n\n'+
  f"Across five reader seeds, prefix F1/EM were {p['f1']:.4f}/{p['em']:.4f}, append {a['f1']:.4f}/{a['em']:.4f}, and fresh direct reading {f['f1']:.4f}/{f['em']:.4f}. Normal-valid counts were {p['valid_count']}/5, {a['valid_count']}/5 and {f['valid_count']}/5 respectively. Table~\\ref{{tab:context-final}} retains all five conditions. No population interval was inferred from this deliberately selected case. The old sixteen-graph archive remained interrupted and its unknown outcomes were not replaced.\n\n"+
  f"The original five-source preparation cost was {context['source_preparation_tokens']:,} tokens. New probe and reader calls consumed {context['new_research_tokens_known']:,} model tokens, with {context['observed_request_wall_seconds']:,.2f} seconds of observed request time. Preparation was charged once to each trace deployment estimate, never amortized over the five experimental repeats.\n\n"+
  r'\input{final_numeric_tables_20261006/context}'+'\n\n')
 body=insert(body,'COMPLETE_CONTEXT_DIAGNOSTIC_RESULTS',result,'\\subsection{Answer-format sensitivity limited the early-state comparison}')
 table=r'\input{final_numeric_tables_20261006/paired}'+'\n\n'
 body=insert(body,'COMPLETE_PAIRED_NUMERIC_TABLE',table,'\\subsection{The second model reached a ceiling in both trace placements}')
 discussion=(f"The output-budget diagnostic changed direct F1 from {oldarm['f1']:.4f} to {budget['f1']:.4f} while holding the same question and sampling settings fixed. Because its hypothesis was selected after truncation and conditions ran sequentially, it supports a bounded output-budget interpretation, not a pure causal estimate of the reasoning-state mechanism. The retained-case context diagnostic also separated a configured allocation failure from execution at a larger allocation; it did not fill the old trial's missing quality outcomes.\n\n")
 body=insert(body,'COMPLETE_DIAGNOSTIC_DISCUSSION',discussion,'\\paragraph{Relation to Trace as State.}')
 backup=ROOT/'revision_snapshots/migration_primary_before_final_diagnostics_20261006.tex'
 if not backup.exists(): backup.write_bytes(old)
 path.write_text(body,encoding='utf-8')
 mainpath=ROOT/'main.tex';text=mainpath.read_text(encoding='utf-8');text=text.replace('Its direct F1 was 0.6967, with eighteen of sixty answers output-truncated.',f"Its direct F1 was 0.6967, with eighteen of sixty answers output-truncated; a sequential diagnostic at a 28,672-token output allowance achieved direct F1 {budget['f1']:.4f}.")
 text=text.replace('Separate native trials exposed configured context overflow and answer-format sensitivity.', 'Separate native trials exposed configured context overflow and answer-format sensitivity; one retained overflow case executed at a larger 196,608-token allocation.')
 text=text.replace('All eight empirical figures','All nine empirical figures')
 mainpath.write_text(text,encoding='utf-8')
 (ROOT/'revision_evidence_20261006/diagnostics_inserted.json').write_text(json.dumps({'complete_audits_required':True,'body_before_sha256':hashlib.sha256(old).hexdigest(),'body_after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Literal complete-study values; qualitative interpretation and final review still required'},indent=2),encoding='utf-8')
 print('Inserted complete diagnostic methods, results, tables and bounded interpretation; final review required.')
if __name__=='__main__':main()
