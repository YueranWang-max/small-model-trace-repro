"""Compare final evidential scope with the earliest archived repository manuscript."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
out=ROOT/'revision_evidence_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit='0bcd1297823f811753f3f31da2630bbfaa39db37'
raw=subprocess.check_output(['git','show',commit+':paper/main.tex'],cwd=REPO)
assert hashlib.sha256(raw).hexdigest()=='c62f1b4ae129d769de85de3e904cb8dee97c5c27f0bd1b094b5ef6a29b228aeb'
old=raw.decode('utf-8-sig');main=(ROOT/'main.tex').read_text(encoding='utf-8');body=(ROOT/'migration_primary_body.tex').read_text(encoding='utf-8')
for name in ['evidence_verification_20261006','evidence_paired4b_20261006','evidence_output_budget_control_20261006','evidence_context_recovery_20261006']:
 assert json.loads((ROOT/name/'summary.json').read_bytes())['complete']
phrases=['two orders of magnitude','calls remain constant','emergent ability','never helps','421 calls','0.99']
anchors=[{'phrase':p,'in_baseline':p in old,'in_current_primary':p in main+'\n'+body} for p in phrases]
assert all(v['in_baseline'] for v in anchors)
assert all(not v['in_current_primary'] for v in anchors if v['phrase']!='0.99')
# A new legitimate numeric result may contain 0.99; compare the old sentence instead.
old_segments=re.split(r'(?<=\.)\s+(?=[A-Z\\])|\n{2,}',old)
numeric_legacy_segments=[x.strip() for x in old_segments if '0.99' in x]
assert numeric_legacy_segments and all(x not in main+'\n'+body for x in numeric_legacy_segments)
anchors[-1]['legacy_contexts_absent']=True
anchors[-1]['numeric_token_alone_is_not_a_claim_check']=True
abstract=lambda s:re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S).group(1)
d={'scope':'Earliest retrievable repository manuscript; comparison of evidence and presentation, not increased solver accuracy or external attestation','baseline_commit':commit,'baseline_sha256':hashlib.sha256(raw).hexdigest(),'current_main_sha256':sha(ROOT/'main.tex'),'current_body_sha256':sha(ROOT/'migration_primary_body.tex'),'removed_broad_claims':anchors,'new_completed_current_calls':566,'paired_model_calls':480,'budget_diagnostic_calls':60,'context_diagnostic_calls':26,'primary_distinct_labelled_graphs':12,'same_graphs_used_for_both_models':True,'context_diagnostic_population':'One deliberately selected prior-overflow case','official_gold_checks':750,'historical_ledgers_checked':590,'historical_missing_responses':'Excluded from empirical claims, never recreated as historical observations','r_figure_count':9,'figure_proofs':['figures/r_typography_20261006/VERIFICATION.md','figures/r_paired_models_20261006/VERIFICATION.md','figures/r_output_budget_20261006/VERIFICATION.md'],'arxiv_author_status':'Unverified placeholder unless subsequently supplied by human','baseline_abstract_whitespace_tokens':len(abstract(old).split()),'current_abstract_whitespace_tokens':len(abstract(main).split()),'abstract_count_method':'Whitespace tokens in abstract TeX source, not typeset word count'}
d['arxiv_author_status']='Archived author block confirmed correct by human and restored; not independent identity verification' if 'Author details pending verification' not in main else 'Author placeholder remains'
(out/'comparison_evidence.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
(out/'COMPARISON.md').write_text('''# Evidence of manuscript revision

The baseline is the earliest retrievable repository manuscript, commit 0bcd1297823f811753f3f31da2630bbfaa39db37; its source digest was rechecked against the archive. This does not prove that no earlier unarchived draft existed.

Five textual broad-claim anchors from the baseline are absent from the current primary narrative. The archived passages carrying the old 0.99 claim are also absent; an isolated numeric token is not treated as a claim check. The incomplete-provenance synthetic/document work and reconstructed-request bridge are excluded from empirical claims. Old responses, freezes and failure categories remain unchanged.

The final evidence adds two complete 240-call model studies on the same twelve questions, a 60-call output-budget diagnostic, and a 26-call diagnostic of one selected context-overflow case: 566 new recorded calls. The same questions across models do not form 24 graph units, and the selected context case is not a population estimate. Independent auditors reconstruct payloads, model/context identity, tokenizer counts, graph labels, visible-answer scores, graph-level intervals and logical costs. Earlier checks cover 750 dataset labels and 590 ledgers; the latter include nonresponse categories and are not 590 valid answers.

All nine current empirical figures are R exports. Portable copied inputs and plotting code reproduce their SVGs byte for byte on the same host. Font and layout checks are measured presentation evidence; this is not evidence that answer accuracy increased.

Writing follows the source paper's question -> intervention -> matched observations -> controls and boundaries -> bounded conclusion. The main narrative leads with completed matched studies, reports numeric uncertainty and exact match, and separates operational capacity, output truncation and placement effects. Old incomplete detail is retained mainly in appendices.

These changes strengthen evidential traceability and make the claim narrower and clearer. They do not certify all historical execution, demonstrate universal small-model improvement, establish a new reasoning mechanism, or guarantee arXiv acceptance. Verified human authorship and submission responsibility remain necessary. The final source-package and offline-relocation checks are reported separately.

Machine-readable counts, claim anchors and current source digests: comparison_evidence.json.
''',encoding='utf-8')
print('Final evidence comparison updated without rerendering or modifying experimental records.')
