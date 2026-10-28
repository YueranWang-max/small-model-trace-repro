from pathlib import Path
import hashlib
import re

root = Path(__file__).resolve().parents[1]
paper = root / 'paper'
main = paper / 'main.tex'
raw = main.read_bytes()
expected = '3e03c5a23dd2add8faae06b77e293351526a0ee8d1162b84df76c5594ba38ce2'
assert hashlib.sha256(raw).hexdigest() == expected, 'Unexpected manuscript revision'
snapshot = paper / 'revision_snapshots/main_before_primary_reorganization_20261004.tex'
assert not snapshot.exists(), 'Do not overwrite snapshot'
snapshot.write_bytes(raw)
text = raw.decode('utf-8')
start = text.index('\\section{Introduction}')
end = text.index('\\section*{Use of Generative AI Tools}')
legacy = text[start:end]
legacy = legacy.replace('Reorganization of the main body around that migration evidence remains incomplete; the different protocols are not pooled.', 'The main body now centers the audited migration evidence; these historical protocols are not pooled with it.')
legacy_file = paper / 'historical_source_free_exploration.tex'
assert not legacy_file.exists()
legacy_file.write_text('% Preserved historical exploration; scoped supplement, not primary migration results.\n' + legacy, encoding='utf-8')
new = text[:start] + '\\input{migration_primary_body}\n\n' + text[end:]
title_start = new.index('\\title{')
author_start = new.index('\\author{')
new = new[:title_start] + '\\title{\\textbf{Reusing Reasoning on Local Small Models:\\\\\nAccuracy, Context and Full-Cost Boundaries}}\n\n' + new[author_start:]
abstract_start = new.index('\\begin{abstract}')
abstract_end = new.index('\\end{abstract}', abstract_start) + len('\\end{abstract}')
abstract = r'''\begin{abstract}
Can reasoning-trace reuse improve a local small model after all preparation calls are paid? Motivated by Trace as State, we evaluate native source generation and source-retaining readers on official graph questions, separating these interventions from reversible representation and thinking-mode controls. An eight-graph Qwen3.5-4B source study achieved F1 0.9286 and exact match 0.875 across sixteen normally stopped, qualified calls. Source competence did not establish deployable reuse: a subsequent long-graph five-trace trial exceeded its configured context on the first graph and ended incomplete. Under that trial's single-prediction token contract, five source inputs alone exceeded the maximum direct-prediction budget on all sixteen selected graphs. An earlier four-graph prefix-reader study improved F1 by 15.64 percentage points at approximately 8.07 times the deployment tokens, without improving exact match; its paired interval crossed zero. Compact graph recoding offers a separate exploratory signal, while disabling thinking reduced quality alongside cost. We preserve interrupted outcomes and unknown usage rather than treating them as completed comparisons. A separately frozen short-state study is ongoing. The completed evidence identifies migration and accounting boundaries; it does not yet confirm an accuracy-preserving end-to-end efficiency gain.
\end{abstract}'''
new = new[:abstract_start] + abstract + new[abstract_end:]
marker = '\\appendix'
insert = r'''\appendix

\section*{Supplement Scope: Historical Source-Free Exploration}
The following preserved exploration evaluates sharded source verification, source-free graph readout and document question answering under earlier protocols. It supplies historical evidence and limitations rather than the primary official-graph migration claim. In particular, score-only legacy archives cannot be strictly rescored, and synthetic syntax permits exact algorithmic solutions. Its figures remain subject to the pending whole-manuscript R regeneration and provenance review.
\input{historical_source_free_exploration}
'''
new = new.replace(marker, insert, 1)
main.write_text(new, encoding='utf-8')
all_text = new + (paper / 'migration_primary_body.tex').read_text(encoding='utf-8') + legacy_file.read_text(encoding='utf-8')
labels = re.findall(r'\\label\{([^}]+)\}', all_text)
references = re.findall(r'\\ref\{([^}]+)\}', all_text)
assert len(labels) == len(set(labels)), 'Duplicate labels'
assert not set(references) - set(labels), f'Missing labels: {set(references)-set(labels)}'
assert new.count('\\begin{document}') == new.count('\\end{document}') == 1
print({'snapshot_sha256': expected, 'new_main_sha256': hashlib.sha256(main.read_bytes()).hexdigest(), 'labels': len(labels), 'references': len(references), 'compiled': False, 'scope': 'source reorganization and lightweight cross-reference check only'})
