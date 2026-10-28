# Better Graph Answers from a Small Model

Simpler Inputs and a Short Reused Trace — research code and archived evidence.

## Scope
The paper studies local small-model reproduction of Trace as State, its boundaries, and a separately frozen compact-input single-trace extension. In the 24-graph confirmation, prefix F1 increases from 0.6542 to 0.7703, at 39.8% more complete model tokens than compact direct. This is not a universal cost-saving claim. Discovery and confirmation must remain separate.

## Start here
- `paper/main.pdf`: current manuscript.
- `paper/main.tex`: source; included TeX files and vector figures are provided.
- `run_user_benefit_20261009.py`: discovery runner (168 archived calls).
- `run_trace_benefit_confirmation_20261010.py`: confirmation runner (264 archived calls).
- `logs/user_benefit_20261009/` and `logs/trace_benefit_confirmation_20261010/`: frozen requests, responses and records where present in the evidence release.
- `paper/`: audits, result tables and R figure scripts. Older dated files retain historical provenance and are not all part of the current paper.

## Environment and reproduction
Original experiments ran on Windows with local Ollama; exact model identities, serving versions, seeds and budgets are in each frozen protocol. Install Python and the dependencies required by the selected script; the supplied import inventory is informational, not a tested lockfile. R/Cairo is used for figures; XeLaTeX and BibTeX compile the multi-file paper. Compile from `paper/`: `xelatex main.tex`, `bibtex main`, then `xelatex main.tex` twice.

This is a source-and-evidence release, not a one-command portable application. Some scripts contain the original absolute Windows paths. Set working directories and adapt path configuration to the new checkout in a separate working copy; do not rewrite frozen records or claim their hashes remain identical. Scripts named run/finish can call a model or resume jobs: inspect them before executing. Do not rerun archived completed or transport-uncertain calls. Audit existing records first. No fresh-environment full rerun has been performed for this release.

## Licences
Code: MIT (`LICENSE`). Third-party datasets/tokenizers retain their own terms (`DATA_LICENSES.md` and accompanying notices); the code licence does not relicense them. The manuscript licence is separate. Model weights, runtimes, credentials and Git history are not included.

## Release manifest
`RELEASE_MANIFEST.json` lists SHA-256 for every delivered file except the manifest itself. Archived evidence manifests describe their original release snapshots, not the entire combined repository. Root-level scripts also include historical exploratory utilities; their presence is not evidence that their outputs support the current manuscript.
