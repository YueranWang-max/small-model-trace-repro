# Data Licenses and Attribution

This repository contains **original code** (MIT license, see `LICENSE`) and
**benchmark data** built from four third-party datasets. The HotpotQA and
MuSiQue original upstream distributions are not included; this repository
redistributes only the listed derived subsets. The official Qasper v0.3
train/dev archive and official test JSON were downloaded locally for research; their
source, checksum, and attribution are recorded below.

## Derived data files included

| File | Source dataset | Upstream license | Questions |
|---|---|---|---|
| `data/realdoc_hotpot_32k.jsonl` | HotpotQA (dev, distractor setting) | CC BY-SA 4.0 | 50 |
| `data/realdoc_musique_32k.jsonl` | MuSiQue (answerable, dev) | CC BY 4.0 | 50 |
| `data/qasper_official/qasper-dev-v0.3.json` | Qasper official development split | CC BY 4.0 | 281 papers; pilot selects 82 questions over 9 papers |
| `data/qasper_official/qasper-train-v0.3.json` | Qasper official training split | CC BY 4.0 | 888 papers; fingerprint study uses paper-level isolation |
| `data/qasper_official/qasper-test-v0.3.json` | Qasper official test split | CC BY 4.0 | 416 papers, 1,451 questions; demand-prior transfer |

SHA-256 checksums:

```
42490b2f653919855e36e26ce9465310818af59cd7ed8ebf6f5c386fffdf86f3  data/realdoc_hotpot_32k.jsonl
1b82e0646026a2162ad213e0eae89b04c28542463957775aadedc9e60cd28799  data/realdoc_musique_32k.jsonl
2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93  data/qasper_official/qasper-dev-v0.3.json
9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4  data/qasper_official/qasper-train-v0.3.json
6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5  data/qasper_official/qasper-test-v0.3.json
```

Each JSONL record contains: the question id, the question text, the answer,
and a ~32K-character document constructed by concatenating the gold paragraphs
with distractor paragraphs (per the build scripts below).

## Derivation

- `build_realdoc.py`, `build_realdoc2.py` — construct the 32K-character
  documents and select the 50 questions per family from the upstream dev sets.
- Downstream distillation caches and per-question records are archived under
  `logs/` (e.g. `logs/realdoc_hotpot_distill_cache.json`).

## Upstream sources (download separately)

**HotpotQA** — Yang et al., *HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering*, EMNLP 2018 (DOI: 10.18653/v1/D18-1259).

- Paper: https://aclanthology.org/D18-1259/
- Data: https://hotpotqa.github.io/ (official distribution,
  `hotpot_dev_distractor_v1.json`)
- License: **CC BY-SA 4.0**. The derived file above is a modification of
  HotpotQA content and is likewise distributed under CC BY-SA 4.0
  (share-alike), with this notice as attribution.

**MuSiQue** — Trivedi et al., *MuSiQue: Multihop Questions via Single-hop
Question Composition*, TACL 2022 (DOI: 10.1162/tacl_a_00475).

- Paper: https://aclanthology.org/2022.tacl-1.31/
- Data: https://github.com/stonybrooknlp/musique (`musique_ans_v1.0_dev.jsonl`)
- License: **CC BY 4.0**. The derived file above is distributed under the
  same license with this notice as attribution.

**Qasper** — Dasigi et al., *A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers*, NAACL 2021 (DOI: 10.18653/v1/2021.naacl-main.365).

- Paper: https://aclanthology.org/2021.naacl-main.365/
- Official data: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
- The official archive SHA-256 is
  `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a`,
  matching the checksum published in the `allenai/qasper` dataset metadata.
- Dataset license: **CC BY 4.0**, per the official `allenai/qasper` dataset card.
  The full paper text in Qasper comes from S2ORC; downstream users should
  follow the upstream data terms when redistributing it.
- Official test archive: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
  SHA-256 `72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72`.
  Only its JSON data member was read; evaluator code was not extracted or executed.
  Provenance is recorded in `data/qasper_official/test_provenance.json`.

## Reproducing the derived files

1. Download the two upstream dev files from the official sources above into
   `data/` (do not commit them).
2. Run the build scripts to regenerate the derived JSONL files; verify the
   SHA-256 checksums above.

## Official GraphWalks short-context native experiments (2026-10-03)

Source: https://huggingface.co/datasets/openai/graphwalks ; upstream dataset card declares MIT. Revision: be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7. File graphwalks_128k_and_shorter.parquet (22,128,284 bytes), SHA256 54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516. Original prompts and labels preserved in data/graphwalks_official; upstream README and metadata archived alongside. All 750 labels in this archived shorter-input file matched an independent graph computation. This file contains 400 parent and 350 BFS questions; the full released repository has additional rows and is not represented by this file alone. This source has a 2026-02-27 correction documented in the upstream README; old synthetic GraphWalks-style experiments are separate derived tasks.

## Official Qwen3 tokenizer and configuration calibration (2026-10-03)

Source: https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ; upstream model card declares Apache-2.0. Archived tokenizer, tokenizer_config, generation_config, config and README in `data/qwen3_tokenizer_official/`, with per-file SHA256 and provenance in metadata.json. Tokenizer JSON SHA256: aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4. These files calibrate input counts and document recommended sampling; they are not experimental answer data or a newly trained model. Local quantized model identity and runtime metadata are recorded separately in request ledgers.

## Current-paper provenance clarification (2026-10-06)

The publisher file page for `graphwalks_128k_and_shorter.parquet` was checked on 6 October 2026; its published SHA-256 exactly matched the archived file. The recorded download revision is retained rather than overwritten. Verification is recorded in `paper/official_dataset_provenance_20261006.json`. Current manuscript evidence is limited to the separately audited official-graph experiments. Older document subsets and synthetic explorations remain historical assets and are not empirical evidence for the current paper.

Model package hashes and observed public-tag drift are recorded under `paper/model_provenance_20261006/`. Each original model license blob is preserved separately. Dataset/code license notices do not establish the human author identity, manuscript license choice or arXiv acceptance.
