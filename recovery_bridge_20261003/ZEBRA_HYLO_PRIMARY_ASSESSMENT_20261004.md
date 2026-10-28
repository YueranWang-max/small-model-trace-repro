# Zebra-HyLo primary-source assessment — 2026-10-04

Scope: supplied WeChat text; primary paper https://arxiv.org/html/2604.24715v1 and released model card https://huggingface.co/amd/HyLo-Llama-6MLA22GDN-64K-SFT . GitHub tree and raw README retrieval failed; code implementation has not been inspected or executed.

## Corrections
- 98% is relative KV-cache reduction at matched sequence length for selected configurations, not total VRAM or end-to-end cost reduction.
- Paper section 4.4: 2M prefill/decode sweep uses TP=8, batch=1, eight MI300X GPUs; tested variants are 6MLA22M2 and 14MLA14M2. This does not establish 2M answer accuracy or single laptop feasibility. Released 6MLA22GDN card explicitly supports 65536 trained/evaluated tokens.
- WeChat 55.1/0.8 values are Table 2 Llama-1B 4MLA12GDN trained at 8K. Its matched 64K-trained RULER-64K is 40.8. Table 3 Llama-3B 6MLA22GDN instead changes RULER-64K from 14.1 to 46.3. The article's approximately 60x comparison combines different backbones.
- Stage I uses 2K intermediate-layer distillation; Stage II uses 8K or 64K output distillation. Do not describe every model as sequential 8K then 64K training.
- Section 3.2 says uniform or baseline-recommended layer layouts, with long-context training rather than layer selection as the study focus. Do not attribute a new universal sensitivity algorithm to HyLo.
- 10B vs 400B is 40x, not the introduction's 400x; inherited pretraining and teacher costs prevent equating this comparison to total compute savings.

## Relevance and priorities (hypotheses, not results)
1. Separate capacity, exact retrieval and multi-hop reasoning qualifications at matched actual input lengths. Being executable alone is insufficient.
2. Current Qwen3.5-4B already mixes GDN and full attention; this does not directly supply an MLA conversion or 98% additional savings for our model.
3. After the live frozen trial terminates, retain priority for qualified native context / KV-precision experiments. No backend changes or additional inference during the current trial.
4. Task-sensitive evidence retention: evaluate sentence/block deletions on development questions to estimate answer sensitivity, preserve exact IDs/numbers/negations/edges and critical evidence, retain pointers to originals, and validate on independent unseen documents/questions. Charge probing and compilation. This is an analogy to architecture allocation, not an implemented HyLo mechanism or proven novelty.
5. Distilling evidence selection or trace-following behaviour is a longer-term branch. It changes the model and requires independent validation and paid teacher/training costs. Full HyLo conversion is not currently justified on this laptop.

No manuscript changes, installations, downloads of weights, runtime changes, or experimental performance claims were made for this assessment.
