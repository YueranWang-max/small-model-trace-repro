"""Descriptive paired analysis with explicit missingness; execute after audit."""
import pathlib,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
a=json.loads((ROOT/'continued_bridge_audit.json').read_text(encoding='utf-8'))
fr=json.loads((ROOT/'bridge_run'/'frozen.json').read_text(encoding='utf-8'))
assert a['known_outcomes']==63 and a['unknown_original_attempts']==[20]
models=['qwen3:1.7b','qwen3:8b'];cases=[]
for case in fr['cases']:
    row=dict(row_index=case['row_index'],type=case['problem_type'],bin=case['bin'],gold_empty=not bool(case['answer_nodes']))
    for model in models:
        rs=[r for r in a['records'] if r['row_index']==case['row_index'] and r['model']==model]
        known=[r for r in rs if r['outcome_known']];missing=2-len(known)
        row[model]=dict(known_repeats=len(known),f1_available=float(np.mean([r['f1'] for r in known])),em_available=float(np.mean([r['em'] for r in known])),f1_bounds=[sum(r['f1'] for r in known)/2,(sum(r['f1'] for r in known)+missing)/2],em_bounds=[sum(r['em'] for r in known)/2,(sum(r['em'] for r in known)+missing)/2])
    row['complete_pair']=all(row[m]['known_repeats']==2 for m in models);cases.append(row)
complete=[r for r in cases if r['complete_pair']];assert len(complete)==15
rng=np.random.default_rng(202610093)
strata=[np.array([i for i,r in enumerate(complete) if r['bin']==[lo,hi] and r['type']==typ]) for lo,hi in [(2000,3500),(14000,16000)] for typ in ['parents','bfs']]
assert [len(s) for s in strata]==[4,3,4,4]
picks=np.concatenate([rng.choice(s,size=(10000,len(s)),replace=True) for s in strata],axis=1)
contrasts={}
for metric in ['f1','em']:
    delta=np.array([r[models[1]][metric+'_available']-r[models[0]][metric+'_available'] for r in complete]);boot=delta[picks].mean(axis=1)
    contrasts[metric]=dict(complete_graphs=15,complete_pair_mean_8b_minus_1_7b=float(delta.mean()),descriptive_95_interval=np.quantile(boot,[.025,.975]).tolist(),positive_graphs=int((delta>0).sum()),negative_graphs=int((delta<0).sum()),equal_graphs=int((delta==0).sum()),all_16_graph_primary_missingness_bounds=[float(np.mean([r[models[1]][metric+'_bounds'][0]-r[models[0]][metric+'_bounds'][1] for r in cases])),float(np.mean([r[models[1]][metric+'_bounds'][1]-r[models[0]][metric+'_bounds'][0] for r in cases]))])
groups=[]
for lo,hi in [(2000,3500),(14000,16000)]:
    for typ in ['parents','bfs']:
        selected=[r for r in cases if r['bin']==[lo,hi] and r['type']==typ]
        group=dict(type=typ,bin=[lo,hi],graphs=4)
        for model in models:
            group[model]={metric+'_missingness_bounds':[float(np.mean([r[model][metric+'_bounds'][j] for r in selected])) for j in [0,1]] for metric in ['f1','em']}
        groups.append(group)
result=dict(primary_complete=False,original_unknown_attempt=20,contrasts=contrasts,groups=groups,cases=cases,always_empty_f1_em=float(np.mean([r['gold_empty'] for r in cases])),limitations=['Complete paired protocol-score estimates describe 15 graphs; missingness bounds describe all 16.','Client request errors score zero under the frozen deployment rule; their actual model answers and usage remain unknown.','Intervals are exploratory and not adjusted across exploration stages.','Model comparison includes weights, training and capacity; no size-only causal interpretation.','Known tokens are a lower bound because original attempt 020 and request-error usage are unknown.','Source competence is not evidence of successful trace transfer.'])
(ROOT/'continued_bridge_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(contrasts,indent=2))
