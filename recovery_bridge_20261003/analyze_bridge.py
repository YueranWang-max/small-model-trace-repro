"""Paired descriptive graph-cluster analysis; run after completion and audit."""
import pathlib,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
audit=json.loads((ROOT/'bridge_audit.json').read_text(encoding='utf-8'))
frozen=json.loads((ROOT/'bridge_run'/'frozen.json').read_text(encoding='utf-8'))
assert audit['complete'] and len(audit['records'])==64
records=audit['records'];models=['qwen3:1.7b','qwen3:8b'];case_results=[]
for case in frozen['cases']:
    out=dict(row_index=case['row_index'],bin=case['bin'],problem_type=case['problem_type'],gold_empty=not bool(case['answer_nodes']))
    for model in models:
        rs=[r for r in records if r['row_index']==case['row_index'] and r['model']==model]
        assert len(rs)==2 and all('f1' in r for r in rs)
        out[model]=dict(f1=float(np.mean([r['f1'] for r in rs])),em=float(np.mean([r['em'] for r in rs])),normal_fraction=float(np.mean([r['valid'] for r in rs])),mean_tokens=float(np.mean([r['prompt_tokens']+r['output_tokens'] for r in rs])) if all(r['usage_known'] for r in rs) else None)
    case_results.append(out)
groups={}
for lo,hi in [(2000,3500),(14000,16000)]:
    for typ in ['parents','bfs']:
        rs=[r for r in case_results if r['bin']==[lo,hi] and r['problem_type']==typ]
        assert len(rs)==4
        groups[f'{typ}_{lo}_{hi}']={model:{metric:float(np.mean([r[model][metric] for r in rs])) for metric in ['f1','em','normal_fraction']} for model in models}
rng=np.random.default_rng(202610092);comparisons={}
for metric in ['f1','em']:
    delta=np.array([r[models[1]][metric]-r[models[0]][metric] for r in case_results])
    # Fixed four strata, graph remains the resampling unit; both seeds move together.
    strata=[np.array([i for i,r in enumerate(case_results) if r['bin']==[lo,hi] and r['problem_type']==typ]) for lo,hi in [(2000,3500),(14000,16000)] for typ in ['parents','bfs']]
    picks=np.concatenate([rng.choice(ids,size=(10000,4),replace=True) for ids in strata],axis=1)
    boot=delta[picks].mean(axis=1)
    comparisons[metric]=dict(mean_8b_minus_1_7b=float(delta.mean()),descriptive_95_interval=np.quantile(boot,[.025,.975]).tolist(),positive_graphs=int((delta>0).sum()),negative_graphs=int((delta<0).sum()),equal_graphs=int((delta==0).sum()))
result=dict(independent_graphs=16,repeats_per_graph_per_model=2,comparisons=comparisons,groups=groups,cases=case_results,always_empty_em=float(np.mean([r['gold_empty'] for r in case_results])),limitations=['Exploratory intervals, no adjustment across research stages.','Model contrast includes weights, training and capacity; not size-only causal.','Historical manifest exclusion check pending D drive restoration.','Source-only study; does not establish Trace as State transfer.','Across-model token comparison does not establish hardware or money savings.'])
(ROOT/'bridge_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(comparisons,indent=2))
