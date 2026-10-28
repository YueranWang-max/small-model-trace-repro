"""Frozen prospective clustered analysis; run after independent terminal audit."""
import pathlib, json, math, collections, csv
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'hybrid_five_trace_replication'

def score(prediction, answer, valid):
    if not valid:
        return dict(f1=0., em=0)
    p, a = set(prediction), set(answer)
    return dict(f1=2*len(p&a)/(len(p)+len(a)) if p or a else 1., em=int(p==a))

def bounds(values, denominator):
    known = [v for v in values if v is not None]
    missing = denominator-len(known)
    return [sum(known)/denominator, (sum(known)+missing)/denominator]

def interval_difference(left, right):
    return [left[0]-right[1], left[1]-right[0]]

def exact_sign_p(positive, negative):
    n = positive+negative
    return min(1., 2*sum(math.comb(n,k) for k in range(min(positive, negative)+1))/2**n) if n else 1.

def main():
    target = OUT / 'analysis.json'
    assert not target.exists(), 'Preserve completed primary analysis.'
    audit = json.loads((OUT / 'independent_audit.json').read_bytes())
    frozen = json.loads((OUT / 'frozen.json').read_bytes())
    assert audit['audit_passed']
    spec = frozen['analysis']
    assert spec['status'] == 'frozen_before_first_call'
    records, cases = audit['records'], frozen['cases']
    arm_names = ['source', 'fresh', 'question_first', 're2', 'prefix', 'append', 'set_mode', 'node_majority']
    graph_results = []
    for index, case in enumerate(cases):
        answer = set(case['answer_nodes'])  # Labels independently verified by separate audit.
        group = [r for r in records if r['case_index']==index]
        source = sorted([r for r in group if r['arm']=='source'], key=lambda r:r['repeat'])
        assert len(source)==5
        all_source_known = all(r['score_determined'] for r in source)
        valid = [r for r in source if r['score_determined'] and r.get('normal_parse',False)]
        votes = {}
        if all_source_known:
            if not valid:
                votes = dict(set_mode=dict(f1=0., em=0, valid=False, prediction=[]),
                             node_majority=dict(**score([],answer,True),valid=True,prediction=[]))
            else:
                normalized = [tuple(sorted(set(r['prediction']))) for r in valid]
                counts = collections.Counter(normalized)
                # Stable maximum selects the earliest source among tied modal sets.
                mode = max(normalized, key=lambda prediction:counts[prediction])
                nodes = collections.Counter(node for prediction in normalized for node in prediction)
                majority = sorted(node for node,count in nodes.items() if count>=3)
                votes = {name: dict(**score(prediction,answer,True), valid=True,prediction=list(prediction))
                         for name,prediction in [('set_mode',mode),('node_majority',majority)]}
        source_token_lower = sum(r.get('prompt_tokens',0)+r.get('output_tokens',0) for r in source if r['usage_known'])
        source_tokens_complete = all(r['usage_known'] for r in source)
        source_client_lower = sum(r.get('client_seconds') or 0 for r in source)
        source_client_complete = all(r.get('client_seconds') is not None for r in source)
        arms = {}
        for name in arm_names:
            if name in ['set_mode','node_majority']:
                vote = votes.get(name)
                arms[name] = dict(f1_bounds=[vote['f1'],vote['f1']] if vote else [0.,1.],
                                  em_bounds=[vote['em'],vote['em']] if vote else [0.,1.],
                                  known_outcomes=1 if vote else 0,
                                  normal_parse_fraction=int(vote['valid']) if vote else None,
                                  deployment_tokens_lower=source_token_lower,
                                  deployment_tokens_complete=source_tokens_complete,
                                  solver_client_seconds_lower=source_client_lower,
                                  solver_client_seconds_complete=source_client_complete,
                                  vote=vote)
                continue
            selected = [r for r in group if r['arm']==name]
            assert len(selected)==5
            known = [r for r in selected if r['score_determined']]
            tokens_lower = sum(r.get('prompt_tokens',0)+r.get('output_tokens',0) for r in selected if r['usage_known'])/5
            complete_cost = all(r['usage_known'] for r in selected)
            client_lower = sum(r.get('client_seconds') or 0 for r in selected)/5
            client_complete = all(r.get('client_seconds') is not None for r in selected)
            if name in ['prefix','append']:
                tokens_lower += source_token_lower
                complete_cost = complete_cost and source_tokens_complete
                client_lower += source_client_lower
                client_complete = client_complete and source_client_complete
            arms[name] = dict(f1_bounds=bounds([r['f1'] if r['score_determined'] else None for r in selected],5),
                              em_bounds=bounds([r['em'] if r['score_determined'] else None for r in selected],5),
                              known_outcomes=len(known),
                              normal_parse_fraction=sum(r.get('normal_parse',False) for r in known)/5,
                              normal_parse_fraction_denominator=5,
                              deployment_tokens_lower=tokens_lower,
                              deployment_tokens_complete=complete_cost,
                              solver_client_seconds_lower=client_lower,
                              solver_client_seconds_complete=client_complete,
                              budget_declines=sum(r['completion']=='budget_declined' for r in selected))
        graph_results.append(dict(case_index=index,row_index=case['row_index'],
                           stratum=('short' if case['length_bin'][0]==2000 else 'long')+'_'+case['problem_type'],
                           arms=arms,always_empty=score([],answer,True),
                           oracle_f1=max(r['f1'] for r in source) if all_source_known else None,
                           oracle_em=max(r['em'] for r in source) if all_source_known else None,
                           source_token_lower=source_token_lower,source_tokens_complete=source_tokens_complete,
                           all_invalid_source_group=all_source_known and not valid,
                           source_group_f1_below_half=arms['source']['f1_bounds'][0]<.5 if all_source_known else None))
    summaries = {}
    for arm in arm_names:
        summaries[arm] = dict(
            **{m+'_bounds':[sum(g['arms'][arm][m+'_bounds'][side] for g in graph_results)/16 for side in [0,1]] for m in ['f1','em']},
            deployment_tokens_lower=sum(g['arms'][arm]['deployment_tokens_lower'] for g in graph_results)/16,
            deployment_tokens_complete=all(g['arms'][arm]['deployment_tokens_complete'] for g in graph_results),
            solver_client_seconds_lower=sum(g['arms'][arm]['solver_client_seconds_lower'] for g in graph_results)/16,
            solver_client_seconds_complete=all(g['arms'][arm]['solver_client_seconds_complete'] for g in graph_results),
            known_graph_outcomes=sum(g['arms'][arm]['known_outcomes']==(1 if arm in ['set_mode','node_majority'] else 5) for g in graph_results),
            normal_parse_fraction=sum(g['arms'][arm]['normal_parse_fraction'] or 0 for g in graph_results)/16)
    comparisons = []
    rng = np.random.default_rng(spec['bootstrap_seed'])
    for comparator in spec['prefix_comparison_family']:
        comparison = dict(comparator=comparator)
        for metric in ['f1','em']:
            all_bounds = [interval_difference(g['arms']['prefix'][metric+'_bounds'],g['arms'][comparator][metric+'_bounds']) for g in graph_results]
            complete = [g for g,b in zip(graph_results,all_bounds) if abs(b[1]-b[0])<=spec['tie_absolute_tolerance']]
            deltas = np.array([g['arms']['prefix'][metric+'_bounds'][0]-g['arms'][comparator][metric+'_bounds'][0] for g in complete],dtype=float)
            replicate_sums = np.zeros(spec['bootstrap_replicates'])
            if complete:
                for stratum in spec['strata']:
                    values = np.array([g['arms']['prefix'][metric+'_bounds'][0]-g['arms'][comparator][metric+'_bounds'][0] for g in complete if g['stratum']==stratum])
                    if len(values):
                        replicate_sums += values[rng.integers(0,len(values),size=(spec['bootstrap_replicates'],len(values)))].sum(axis=1)
                ci = np.quantile(replicate_sums/len(complete),[.025,.975]).tolist()
                mean = float(deltas.mean())
            else:
                ci,mean = None,None
            positive = int(sum(deltas>spec['tie_absolute_tolerance']))
            negative = int(sum(deltas<-spec['tie_absolute_tolerance']))
            comparison[metric] = dict(primary_all16_difference_bounds=[sum(b[s] for b in all_bounds)/16 for s in [0,1]],
                                    complete_pair_graph_count=len(complete),complete_pair_rows=[g['row_index'] for g in complete],
                                    complete_pair_mean_difference=mean,descriptive_bootstrap_ci=ci,
                                    positive_graphs=positive,negative_graphs=negative,tied_graphs=len(complete)-positive-negative,
                                    exact_two_sided_sign_p=exact_sign_p(positive,negative),
                                    sign_test_is_direction_not_mean=True)
        comparisons.append(comparison)
    ordered = sorted(enumerate(comparisons),key=lambda pair:pair[1]['f1']['exact_two_sided_sign_p'])
    adjusted = 0.
    for rank,(_,comparison) in enumerate(ordered):
        adjusted = max(adjusted,min(1.,(6-rank)*comparison['f1']['exact_two_sided_sign_p']))
        comparison['f1']['holm_adjusted_sign_p'] = adjusted
    returned_qualification = all(r['qualified'] for r in records if r['completion']=='returned')
    all_complete = audit['primary_complete'] and audit['unknown_scored_outcomes']==0 and returned_qualification and records[0]['qualified']

    def evidence(comparison):
        result = comparison['f1']
        return bool(all_complete and result['complete_pair_graph_count']==16
                    and result['complete_pair_mean_difference']>=spec['practical_prefix_gain_minimum']
                    and result['descriptive_bootstrap_ci'][0]>0 and result['holm_adjusted_sign_p']<.05)

    ordering = evidence(next(c for c in comparisons if c['comparator']=='append'))
    accuracy_controls = [c for c in comparisons if c['comparator']!='append']
    em_noninferior_points = all(c['em']['primary_all16_difference_bounds'][0]>=-spec['tie_absolute_tolerance'] for c in accuracy_controls)
    accuracy = bool(all_complete and all(evidence(c) for c in accuracy_controls) and em_noninferior_points
                    and summaries['prefix']['normal_parse_fraction']>=.95)
    non_trace = ['fresh','question_first','re2','set_mode','node_majority']
    costs_complete = all(summaries[name]['deployment_tokens_complete'] for name in ['prefix']+non_trace)
    best_cost = min(summaries[name]['deployment_tokens_lower'] for name in non_trace)
    cost_candidate = bool(accuracy and costs_complete and summaries['prefix']['deployment_tokens_lower']<=best_cost)
    result = dict(analysis_complete=True,all_planned_outcomes_known_and_all_returned_responses_qualified=all_complete,
                  graph_units=16,summary=summaries,comparisons=comparisons,graphs=graph_results,
                  ordering_evidence=ordering,accuracy_candidate=accuracy,cost_candidate=cost_candidate,
                  important_progress_established=False,independent_confirmation_required=True,
                  em_noninferiority_is_point_estimate_not_formal_noninferiority_test=True,
                  always_empty_f1=sum(g['always_empty']['f1'] for g in graph_results)/16,
                  known_research_tokens=audit['known_research_tokens'],
                  unknown_attempted_usage_calls=audit['unknown_attempted_usage_calls'],
                  calibration_tokens=sum(r.get('prompt_tokens',0)+r.get('output_tokens',0) for r in records if r['phase']=='input_calibration' and r['usage_known']),
                  preparation_seconds=audit['preparation_seconds'],
                  bundle_and_all_reader_job_preparation_seconds=audit['bundle_and_all_reader_job_preparation_seconds'],
                  full_research_workflow_seconds=audit['full_research_workflow_seconds'],
                  limitations=['Bootstrap intervals are descriptive, not simultaneous confidence intervals.',
                               'Sign tests concern direction across graph pairs, not mean F1.',
                               'Unknown deployment outcomes are bounded; complete-pair results are secondary.',
                               'EM rule checks observed mean, not statistical noninferiority.',
                               'Five source calls are paid in full per trace or vote deployment.',
                               'Solver client wall includes requests but excludes shared preparation and qualification metadata; those are recorded separately, not silently amortized into a single deployment latency.',
                               'This experiment does not alone establish a novel method or important confirmed progress.'])
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT / 'audited_graph_arms.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['row_index','stratum','arm','f1_lower','f1_upper','em_lower','em_upper','deployment_tokens_lower','deployment_tokens_complete','normal_parse_fraction']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for g in graph_results:
            for name,a in g['arms'].items():
                writer.writerow(dict(row_index=g['row_index'],stratum=g['stratum'],arm=name,
                                     f1_lower=a['f1_bounds'][0],f1_upper=a['f1_bounds'][1],em_lower=a['em_bounds'][0],em_upper=a['em_bounds'][1],
                                     deployment_tokens_lower=a['deployment_tokens_lower'],deployment_tokens_complete=a['deployment_tokens_complete'],normal_parse_fraction=a['normal_parse_fraction']))
    print(json.dumps({k:v for k,v in result.items() if k not in ['graphs','comparisons']},indent=2))

if __name__ == '__main__':
    main()
