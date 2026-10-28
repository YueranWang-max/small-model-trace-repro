"""Frozen graph-unit analysis. Run only after terminal independent audit."""
import hashlib
import json
import math
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'partial_state_study_20261004'
ARMS = ['direct','question_first','re2','task_cue','prefix','append','neutral_prefix']
CONTROLS = ['direct','question_first','re2','task_cue','neutral_prefix']


def sign_p(difference):
    positive = int(np.sum(difference > 1e-12))
    negative = int(np.sum(difference < -1e-12))
    n = positive+negative
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(positive,negative)+1))/2**n) if n else 1.
    return dict(p=p,positive=positive,negative=negative,ties=int(len(difference)-n),test='exact two-sided paired sign; nonzero directions')


def main():
    destination = OUT/'analysis.json'
    assert not destination.exists()
    audit_path = OUT/'independent_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['audit_passed']
    frozen = json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    for file,expected in frozen['dependencies'].items():
        assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest() == expected
    result = dict(status='complete' if audit['primary_complete'] else 'interrupted',
                  primary_complete=audit['primary_complete'],independent_graph_units=16,repetitions_per_graph=2,
                  known_reader_slots=audit['known_reader_slots'],planned_reader_slots=224,
                  audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                  analysis_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  arm_summaries={},comparisons={},promising_candidate_gate=False,
                  bootstrap_intervals='descriptive graph-cluster percentile95%; not simultaneous')
    completion = json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    result['actual_research_workflow_seconds'] = completion.get('actual_workflow_seconds')
    result['deployment_timing_scope'] = ('Reconstructed source+reader measured preparation/request-protocol components; '
                                       'not contiguous deployment measurement. Excludes final logging persist and one-off research preparation.')
    for arm in ARMS:
        records = [s for s in audit['reader_results'] if s['arm'] == arm]
        assert len(records) == 32
        known = [s for s in records if s['known']]
        missing = 32-len(known)
        result['arm_summaries'][arm] = dict(planned=32,known=len(known),unknown=missing,
            normal_valid=sum(s.get('normal_valid',False) for s in known),
            observed_budget_declines=sum(s['status']=='budget_decline' for s in records),
            f1_lower=sum(s['f1'] for s in known)/32,f1_upper=(sum(s['f1'] for s in known)+missing)/32,
            em_lower=sum(s['em'] for s in known)/32,em_upper=(sum(s['em'] for s in known)+missing)/32,
            deployment_tokens_lower_bound=sum(s['tokens_lower'] for s in records),
            deployment_component_seconds_lower=sum(s['deployment_component_seconds_lower'] for s in records),
            deployment_cost_complete=all(s.get('tokens_complete',False) and s['known'] for s in records))
    complete = (audit['primary_complete'] and audit['known_reader_slots']==224
                and audit['attempted_unknown_usage_calls']==0)
    if complete:
        graph_rows = [c['row_index'] for c in frozen['cases']]
        means = {}
        for arm in ARMS:
            matrix = [[s for s in audit['reader_results'] if s['arm']==arm and s['row_index']==row] for row in graph_rows]
            assert all(len(group)==2 and all(s['known'] for s in group) for group in matrix)
            means[arm] = np.array([np.mean([s['f1'] for s in group]) for group in matrix])
            summary = result['arm_summaries'][arm]
            summary['mean_complete_deployment_tokens'] = summary['deployment_tokens_lower_bound']/32
            summary['mean_measured_deployment_component_seconds'] = summary['deployment_component_seconds_lower']/32
        rng = np.random.default_rng(202610400)
        resamples = rng.integers(0,16,size=(20000,16))
        p_values = []
        for control in CONTROLS+['append']:
            difference = means['prefix']-means[control]
            boot = difference[resamples].mean(axis=1)
            test = sign_p(difference)
            comparison = dict(f1_difference=float(difference.mean()),
                              f1_ci_low=float(np.quantile(boot,.025)),f1_ci_high=float(np.quantile(boot,.975)),
                              pergraph_f1_differences=difference.tolist(),
                              em_difference=result['arm_summaries']['prefix']['em_lower']-result['arm_summaries'][control]['em_lower'],
                              sign_test=test,confirmatory_family=control in CONTROLS)
            result['comparisons'][control] = comparison
            if control in CONTROLS:
                p_values.append((test['p'],control))
        ordered = sorted(p_values)
        previous = 0.
        for index,(p,control) in enumerate(ordered):
            adjusted = max(previous,min(1.,p*(len(ordered)-index)))
            result['comparisons'][control]['holm_adjusted_sign_p'] = adjusted
            previous = adjusted
        quality_gate = all(result['comparisons'][control]['f1_difference']>=.05
                           and result['comparisons'][control]['f1_ci_low']>0
                           and result['comparisons'][control]['holm_adjusted_sign_p']<.05
                           and result['comparisons'][control]['em_difference']>=0 for control in CONTROLS)
        cheapest = min(result['arm_summaries'][arm]['mean_complete_deployment_tokens'] for arm in ARMS[:4])
        cost_gate = result['arm_summaries']['prefix']['mean_complete_deployment_tokens'] <= cheapest
        reliability_gate = result['arm_summaries']['prefix']['normal_valid'] >= 31
        result.update(quality_gate=quality_gate,cost_gate=cost_gate,reliability_gate=reliability_gate,
                      cheapest_nonstate_mean_tokens=cheapest,
                      promising_candidate_gate=quality_gate and cost_gate and reliability_gate)
    result['limitations'] = ['A gate pass still requires separate independent confirmation and does not establish broad document-QA performance.',
                             'Logical token cost is not latency, energy, money or GPU operation count.',
                             'Missingness bounds are not confidence intervals; repeated seeds are not independent graphs.']
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons']},indent=2))


if __name__ == '__main__':
    main()
