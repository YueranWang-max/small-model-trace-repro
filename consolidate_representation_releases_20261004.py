"""Consolidate already audited completed studies, without inference or reanalysis."""
import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DEST = ROOT/'recovery_bridge_20261003/representation_release_20261004.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not DEST.exists()
    stages = {}
    for name,calls in [('identity_recode',48),('reasoning_mode',64)]:
        base = ROOT/'logs'/('official_graph_'+name)
        audit = read(base/'independent_audit.json')
        summary = read(base/'summary.json')
        frozen = read(base/'frozen.json')
        records = read(base/'results.json')
        assert audit['passed'] and audit['qualified_calls'] == calls == len(records)
        assert summary['qualified_calls'] == summary['planned_calls'] == calls
        assert summary['unknown_usage_calls'] == 0
        for file,expected in audit['assets'].items():
            assert sha(ROOT/file) == expected
        arm_tokens = {}
        for record in records:
            raw = record['raw']
            assert type(raw['prompt_eval_count']) is int and type(raw['eval_count']) is int
            assert raw['prompt_eval_count'] >= 0 and raw['eval_count'] >= 0
            arm_tokens[record['arm']] = arm_tokens.get(record['arm'],0)+raw['prompt_eval_count']+raw['eval_count']
        assert sum(arm_tokens.values()) == summary['known_tokens']
        datafile = ROOT/'paper/figures/data'/('official_graph_'+name+'.csv')
        with datafile.open(encoding='utf-8',newline='') as handle:
            figure_rows = list(csv.DictReader(handle))
        for row,expected in zip(figure_rows,summary['summary'],strict=True):
            assert row['arm'] == expected['arm']
            assert abs(float(row['f1'])-expected['f1']) < 1e-12
            assert abs(float(row['em'])-expected['em']) < 1e-12
        assets = [base/'frozen.json',base/'independent_audit.json',base/'results.json',base/'summary.json',datafile,
                  ROOT/'paper/figures'/('plot_official_graph_'+name+'.R'),
                  ROOT/'paper/figures'/('official_graph_'+name+'.pdf'),
                  ROOT/'paper/figures'/('official_graph_'+name+'.png')]
        entry = dict(audited_calls=calls,unknown_usage_calls=0,model_identity=frozen['identity'],
                     independent_graph_units=len(frozen['selected']),seed_repetitions=len(frozen['seeds']),
                     physical_research_tokens=sum(arm_tokens.values()),physical_arm_token_sums=arm_tokens,
                     figure_values_match_saved_summary=True,assets={str(p.relative_to(ROOT)):sha(p) for p in assets},
                     scope='Existing independent audit certification and figure/data provenance; no new scoring or inference.')
        if name == 'identity_recode':
            assert arm_tokens == {s['arm']:s['tokens'] for s in summary['summary']}
            entry.update(cost_denominator='Each physical arm sum includes16calls:8graphs×2seeds; mean single reading is arm_sum/16.',
                         compact_dictionary_bytes=summary['compact_dictionary_bytes_total'],
                         preprocessing_seconds=summary['preprocess_wall_sec_total'],gate=summary['followup_gate'])
        else:
            for s in summary['summary']:
                if s['arm'] in ['on','off']:
                    assert arm_tokens[s['arm']]/2 == s['deployment_tokens']
            types = {c['row_index']:c['problem_type'] for c in frozen['selected']}
            route = [r for r in records if r['arm']==('off' if types[r['row_index']]=='parents' else 'on')]
            assert len(route) == 32
            route_tokens = sum(r['raw']['prompt_eval_count']+r['raw']['eval_count'] for r in route)
            expected_route = next(s for s in summary['summary'] if s['arm']=='route')
            assert route_tokens/2 == expected_route['deployment_tokens']
            entry.update(cost_denominator='Figure deployment_tokens is the sum of two-seed mean costs over16graphs: physical32call_sum/2. Mean single reading is physical_sum/32.',
                         derived_route_actual_selected_call_tokens=route_tokens,
                         derived_route_definition='Parents use saved off calls; BFS use saved on calls. No third physical32callarm was generated.',
                         gates=summary['gates'])
        stages[name] = entry
    result = dict(status='two_completed_release_provenance_checks_passed',inference_calls=0,
                  source_sha256=sha(pathlib.Path(__file__)),stages=stages,
                  limitations=['No claim that either stage passed its prespecified improvement gate.',
                               'Different models, interventions and task strata cannot be pooled as one trace-reuse effect.',
                               'Logical token totals and separately measured preprocessing are not comprehensive hardware cost or contiguous end-to-end latency.'])
    DEST.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],calls=sum(s['audited_calls'] for s in stages.values()),
                         research_tokens=sum(s['physical_research_tokens'] for s in stages.values()),
                         physical_arm_token_sums={k:v['physical_arm_token_sums'] for k,v in stages.items()}),indent=2))


if __name__ == '__main__':
    main()
