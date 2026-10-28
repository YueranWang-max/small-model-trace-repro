"""Post-terminal display export; never scores answers or changes frozen analysis."""
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent / 'partial_state_study_20261004'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Reading active attempts is intentionally unsupported.
    required = ['completion.json', 'independent_audit.json', 'analysis.json', 'frozen.json']
    data = {name: json.loads((OUT / name).read_text(encoding='utf-8')) for name in required}
    audit = data['independent_audit.json']
    analysis = data['analysis.json']
    assert audit['audit_passed'] is True
    assert analysis['audit_sha256'] == sha(OUT / 'independent_audit.json')
    assert analysis['analysis_source_sha256'] == sha(Path(__file__).resolve().parent / 'analyze_partial_state_20261004.py')
    assert audit['frozen_sha256'] == sha(OUT / 'frozen.json')
    assert analysis['primary_complete'] == audit['primary_complete'] == data['completion.json']['primary_complete']
    for name, digest in data['frozen.json']['dependencies'].items():
        assert sha(Path(name)) == digest, name
    destinations = ['display_arm_summary.csv', 'display_comparisons.csv', 'display_release.json']
    assert not any((OUT / name).exists() for name in destinations), 'Never overwrite a release'
    rows = []
    for arm in ['direct', 'question_first', 're2', 'task_cue', 'prefix', 'append', 'neutral_prefix']:
        s = analysis['arm_summaries'][arm]
        assert s['planned'] == 32 and s['known'] + s['unknown'] == 32
        assert 0 <= s['f1_lower'] <= s['f1_upper'] <= 1
        rows.append(dict(arm=arm, planned=32, known=s['known'], unknown=s['unknown'],
                         normal_valid=s['normal_valid'], budget_declines=s['observed_budget_declines'],
                         f1_lower=s['f1_lower'], f1_upper=s['f1_upper'],
                         em_lower=s['em_lower'], em_upper=s['em_upper'],
                         deployment_cost_complete=int(s['deployment_cost_complete']),
                         deployment_tokens_sum_lower=s['deployment_tokens_lower_bound'],
                         planned_prediction_tokens_lower=s['deployment_tokens_lower_bound']/32))
    with (OUT / destinations[0]).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fields = ['control', 'f1_difference', 'f1_ci_low', 'f1_ci_high', 'holm_adjusted_sign_p', 'confirmatory_family']
    with (OUT / destinations[1]).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for control, s in analysis['comparisons'].items():
            writer.writerow(dict(control=control, f1_difference=s['f1_difference'],
                                 f1_ci_low=s['f1_ci_low'], f1_ci_high=s['f1_ci_high'],
                                 holm_adjusted_sign_p=s.get('holm_adjusted_sign_p', ''),
                                 confirmatory_family=int(s['confirmatory_family'])))
    manifest = dict(scope='Display export only: frozen audit/analysis supply all scores and inference.',
                    primary_complete=analysis['primary_complete'],
                    promising_candidate_gate=analysis['promising_candidate_gate'],
                    known_reader_slots=analysis['known_reader_slots'], planned_reader_slots=224,
                    inputs={name: sha(OUT / name) for name in required},
                    csv_outputs={name: sha(OUT / name) for name in destinations[:2]},
                    exporter_sha256=sha(Path(__file__)),
                    cost_denominator='Sum of charged deployment tokens divided by 32 planned predictions per arm; incomplete costs remain lower bounds.',
                    quality_bounds='Identification bounds from unknown outcomes, never confidence intervals.',
                    comparison_intervals='Only supplied by complete frozen analysis; descriptive graph bootstrap, not simultaneous.')
    (OUT / destinations[2]).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
