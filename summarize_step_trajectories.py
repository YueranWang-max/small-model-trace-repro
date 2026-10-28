"""Report complete correct trajectories separately from correct endpoints."""
import json
from pathlib import Path

for name in ['controlled_step_execution', 'controlled_step_schema']:
    folder = Path(__file__).resolve().parent / 'logs' / name
    frozen = json.loads((folder / 'frozen.json').read_text(encoding='utf-8'))
    data = json.loads((folder / 'reader.json').read_text(encoding='utf-8'))
    results = []
    for row in data['rows']:
        task = next(t for t in frozen['tasks'] if t['id'] == row['id'])
        for arm in ['chain_full', 'chain_local']:
            calls = row['arms'][arm]['calls']
            perfect = len(calls) == 6 and all(
                call['output_state'] == task['trajectory'][i + 1]
                for i, call in enumerate(calls))
            results.append(dict(model=row['model'], task=row['id'], arm=arm,
                                final_em=row['arms'][arm]['final_em'], perfect=perfect))
    (folder / 'trajectory_audit.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    for model in sorted(set(r['model'] for r in results)):
        for arm in ['chain_full', 'chain_local']:
            subset = [r for r in results if r['model'] == model and r['arm'] == arm]
            print(name, model, arm, 'perfect', sum(r['perfect'] for r in subset), '/', len(subset),
                  'correct_endpoint_with_wrong_path', sum(r['final_em'] == 1 and not r['perfect'] for r in subset))
