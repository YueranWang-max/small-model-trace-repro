"""Post-terminal descriptive diagnostics; never modifies frozen scoring or records."""
import collections
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'partial_state_study_20261004'
audit_path = OUT / 'independent_audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
arms = collections.defaultdict(collections.Counter)
details = []
for row in audit['reader_results']:
    if not row['known']:
        arms[row['arm']]['unknown'] += 1
        continue
    record = json.loads((OUT / f"attempt_{row['ordinal']:03d}.json").read_text(encoding='utf-8'))
    raw = record.get('raw', {})
    reason = raw.get('done_reason')
    if row['status'] != 'qualified_response':
        category = row['status']
    elif reason == 'length':
        category = 'output_limit'
    elif not row['normal_valid']:
        category = 'normal_stop_malformed'
    elif row['em'] == 1:
        category = 'normal_valid_exact'
    else:
        category = 'normal_valid_incorrect'
    arms[row['arm']][category] += 1
    details.append(dict(ordinal=row['ordinal'], row_index=row['row_index'], repeat=row['repeat'],
                        arm=row['arm'], category=category, f1=row['f1'], em=row['em'],
                        cap=record.get('request', {}).get('options', {}).get('num_predict'),
                        input_tokens=raw.get('prompt_eval_count'), output_tokens=raw.get('eval_count'),
                        thinking_characters=len(raw.get('message', {}).get('thinking', '')),
                        visible_characters=len(raw.get('message', {}).get('content', ''))))

by_pair = collections.defaultdict(dict)
for row in details:
    by_pair[(row['row_index'], row['repeat'])][row['arm']] = row
paired = {}
for control in ['direct', 'question_first', 're2', 'task_cue', 'neutral_prefix', 'append']:
    pairs = [(v['prefix'], v[control]) for v in by_pair.values() if 'prefix' in v and control in v]
    paired[control] = dict(known_pairs=len(pairs),
        prefix_f1=sum(a['f1'] for a,b in pairs)/len(pairs) if pairs else None,
        control_f1=sum(b['f1'] for a,b in pairs)/len(pairs) if pairs else None,
        prefix_exact=sum(a['em'] for a,b in pairs),control_exact=sum(b['em'] for a,b in pairs))

result = dict(scope='Post-interruption descriptive diagnosis; no confirmatory inference or score repair.',
              audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
              arm_failure_counts={k:dict(v) for k,v in arms.items()}, paired_observed_only=paired,
              records=details,
              limitations=['Observed pairs are incomplete and repeated seeds are not independent graphs.',
                           'Changing decoding or caps is development, requiring fresh independent confirmation.',
                           'Length outputs retain their original zero scores; no thinking-answer salvage.'])
destination = ROOT / 'PARTIAL_STATE_FAILURE_DIAGNOSIS_20261005.json'
with destination.open('x', encoding='utf-8') as f:
    json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
