"""Bounded eight-graph source analysis after terminal independent audit."""
import json, pathlib, csv
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'hybrid_source_qualification'

def main():
    assert not (OUT/'analysis.json').exists()
    audit=json.loads((OUT/'independent_audit.json').read_bytes())
    frozen=json.loads((OUT/'frozen.json').read_bytes())
    assert audit['audit_passed']
    groups=[]
    for lower,kind in [(2000,'parents'),(2000,'bfs'),(14000,'parents'),(14000,'bfs')]:
        cases=[c for c in frozen['cases'] if c['length_bin'][0]==lower and c['problem_type']==kind]
        ids=[c['row_index'] for c in cases]
        records=[r for r in audit['records'] if r['phase']=='scored_source' and r['row_index'] in ids]
        assert len(cases)==2 and len(records)==4
        known=[r for r in records if r['score_determined']]
        missing=4-len(known)
        groups.append(dict(length_bin=cases[0]['length_bin'],kind=kind,row_indices=ids,
                           independent_graphs=2,planned_source_calls=4,known_source_outcomes=len(known),
                           normal_parse=sum(r.get('normal_parse',False) for r in known),
                           f1_bounds=[sum(r['f1'] for r in known)/4,(sum(r['f1'] for r in known)+missing)/4],
                           em_bounds=[sum(r['em'] for r in known)/4,(sum(r['em'] for r in known)+missing)/4],
                           known_tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in records if r['usage_known']),
                           unknown_attempted_usage=sum(r['attempted'] and not r['usage_known'] for r in records),
                           known_client_seconds=sum(r.get('client_seconds') or 0 for r in records),
                           client_seconds_complete=all(r.get('client_seconds') is not None for r in records)))
    result=dict(independent_graphs=8,planned_source_calls=16,groups=groups,
                primary_bounds=audit['primary_bounds'],normal_parse=audit['normal_parse'],
                frozen_source_gate_passed=audit['frozen_gate_passed'],
                known_research_tokens=audit['known_research_tokens'],calibration_tokens=audit['calibration_tokens'],
                unknown_attempted_usage_calls=audit['unknown_attempted_usage_calls'],
                preparation_seconds=audit['preparation_seconds'],
                known_scored_tokens=sum(g['known_tokens'] for g in groups),
                known_scored_client_seconds=sum(g['known_client_seconds'] for g in groups),
                client_seconds_complete=all(g['client_seconds_complete'] for g in groups),
                always_empty_f1_em=sum(not c['answer_nodes'] for c in frozen['cases'])/8,
                source_gate_is_necessary_for_trace_test=False,
                trace_reuse_tested=False,important_progress_established=False,
                limitations=['Two source repeats per graph are not independent graph cases.',
                             'Model family, training, tokenizer, profile and output cap differ from previous model bridges.',
                             'Longer cases also contain larger graphs; this is not a pure context-length intervention.',
                             'Source accuracy does not establish trace-reuse benefit or end-to-end deployment savings.',
                             'The frozen source gate retains its original verdict, independent of permission to study traces.'])
    duration_keys=['total_duration_seconds','load_duration_seconds','prompt_eval_duration_seconds','eval_duration_seconds']
    result['server_duration_totals']={phase:{key:dict(known_seconds=sum(r.get(key) or 0 for r in audit['records'] if r['phase']==phase),
                                                   complete=all(r.get(key) is not None for r in audit['records'] if r['phase']==phase))
                                           for key in duration_keys}
                                     for phase in ['scored_source','input_calibration']}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
