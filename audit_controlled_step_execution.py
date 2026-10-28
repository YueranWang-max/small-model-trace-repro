"""Verify source maps, oracle inputs, model-only propagation and schema values."""
import json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(v):return hashlib.sha256(v).hexdigest()

def audit(name):
    outdir=ROOT/'logs'/name;f=json.loads((outdir/'frozen.json').read_text(encoding='utf-8'));d=json.loads((outdir/'reader.json').read_text(encoding='utf-8'));count=oracle=propagated=valid=0
    for t in f['tasks']:
        assert sha(t['source'].encode())==t['source_sha256'];state=t['initial'];trajectory=[state]
        for j,line in enumerate(t['step_lines']):
            assert line in t['source'];mapping={int(a):int(b) for a,b in re.findall(r'S(\d+)->S(\d+)',line)};assert mapping=={int(k):v for k,v in t['maps'][j].items()};assert sorted(mapping)==list(range(8));state=mapping[state];trajectory.append(state)
        assert trajectory==t['trajectory'] and f'S{state:02d}'==t['answer']
    for r in d['rows']:
        t=next(t for t in f['tasks'] if t['id']==r['id'])
        for a,ar in r['arms'].items():
            for i,v in enumerate(ar['calls']):
                if a.startswith('oracle'):assert v['input_state']==t['trajectory'][i];oracle+=1
                elif i:assert v['input_state']==ar['calls'][i-1]['output_state'];propagated+=1
                else:assert v['input_state']==t['initial']
                if name.endswith('schema'):
                    raw=v.get('raw',{});obj=json.loads(raw['native_schema_content']);assert set(obj)=={'state'} and obj['state'] in f['schema']['properties']['state']['enum'];assert int(obj['state'][1:])==v['output_state'];assert raw['schema_valid'];valid+=1
                count+=1
    out={'source_maps_and_groundtruth_verified':True,'recorded_calls':count,'oracle_input_checks':oracle,'model_only_feedback_checks':propagated,'native_json_values_verified':valid,'not_semantic_generalization':True};(outdir/'execution_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');return out
if __name__=='__main__':
    for name in ['controlled_step_execution','controlled_step_schema']:print(name,audit(name))
