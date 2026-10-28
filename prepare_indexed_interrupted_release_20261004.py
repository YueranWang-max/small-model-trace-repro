"""Build R inputs from preserved response and separate selection audits; no inference."""
import pathlib,json,csv,hashlib
root=pathlib.Path(__file__).resolve().parent
out=root/'logs/official_graph_indexed_source'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
a=read(out/'interrupted_independent_audit.json');sel=read(out/'selection_reconstruction_audit_20261004.json');f=read(out/'frozen.json');r=read(out/'results.json')
assert a['partial_audit_passed'] and sel['selection_reconstruction_passed'] and len(r)==94
with (out/'interrupted_figure_summary.csv').open('w',newline='',encoding='utf-8') as h:
 w=csv.writer(h);w.writerow(['arm','known','planned','f1_lower','f1_upper','em_lower','em_upper','tokens_lower','cost_complete'])
 for d in a['summary']:w.writerow([d['arm'],d['known_responses'],16,*d['f1_missingness_bounds'],*d['em_missingness_bounds'],d['known_token_lower_bound'],d['known_responses']==16])
with (out/'interrupted_figure_graphs.csv').open('w',newline='',encoding='utf-8') as h:
 w=csv.writer(h);w.writerow(['row_index','arm','known','f1_lower','f1_upper'])
 for c in f['selected']:
  for d in a['summary']:
   rr=[x for x in r if x['row_index']==c['row_index'] and x['arm']==d['arm']];total=sum(x['score']['f1'] for x in rr)
   w.writerow([c['row_index'],d['arm'],len(rr),total/2,(total+2-len(rr))/2])
release=dict(status='interrupted_response_and_selection_audits_passed',primary_complete=False,saved_calls=94,planned_calls=96,selection_reconstruction_complete=True,unknown_attempted_usage_calls=1,not_reached_calls=1,known_token_lower_bound=a['known_token_lower_bound'],followup_gate=a['followup_gate'],graph_units=8,preprocessing_seconds=sum(c['preprocess_wall_sec'] for c in f['selected']),compact_dictionary_bytes=sum(c['compact_dictionary_json_bytes'] for c in f['selected']),audit_inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'frozen.json',out/'interrupted_independent_audit.json',out/'selection_reconstruction_audit_20261004.json')})
(out/'interrupted_release_summary_20261004.json').write_text(json.dumps(release,indent=2),encoding='utf-8');print(json.dumps(release,indent=2))
