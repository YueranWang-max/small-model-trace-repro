"""Release audit of the 128-call long/sampling/five-trace stage."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    oldpath=ROOT/'logs/official_graph_native_stage_audit_20261003.json';old=json.loads(oldpath.read_text(encoding='utf-8'));changed=[]
    allowed={'paper/OFFICIAL_GRAPH_NATIVE_INTERFACE_20261003.md','paper/OFFICIAL_GRAPH_NATIVE_ORDER_20261003.md','README.md','paper/REVISION_AUDIT.md','paper/RESEARCH_REBASE_20261003.md','DATA_LICENSES.md'}
    for n,v in list(old['assets'].items()):
        current=sha(ROOT/n)
        if current!=v:
            assert n.replace('\\','/') in allowed,n
            changed.append({'path':n,'previous_sha256':v,'current_sha256':current});old['assets'][n]=current
    if changed:
        old.setdefault('subsequent_document_qualifications',[]).append({'reason':'Qwen recommended thinking sampling qualification; old frozen experimental data/scores unchanged','changes':changed});save(oldpath,old)
    assets={};calls=[];counts={}
    for name,n in [('official_graph_long_boundary',32),('official_graph_sampling_control',16),('official_graph_five_trace',80)]:
        out=ROOT/'logs'/name;fr=json.loads((out/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((out/'results.json').read_text(encoding='utf-8'));audit=json.loads((out/'independent_audit.json').read_text(encoding='utf-8'));assert audit['passed'] and len(rs)==n
        for dep,v in fr['dependencies'].items():assert sha(ROOT/dep)==v,dep
        for p in out.glob('attempt_*.json'):assert json.loads(p.read_text(encoding='utf-8'))['completion']=='returned';assets[str(p.relative_to(ROOT))]=sha(p)
        for file in ['frozen.json','results.json','summary.json','independent_audit.json']:
            p=out/file;assets[str(p.relative_to(ROOT))]=sha(p)
        for path in [ROOT/('run_'+name+'.py'),ROOT/('audit_'+name+'.py'),ROOT/('protocol_'+name+'_20261003.md')]:assets[str(path.relative_to(ROOT))]=sha(path)
        for ext in ['R','pdf','png']:
            p=ROOT/'paper/figures'/('plot_'+name+'.R' if ext=='R' else name+'.'+ext);assert p.exists();assets[str(p.relative_to(ROOT))]=sha(p)
        for tail in ['', '_cases']:
            p=ROOT/'paper/figures/data'/(name+tail+'.csv');assert p.exists();assets[str(p.relative_to(ROOT))]=sha(p)
        calls.extend(rs);counts[name]=n
    for name in ['OFFICIAL_GRAPH_LONG_AND_SAMPLING_20261003.md','OFFICIAL_GRAPH_FIVE_TRACE_20261003.md','OFFICIAL_GRAPH_MIGRATION_REVIEW_20261003.md']:
        p=ROOT/'paper'/name;assets[str(p.relative_to(ROOT))]=sha(p)
    qualified=sum(bool(r['api_input_matches']) for r in calls);assert qualified==128;assert all(all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in calls)
    result={'passed':True,'saved_calls':128,'planned_calls':128,'warmups':0,'counts':counts,'known_tokens':sum(r['raw']['prompt_eval_count']+r['raw']['eval_count'] for r in calls),'current_phase_cost_complete':True,'historical_cost_complete':False,'historical_unknown_attempts_retained':old['additional_uncertain_attempts'],'http_errors':sum(bool(r['error']) for r in calls),'length_terminations':sum(r['raw'].get('done_reason')=='length' for r in calls),'input_count_qualified_calls':qualified,'frozen_dependencies_verified':True,'main_manuscript_updated':False,'figures':'R4.6.1 PDF and600dpiPNG; visually inspected','review':'three simulated emphasis reviews; current-stage scope','assets':assets}
    save(ROOT/'logs/official_graph_state_stage_audit_20261003.json',result);print(json.dumps({k:v for k,v in result.items() if k!='assets'},indent=2))
if __name__=='__main__':main()
