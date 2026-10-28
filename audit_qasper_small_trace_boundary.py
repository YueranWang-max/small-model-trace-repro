"""Reconstruct prompt hashes and rescore archived 4B outputs separately."""
import json
import run_qasper_small_trace_boundary as pilot
from rescore_qasper_demand_transfer import references,f1,normalize
old=pilot.old;OUT=pilot.OUT;ROOT=pilot.ROOT

def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));arch=json.loads((ROOT/'logs/qasper_visible_trace_replication/results.json').read_text(encoding='utf-8'));checked=quotes=0;oldrows=[]
    for p in fr['papers']:
        for q in p['qas']:
            r=next(r for r in d['rows'] if r['id']==q['question_id']);query=old.query_text(p['title'],q['question']);feedback=r['feedback'];quotes+=bool(feedback['quote']);assert not feedback['quote'] or feedback['quote'] in p['source']
            for a in pilot.ARMS:
                prompt=old.second_prompt('reread' if a=='question_pre' else a,p['source'],query,feedback['trace'],feedback['quote'])
                if a=='question_pre':prompt='Question: '+query+'\n'+prompt
                assert old.sha(prompt.encode())==r['arms'][a]['prompt_sha256'];checked+=1
            ap=next(pp for pp in arch['papers'] if pp['paper_id']==p['paper_id']);ar=next(rr for rr in ap['records'] if rr['id']==q['question_id']);gold=references(q);oldrows.append({'id':r['id'],'paper_id':p['paper_id'],'scores':{a:max(f1('Unanswerable' if normalize(v['answer'])=='unknown' else v['answer'],g) for g in gold) for a,v in ar['arms'].items()}})
    out={'reconstructed_second_prompt_hashes':checked,'exact_quotes':quotes,'archived_4b_reference_max_multiset_f1':{a:sum(r['scores'][a] for r in oldrows)/len(oldrows) for a in ('reread','trace_pre','trace_post','quote_pre')},'old_primary_scores_not_modified':True,'same_source_questions_different_own_traces_architectures_and_model_settings':True,'old_rescored_rows':oldrows};(OUT/'prompt_audit_old_rescore.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print({k:v for k,v in out.items() if k!='old_rescored_rows'})
if __name__=='__main__':main()
