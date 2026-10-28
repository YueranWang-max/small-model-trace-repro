"""Exact transport recovery and matched request repeat; no score replacement."""
import json
from pathlib import Path
from run_qasper_interface_replication import ROOT,OUT,SOURCE,call,sha,save,references,f1,normalize
DEST=ROOT/'logs/qasper_interface_transport_recovery'
PROTOCOL=ROOT/'protocol_qasper_interface_transport_recovery_20261003.md'

def main():
    DEST.mkdir(parents=True,exist_ok=True);ctrl=json.loads((ROOT/'logs/qasper_interface_full_binding/results.json').read_text(encoding='utf-8'));pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));errors=[r for r in ctrl['records'] if 'error' in r['result']];assert len(errors)==1 and 'HTTPError 500' in errors[0]['result']['error'];row=errors[0];m=row['model'];qid=row['id'];pr=next(r for r in pilot['models'][m]['records'] if r['id']==qid);cases={'full_bind':row['result']['request'],'wide_bind':pr['arms']['wide_bind']['request']};fp=DEST/'frozen.json'
    if not fp.exists():save(fp,{'script_sha256':sha(Path(__file__).read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'control_results_sha256':sha((ROOT/'logs/qasper_interface_full_binding/results.json').read_bytes()),'model':m,'paper_id':row['paper_id'],'id':qid,'requests':cases})
    fr=json.loads(fp.read_text(encoding='utf-8'));assert fr['requests']==cases and fr['script_sha256']==sha(Path(__file__).read_bytes()) and fr['protocol_sha256']==sha(PROTOCOL.read_bytes()) and fr['control_results_sha256']==sha((ROOT/'logs/qasper_interface_full_binding/results.json').read_bytes());path=DEST/'results.json';data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'arms':{}};src=json.loads(SOURCE.read_text(encoding='utf-8'));gq=next(q for q in src[row['paper_id']]['qas'] if q['question_id']==qid)
    for a,req in cases.items():
        if a in data['arms']:continue
        r=call(m,req['messages'][1]['content'],req['format'],req['options']['num_predict']);assert r['request']==req;pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred;r['f1']=max(f1(aligned,g) for g in references(gq));data['arms'][a]=r;save(path,data);print(a,'error' if 'error' in r else 'done',flush=True)
    original_sum=sum(r['result']['f1'] for r in ctrl['records'] if r['model']==m);summary={'calls':2,'model':m,'case_id':qid,'original_failure_zero_preserved':True,'full_bind_original_mean':original_sum/24,'full_bind_recovery_sensitivity_mean':(original_sum+data['arms']['full_bind']['f1'])/24,'original_wide_bind_f1':pr['arms']['wide_bind'].get('f1'),'repeated_wide_bind_f1':data['arms']['wide_bind']['f1'],'full_recovery_error':data['arms']['full_bind'].get('error'),'not_primary_or_independent_confirmation':True};save(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
