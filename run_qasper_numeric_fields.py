"""Exact-source numeric record pilot with required fields and baseline fallback."""
import json
import random
import re
import sys
import time
from pathlib import Path
from repro_lib import ollama_chat
from run_qasper_fingerprint import prepare
from run_qasper_prefix_vs_bm25 import ROOT,sha,MODEL
from run_qasper_visible_trace_replication import call,query_text
from run_qasper_contract_replication import parsed
from rescore_qasper_demand_transfer import references,f1,normalize
from audit_qasper_measurement import scalar

OUT=ROOT/'logs/qasper_numeric_fields';SOURCE=ROOT/'data/qasper_official/qasper-train-v0.3.json'
PROTOCOL=ROOT/'protocol_qasper_numeric_fields_20261002.md'
PAT=r'how (many|much|large|big)|\b(number|percentage|percent|rate|size)\b'

def required(question):
    return {'unit':bool(re.search(r'percentage|percent',question,re.I)),
        'condition':bool(re.search(r'\b(when|under|during)\b',question,re.I)),
        'comparator':bool(re.search(r'compar|baseline|improv|difference',question,re.I))}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);source=json.loads(SOURCE.read_text(encoding='utf-8'))
    split=json.loads((ROOT/'logs/qasper_fingerprint/split.json').read_text(encoding='utf-8'));cohorts={}
    for label,ids in [('validation',split['validation']),('evaluation',split['heldout'][86:])]:
        tasks=[]
        for pid in ids:
            p=prepare(pid,source[pid])
            for q in p['questions']:
                if re.search(PAT,q['text'],re.I):
                    selected=q['rank'][:8]
                    tasks.append({'paper_id':pid,'id':q['id'],'question':q['text'],'title':source[pid]['title'],
                        'required':required(q['text']),'paragraphs':p['paragraphs'],'selected':selected})
        cohorts[label]=tasks[:12] if label=='validation' else tasks
    assert len(cohorts['validation'])==12 and len(cohorts['evaluation'])==29
    f={'source_sha256':sha(SOURCE.read_bytes()),'protocol_sha256':sha(PROTOCOL.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
        'dependencies':{n:sha((ROOT/n).read_bytes()) for n in ['audit_qasper_measurement.py','run_qasper_contract_replication.py','run_qasper_fingerprint.py','rescore_qasper_demand_transfer.py','repro_lib.py','run_qasper_visible_trace_replication.py']},'cohorts':cohorts}
    path=OUT/'frozen.json';assert not path.exists();path.write_text(json.dumps(f,indent=2,ensure_ascii=False),encoding='utf-8');print({k:len(v) for k,v in cohorts.items()})

def verify(content,task):
    text=content.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    try:obj=json.loads(text)
    except Exception:return {'schema':False,'accepted':False,'reason':'json_parse'}
    if not isinstance(obj,dict) or obj.get('status') not in ('supported','unknown') or not isinstance(obj.get('records'),list):return {'schema':False,'accepted':False,'reason':'schema'}
    if obj['status']=='unknown' or not obj['records']:return {'schema':True,'accepted':False,'reason':'unknown'}
    records=[]
    for item in obj['records']:
        if not isinstance(item,dict) or any(not isinstance(item.get(k),str) for k in ['value','unit','condition','comparator','quote']) or type(item.get('paragraph_id')) is not int:return {'schema':True,'accepted':False,'reason':'record_schema'}
        pid=item['paragraph_id']
        if pid not in task['selected']:return {'schema':True,'accepted':False,'reason':'paragraph_outside_context'}
        body=task['paragraphs'][pid]['text'];quote=item['quote'];offset=body.find(quote)
        if len(quote)<10 or offset<0:return {'schema':True,'accepted':False,'reason':'quote_not_exact'}
        value=item['value']
        parsed_value=scalar(value)
        if parsed_value is None or parsed_value[1]:return {'schema':True,'accepted':False,'reason':'unsupported_scalar'}
        match=re.search(r'(?<![\w.,+\-])'+re.escape(value)+r'(?![\w.,+\-])',quote)
        if not match:return {'schema':True,'accepted':False,'reason':'value_not_complete_token'}
        tail=quote[match.end():].lstrip()
        if (tail.startswith('%') or re.match(r'percent\b',tail)) and item['unit'] not in ('%','percent'):
            return {'schema':True,'accepted':False,'reason':'percentage_unit_dropped'}
        for field in ('unit','condition','comparator'):
            if task['required'][field] and not item[field]:return {'schema':True,'accepted':False,'reason':'missing_'+field}
            if item[field] and item[field] not in quote:return {'schema':True,'accepted':False,'reason':field+'_not_exact'}
        quote_byte_start=len(body[:offset].encode('utf-8'));quote_bytes=quote.encode('utf-8')
        assert body.encode('utf-8')[quote_byte_start:quote_byte_start+len(quote_bytes)]==quote_bytes
        records.append({**item,'char_start':offset,'char_length':len(quote),'utf8_byte_start':quote_byte_start,
            'utf8_byte_length':len(quote_bytes),'paragraph_sha256':sha(body.encode('utf-8'))})
    parts=[]
    for r in records:
        value=(r['value']+' '+r['unit']).strip();qualifiers=[r[k] for k in ('condition','comparator') if r[k]]
        parts.append(value+(' ('+'; '.join(qualifiers)+')' if qualifiers else ''))
    return {'schema':True,'accepted':True,'reason':'source_and_fields_checked_only','records':records,'answer':', '.join(parts)}

def extract(task,material):
    schema='{"status":"supported or unknown","records":[{"value":"verbatim scalar","unit":"verbatim unit or empty","condition":"verbatim condition or empty","comparator":"verbatim comparison target or empty","paragraph_id":0,"quote":"exact contiguous source substring"}]}'
    prompt='Extract only the quantities that answer this question from the supplied evidence. Output only JSON with this schema: '+schema+'\nValues, units, conditions, comparators and quotes must be verbatim substrings, not paraphrases. Use one record per requested quantity. If no readable supporting values exist, status unknown and records empty. Do not infer numbers from table captions. Required fields: '+json.dumps(task['required'])+'\nEvidence:\n'+material+'\nQuestion: '+query_text(task['title'],task['question'])
    t=time.perf_counter()
    try:
        g=ollama_chat(MODEL,'Extract research-paper evidence faithfully. Return only the requested JSON.',prompt,num_ctx=32768,think=False,temperature=0,num_predict=1024)
        result={'content':g.content,'prompt_tokens':g.raw.get('prompt_eval_count'),'generation_tokens':g.raw.get('eval_count'),'wall_sec':time.perf_counter()-t,'load_ns':g.raw.get('load_duration'),'prompt_sha256':sha(prompt.encode())}
    except Exception as exc:result={'content':'','error':repr(exc),'wall_sec':time.perf_counter()-t}
    result['verification']=verify(result['content'],task);return result

def run(label):
    fp=OUT/'frozen.json';f=json.loads(fp.read_text(encoding='utf-8'))
    for path,key in [(SOURCE,'source_sha256'),(PROTOCOL,'protocol_sha256'),(Path(__file__),'script_sha256')]:assert sha(path.read_bytes())==f[key]
    for n,h in f['dependencies'].items():assert sha((ROOT/n).read_bytes())==h
    if label=='evaluation':assert json.loads((OUT/'validation_summary.json').read_text(encoding='utf-8'))['schema_gate_pass']
    src=json.loads(SOURCE.read_text(encoding='utf-8'));path=OUT/(label+'_results.json')
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'frozen_sha256':sha(fp.read_bytes()),'records':[]}
    assert data['frozen_sha256']==sha(fp.read_bytes())
    if 'warmup' not in data:
        data['warmup']=call('Reply exactly: Final Answer: ready',{'answer':'ready','answer_aliases':[]},think=False,cap=16)
        path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for i,task in enumerate(f['cohorts'][label]):
        row=next((r for r in data['records'] if r['id']==task['id']),None)
        if row is None:row={'id':task['id'],'paper_id':task['paper_id']};data['records'].append(row)
        material='\n\n'.join(f"[P{j}] Section: {task['paragraphs'][j]['heading']}\n"+task['paragraphs'][j]['text'] for j in task['selected'])
        q=next(q for q in src[task['paper_id']]['qas'] if q['question_id']==task['id']);gold=references(q)
        for arm in (['base','extract'] if i%2==0 else ['extract','base']):
            if arm in row:continue
            if arm=='extract':row[arm]=extract(task,material)
            else:
                prompt='Answer using only the evidence below. If insufficient, say unknown.\nAnswer value format: Write a concise numeric answer.\nEvidence:\n'+material+'\n\nQuestion: '+query_text(task['title'],task['question'])+'\nOutput exactly one line in this format: Final Answer: <short answer value>'
                result=call(prompt,{'answer':gold[0],'answer_aliases':gold[1:]},think=False,cap=1024);answer,recovered=parsed(result);result['answer']=answer;result['recovered']=recovered;row[arm]=result
            path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');print(label,i+1,task['paper_id'],arm,flush=True)

def analyze(label):
    f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));data=json.loads((OUT/(label+'_results.json')).read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'))
    rows=data['records'];assert len(rows)==len(f['cohorts'][label]) and all('base' in r and 'extract' in r for r in rows)
    results=[];reasons={};total_extract_sec=total_policy_sec=0;tokens={a:0 for a in ('base','extract','policy')};memory=0
    for r in rows:
        q=next(q for q in src[r['paper_id']]['qas'] if q['question_id']==r['id']);gold=references(q);v=r['extract']['verification'];reason=v['reason'];reasons[reason]=reasons.get(reason,0)+1
        pred={'base':r['base']['answer'],'policy':v['answer'] if v['accepted'] else r['base']['answer']}
        scores={a:max(f1('Unanswerable' if normalize(s)=='unknown' else s,g) for g in gold) for a,s in pred.items()}
        results.append({'id':r['id'],'paper_id':r['paper_id'],'question':q['question'],'accepted':v['accepted'],'answers':pred,'scores':scores})
        ex=r['extract'];total_extract_sec+=ex['wall_sec'];total_policy_sec+=ex['wall_sec']+(0 if v['accepted'] else r['base']['wall_sec'])
        for a,k in [('base','base'),('extract','extract')]:tokens[a]+=(r[k].get('prompt_tokens') or 0)+(r[k].get('generation_tokens') or 0)
        if not v['accepted']:tokens['policy']+=(r['base'].get('prompt_tokens') or 0)+(r['base'].get('generation_tokens') or 0)
        if v['accepted']:memory+=len(json.dumps(v,ensure_ascii=False).encode('utf-8'))
    tokens['policy']+=tokens['extract'];n=len(rows)
    out={'questions':n,'papers':len({r['paper_id'] for r in rows}),'schema_valid':sum(r['extract']['verification']['schema'] for r in rows),
        'accepted':sum(r['extract']['verification']['accepted'] for r in rows),'reasons':reasons,
        'schema_gate_pass':sum(r['extract']['verification']['schema'] for r in rows)/n>=.8,
        'f1':{a:sum(r['scores'][a] for r in results)/n for a in ('base','policy')},'total_tokens':tokens,
        'base_wall_sec':sum(r['base']['wall_sec'] for r in rows),'extract_wall_sec':total_extract_sec,'policy_wall_sec':total_policy_sec,
        'warmup_wall_sec':data['warmup']['wall_sec'],'accepted_record_json_bytes':memory,
        'accepted_wins':sum(r['accepted'] and r['scores']['policy']>r['scores']['base']+1e-9 for r in results),
        'accepted_losses':sum(r['accepted'] and r['scores']['policy']<r['scores']['base']-1e-9 for r in results),'results':results}
    groups=[[r for r in results if r['paper_id']==pid] for pid in sorted({r['paper_id'] for r in results})];rng=random.Random(20261002);boot=[]
    for _ in range(10000):
        sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(r['scores']['policy']-r['scores']['base'] for r in sample)/len(sample))
    boot.sort();out['paired']={'difference':out['f1']['policy']-out['f1']['base'],'paper_ci95':[boot[250],boot[9750]]};out['answer_gate_pass']=boot[250]>0
    (OUT/(label+'_summary.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:analyze(sys.argv[2])
