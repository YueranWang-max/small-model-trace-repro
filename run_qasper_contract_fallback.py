"""Source-only lexical contract deficits cause original paragraph fallback."""
import json,re,sys,time,math
import run_qasper_relation_closure as source_stage
import run_qasper_fact_blocks as statistics
from run_qasper_fact_blocks import ROOT,SOURCE,sha,reader,family,STOP,PRED
OUT=ROOT/'logs/qasper_contract_fallback'; ARMS=statistics.ARMS
NUMBER=r'\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b'
UNIT=r'\b(?:words?|tokens?|samples?|examples?|instances?|hours?|layers?|dimensions?|groups?|octaves?|datasets?|percent\w*|parameters?|tasks?|languages?|tweets?|documents?|epochs?)\b|%'
SCOPE=r'\b(?:not|no|never|without|only|all|any|both|except|other than|can|consider\w*|us\w*|report\w*|evaluat\w*|test\w*)\b'
STRUCT=r'\b(?:consists? of|composed of|component\w*|sub-network\w*|subnetwork\w*|concatenat\w*|incorporat\w*|followed by)\b'

def candidates(p,query):
    fam=family(query);terms=set(re.findall(r'\b[a-z][a-z0-9-]+\b',query.lower()))-STOP;out=[]
    for u in p['units']:
        text=u['text'];overlap=bool(terms&set(re.findall(r'\b[a-z][a-z0-9-]+\b',text.lower())))
        if fam=='quantity':ok=bool(re.search(NUMBER,text,re.I) and re.search(UNIT,text,re.I))
        elif fam=='boolean':ok=overlap and bool(re.search(SCOPE,text,re.I))
        elif fam=='method':ok=bool((overlap and re.search(PRED['method'],text,re.I)) or re.search(STRUCT,text,re.I))
        else:ok=bool(re.search(PRED[fam],text,re.I))
        if ok:out.append({'pid':u['pid'],'sid':u['sid'],'family':fam,'original':text})
    return out

def prepare():
    source_stage.OUT=OUT;source_stage.prepare();start=time.perf_counter();fp=OUT/'frozen.json';sp=OUT/'selected.json';f=json.loads(fp.read_text(encoding='utf-8'));s=json.loads(sp.read_text(encoding='utf-8'));weights=json.loads((ROOT/'logs/qasper_attention_guard/frozen.json').read_text(encoding='utf-8'))['weights']
    for t in s['tasks']:
        p=next(p for p in f['papers'] if p['paper_id']==t['paper_id']);base={(u['pid'],u['sid']) for u in t['arms']['elements']['pointers']};cs=candidates(p,t['question']);missing=[c for c in cs if (c['pid'],c['sid']) not in base];fallback={c['pid'] for c in missing};chosen=base|{(u['pid'],u['sid']) for u in p['units'] if u['pid'] in fallback};wide=set()
        for i,para in enumerate(p['paragraphs']):
            us=[u for u in p['units'] if u['pid']==i];n=sum(pi==i for pi,j in chosen);rank=sorted(us,key=lambda u:(-sum(weights[x] for x in u['features']),u['sid']));wide.update((i,u['sid']) for u in rank[:n])
        t.pop('relation_chains',None);t['contract_candidates']=cs;t['missing_before_fallback']=missing;t['fallback_paragraph_ids']=sorted(fallback);t['arms']['block']=reader.render(p,chosen);t['arms']['wide']=reader.render(p,wide)
    calibration=json.loads((ROOT/'logs/qasper_attention_reader/frozen.json').read_text(encoding='utf-8'))['calibration'];f['contract_fitting_coverage']=[{'question_id':c['question_id'],'target_string_covered':any(c['target'] in r['original'] for r in candidates({'units':[{'pid':i,'sid':0,'text':u['text']} for i,u in enumerate(c['source_units'])]},c['question']))} for c in calibration];f['contract_expressions']={'number':NUMBER,'unit':UNIT,'scope':SCOPE,'structure':STRUCT};f['hashes'].update({str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'run_qasper_contract_fallback.py',ROOT/'protocol_qasper_contract_fallback_20261003.md']});fp.write_text(json.dumps(f,indent=2),encoding='utf-8');s['frozen_sha256']=sha(fp.read_bytes());s['contract_preparation_wall_sec']=time.perf_counter()-start;s['preparation_wall_sec']+=s['contract_preparation_wall_sec'];sp.write_text(json.dumps(s,indent=2),encoding='utf-8');print('fallbacks',sum(len(t['fallback_paragraph_ids']) for t in s['tasks']),flush=True)

def amend():
    assert not (OUT/'reader.json').exists();fp=OUT/'frozen.json';sp=OUT/'selected.json';f=json.loads(fp.read_text(encoding='utf-8'));s=json.loads(sp.read_text(encoding='utf-8'));weights=json.loads((ROOT/'logs/qasper_attention_guard/frozen.json').read_text(encoding='utf-8'))['weights']
    for t in s['tasks']:
        p=next(p for p in f['papers'] if p['paper_id']==t['paper_id']);keys={(u['pid'],u['sid']) for u in t['arms']['elements']['pointers']};target=t['arms']['block']['sentences_before_closure'];rank=sorted([u for u in p['units'] if (u['pid'],u['sid']) not in keys],key=lambda u:(-sum(weights[x] for x in u['features']),u['pid'],u['sid']));keys.update((u['pid'],u['sid']) for u in rank[:target-len(keys)]);t['arms']['wide']=reader.render(p,keys)
    f['pre_reader_amendment']='global sentence count control; no reader/gold yet'
    for p in [ROOT/'run_qasper_contract_fallback.py',ROOT/'protocol_qasper_contract_fallback_20261003.md']:f['hashes'][str(p.relative_to(ROOT))]=sha(p.read_bytes())
    fp.write_text(json.dumps(f,indent=2),encoding='utf-8');s['frozen_sha256']=sha(fp.read_bytes());s['control']='global_preclosure_sentence_count';sp.write_text(json.dumps(s,indent=2),encoding='utf-8');print('amended before reader',flush=True)

def run():reader.OUT=OUT;reader.ARMS=ARMS;reader.run()
def analyze():
    statistics.OUT=OUT;statistics.analyze();p=OUT/'summary.json';d=json.loads(p.read_text(encoding='utf-8'));d['development_reuse']=False;d['previously_scanned_partition']=True;p.write_text(json.dumps(d,indent=2),encoding='utf-8')
if __name__=='__main__':{'prepare':prepare,'amend':amend,'run':run,'analyze':analyze}[sys.argv[1]]()
