"""Post-hoc exact bare-code diagnostic; never modifies frozen strict scores."""
import json,re,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/controlled_state_boundary';ARMS=('no_state','correct_pre','correct_post','wrong_pre')

def recover(v):
    if v['answer']:return v['answer'],'strict_label'
    text=v.get('content','').strip();m=re.fullmatch(r'[VS]\d+',text,re.I)
    return (m.group().upper(),'bare_code') if m else ('','unparsed')

def main():
    d=json.loads((OUT/'reader.json').read_text(encoding='utf-8'));f=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert len(d['rows'])==32 and all(set(r['arms'])==set(ARMS) for r in d['rows']);rows=[]
    for r in d['rows']:
        t=next(t for t in f['tasks'] if t['id']==r['id']);rec={**{k:r[k] for k in ('model','id','base_id','kind','noise_rows')},'arms':{}}
        for a,v in r['arms'].items():
            value,reason=recover(v);rec['arms'][a]={'value':value,'recovery':reason,'em':int(value==t['answer']),'follows_wrong_state':int(value==t['wrong_state_answer'])}
        rows.append(rec)
    out={'posthoc_format_only_diagnostic':True,'strict_primary_unchanged':True,'bare_code_rule':'entire stripped output fullmatch [VS]digits; no choosing among multiple codes','models':{},'rows':rows}
    for model in sorted({r['model'] for r in rows}):
        rs=[r for r in rows if r['model']==model];groups=[[r for r in rs if r['base_id']==b] for b in sorted({r['base_id'] for r in rs})];m={'arms':{},'strata':{},'paired':{}}
        for a in ARMS:m['arms'][a]={'em':sum(r['arms'][a]['em'] for r in rs)/16,'bare_recoveries':sum(r['arms'][a]['recovery']=='bare_code' for r in rs),'unparsed':sum(r['arms'][a]['recovery']=='unparsed' for r in rs)}
        for kind in ('binding','update'):
            for n in (16,128):
                sample=[r for r in rs if r['kind']==kind and r['noise_rows']==n];m['strata'][f'{kind}_{n}']={a:sum(r['arms'][a]['em'] for r in sample)/len(sample) for a in ARMS}
        for b in ('no_state','correct_post'):
            rng=random.Random(20261003);bs=[]
            for _ in range(10000):
                sample=[r for _ in groups for r in rng.choice(groups)];bs.append(sum(r['arms']['correct_pre']['em']-r['arms'][b]['em'] for r in sample)/16)
            bs.sort();m['paired']['pre_minus_'+b]={'diff':m['arms']['correct_pre']['em']-m['arms'][b]['em'],'ci95':[bs[250],bs[9750]]}
        m['wrong_state_follow_rate']=sum(r['arms']['wrong_pre']['follows_wrong_state'] for r in rs)/16;out['models'][model]=m
    (OUT/'format_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
