"""Exploratory native same-T order pilot, conditional on the frozen interface gate."""
import json
from pathlib import Path
from run_official_graph_native_probe import ROOT,OPENER,sha,save,call,score
OUT=ROOT/'logs/official_graph_native_order'
PROTOCOL=ROOT/'protocol_official_graph_native_order_20261003.md'
PD='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
def serialize(s):
    s=s[:50000]+'...' if len(s)>50000 else s
    return PD+'\n<first_run_reasoning_traces>\n[Trace 1] '+s.strip()+'\n</first_run_reasoning_traces>'
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    paths=[ROOT/'logs/official_graph_budget_extension/frozen.json',ROOT/'logs/official_graph_budget_extension/results.json',Path(__file__),ROOT/'run_official_graph_native_probe.py',PROTOCOL]
    deps={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
    if not (OUT/'frozen.json').exists():
        fr=json.loads(paths[0].read_text(encoding='utf-8'));rs=json.loads(paths[1].read_text(encoding='utf-8'));qualified=[];cases=[]
        for model in ['qwen3.5:4b','qwen3:1.7b']:
            a=[r for r in rs if r['model']==model and r['cap']==8192]
            if len(a)==2 and all(r['score']['valid'] for r in a):
                qualified.append(model)
                for p in fr['selected']:
                    source=next(r for r in a if r['row_index']==p['row_index']);reasoning=source['raw']['message']['thinking'];assert reasoning
                    cases.append({'model':model,'paper_row':p,'source_request_sha256':source['request_sha256'],'trace':serialize(reasoning),'source_score':source['score']})
        with OPENER.open('http://127.0.0.1:11434/api/tags') as h:inventory=json.load(h)
        save(OUT/'frozen.json',{'dependencies':deps,'qualified_models':qualified,'cases':cases,'model_inventory':inventory,'arm_orders':[['pre','reread','post'],['post','reread','pre']],'producer_cost_already_in_extension':True})
    frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));assert frozen['dependencies']==deps
    assert not (OUT/'results.json').exists()
    records=[]
    for i,c in enumerate(frozen['cases']):
        p=c['paper_row'];x=p['prompt'];head,op=x.rsplit('Operation:',1);T=c['trace']
        prompts={'reread':x,'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op}
        assert prompts['post'].endswith('Operation:'+op) and prompts['pre'].endswith(x)
        for arm in frozen['arm_orders'][i%2]:
            r={'model':c['model'],'row_index':p['row_index'],'arm':arm,**call(c['model'],prompts[arm],8192)};r['score']=score(r['raw'],p['answer_nodes'],r['error']);records.append(r);save(OUT/'results.json',records)
            print(c['model'],p['row_index'],arm,r['raw'].get('done_reason'),r['score'],flush=True)
    print('Complete',len(records),'native same-T exploratory calls',flush=True)
if __name__=='__main__':main()
