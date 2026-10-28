"""Independent audit of the adaptive extension, preserving native probe results."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_budget_extension'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'));ps={x['row_index']:x for x in fr['selected']}
    assert len(rs)==8 and {(r['model'],r['row_index'],r['cap']) for r in rs}=={(m,i,c) for m in ['qwen3.5:4b','qwen3:1.7b'] for i in ps for c in [4096,8192]}
    assert fr['protocol_sha256']==sha(ROOT/'protocol_official_graph_budget_extension_20261003.md') and all(fr['dependencies'][n]==sha(ROOT/n) for n in fr['dependencies'])
    assert fr['data_sha256']==sha(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet')
    for r in rs:
        q=r['request'];assert q['messages'][1]['content']==ps[r['row_index']]['prompt'] and q['think'] is True and q['options']['num_predict']==r['cap'] and q['options']['num_ctx']==32768 and q['options']['temperature']==0 and r['attempts']==1
        assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
        raw=r['raw'];c=(raw.get('message') or {}).get('content','') or '';m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',c.rstrip().split('\n')[-1].strip());valid=bool(m) and not r['error'] and raw.get('done_reason')=='stop'
        pred=set(x.strip() for x in m[1].split(',') if x.strip()) if m else set();gold=set(ps[r['row_index']]['answer_nodes']);f=(2*len(pred&gold)/(len(pred)+len(gold)) if pred or gold else 1.) if valid else 0.
        assert r['score']['valid']==valid and r['score']['em']==float(valid and pred==gold) and abs(r['score']['f1']-f)<1e-12
    summary=[]
    for model in ['qwen3.5:4b','qwen3:1.7b']:
        for cap in [4096,8192]:
            a=[r for r in rs if r['model']==model and r['cap']==cap];summary.append({'model':model,'cap':cap,'valid':sum(r['score']['valid'] for r in a),'n':2,'em':sum(r['score']['em'] for r in a)/2,'f1':sum(r['score']['f1'] for r in a)/2,'length':sum(r['raw'].get('done_reason')=='length' for r in a),'errors':sum(bool(r['error']) for r in a),'known_tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a),'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in a)})
    save(OUT/'summary.json',summary);save(OUT/'independent_audit.json',{'passed':True,'persisted_calls':8,'attempts':8,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'summary.json']}})
    report=ROOT/'paper/OFFICIAL_GRAPH_NATIVE_INTERFACE_20261003.md'
    text=report.read_text(encoding='utf-8')+'\n## 自适应新题预算扩展\n\n两道未运行官方题，各模型4096/8192独立请求，共8次；冻结协议另存，不替换前16格。小样本且可能共享图分布，不称独立确认。\n\n| 模型 | cap | 有效答案 | EM | setF1 | length | 已知token |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summary:text+=f"| {s['model']} | {s['cap']} | {s['valid']}/2 | {s['em']*100:.2f}% | {s['f1']*100:.2f}% | {s['length']} | {s['known_tokens']} |\n"
    text+='\n累计24落盘响应，另1次中断请求送达/完成未知。已知使用量为整个阶段下界，不能声称完整端到端成本比较。预算提升改变输出资源，不能解释成状态方法有效。门槛8192两题均可评分，仅是后续顺序试点接口门槛。\n';report.write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2));print('Audit passed,24 saved responses plus1 uncertain attempt.')
if __name__=='__main__':main()
