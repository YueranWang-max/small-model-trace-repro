"""Independent prompt/trace/scorer audit and paired analysis of the new-graph run."""
import csv,hashlib,itertools,json,re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_order_replication'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def strict(r,gold):
    c=(r['raw'].get('message') or {}).get('content','') or ''
    m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',c.rstrip().split('\n')[-1].strip());valid=bool(m) and r['error'] is None and r['raw'].get('done_reason')=='stop'
    a=set(x.strip() for x in m[1].split(',') if x.strip()) if m else set();b=set(gold)
    f=(2*len(a&b)/(len(a)+len(b)) if a or b else 1.) if valid else 0.
    assert r['score']['valid']==valid and r['score']['em']==float(valid and a==b) and abs(r['score']['f1']-f)<1e-12
def trace(s):
    p='Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.'
    if len(s)>50000:s=s[:50000]+'...'
    return p+'\n<first_run_reasoning_traces>\n[Trace 1] '+s.strip()+'\n</first_run_reasoning_traces>'
def audit_order(name,replication=False):
    folder=ROOT/'logs'/name;fr=json.loads((folder/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((folder/'results.json').read_text(encoding='utf-8'));assert all(v==sha(ROOT/n) for n,v in fr['dependencies'].items())
    if replication:
        cases=fr['selected'];assert len(rs)==32;index={(r['row_index'],r['arm']):r for r in rs};assert len(index)==32
        assert len({p['graph_head_sha256'] for p in cases})==8
    else:
        cases=[c['paper_row'] for c in fr['cases']];assert len(rs)==6;index={(r['row_index'],r['arm']):r for r in rs};assert len(index)==6
        ext=json.loads((ROOT/'logs/official_graph_budget_extension/results.json').read_text(encoding='utf-8'))
    for p in cases:
        x=p['prompt'];head,op=x.rsplit('Operation:',1)
        source=index[(p['row_index'],'source')] if replication else next(r for r in ext if r['model']=='qwen3:1.7b' and r['row_index']==p['row_index'] and r['cap']==8192)
        strict(source,p['answer_nodes']);T=trace((source['raw'].get('message') or {}).get('thinking','') or '')
        for arm in ['pre','post','reread']:
            r=index[(p['row_index'],arm)];expected={'pre':T+'\n'+x,'post':head+T+'\nOperation:'+op,'reread':x}[arm]
            assert r['request']['messages'][1]['content']==expected
            assert r['request']['messages'][0]==source['request']['messages'][0] and r['request']['options']==source['request']['options'] and r['request']['think'] is True
            assert r['request_sha256']==hashlib.sha256(json.dumps(r['request'],sort_keys=True).encode()).hexdigest() and r['attempts']==1
            strict(r,p['answer_nodes'])
            if replication:assert r['trace_sha256']==hashlib.sha256(T.encode()).hexdigest() and r['source_request_sha256']==source['request_sha256']
    if replication:
        data=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist();used=set()
        for prior in ['official_graph_native_probe','official_graph_budget_extension']:
            for p in json.loads((ROOT/'logs'/prior/'frozen.json').read_text(encoding='utf-8'))['selected']:used.add(hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest())
        reconstructed=[]
        for typ in ['parents','bfs']:
            count=0
            for i,p in enumerate(data):
                h=hashlib.sha256(p['prompt'].rsplit('Operation:',1)[0].encode()).hexdigest()
                if p['problem_type']!=typ or not 2000<=p['prompt_chars']<3500 or h in used:continue
                reconstructed.append(i);used.add(h);count+=1
                if count==4:break
        expected=[reconstructed[j+offset] for j in range(4) for offset in [0,4]]
        assert [p['row_index'] for p in cases]==expected
        for i,p in enumerate(cases):
            assert p['prompt']==data[p['row_index']]['prompt'] and p['answer_nodes']==data[p['row_index']]['answer_nodes']
            actual=[r['arm'] for r in rs if r['row_index']==p['row_index']];assert actual==['source']+fr['orders'][i]
    save(folder/'independent_audit.json',{'passed':True,'saved_calls':len(rs),'same_T_x_q_verified':True,'reasoning_only_T_no_gold_correction':True,'strict_visible_rescore':True,'fresh_graph_selection_reconstructed':replication,'assets':{str(p.relative_to(ROOT)):sha(p) for p in [folder/'frozen.json',folder/'results.json']}})
    return fr,rs,index
def main():
    pf,pr,pi=audit_order('official_graph_native_order')
    fr,rs,index=audit_order('official_graph_order_replication',True);ids=[p['row_index'] for p in fr['selected']]
    rng=np.random.default_rng(202610035);draw=rng.integers(0,8,size=(20000,8));vectors={a:np.array([index[(i,a)]['score']['f1'] for i in ids]) for a in ['source','reread','pre','post']};summary=[]
    sources=[index[(i,'source')] for i in ids]
    def tokens(a):return sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a)
    for arm,v in vectors.items():
        a=[index[(i,arm)] for i in ids];ci=np.quantile(v[draw].mean(axis=1),[.025,.975]);summary.append({'arm':arm,'n':8,'f1':float(v.mean()),'f1_lo':float(ci[0]),'f1_hi':float(ci[1]),'em':sum(r['score']['em'] for r in a)/8,'valid':sum(r['score']['valid'] for r in a),'length':sum(r['raw'].get('done_reason')=='length' for r in a),'errors':sum(bool(r['error']) for r in a),'reader_tokens':tokens(a),'total_path_tokens':tokens(a)+(tokens(sources) if arm!='source' else 0),'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in (a+sources if arm!='source' else a))})
    comps=[];signs=np.array(list(itertools.product([-1,1],repeat=8)))
    for other in ['post','reread']:
        diff=vectors['pre']-vectors[other];ci=np.quantile(diff[draw].mean(axis=1),[.025,.975]);stat=abs(diff.mean());p=float(np.mean(np.abs((signs*diff).mean(axis=1))>=stat-1e-12))
        comps.append({'contrast':'pre-'+other,'delta':float(diff.mean()),'lo':float(ci[0]),'hi':float(ci[1]),'exact_two_sided_p':p})
    order=sorted(range(2),key=lambda i:comps[i]['exact_two_sided_p']);running=0.
    for rank,i in enumerate(order):running=max(running,(2-rank)*comps[i]['exact_two_sided_p']);comps[i]['holm_p']=min(1.,running)
    by={s['arm']:s for s in summary};gate=all(c['lo']>0 and c['holm_p']<.05 for c in comps) and by['pre']['em']>=max(by['post']['em'],by['reread']['em']) and all(r['score']['valid'] for r in rs)
    result={'summary':summary,'paired_contrasts':comps,'advance_gate':gate,'all32_valid':all(r['score']['valid'] for r in rs),'source_rows':ids,'pilot_f1':{a:sum(r['score']['f1'] for r in pr if r['arm']==a)/2 for a in ['pre','post','reread']}}
    save(OUT/'summary.json',result)
    with (ROOT/'paper/figures/data/official_graph_native_order.csv').open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (ROOT/'paper/figures/data/official_graph_native_order_pairs.csv').open('w',encoding='utf-8',newline='') as h:
        w=csv.writer(h);w.writerow(['row_index','type','source','reread','pre','post'])
        for p in fr['selected']:w.writerow([p['row_index'],p['problem_type']]+[index[(p['row_index'],a)]['score']['f1'] for a in ['source','reread','pre','post']])
    lines=['# 原生轨迹顺序：官方短图重复（2026-10-03）','','**在八个新官方短图上检验同一份自产推理前置是否优于后置和无反馈重读；结论以预设门槛为准。**','','## 从接口到顺序','','先16格预算接口、8格自适应预算扩展、6格同题顺序试点，再32格新图重复，62个落盘响应，另1个中断请求送达/完成未知。没有把预算失败隐藏，也没有从thinking抽取可见答案。两题试点setF1 pre94.44/post87.50/reread37.50%，前后EM相同；新图结果不得被试点替代。','','官方修正数据750行独立图标签一致。新8图按类型/字符档/图头SHA选取，排除之前图，未按答案选择。4Parents/4BFS，约2.6K字符；图头不同不代表图分布独立同分布或匿名拓扑完全不重合。qwen3:1.7b（本地2.0B、Q4_K_M）；每图原生source一次、三读出一次，think=true/8192输出cap。任何source错误、未结束、空reasoning均保留，不筛掉问题。','','T仅由该图实际source reasoning构成（推理内猜测答案可存在），可见content不入T、无gold修复。pre/post原图x和末尾Operation/q逐字相同。与原研究相比，1trace/1repeat、短图、客户端预算、温度与量化运行时不同，末尾没有额外重复答案指令；只迁移原生重放和顺序机制，不称完整256K复现。','','## 八图结果','','| 臂 | setF1 | EM | 有效答案 | length | source+reader token |','|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['arm']} | {100*s['f1']:.2f}% | {100*s['em']:.2f}% | {s['valid']}/8 | {s['length']} | {s['total_path_tokens']} |")
    lines+=['','source行只有首轮成本；其余部署路径已逐题加回生产者，不能以reader独算。缓存/后台工作/输出长度不同限制墙钟因果解释；全阶段曾中断，成本不完整。','','| 配对setF1比较 | 差值pp | 95%点态图cluster区间pp | 精确双侧p | Holm p |','|---|---:|---:|---:|---:|']
    for c in comps:lines.append(f"| {c['contrast']} | {100*c['delta']:+.2f} | [{100*c['lo']:+.2f}, {100*c['hi']:+.2f}] | {c['exact_two_sided_p']:.4f} | {c['holm_p']:.4f} |")
    lines+=['','20,000 paired graph bootstrap，seed202610035；精确符号翻转2^8，两个主要比较Holm校正。n8只支持有限不确定性描述，不称跨模型/长文通用。','','预设推进门槛：'+('通过' if gate else '**未通过**')+'。需要两F1对比区间下界>0且Holm p<.05、pre EM不低于另外两臂、所有source/reader正常合法；不得只挑有利指标。','','## 证据范围','','来源：[原论文](https://arxiv.org/html/2609.02702)、[官方GraphWalks](https://huggingface.co/datasets/openai/graphwalks)。首要目标仍是小模型迁移和边界，当前不是通用原文压缩方法或创新证明。主论文tex/PDF尚未重写，完整实验/引用/论文审查尚未完成。']
    (ROOT/'paper/OFFICIAL_GRAPH_NATIVE_ORDER_20261003.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
