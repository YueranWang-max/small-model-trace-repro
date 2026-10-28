"""Independent raw-response audit and descriptive report; execute after model calls."""
import csv,hashlib,json,re
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'logs/official_graph_native_probe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    fr=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'));rs=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    assert len(rs)==16 and len({(r['model'],r['row_index'],r['cap']) for r in rs})==16
    assert fr['runner_sha256']==sha(ROOT/'run_official_graph_native_probe.py') and fr['protocol_sha256']==sha(ROOT/'protocol_official_graph_native_probe_20261003.md')
    assert fr['data_sha256']==sha(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet')=='54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516'
    data=pq.read_table(ROOT/'data/graphwalks_official/graphwalks_128k_and_shorter.parquet').to_pylist();selected={r['row_index']:r for r in fr['selected']};literal_diff=[]
    for r in rs:
        p=selected[r['row_index']];assert p['prompt']==data[r['row_index']]['prompt'] and p['answer_nodes']==data[r['row_index']]['answer_nodes']
        q=r['request'];assert q['messages'][1]['content']==p['prompt'] and q['think'] is True and q['options']['num_predict']==r['cap'] and q['options']['num_ctx']==32768 and q['options']['temperature']==0 and r['attempts']==1
        assert r['request_sha256']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
        raw=r['raw'];text=(raw.get('message') or {}).get('content','') or '';m=re.fullmatch(r'Final Answer:\s*\[([^\[\]\n]*)\]',text.rstrip().split('\n')[-1].strip())
        valid=bool(m) and r['error'] is None and raw.get('done_reason')=='stop';a=set(x.strip() for x in m[1].split(',') if x.strip()) if m else set();b=set(p['answer_nodes'])
        f=(2*len(a&b)/(len(a)+len(b)) if a or b else 1.) if valid else 0.
        assert r['score']['valid']==valid and r['score']['em']==int(valid and a==b) and abs(r['score']['f1']-f)<1e-12
        # Literal dataset README checks the actual final line, including trailing newlines.
        last=text.split('\n')[-1];lm=re.search(r'\[.*\]',last) if 'Final Answer:' in last else None
        if bool(lm)!=bool(m):literal_diff.append([r['model'],r['row_index'],r['cap']])
    recovery=next(r for r in rs if r['model']=='qwen3.5:4b' and r['row_index']==400 and r['cap']==4096)
    assert not recovery['score']['valid'] and recovery['score']['em']==0 and recovery['score']['f1']==0
    summary=[]
    for model in ['qwen3.5:4b','qwen3:1.7b']:
        for cap in [1024,4096]:
            a=[r for r in rs if r['model']==model and r['cap']==cap];s={'model':model,'cap':cap,'n':4,'valid':sum(r['score']['valid'] for r in a),'em':sum(r['score']['em'] for r in a)/4,'f1':sum(r['score']['f1'] for r in a)/4,'length':sum(r['raw'].get('done_reason')=='length' for r in a),'errors':sum(bool(r['error']) for r in a),'thinking_nonempty':sum(bool((r['raw'].get('message') or {}).get('thinking')) for r in a),'known_tokens':sum(r['raw'].get('prompt_eval_count',0)+r['raw'].get('eval_count',0) for r in a),'unknown_usage':sum(not all(k in r['raw'] for k in ['prompt_eval_count','eval_count']) for r in a),'wall_sec':sum(r['wall_sec'] for r in a)};s['interface_gate']=s['valid']>=3;summary.append(s)
    save(OUT/'summary.json',summary)
    audit={'passed':True,'persisted_calls':16,'formal_unique_cells':16,'interruption':json.loads((OUT/'interruption.json').read_text(encoding='utf-8')),'total_attempts_known_or_possible':17,'total_cost_complete':False,'literal_readme_parser_differences':literal_diff,'full_dataset_oracle_audit':fr['oracle_audit'],'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen.json',OUT/'results.json',OUT/'interruption.json',OUT/'summary.json']}}
    save(OUT/'independent_audit.json',audit)
    dst=ROOT/'paper/figures/data/official_graph_native_probe.csv'
    with dst.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# 官方图任务原生推理接口与输出预算边界（2026-10-03）','','**本阶段检查小模型是否能结束原生推理并给出答案；尚未检验状态前置能否提高推理准确性。**','','## 数据与设计','','官方 openai/graphwalks 修正版，完整保留 released prompt，SHA256 与官方LFS一致。全部750行以独立图算法核对标签一致。按类型与字符档首行选择4题；约2.6K、7.6K字符，不是token档，不是原论文256K复现。两模型think=true，各题独立1024/4096输出cap，共16落盘响应。预算包含隐藏推理。','','| 模型 | 输出cap | 正常可见答案 | EM | setF1 | 长度终止 | API错误 | 已知token |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['model']} | {s['cap']} | {s['valid']}/4 | {100*s['em']:.2f}% | {100*s['f1']:.2f}% | {s['length']}/4 | {s['errors']} | {s['known_tokens']} |")
    lines+=['','## 必须保留的资格限制','','一次执行进程在第三个已保存响应之后意外退出。第四个请求是否送达/完成未知；恢复后重发该单元，恢复响应单独保存，不把未知原响应解释为成功。矩阵16格加1次可能的中断尝试，全部成本未知，已知token仅下界，不能给总成本或速度结论。独立审计验证完整原prompt、请求选项、SHA和可见答案重新评分。','','原README最后一行解析与本试点最后非空行严格格式解析的差异格数：'+str(len(literal_diff))+'。隐藏thinking不作为可见答案；缺答案、length、HTTP失败均零分，合法空集可得满分。与原论文无客户端输出cap不同，预算失败不是原论文理论的反例。','','## 发现与下一步','','每个预算档至少3/4可评分答案才通过接口门槛。实际结果见 summary.json 的interface_gate；未通过的档不能直接开展该模型/档位的状态顺序确认。下一阶段应独立新题探测输出预算或先选可用接口，不在这4题上调提示。门槛仅说明可评分，不说明答对率或迁移成功。','','旧repro_lib.extract_answer会回退到thinking，extract_list会寻找非终端中间答案，run_pilot汇总还会排除整题HTTP失败。这些是历史探索评分口径，不能描述为忠实原论文复现。历史pilot日志缺完整可见答案和终止信息，不能恢复严格分数；保持旧分数并限制其证据用途。','','来源：[原论文附录B/C](https://arxiv.org/html/2609.02702)、[官方数据与修正记录](https://huggingface.co/datasets/openai/graphwalks)。本阶段不是完整论文审稿；主tex/PDF未更新。']
    (ROOT/'paper/OFFICIAL_GRAPH_NATIVE_INTERFACE_20261003.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (ROOT/'paper/OFFICIAL_GRAPH_NATIVE_INTERFACE_20261003.md').open('a',encoding='utf-8') as h:
        h.write('\n中断格主分析按缺失输出记0；恢复的400/4096响应也为length、0分，因此上表质量数值与缺失记0完全一致。恢复原始响应作为补充诊断保留，不替代未知原响应；其已知token计入恢复支出，整个阶段使用量仍为下界。\n')
    print(json.dumps(summary,indent=2));print('Audit passed; interruption and unknown cost retained.')
if __name__=='__main__':main()
