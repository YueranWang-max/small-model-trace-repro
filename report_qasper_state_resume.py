"""Publish audited exploratory evidence without updating primary paper claims."""
import csv,json
from run_qasper_state_resume import ROOT,OUT,ARMS,save,sha

def main():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'));audit=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'));assert audit['audit_pass']
    diag=json.loads((OUT/'diagnostics.json').read_text(encoding='utf-8'));lines=['# 真实论文：未完成要素状态与局部原文接续（2026-10-03）','','## 一句话结论','','本轮检验“把已找到的内容和待查要素写成状态，再补读完整原文段落”是否能把合成操作任务的收益迁移到真实论文；以冻结门槛和全部生产者成本判断结果。','','## 设计与范围','','6篇Qasper训练集论文、每篇2题，两种本机模型，168正式调用＋2预热。论文来自既有结构指纹分区中至少2题论文的第25–30项；这是本策略新读出，不能称项目从未接触的数据。所有问题、源块及协议先冻结，答案标注只在全部调用结束后评分。每模型/题独立生成一份状态，六种读出共享同题。','','|方案|输入和成本|','|---|---|','|full|全部摘要及正文段落，一次调用|','|base|普通BM25前8段，一次调用|','|wide|普通BM25前12段，一次调用|','|reread|与wide完全相同的另一次调用；只计其本次读出|','|state_wide|实际生成的已知/缺项/草稿状态＋前12段；生产者加读出|','|state_adaptive|同一状态＋前8段及缺项检索最多4个新段；生产者加读出|','','state_wide对reread比较在相同原文下加入状态；state_adaptive对state_wide比较动态路由。段数上限相同但token并不严格匹配；本轮不是原始Trace as State基准或纯粹前后顺序实验。生产者固定先运行，其余方案随机排列。reread控制相同请求的重复波动，未控制相等的总生成预算。','','## 结果','','主指标为多参考答案Counter token-F1；仅精确规范化unknown转Unanswerable；公开标注不改写，失败计0。置信区间是按6篇论文整群抽样的探索性区间，无统计确证或多重检验校正。']
    csvrows=[]
    for model,v in s['models'].items():
        lines+=['',f'### {model}','','|方案|F1 %|文本EM %|总token|调用墙钟秒|失败|证据全覆盖/可映射题|','|---|---:|---:|---:|---:|---:|---:|']
        for arm,x in v['arms'].items():
            lines.append(f"|{arm}|{100*x['f1']:.2f}|{100*x['em']:.2f}|{x['total_tokens']}|{x['wall_sec']:.3f}|{x['errors']}|{x['coverage_all_hit']}/{x['coverage_eligible']}|")
            csvrows.append({'model':model,'arm':arm,'f1':x['f1'],'total_tokens':x['total_tokens'],'wall_sec':x['wall_sec']})
        lines+=['','|配对差值|F1百分点|论文簇95%区间|','|---|---:|---|']
        for contrast,c in v['paired'].items():lines.append(f"|{contrast}|{100*c['difference']:+.2f}|[{100*c['paper_ci95'][0]:+.2f}, {100*c['paper_ci95'][1]:+.2f}]|")
        lines+=['',f"冻结推进门槛：{'通过，仅可进入独立重复' if v['advance'] else '未通过；不宣布自然文档迁移成功'}。预热{v['warmup_sec']:.3f}秒；本模型动态路由准备{v['route_sec']:.3f}秒，尚需加到state_adaptive的调用墙钟。"]
    lines+=['','## 失败诊断与解释边界','','生产者known只是模型声称的已知事实，未经过机械事实核验；提示要求核验，不等于实际核验。缺项检索允许查询漂移，草稿可能形成错误锚定。若状态收益不足，不能由合成表格任务成功推断自然文本成功。JSON预算耗尽等输出错误保留原始文本，计零分；未追加质量重试。所有局部源块完整保留，证据全覆盖只表示已标注段落覆盖，不证明语义足够。正文未包括浮动表格和图像，相关问题可能无法回答。','','程序审计验证冻结文件、源哈希/段落偏移、每份真实生成状态、动态路由重算、请求哈希、相同wide/reread请求，以及正式HTTP调用数。该审计验证流程一致性，不验证模型事实正确性。',f"正式{audit['formal_calls']}调用/{audit['http_attempts']}HTTP尝试，记录失败{audit['call_errors']}；绑定检查{audit['state_bindings']}，完整源块检查{audit['source_blocks_verified']}。",'',f"一次性正文准备{ s['prepare_sec']:.3f}秒，预热单列；不能宣称冷启动/缓存复用端到端加速。所有state成本各自加同一生产者一次，评估共享不等于部署免费。",'','## 下一步收敛','','若未通过门槛，停止在这12题上优化状态措辞；先使用封闭受控的学术事实任务区分缺证据、取错事实、条件绑定与答案输出错误，再用新题检验。若通过，也只进入事先冻结的新论文重复。主论文必须保留真实文档边界；不把toy增益或本试点小样本写成通用压缩结论。','','## 记录','','协议protocol_qasper_state_resume_20261003.md；日志logs/qasper_state_resume；R图paper/figures/qasper_state_resume.pdf和.png；范围审稿paper/QASPER_STATE_RESUME_REVIEW_20261003.md。']
    path=ROOT/'paper/QASPER_STATE_RESUME_20261003.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cp=ROOT/'paper/figures/data/qasper_state_resume.csv'
    with cp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print(path)

if __name__=='__main__':main()
