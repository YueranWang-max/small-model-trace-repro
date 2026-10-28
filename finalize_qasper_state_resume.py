"""Bounded diagnostic synthesis; frozen pilot outcomes remain unchanged."""
import csv,json,random
from run_qasper_state_resume import ROOT,OUT,MODELS,save,sha

def main():
    p=json.loads((OUT/'scored_results.json').read_text(encoding='utf-8'));s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'));aout=ROOT/'logs/qasper_state_resume_ablation';ar=json.loads((aout/'results.json').read_text(encoding='utf-8'));asum=json.loads((aout/'summary.json').read_text(encoding='utf-8'));oracle=json.loads((ROOT/'logs/qasper_state_resume_oracle/summary.json').read_text(encoding='utf-8'))
    path=ROOT/'paper/QASPER_STATE_RESUME_20261003.md';text=path.read_text(encoding='utf-8');extra=['','## 完成后的独立源选择诊断与自适应接口消融','','另有24次特权证据定位诊断、48次相同问题接口消融，总计240正式调用＋2预热。它们不属于原168次冻结主检验，也不改变原推进门槛。消融是在看到主结果后提出，不能当独立重复；证据诊断使用标注位置，不能部署。','','|模型|仅标注段落F1 %|无映射回退全文题数|只留缺项状态F1 %|完整状态＋答案绑定F1 %|','|---|---:|---:|---:|---:|'];diagnostic={};csvrows=[]
    for model in MODELS:
        rows=p['models'][model]['records'];new=[r for r in ar['records'] if r['model']==model];ids={r['id']:r for r in new};prod_tokens=sum((r['producer'].get('raw',{}).get('prompt_eval_count') or 0)+(r['producer'].get('raw',{}).get('eval_count') or 0) for r in rows);prod_sec=sum(r['producer']['wall_sec'] for r in rows)
        v=asum[model];extra.append(f"|{model}|{100*oracle[model]['f1']:.2f}|{oracle[model]['fallback_questions']}|{100*v['queries_only']['f1']:.2f}|{100*v['state_extract']['f1']:.2f}|")
        diagnostic[model]={'arms':v,'paired':{}}
        for arm,x in v.items():
            x['deployment_tokens']=x['reader_tokens']+prod_tokens;x['deployment_wall_sec']=x['reader_wall_sec']+prod_sec
            for ref in ['state_wide','wide','base']:
                groups=[[r for r in rows if r['paper_id']==pid] for pid in dict.fromkeys(r['paper_id'] for r in rows)];rng=random.Random(202610031);boot=[]
                for _ in range(10000):
                    sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(ids[r['id']]['arms'][arm]['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
                boot.sort();diagnostic[model]['paired'][arm+'_minus_'+ref]={'difference':x['f1']-s['models'][model]['arms'][ref]['f1'],'paper_ci95':[boot[250],boot[9750]]}
        for arm in ['wide','state_wide']:
            csvrows.append({'model':model,'arm':arm,'f1':s['models'][model]['arms'][arm]['f1']})
        for arm in ['queries_only','state_extract']:csvrows.append({'model':model,'arm':arm,'f1':v[arm]['f1']})
    extra+=['','消融的答案绑定只看题目：前置助动词问题限定Yes/No/unknown，其余只输出所问要素，不写解释和题目复述；保持完整状态及同一12段原文，不改解析器、不调用标准答案。缺项组只删除模型声称的已知内容和草稿，并保留实际缺项查询；同时改变了状态长度，因此不能单独归因于草稿。','','|模型/新方案|生产者＋读者token|生产者＋读者墙钟秒|平均答案词数|失败|','|---|---:|---:|---:|---:|']
    for model,v in asum.items():
        for arm,x in v.items():extra.append(f"|{model}/{arm}|{x['deployment_tokens']}|{x['deployment_wall_sec']:.3f}|{x['mean_answer_words']:.2f}|{x['errors']}|")
    extra+=['','|模型/消融对照|F1百分点|探索性论文簇95%区间|','|---|---:|---|']
    for model,v in diagnostic.items():
        for k,c in v['paired'].items():extra.append(f"|{model}/{k}|{100*c['difference']:+.2f}|[{100*c['paper_ci95'][0]:+.2f}, {100*c['paper_ci95'][1]:+.2f}]|")
    extra+=['','### 可核查的个案','','4B“有哪些归一化技术”：普通12段答CMN/CMVN，F1 85.71%；加状态仍含这些要素，但添加额外说明，F1 47.62%。4B“模型是否与先进模型比较”：Yes得100%，扩写为Yes加解释得15.38%。这两题显示词面指标受答案冗长影响，不能把全部F1下降解释成知识或推理退化。另有“是否分别处理抽象意义与概念”从unknown改Yes；该题公开标注为不可回答，原文能否支持这个判断仍需语义核验，不把标注当作已证明的哲学事实。','','标注证据段落组每模型2题无法映射而回退全文，其余10题只保留标注段落。这是上下文删除诊断，不能称绝对阅读能力上限，也不能直接归因于状态机制。','','### 阶段结论','','原主检验两模型均未通过。更小模型的所有方案绝对F1偏低，输出预算耗尽问题也未消失。4B状态增加了答案冗长，既有证据覆盖的改善没有转化成冻结主成绩；后续是否存在接口恢复，以本节消融数值为准，必须在新论文重复后才能称方法有效。当前更可靠的研究区分是：证据定位、状态内容可信度、条件/问题绑定、答案输出接口，应分开检验。不能笼统称“压缩失败”或“状态接续成功”。','','本报告只补充研究日志和边界证据，不宣布主论文已完成无重大问题审稿；原基准、完整成本和新样本确认仍缺。消融图由R绘制，所有数据源与SHA见diagnostic_synthesis.json。']
    path.write_text(text+'\n'.join(extra)+'\n',encoding='utf-8');save(aout/'diagnostic_synthesis.json',diagnostic)
    with (ROOT/'paper/figures/data/qasper_state_interface.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','arm','f1']);w.writeheader();w.writerows(csvrows)
    audit={'formal_calls':240,'warmups':2,'papers':6,'questions':12,'independent_confirmation':False,'main_paper_updated':False,'sha256':{str(x.relative_to(ROOT)):sha(x.read_bytes()) for x in [path,OUT/'summary.json',OUT/'independent_audit.json',aout/'diagnostic_synthesis.json',aout/'independent_audit.json',ROOT/'logs/qasper_state_resume_oracle/independent_audit.json']}}
    save(ROOT/'logs/qasper_state_resume_stage_audit_20261003.json',audit);print(json.dumps(diagnostic,indent=2))

if __name__=='__main__':main()
