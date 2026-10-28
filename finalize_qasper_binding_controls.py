"""Audit equal-binding controls and qualify visible-state conclusions."""
import csv,json,random
from pathlib import Path
from run_qasper_state_resume import ROOT,OUT,MODELS,SOURCE,sha,save,references,f1,normalize

def main():
    dest=ROOT/'logs/qasper_state_resume_binding_control';fr=json.loads((dest/'frozen.json').read_text(encoding='utf-8'));data=json.loads((dest/'results.json').read_text(encoding='utf-8'));summary=json.loads((dest/'summary.json').read_text(encoding='utf-8'));old=json.loads((ROOT/'logs/qasper_state_resume_ablation/frozen.json').read_text(encoding='utf-8'));ab=json.loads((ROOT/'logs/qasper_state_resume_ablation/results.json').read_text(encoding='utf-8'));asum=json.loads((ROOT/'logs/qasper_state_resume_ablation/diagnostic_synthesis.json').read_text(encoding='utf-8'));pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert fr['protocol_sha256']==sha((ROOT/'protocol_qasper_state_resume_binding_control_20261003.md').read_bytes()) and fr['script_sha256']==sha((ROOT/'run_qasper_state_resume_binding_control.py').read_bytes()) and fr['pilot_sha256']==sha((OUT/'results.json').read_bytes()) and fr['ablation_frozen_sha256']==sha((ROOT/'logs/qasper_state_resume_ablation/frozen.json').read_bytes()) and data['frozen_sha256']==sha((dest/'frozen.json').read_bytes())
    count=errors=0;paired={}
    for r in data['records']:
        model=r['model'];c=next(c for c in old['cases'] if c['model']==model and c['id']==r['id']);pr=next(x for x in pilot['models'][model]['records'] if x['id']==r['id']);suffix='\nFinal answer binding: '+c['prompts']['state_extract'].split('\nFinal answer binding: ',1)[1];goldq=next(q for q in src[r['paper_id']]['qas'] if q['question_id']==r['id'])
        assert set(r['arms'])=={'wide_extract','base_extract'}
        for a,x in r['arms'].items():
            expected=pr['arms'][a.split('_')[0]]['request']['messages'][1]['content']+suffix
            assert x['request']['messages'][1]['content']==expected and x['request_sha256']==sha(json.dumps(x['request'],sort_keys=True).encode()) and x['http_attempts']==1 and x['request']['model']==model
            if 'error' not in x:assert json.loads(x['raw']['message']['content'])==x['parsed']
            pred=x['parsed'].get('answer','');aligned='Unanswerable' if normalize(pred)=='unknown' else pred;assert abs(x['official_f1']-max(f1(aligned,g) for g in references(goldq)))<1e-12;count+=1;errors+=('error' in x)
    assert count==48
    for model in MODELS:
        rows=[r for r in data['records'] if r['model']==model];ids={r['id']:r for r in ab['records'] if r['model']==model};groups=[[r for r in rows if r['paper_id']==pid] for pid in dict.fromkeys(r['paper_id'] for r in rows)];paired[model]={}
        for ref in ['wide_extract','base_extract']:
            rng=random.Random(202610032);boot=[]
            for _ in range(10000):
                sample=[r for _ in groups for r in groups[rng.randrange(len(groups))]];boot.append(sum(ids[r['id']]['arms']['state_extract']['official_f1']-r['arms'][ref]['official_f1'] for r in sample)/len(sample))
            boot.sort();paired[model]['state_extract_minus_'+ref]={'difference':asum[model]['arms']['state_extract']['f1']-summary[model][ref]['f1'],'paper_ci95':[boot[250],boot[9750]]}
    save(dest/'independent_audit.json',{'formal_calls':48,'http_attempts':48,'errors':errors,'audit_pass':True,'results_sha256':sha((dest/'results.json').read_bytes())});save(dest/'paired_summary.json',paired)
    path=ROOT/'paper/QASPER_STATE_RESUME_20261003.md';lines=['','## 相同答题格式、没有状态的必要对照','','进一步冻结48次相同题目控制，总数288正式调用＋2预热。普通前8/前12段使用与state_extract逐字相同的最终答案绑定指令，删除整个自产状态，采用同一输出schema和预算。比较才有资格讨论状态的额外价值；全部是同批问题的自适应诊断，不能称新论文复现。','','|模型|top8＋绑定F1 %|top12＋绑定F1 %|状态＋top12＋绑定F1 %|','|---|---:|---:|---:|']
    for model in MODELS:lines.append(f"|{model}|{100*summary[model]['base_extract']['f1']:.2f}|{100*summary[model]['wide_extract']['f1']:.2f}|{100*asum[model]['arms']['state_extract']['f1']:.2f}|")
    lines+=['','|模型/状态减无状态|F1百分点|探索性论文簇95%区间|','|---|---:|---|']
    for model,v in paired.items():
        for k,x in v.items():lines.append(f"|{model}/{k}|{100*x['difference']:+.2f}|[{100*x['paper_ci95'][0]:+.2f}, {100*x['paper_ci95'][1]:+.2f}]|")
    lines+=['','|模型/无状态控制|总token|调用墙钟秒|平均答案词数|失败|','|---|---:|---:|---:|---:|']
    for model,v in summary.items():
        for k,x in v.items():lines.append(f"|{model}/{k}|{x['tokens']}|{x['wall_sec']:.3f}|{x['mean_answer_words']:.2f}|{x['errors']}|")
    lines+=['','### 更新后的解释','','不能把state_extract相对state_wide的改善直接归功于状态：两者只改变答题约束，状态和原文均相同。状态的额外收益只看与本节相同答题约束的无状态控制比较；若状态无优势，应优先保留更简单、更便宜的普通检索，而非继续叠加草稿。F1恢复不代表事实正确性已全面恢复，仍需语义复核和未用于本次诊断的新问题。','','各次调用墙钟完整保留，但模型固定分块、冷启动/缓存差异和部分本机审计与绘图同期运行影响运行条件。因此本轮墙钟是观察记录，不是干净的速度基准；不作提速或生命周期成本优势结论。tokens计入生产者，可作这批请求的确定性用量比较。','','下一阶段停止改这12题的提示：优先在新的论文问题上固定“局部完整原文＋清晰答案接口”的廉价基线，再比较有无状态；更高层创新必须超过该基线，不能把格式修复包装为方法突破。']
    path.write_text(path.read_text(encoding='utf-8')+'\n'.join(lines)+'\n',encoding='utf-8')
    cp=ROOT/'paper/figures/data/qasper_state_interface.csv';csvrows=list(csv.DictReader(cp.open(encoding='utf-8')))
    for model,v in summary.items():
        for arm,x in v.items():csvrows.append({'model':model,'arm':arm,'f1':x['f1']})
    with cp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['model','arm','f1']);w.writeheader();w.writerows(csvrows)
    audit=json.loads((ROOT/'logs/qasper_state_resume_stage_audit_20261003.json').read_text(encoding='utf-8'));audit['formal_calls']=288;audit['sha256'].update({str(x.relative_to(ROOT)):sha(x.read_bytes()) for x in [path,dest/'independent_audit.json',dest/'paired_summary.json',cp]});save(ROOT/'logs/qasper_state_resume_stage_audit_20261003.json',audit);print(json.dumps({'summary':summary,'paired':paired},indent=2))

if __name__=='__main__':main()
