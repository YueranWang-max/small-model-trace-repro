"""Audit adaptive full-format control; preserve original primary gates."""
import csv,json
from run_qasper_interface_replication import ROOT,OUT,MODELS,SOURCE,sha,save,references,f1,normalize

def main():
    dest=ROOT/'logs/qasper_interface_full_binding';fr=json.loads((dest/'frozen.json').read_text(encoding='utf-8'));data=json.loads((dest/'results.json').read_text(encoding='utf-8'));s=json.loads((dest/'summary.json').read_text(encoding='utf-8'));pilot=json.loads((OUT/'results.json').read_text(encoding='utf-8'));src=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert fr['pilot_sha256']==sha((OUT/'results.json').read_bytes()) and fr['protocol_sha256']==sha((ROOT/'protocol_qasper_interface_full_binding_20261003.md').read_bytes()) and fr['script_sha256']==sha((ROOT/'run_qasper_interface_full_binding.py').read_bytes()) and data['frozen_sha256']==sha((dest/'frozen.json').read_bytes())
    errors=[];count=0
    for m in MODELS:
        rows=[r for r in data['records'] if r['model']==m];assert len(rows)==24 and len({r['id'] for r in rows})==24
        for row in rows:
            pr=next(r for r in pilot['models'][m]['records'] if r['id']==row['id']);suffix='\nFinal answer binding: '+pr['arms']['wide_bind']['request']['messages'][1]['content'].split('\nFinal answer binding: ',1)[1];expected=pr['arms']['full']['request']['messages'][1]['content']+suffix;r=row['result'];req=r['request'];assert req['messages'][1]['content']==expected and req['model']==m and req['options']==pr['arms']['full']['request']['options'] and req['format']==pr['arms']['full']['request']['format'] and r['http_attempts']==1 and r['request_sha256']==sha(json.dumps(req,sort_keys=True).encode())
            if 'error' not in r:assert json.loads(r['raw']['message']['content'])==r['parsed']
            else:errors.append({'model':m,'id':row['id'],'error':r['error'],'done_reason':r.get('raw',{}).get('done_reason')})
            q=next(q for q in src[row['paper_id']]['qas'] if q['question_id']==row['id']);pred=r['parsed'].get('answer','');pred=pred if isinstance(pred,str) else '';aligned='Unanswerable' if normalize(pred)=='unknown' else pred;assert abs(r['f1']-max(f1(aligned,g) for g in references(q)))<1e-12;count+=1
    assert count==48;save(dest/'independent_audit.json',{'calls':count,'http_attempts':count,'errors':errors,'audit_pass':True,'results_sha256':sha((dest/'results.json').read_bytes())})
    path=ROOT/'paper/QASPER_INTERFACE_REPLICATION_20261003.md';lines=['','## 必要的全文同格式对照：另48调用','','该对照是在看到336调用结果后提出，已另行冻结、完整运行，不是原预设确认部分。加上本对照共384正式调用＋2预热，仍是同24题。原336分数及门槛不修改。全文逐字采用与局部阅读相同的答案绑定，只改变可用原文范围。','','|模型|全文同格式F1 %|总token|墙钟秒|失败|','|---|---:|---:|---:|---:|']
    for m,v in s.items():lines.append(f"|{m}|{100*v['f1']:.2f}|{v['tokens']}|{v['wall_sec']:.3f}|{v['errors']}|")
    lines+=['','|模型/相对全文同格式|F1百分点|探索性论文簇95%区间|严格质量/token门槛|','|---|---:|---|---|']
    for m,v in s.items():
        for k,x in v['paired'].items():lines.append(f"|{m}/{k}|{100*x['difference']:+.2f}|[{100*x['paper_ci95'][0]:+.2f}, {100*x['paper_ci95'][1]:+.2f}]|{v['gates'][k.split('_minus_')[0]]}|")
    lines+=['','### 修正后的结论','','tag1.7b前8段＋绑定相对原全文F1增加11.86百分点，原预设组合方案门槛通过；但它同时改变原文范围与答案接口，不能当作纯粹压缩贡献。补齐全文同格式后，前8/前12段仍有正的观察差值与探索性区间；全文出现1次HTTP500，本节严格门槛要求两侧无失败，因而未通过。该失败按0分保留，不能靠重试替换分数建立无损结论；失败不能归因于输出预算耗尽。该调用没有token回执，表中tag1.7b全文同格式150161token是已知用量下界，不能视为全部成本。','','4B同格式全文44.17%，前8段＋绑定48.53%、前12段＋绑定46.59%，差值区间较宽且低于-3百分点的下界，未确认保留质量。两个模型的额外状态相对同格式局部源均下降，因此本状态变体不继续按成功方法推进。收益不能等同语义正确率或方法创新；格式修复是必要工程边界，新颖性未建立。','','### 下一步工作与论文主线','','保留本轮已重复的模型差异：较小模型的答案接口收益明确，4B的广义接口主门槛未过。停止在这些题目优化状态草稿或挑源段数。回到第一目标，优先复现原任务中的相同T/x/q顺序与长度边界；随后检验更可信、与任务操作绑定的自产状态，不能将当前失败归咎于唐杰老师原方法。自然文档的正向线索须有新的病例确认、失败原因核验和语义评价。','','### 审计与图','','原336调用与新48调用分别独立重建，合计384正式HTTP尝试；控制错误细节：'+json.dumps(errors,ensure_ascii=False)+'。R图同时展示full/full_bind，区分输出接口与源长度。所有raw、协议、请求哈希和错误保留；图表未使用图像大模型。']
    path.write_text(path.read_text(encoding='utf-8')+'\n'.join(lines)+'\n',encoding='utf-8')
    cp=ROOT/'paper/figures/data/qasper_interface_replication.csv';rows=list(csv.DictReader(cp.open(encoding='utf-8')))
    for m,v in s.items():rows.append({'model':m,'arm':'full_bind','f1':v['f1'],'tokens':v['tokens'],'wall_sec':v['wall_sec']})
    with cp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['model','arm','f1','tokens','wall_sec']);w.writeheader();w.writerows(rows)
    print(json.dumps({'calls':count,'errors':errors,'summary':s},indent=2))

if __name__=='__main__':main()
