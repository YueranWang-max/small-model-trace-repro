import os
import hashlib, os, datetime

BASE = os.path.dirname(os.path.abspath(__file__))
def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''):
            h.update(c)
    return h.hexdigest()

def ts(p):
    st = os.stat(p)
    c = datetime.datetime.fromtimestamp(st.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
    m = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    return c, m

files = [
    ('protocol_stageDE.md (本文件，追加审计段之前的冻结原版)', 'protocol_stageDE.md'),
    ('run_stageD.py', 'run_stageD.py'),
    ('run_stageE.py', 'run_stageE.py'),
    ('analyze_stageD.py (m=4 修订后)', 'analyze_stageD.py'),
    ('analyze_stageE.py', 'analyze_stageE.py'),
    ('verify_stageE_extra.py (Stage E 事后比较)', 'verify_stageE_extra.py'),
    ('logs/stageD_hotpot.jsonl (主数据)', r'logs\stageD_hotpot.jsonl'),
    ('logs/stageD_musique.jsonl (已废弃：并发双写 102 行)', r'logs\stageD_musique.jsonl'),
    ('logs/stageD_musique_rerun.jsonl (干净复跑，最终采用)', r'logs\stageD_musique_rerun.jsonl'),
    ('logs/stageE_llama.jsonl (主数据)', r'logs\stageE_llama.jsonl'),
    ('logs/stageDE_final_analysis.txt (m=3 终局分析，已废弃)', r'logs\stageDE_final_analysis.txt'),
    ('logs/stageD_m4_final.txt (m=4 最终分析，采用)', r'logs\stageD_m4_final.txt'),
    ('logs/stageE_extra_comparisons.txt (事后解释性比较)', r'logs\stageE_extra_comparisons.txt'),
    ('logs/ollama_show_archive.txt (模型配置归档)', r'logs\ollama_show_archive.txt'),
]

lines = []
lines.append('')
lines.append('## 6. Post-run Audit Trail（2026-09-21 追加；七轮外部审查要求，不改写上文冻结内容）')
lines.append('')
lines.append('### 6.1 分析协议的事后修订（critical：best-RAG 数据依赖选择）')
lines.append('- 原冻结方案：主比较 m=3（exq−eq、txq−tq、best_rag−eq），best_rag 按「族内均分 F1 更高者」从 bm25_top4/dense_top4 中选取。')
lines.append('- 缺陷：即使选择规则事前冻结，在同一数据上先选择后检验仍低估多重比较错误率（外部审查七轮 critical-2 指出）。')
lines.append('- 修订：2026-09-21，外部审查后把 bm25_top4−eq 与 dense_top4−eq **都**纳入主族，m=3 → **m=4**（analyze_stageD.py 头部注明 post-hoc analysis revision）。**只改分析，不改任何实验数据与条件定义**。')
lines.append('')
lines.append('### 6.2 MuSiQue 数据完整性事故与干净复跑（critical：事后去重）')
lines.append('- 事故：夜间链式驱动器与一个旧后台任务**并发双写** logs/stageD_musique.jsonl，致 102 行 / 45 unique id；处置时按 id 保留最后一次做了事后去重并 os.replace 覆盖原文件（无备份）——事后数据处理，审计性受损。')
lines.append('- 处置：废弃 logs/stageD_musique.jsonl（保留在仓库仅供审计，不用于任何分析）；执行**干净单进程技术复跑** 50 题 → logs/stageD_musique_rerun.jsonl。')
lines.append('- 复跑定性：这是「看到初始结果后」的数据完整性复跑（data-integrity rerun），论文与协议持续如此标记；MuSiQue 族结论以复跑数据为准。')
lines.append('')
lines.append('### 6.3 时间线与文件清单（SHA-256）')
lines.append('')
lines.append('| 文件 | 创建时间 | 最后写入 | SHA-256（前 16 位） |')
lines.append('|---|---|---|---|')

import sys
for label, rel in files:
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        lines.append('| %s | 缺失 | | |' % label)
        continue
    c, m = ts(p)
    h = sha(p)[:16]
    lines.append('| %s | %s | %s | `%s` |' % (label, c, m, h))

# full hashes written to a separate sidecar
side = os.path.join(BASE, 'logs', 'stageDE_audit_sha256.txt')
with open(side, 'w', encoding='utf-8') as f:
    for label, rel in files:
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            f.write('%s  %s\n' % (sha(p), rel))

with open(os.path.join(BASE, 'protocol_stageDE.md'), 'a', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('audit trail appended; full hashes ->', side)
