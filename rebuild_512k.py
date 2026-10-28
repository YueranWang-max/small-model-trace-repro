# -*- coding: utf-8 -*-
"""从运行日志重建 512K 转录结果 JSON（原进程在落盘时撞 PermissionError，数据不能丢）。"""
import json
import os

# 来自主进程完整日志（QTYkEY.stdout）的转录完成段
out = {
    "size": 512000,
    "graph_chars": 519524,
    "n_chunks": 131,
    "models": ["openbmb/minicpm5-2b", "qwen3.5:4b"],
    "transcribe_calls": 413,
    "elapsed_sec": 20011,
    "db_size": 33715,
    "truth_edges": 34093,
    "recall": 0.9889,
    "n_missing": 378,
    "collapsed_chunks": [
        {"chunk": 83, "lines": 264, "verified": 1,
         "note": "单次扫描塌陷：384 候选仅 1 条通过机械核验，~263 行漏检，占全部漏检的 ~70%"},
        {"chunk": 131, "lines": 35, "verified": 35, "note": "末片正常（35 行全收）"},
    ],
    "questions": None,
    "questions_note": ("边库落盘失败（PermissionError）导致问答阶段未执行；"
                       "同机制问答已在 128K(5/5=1.000) 与 256K(4/5=1.000, BFS 0.965) 档充分验证。"),
    "reconstruction": "由主进程日志重建，转录数值段逐片核对无缺",
}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "transcribe_512k.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("written:", p, os.path.getsize(p), "bytes")

# 验证：重写边库文件测试 PermissionError 是否复现
try:
    dbp = p.replace(".json", "_db.json")
    with open(dbp, "w", encoding="utf-8") as f:
        f.write('{"db": [], "note": "边库未能从崩溃进程恢复；可由 run_transcribe.py --size 512000 复建"}\n')
    print("db write test: OK ->", dbp)
    os.remove(dbp)
    print("db write test: cleaned up")
except PermissionError as e:
    print("db write test: PermissionError persists:", e)
