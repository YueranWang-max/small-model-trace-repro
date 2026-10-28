# -*- coding: utf-8 -*-
"""
queue_sweep2 —— 同代际扫描续跑(前队列 9JEiit 被宿主回收后接力)。
4b 从 records_so_far checkpoint 恢复(3 题已缓存, 重做 idx14 S3+汇总), 8b 全新跑。
日志追加到 logs/queue_sweep.log, 结束时必须写入"扫描队列结束"标记,
hotpot 队列(queue_realdoc2.py)靠该标记触发。任何路径都保证写标记(finally)。
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SWEEP_LOG = os.path.join(BASE, "logs", "queue_sweep.log")
MODELS = [("qwen3:4b", "qwen3_4b"), ("qwen3:8b", "qwen3_8b")]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] [sweep2] {m}"
    print(line, flush=True)
    with open(SWEEP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_model(model, tag):
    out = f"logs/scale8b_{tag}.json"
    for attempt in (1, 2):
        log(f"=== {model} 第 {attempt} 次尝试 ===")
        try:
            r = subprocess.run(
                [PY, "run_scale8b.py", "--model", model, "--out", out,
                 "--n-hard", "4"],
                cwd=BASE, capture_output=True, text=True, timeout=8 * 3600)
            log(f"exit={r.returncode}\n{(r.stdout or '')[-1200:]}")
        except subprocess.TimeoutExpired as e:
            log(f"超时: {e}")
            r = None
        ok = False
        try:
            d = json.load(open(os.path.join(BASE, out), encoding="utf-8"))
            s = d.get("summary")
            if s:
                log(f"{model} 完成: S1={s['S1_f1']} S2={s['S2_f1']} "
                    f"S3={s['S3_f1']} (n={s['n_questions']})")
                ok = True
        except Exception as e:
            log(f"汇总读取失败: {e}")
        if ok:
            return True
        if attempt == 1:
            log("失败, 120s 后重试 (checkpoint 断点续跑)")
            time.sleep(120)
    log(f"{model} 两次尝试均失败")
    return False


def main():
    log("续跑队列启动 (4b 恢复 + 8b)")
    results = {}
    try:
        for model, tag in MODELS:
            results[model] = "OK" if run_model(model, tag) else "FAIL"
    finally:
        log(f"===== 扫描队列结束: {results} =====")


if __name__ == "__main__":
    main()
