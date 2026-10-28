# -*- coding: utf-8 -*-
"""
queue_realdoc2 —— 第二文档族 (HotpotQA) 队列。
等待 queue_sweep (同代际扫描) 结束后, 用校正预算 (PRED_READ=12288) 跑
run_realdoc.py 于 data/realdoc_hotpot_32k.jsonl, 50 题。
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SWEEP_LOG = "logs/queue_sweep.log"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sweep_done():
    p = os.path.join(BASE, SWEEP_LOG)
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return "扫描队列结束" in f.read()
    except Exception:
        return False


def main():
    log("等待同代际扫描队列结束 ...")
    while not sweep_done():
        time.sleep(300)
    log("扫描已结束, 启动 Hotpot 族 realdoc (PRED_READ=12288)")
    for attempt in (1, 2):
        r = subprocess.run(
            [PY, "run_realdoc.py",
             "--data", "data/realdoc_hotpot_32k.jsonl",
             "--out", "logs/realdoc_hotpot_2b.jsonl",
             "--pred-read", "12288",
             "--distill-cache", "logs/realdoc_hotpot_distill_cache.json"],
            cwd=BASE, capture_output=True, text=True, timeout=12 * 3600)
        log(f"exit={r.returncode}\n{(r.stdout or '')[-2000:]}")
        if r.returncode == 0:
            break
        if attempt == 1:
            log("失败, 300s 后重试 (断点续跑)")
            time.sleep(300)
    log("Hotpot 族队列结束")


if __name__ == "__main__":
    main()
