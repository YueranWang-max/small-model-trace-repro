# -*- coding: utf-8 -*-
"""
queue_qwen25 —— 串行队列尾巴: 等 realdoc 50 题全部落盘后, 自动接跑
qwen2.5:7b 对照轴 (同代分离实验: 旧代 7.6B vs 新代 4.7B/8.8B)。
严格串行保证: 仅当 realdoc 行数 >= 50 时才启动, 不与任何实验并发。
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REALDOC = os.path.join(BASE, "logs", "realdoc_2b.jsonl")
OUT = os.path.join(BASE, "logs", "scale8b_qwen25.json")
QLOG = os.path.join(BASE, "logs", "scale8b_qwen25_run.log")
POLL_SEC = 300
GIVEUP_SEC = 24 * 3600


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def realdoc_count():
    n = 0
    try:
        with open(REALDOC, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    n += 1
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return n


def main():
    log(f"队列启动: 等待 realdoc >= 50 (当前 {realdoc_count()}), "
        f"每 {POLL_SEC}s 轮询, 最长等 {GIVEUP_SEC // 3600}h")
    t0 = time.time()
    while realdoc_count() < 50:
        if time.time() - t0 > GIVEUP_SEC:
            log("放弃: 等待超时 24h, realdoc 未完成")
            sys.exit(2)
        time.sleep(POLL_SEC)
    log(f"realdoc 完成 ({realdoc_count()}/50), 60s 冷却后启动 qwen2.5:7b 对照轴")
    time.sleep(60)
    cmd = [PY, os.path.join(BASE, "run_scale8b.py"),
           "--model", "qwen2.5:7b",
           "--ref", os.path.join(BASE, "logs", "chain_2b_3k_s13.json"),
           "--out", OUT]
    log(f"启动: {' '.join(cmd)}")
    with open(QLOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=BASE).returncode
    log(f"qwen2.5:7b 对照轴退出码 {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
