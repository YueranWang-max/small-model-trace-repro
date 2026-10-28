# -*- coding: utf-8 -*-
"""
queue_budget_matched —— 串行队列尾巴 2: 等 qwen2.5:7b 对照轴完成后,
自动接跑 Budget-Matched Stage A (qwen3:1.7b, 20 题 x C0-C8)。
严格串行: 仅当 scale8b_qwen25.json 4 题全部含 S1/S2/S3 时启动。
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
Q25 = os.path.join(BASE, "logs", "scale8b_qwen25.json")
RLOG = os.path.join(BASE, "logs", "budget_matched_stageA_run.log")
POLL_SEC = 300
GIVEUP_SEC = 30 * 3600


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def q25_done():
    try:
        d = json.load(open(Q25, encoding="utf-8"))
        recs = d.get("records") or d.get("records_so_far") or []
        return len(recs) >= 4 and all(
            all(k in r for k in ("S1", "S2", "S3")) for r in recs)
    except Exception:
        return False


def main():
    log(f"队列启动: 等待 qwen2.5:7b 对照轴完成, 每 {POLL_SEC}s 轮询")
    t0 = time.time()
    while not q25_done():
        if time.time() - t0 > GIVEUP_SEC:
            log("放弃: 等待超时 30h")
            sys.exit(2)
        time.sleep(POLL_SEC)
    log("qwen2.5:7b 对照轴完成, 60s 冷却后启动 Budget-Matched Stage A")
    time.sleep(60)
    cmd = [PY, os.path.join(BASE, "run_budget_matched.py"),
           "--out", os.path.join(BASE, "logs", "budget_matched_stageA.json")]
    log(f"启动: {' '.join(cmd)}")
    with open(RLOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=BASE).returncode
    log(f"Stage A 退出码 {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
