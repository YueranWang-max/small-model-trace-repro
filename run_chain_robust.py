# -*- coding: utf-8 -*-
"""
run_chain_robust —— 自愈串行链 runner。
阶段1: run_scale8b (qwen3-vl:8b 续跑)  阶段2: run_realdoc (50 题)。
每阶段最多 8 次尝试, 失败后退避 120s 重试; 断点机制保证重试只补缺失。
之后 qwen2.5:7b 对照轴由 queue_qwen25.py 在 realdoc>=50 时自动接跑。
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MASTER = os.path.join(BASE, "logs", "scale8b_master.log")
MAX_TRY = 8
BACKOFF = 120


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def marker(m):
    with open(MASTER, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def run_phase(name, args, logpath):
    for i in range(1, MAX_TRY + 1):
        log(f"{name} 第 {i}/{MAX_TRY} 次尝试")
        with open(logpath, "a", encoding="utf-8") as f:
            rc = subprocess.run([PY] + args, stdout=f,
                                stderr=subprocess.STDOUT,
                                cwd=BASE).returncode
        if rc == 0:
            log(f"{name} 成功")
            return True
        log(f"{name} 失败 (rc={rc}), {BACKOFF}s 后重试")
        time.sleep(BACKOFF)
    log(f"{name} 放弃: 连续 {MAX_TRY} 次失败")
    return False


def main():
    log("自愈链启动")
    ok1 = run_phase(
        "阶段1 qwen3-vl:8b",
        ["run_scale8b.py", "--model", "qwen3-vl:8b",
         "--ref", os.path.join(BASE, "logs", "chain_2b_3k_s13.json"),
         "--out", os.path.join(BASE, "logs", "scale8b_qwen3vl.json")],
        os.path.join(BASE, "logs", "scale8b_qwen3vl_run.log"))
    marker("RESUME_DONE" if ok1 else "RESUME_FAILED")

    ok2 = run_phase(
        "阶段2 realdoc",
        ["run_realdoc.py", "--n-instances", "50",
         "--out", os.path.join(BASE, "logs", "realdoc_2b.jsonl")],
        os.path.join(BASE, "logs", "realdoc_run.log"))
    marker("REALDOC_DONE" if ok2 else "REALDOC_FAILED")
    marker("CHAIN5_FINISHED" if (ok1 and ok2) else "CHAIN5_PARTIAL")
    log(f"链结束 ok1={ok1} ok2={ok2}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
