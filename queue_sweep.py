# -*- coding: utf-8 -*-
"""
queue_sweep —— 同代际参数扫描串行队列。
等待 realdoc 预算复跑 (14 题 jsonl 齐且进程退出) 后, 依次跑 qwen3:1.7b/4b/8b
的三制式读出 (seed-13 难题 replay, 共享 _e_layered 种子)。每模型失败重试 1 次。
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MODELS = [("qwen3:1.7b", "qwen3_1.7b"), ("qwen3:4b", "qwen3_4b"),
          ("qwen3:8b", "qwen3_8b")]
WAIT_TARGET = "logs/realdoc_2b_pred12k.jsonl"
WAIT_N = 14


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ollama_alive():
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=10)
        return True
    except Exception:
        return False


def wait_frontend():
    log("等待 realdoc 预算复跑收官 ...")
    while True:
        n = 0
        p = os.path.join(BASE, WAIT_TARGET)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f)
        if n >= WAIT_N:
            log(f"复跑已收官 ({n} 题), 开始扫描")
            return
        if not ollama_alive():
            log("警告: ollama 无响应, 等待 120s 后重查")
        time.sleep(60)


def run_model(model, tag):
    out = f"logs/scale8b_{tag}.json"
    for attempt in (1, 2):
        log(f"=== {model} 第 {attempt} 次尝试 ===")
        r = subprocess.run(
            [PY, "run_scale8b.py", "--model", model, "--out", out,
             "--n-hard", "4"],
            cwd=BASE, capture_output=True, text=True, timeout=8 * 3600)
        tail = (r.stdout or "")[-1500:]
        log(f"exit={r.returncode}\n{tail}")
        if r.returncode == 0:
            try:
                d = json.load(open(os.path.join(BASE, out), encoding="utf-8"))
                s = d.get("summary", {})
                log(f"{model} 完成: S1={s.get('S1_f1')} S2={s.get('S2_f1')} "
                    f"S3={s.get('S3_f1')}")
                return True
            except Exception as e:
                log(f"汇总读取失败: {e}")
        if attempt == 1:
            log("失败, 120s 后重试")
            time.sleep(120)
    log(f"{model} 两次尝试均失败")
    return False


def main():
    wait_frontend()
    results = {}
    for model, tag in MODELS:
        ok = run_model(model, tag)
        results[model] = "OK" if ok else "FAIL"
    log(f"===== 扫描队列结束: {results} =====")
    for model, tag in MODELS:
        p = os.path.join(BASE, f"logs/scale8b_{tag}.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                s = d.get("summary")
                if s:
                    log(f"{model}: S1={s['S1_f1']} S2={s['S2_f1']} "
                        f"S3={s['S3_f1']} (n={s['n_questions']})")
            except Exception:
                pass


if __name__ == "__main__":
    main()
