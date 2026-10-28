# -*- coding: utf-8 -*-
"""Stage D/E 收官器:
1. 轮询 chain_driver.log 等待 'stageD_musique 结束' 标记(或唯一 id 达 50)
2. 对 stageD_hotpot/stageD_musique jsonl 按 id 去重(保留最后一次出现)
3. 依次运行 analyze_stageD.py / analyze_stageE.py, 输出落盘 logs/stageDE_final_analysis.txt
"""
import os
import json, os, subprocess, sys, time, io

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = os.path.join(BASE, "logs", "stageDE_final_analysis.txt")
DRIVER_LOG = os.path.join(BASE, "logs", "chain_driver.log")

def n_unique(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                ids.add(json.loads(l)["id"])
    return len(ids)

def wait_done(timeout=7200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(DRIVER_LOG, encoding="utf-8", errors="replace") as f:
                if "stageD_musique 结束" in f.read():
                    return "driver标记"
        except OSError:
            pass
        try:
            if n_unique(os.path.join(BASE, "logs", "stageD_musique.jsonl")) >= 50:
                time.sleep(30)
                return "唯一id=50"
        except OSError:
            pass
        time.sleep(20)
    return "超时"

def dedup(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    last = {}
    for r in recs:
        last[r["id"]] = r
    if len(last) == len(recs):
        return len(recs), len(recs)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in last.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return len(recs), len(last)

def main():
    buf = io.StringIO()
    how = wait_done()
    buf.write(f"[watch] 完成判定: {how} ({time.strftime('%H:%M:%S')})\n")
    time.sleep(10)
    for name in ["stageD_hotpot.jsonl", "stageD_musique.jsonl"]:
        p = os.path.join(BASE, "logs", name)
        before, after = dedup(p)
        buf.write(f"[dedup] {name}: {before} -> {after} 行\n")
    for script in ["analyze_stageD.py", "analyze_stageE.py"]:
        buf.write(f"\n{'='*20} {script} {'='*20}\n")
        try:
            r = subprocess.run([PY, os.path.join(BASE, script)], capture_output=True, text=True,
                               cwd=BASE, encoding="utf-8", errors="replace", timeout=1800)
            buf.write(r.stdout or "")
            if r.stderr:
                buf.write("\n[stderr]\n" + r.stderr[-2000:])
            buf.write(f"\n[{script} rc={r.returncode}]\n")
        except Exception as e:
            buf.write(f"\n[{script} 异常: {e}]\n")
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("FINISH_DONE")

if __name__ == "__main__":
    main()
