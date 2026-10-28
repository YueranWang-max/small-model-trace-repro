# -*- coding: utf-8 -*-
"""
run_stageDE_chain —— Stage D/E 三段串行驱动器。
单一 python 进程避免 bash shim 对长复合命令的破坏;
逐段子进程 + EXIT 标记落盘; 段内 checkpoint 断点续跑, 驱动器死亡后重启即续。
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SEGS = [
    ("stageD_hotpot",
     [PY, "run_stageD.py", "--data", "data/realdoc_hotpot_32k.jsonl",
      "--out", "logs/stageD_hotpot.jsonl"],
     "logs/stageD_hotpot_run.log"),
    ("stageE_llama",
     [PY, "run_stageE.py"],
     "logs/stageE_run.log"),
    ("stageD_musique",
     [PY, "run_stageD.py", "--data", "data/realdoc_musique_32k.jsonl",
      "--out", "logs/stageD_musique.jsonl"],
     "logs/stageD_musique_run.log"),
]


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cmd, logfile in SEGS:
        if only and name != only:
            continue
        marker = f"EXIT_{name.upper()}"
        # 已有成功标记则跳过 (断点续跑语义)
        lf = os.path.join(BASE, logfile)
        if os.path.exists(lf):
            with open(lf, encoding="utf-8", errors="replace") as f:
                if f"{marker}=0" in f.read():
                    log(f"{name}: 已有 {marker}=0, 跳过")
                    continue
        log(f"启动 {name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        with open(lf, "a", encoding="utf-8", errors="replace") as fo:
            rc = subprocess.call(cmd, cwd=BASE, stdout=fo,
                                 stderr=subprocess.STDOUT)
            fo.write(f"\n{marker}={rc}\n")
            fo.flush()
        log(f"{name} 结束 rc={rc} ({(time.time()-t0)/60:.1f} min)")
        if rc != 0:
            log(f"{name} 非零退出, 继续下一段 (断点可续)")
    log("链完成")


if __name__ == "__main__":
    main()
