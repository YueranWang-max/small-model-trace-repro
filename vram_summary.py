# -*- coding: utf-8 -*-
"""vram_stageDE.csv whole-run 概况 v2: 处理 nvidia-smi 默认 CSV 带单位格式"""
import os
import io, statistics
from datetime import datetime

mem, pw, ts = [], [], []
for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "vram_stageDE.csv"), encoding="utf-8", errors="replace"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    try:
        t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
        m = float(parts[1].replace("MiB", "").strip())
        p = float(parts[2].replace("W", "").strip())
    except ValueError:
        continue
    ts.append(t); mem.append(m); pw.append(p)

iv = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
out = io.StringIO()
out.write("vram_stageDE.csv whole-run profile (n=%d samples)\n" % len(ts))
if ts:
    out.write("span: %s -> %s (%.1f h)\n" % (ts[0], ts[-1], (ts[-1] - ts[0]).total_seconds() / 3600))
if iv:
    out.write("sampling interval: median=%.1fs mean=%.1fs\n" % (statistics.median(iv), statistics.mean(iv)))
if mem:
    out.write("VRAM used: mean=%.0f MiB peak=%.0f MiB (%.2f GiB)\n" % (statistics.mean(mem), max(mem), max(mem) / 1024))
if pw:
    out.write("Power: mean=%.1f W peak=%.1f W\n" % (statistics.mean(pw), max(pw)))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "vram_summary.txt"), "w", encoding="utf-8").write(out.getvalue())
print(out.getvalue())
