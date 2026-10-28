# -*- coding: utf-8 -*-
"""读取 pilot 结果 JSON, 输出结构化对比 (总体 + 按操作类型分项)。
用法: python analyze.py logs/pilot_2b_6k.json [more.json ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def summarize(records, label):
    ok = [r for r in records if "error" not in r]
    n = len(ok)
    if n == 0:
        return None
    lines = [f"\n===== {label}  (有效题目 {n}) ====="]
    hdr = f"{'条件':<14}{'EM':>8}{'F1':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for cond, name in (("baseline", "first pass"), ("append", "trace append"),
                       ("state", "trace as state")):
        em = sum(r[cond]["em"] for r in ok) / n
        f1 = sum(r[cond]["f1"] for r in ok) / n
        lines.append(f"{name:<14}{em:>8.3f}{f1:>8.3f}")
    wins = sum(1 for r in ok if r["state"]["f1"] > r["append"]["f1"])
    ties = sum(1 for r in ok if abs(r["state"]["f1"] - r["append"]["f1"]) < 1e-9)
    lines.append(f"\n逐题 (F1): state 胜 {wins} / 平 {ties} / 负 {n - wins - ties}")
    d = sum(r["state"]["f1"] - r["append"]["f1"] for r in ok) / n
    lines.append(f"state - append 平均 F1 差 = {d:+.3f}")
    # 分操作类型
    for op in ("bfs", "parents"):
        sub = [r for r in ok if r.get("op_type") == op]
        if not sub:
            continue
        m = len(sub)
        b = sum(r["baseline"]["f1"] for r in sub) / m
        a = sum(r["append"]["f1"] for r in sub) / m
        s = sum(r["state"]["f1"] for r in sub) / m
        lines.append(f"  [{op:<7}] n={m}  first={b:.3f}  append={a:.3f}  state={s:.3f}")
    return "\n".join(lines)


def main():
    out = []
    for path in sys.argv[1:]:
        if not os.path.isabs(path):
            path = os.path.join(HERE, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        label = os.path.basename(path)
        s = summarize(data["records"], label)
        meta = data.get("summary", {})
        if meta:
            out.append(f"\n[{label}] 模型={meta.get('model')} "
                       f"目标字符={meta.get('target_chars')} n_tr={meta.get('n_tr')} "
                       f"用时={meta.get('elapsed_sec')}s")
        if s:
            out.append(s)
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "logs", "analysis.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
