# -*- coding: utf-8 -*-
"""
analyze_stageC —— Stage C 收官分析（严格按 protocol_stageC.md §4，冻结算法）。
主比较: bm25_top2/bm25_top4/sliding12k 各减 sdr_state 的逐题配对差。
CI: 配对 bootstrap B=10,000 (seed 20260921, percentile)。
p: 配对符号翻转随机化检验 MC 10^6 (seed 20260921, add-one 因 MC)。
多重: Holm 步降 m=3, alpha=0.05。ITT 口径。
"""
from __future__ import annotations

import json
import os
import random
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "logs/stageC_baselines.jsonl")
BOOT = 10_000
MC = 1_000_000
SEED = 20260921
PRIMARY = [("bm25_top2", "sdr_state"), ("bm25_top4", "sdr_state"),
           ("sliding12k", "sdr_state")]


def load():
    recs = {}
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                recs[r["id"]] = r
            except Exception:
                pass
    return recs


def paired_diffs(recs, a, b):
    d = []
    for r in recs.values():
        ca = r["conds"].get(a, {})
        cb = r["conds"].get(b, {})
        d.append(ca.get("f1", 0.0) - cb.get("f1", 0.0))  # ITT: 缺失/失败=0
    return d


def boot_ci(d, rng):
    n = len(d)
    obs = sum(d) / n
    means = []
    for _ in range(BOOT):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo, hi = means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]
    return obs, lo, hi


def signflip_p(d, rng):
    """MC 符号翻转 (2^k 不可枚举), add-one 平滑; 返回 (p_two, mc_se)。"""
    n = len(d)
    obs = abs(sum(d) / n)
    ge = 0
    for _ in range(MC):
        s = sum(v if rng.random() < 0.5 else -v for v in d)
        if abs(s / n) >= obs - 1e-12:
            ge += 1
    p = (ge + 1) / (MC + 1)
    se = (p * (1 - p) / (MC + 1)) ** 0.5
    return p, se


def main():
    recs = load()
    n = len(recs)
    print(f"n = {n}")
    if n == 0:
        sys.exit("无数据")
    rng = random.Random(SEED)
    res = {}
    for a, b in PRIMARY:
        d = paired_diffs(recs, a, b)
        obs, lo, hi = boot_ci(d, rng)
        p, se = signflip_p(d, rng)
        res[f"{a}-{b}"] = (obs, lo, hi, p, se)
        print(f"\n{a} - {b}: delta={obs:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]")
        print(f"  MC sign-flip p={p:.6f} (se={se:.2e})")

    print("\n===== Holm step-down (m=3, alpha=0.05) =====")
    for tag, idx in [("two-sided", 3)]:
        order = sorted(res.items(), key=lambda kv: kv[1][idx])
        prev = 0.0
        for i, (k, v) in enumerate(order):
            adj = min(1.0, max(prev, (3 - i) * v[idx]))
            prev = max(prev, adj)
            thr = 0.05 / (3 - i)
            print(f"  rank{i+1} {k}: p={v[idx]:.6f} vs {thr:.4f} "
                  f"-> adj_p={adj:.4f} "
                  f"{'<0.05' if adj < 0.05 else '>=0.05 未过'}")

    # ---- 次要 (不校正) ----
    print("\n===== 次要指标 =====")
    conds = ["direct_full", "bm25_top2", "bm25_top4", "sliding12k", "sdr_state"]
    for c in conds:
        f1s, ems, fails, calls = [], [], 0, []
        for r in recs.values():
            cc = r["conds"].get(c, {})
            f1s.append(cc.get("f1", 0.0))
            ems.append(cc.get("em", 0.0))
            raw = cc.get("raw") or []
            if not raw or all(not (x or "").strip() for x in raw):
                fails += 1
            if "n_calls" in cc:
                calls.append(cc["n_calls"])
        extra = ""
        if c.startswith("bm25"):
            gr = [r["conds"][c]["gold_recall"] for r in recs.values()
                  if c in r["conds"]]
            extra = f"  gold_recall={sum(gr)/len(gr):.3f}"
        cl = f"  calls={sum(calls)/len(calls):.1f}" if calls else ""
        print(f"  {c}: F1={sum(f1s)/n:.4f} EM={sum(ems)/n:.4f} "
              f"fail={fails}/{n}{extra}{cl}")


if __name__ == "__main__":
    main()
