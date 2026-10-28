# -*- coding: utf-8 -*-
"""
analyze_stageE —— Stage E 跨家族预注册分析 (protocol_stageDE.md §2)。
主比较 m=2: best_rag-eq, exq-eq; bootstrap CI 1e4 (seed 20260923);
MC sign-flip 1e6 (seed 20260923, add-one); Holm 步降。
"""
from __future__ import annotations

import json
import math
import os
import random

BASE = os.path.dirname(os.path.abspath(__file__))
BOOT_B = 10_000
BOOT_SEED = 20260923
MC = 1_000_000
MC_SEED = 20260923


def load(path):
    if not os.path.exists(path):
        return []
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    # 同 id 去重（双进程并发期可能重复追加同一题），保留最后一条
    uniq = {}
    for r in recs:
        uniq[r.get("id", f"_idx{len(uniq)}")] = r
    return list(uniq.values())


def boot_ci(diffs):
    rng = random.Random(BOOT_SEED)
    n = len(diffs)
    stats = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOT_B))
    return stats[int(0.025 * BOOT_B)], stats[int(0.975 * BOOT_B)]


def signflip_p(diffs):
    obs = sum(diffs) / len(diffs)
    rng = random.Random(MC_SEED)
    n = len(diffs)
    ge = le = 0
    for _ in range(MC):
        s = sum(diffs[i] if rng.random() < 0.5 else -diffs[i]
                for i in range(n))
        m = s / n
        if m >= obs - 1e-12:
            ge += 1
        if m <= obs + 1e-12:
            le += 1
    p = (min(le, ge) * 2 + 1) / (MC + 1)
    se = math.sqrt((min(le, ge) / MC) * (1 - min(le, ge) / MC) / MC)
    return obs, p, se


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    recs = load(os.path.join(BASE, "logs/stageE_llama.jsonl"))
    n = len(recs)
    print(f"Stage E (llama3.2:3b, shared qwen states) n={n}")
    if not n:
        print("无数据")
        return
    conds = ["direct_full", "eq", "exq", "bm25_top4"]
    means = {}
    for c in conds:
        v = [r["conds"][c]["f1"] for r in recs]
        em = sum(r["conds"][c]["em"] for r in recs) / n
        lat = sum(r["conds"][c].get("elapsed_sec", 0) for r in recs) / n
        means[c] = sum(v) / len(v)
        print(f"  {c}: F1={means[c]:.4f} EM={em:.4f} lat={lat:.1f}s")

    primaries = [("bm25_top4", "eq"), ("exq", "eq")]
    diffs = []
    for hi, lo in primaries:
        d = [r["conds"][hi]["f1"] - r["conds"][lo]["f1"] for r in recs]
        diffs.append(d)
    ps, cis = [], []
    for d in diffs:
        _, p, _ = signflip_p(d)
        ps.append(p)
        cis.append(boot_ci(d))
    adj = holm(ps)
    print("\n--- 预注册主比较 (m=2 Holm) ---")
    for (hi, lo), d, ci, p, a in zip(primaries, diffs, cis, ps, adj):
        obs = sum(d) / len(d)
        verdict = "PASS Holm<0.05" if a < 0.05 else "not significant"
        print(f"  {hi}-{lo}: delta={obs:+.4f} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"p={p:.6f} adj_p={a:.4f} -> {verdict}")

    # 分层 exploratory
    for h in sorted({r["hops"] for r in recs}):
        rs = [r for r in recs if r["hops"] == h]
        if len(rs) >= 5:
            d = [r["conds"]["exq"]["f1"] - r["conds"]["eq"]["f1"] for r in rs]
            print(f"  [exploratory] h{h} (n={len(rs)}): "
                  f"exq-eq={sum(d)/len(d):+.4f}")


if __name__ == "__main__":
    main()
