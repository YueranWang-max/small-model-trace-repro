# -*- coding: utf-8 -*-
"""Stage B 精确配对随机化检验（符号翻转）——ChatGPT 三轮审查 critical#2 复算。
统计量=逐题配对差均值；零对不翻转；C2-C0 全枚举 2^15，其余 Monte Carlo 1e6。"""
import os
import json, random, itertools, math

recs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "budget_matched_stageB.json"), encoding="utf-8"))["records"]
B_MC = 1_000_000
rng = random.Random(20260920)

def exact_test(a, b):
    d = [r[a]["f1"] - r[b]["f1"] for r in recs]
    n = len(d)
    obs = sum(d) / n
    nz = [x for x in d if x != 0]
    k = len(nz)
    obs_stat = abs(obs)
    if k <= 20:  # 全枚举
        total, ge = 2 ** k, 0
        for mask in range(total):
            s = 0.0
            for j in range(k):
                s += nz[j] if (mask >> j) & 1 else -nz[j]
            if abs(s) / n >= obs_stat - 1e-12:
                ge += 1
        p_two = ge / total  # 完全精确枚举：全部状态可见，无需 add-one 平滑
        method = f"exact enumeration 2^{k}"
    else:  # Monte Carlo 符号翻转
        total, ge = B_MC, 0
        an = [abs(x) for x in nz]
        for _ in range(B_MC):
            s = sum(sign * v for sign, v in zip((rng.random() < .5 and 1 or -1 for _ in range(k)), an))
            if abs(s) / n >= obs_stat - 1e-12:
                ge += 1
        p_two = (ge + 1) / (B_MC + 1)
        method = f"MC sign-flip {B_MC:,} (2^{k} 不可枚举)"
    # 单侧（预注册方向）：C2-C0<0 取左尾，C3-C0>0 取右尾，C6-C3 双侧
    return obs, k, p_two, method

def one_sided(a, b, direction):
    d = [r[a]["f1"] - r[b]["f1"] for r in recs]
    n = len(d); obs = sum(d) / n
    nz = [x for x in d if x != 0]; k = len(nz)
    obs_stat = abs(obs)
    if k <= 20:
        total, ge = 2 ** k, 0
        for mask in range(total):
            s = sum(nz[j] if (mask >> j) & 1 else -nz[j] for j in range(k))
            if (s / n <= -obs_stat + 1e-12) if direction == "neg" else (s / n >= obs_stat - 1e-12):
                ge += 1
        return ge / total  # 完全精确枚举：无 add-one
    an = [abs(x) for x in nz]; ge = 0
    for _ in range(B_MC):
        s = sum(sign * v for sign, v in zip((rng.random() < .5 and 1 or -1 for _ in range(k)), an))
        if (s / n <= -obs_stat + 1e-12) if direction == "neg" else (s / n >= obs_stat - 1e-12):
            ge += 1
    return (ge + 1) / (B_MC + 1)

print("=" * 70)
print("精确配对随机化检验（符号翻转），统计量=配对差均值")
res2 = {}
for (a, b, dirn) in [("C2", "C0", "neg"), ("C3", "C0", "pos"), ("C6", "C3", None)]:
    obs, k, p2, method = exact_test(a, b)
    p1 = one_sided(a, b, dirn) if dirn else p2
    res2[f"{a}-{b}"] = (p2, p1)
    print(f"\n{a}-{b}: delta={obs:+.4f}  非零对={k}")
    print(f"  {method}")
    print(f"  双侧 p={p2:.6f}   单侧(预注册方向) p={p1:.6f}")

print("\n" + "=" * 70)
print("Holm 步降 (m=3, alpha=0.05)")
for label, idx in [("双侧", 0), ("单侧(方向性)", 1)]:
    order = sorted(res2.items(), key=lambda kv: kv[1][idx])
    print(f"\n  [{label}]")
    prev = 0.0
    for i, (k, (p2, p1)) in enumerate(order):
        p = (p2, p1)[idx]
        adj = min(1.0, max(prev, (3 - i) * p)); prev = max(prev, adj)
        thresh = 0.05 / (3 - i)
        print(f"    rank{i+1} {k}: p={p:.6f} vs {thresh:.4f} -> adj_p={adj:.4f} "
              f"{'<0.05 显著' if adj < 0.05 else '>=0.05 不显著'}")
