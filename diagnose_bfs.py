# -*- coding: utf-8 -*-
"""BFS 失败根因诊断:
1) 还原 v2 实验里的 16K BFS 题 (seed 11, 第3次采样), 算真实层级结构
2) 蒙特卡洛: 每层"扫描召回率" r 对 frontier F1 的影响 —— 误差传播定量
"""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import generate_sample, sample_from_target_chars, f1_score

rng = random.Random(11 + 16000)
s1 = sample_from_target_chars(rng, 16000, op_type="parents")
s2 = sample_from_target_chars(rng, 16000, op_type="parents")
s3 = sample_from_target_chars(rng, 16000, op_type="bfs")   # v2 的第3题

adj = {}
for u, v in s3.edges:
    adj.setdefault(u, []).append(v)

import re
m = re.search(r"from node (n\d+) with depth (\d+)", s3.op_text)
start, depth = m.group(1), int(m.group(2))

# 真实层级
visited = {start}
frontier = [start]
levels = []
for d in range(1, depth + 1):
    nxt = []
    for u in frontier:
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v)
                nxt.append(v)
    frontier = nxt
    levels.append(sorted(set(frontier)))

print(f"题: {s3.op_text}  图 {s3.graph_chars} 字符, {len(s3.edges)} 边, "
      f"{len(set(u for u,_ in s3.edges))} 源节点")
for d, lv in enumerate(levels, 1):
    print(f"  层{d}: frontier={len(lv):>3}  (至此visited={len(visited)})")
print(f"  标准答案(层{depth}) = {len(levels[-1])} 节点")

# 每层需要模型从多少 chunk 里找多少条边 (v2: 全部 chunk × frontier 集合扫描)
n_chunks = 4  # 16K / 4K
total_edges_last = sum(len(adj.get(u, ())) for u in levels[-2]) if depth >= 2 else 0
print(f"\n v2 的层间扫描: 每层要模型在 {n_chunks} 片里对照 frontier 集合找边")
print(f"  层{depth} 的扫描任务要对照的 F 集合大小 = {len(levels[-2])} (前层frontier)")

# ---- 蒙特卡洛: 每层每条边以概率 r 被扫到, 看最终 frontier F1 ----
def simulate(r, trials=300):
    f1s = []
    for _ in range(trials):
        vis = {start}
        fr = [start]
        for d in range(1, depth + 1):
            nxt = []
            for u in fr:
                for v in adj.get(u, ()):
                    if v not in vis and random.random() < r:
                        vis.add(v)
                        nxt.append(v)
            fr = nxt
        got = sorted(set(fr))
        f1s.append(f1_score(got, s3.answer))
    return sum(f1s) / len(f1s)

print("\n蒙特卡洛 (每层每条边被扫到的概率 r -> 最终 frontier F1):")
for r in (1.0, 0.98, 0.95, 0.9, 0.8, 0.6):
    print(f"  r={r:.2f}  ->  期望 F1 = {simulate(r):.3f}")

# 反向: 假阳性 (幻觉边) 对 precision 的影响: 每层每节点以 q 概率多出一假边
def simulate_fp(r, q, trials=300):
    f1s = []
    all_nodes = sorted(adj.keys())
    for _ in range(trials):
        vis = {start}
        fr = [start]
        for d in range(1, depth + 1):
            nxt = []
            for u in fr:
                for v in adj.get(u, ()):
                    if v not in vis and random.random() < r:
                        vis.add(v)
                        nxt.append(v)
            # 幻觉: frontier 中每个节点以 q 概率凭空连向一个随机节点
            for u in fr:
                if random.random() < q:
                    v = random.choice(all_nodes)
                    if v not in vis:
                        vis.add(v)
                        nxt.append(v)
            fr = nxt
        got = sorted(set(fr))
        f1s.append(f1_score(got, s3.answer))
    return sum(f1s) / len(f1s)

print("\n加幻觉 (召回 r=0.95, 每节点每层 q 概率产生一条假边):")
for q in (0.0, 0.02, 0.05, 0.1):
    print(f"  q={q:.2f}  ->  期望 F1 = {simulate_fp(0.95, q):.3f}")
