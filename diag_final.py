# -*- coding: utf-8 -*-
"""诊断: 2B 面对 ~77 条边的大状态, final-pass 到底输出什么?"""
import os, re, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import ollama_chat, sample_from_target_chars, GRAPH_PREAMBLE

rng = random.Random(11 + 16000)
s1 = sample_from_target_chars(rng, 16000, op_type="parents")
s3 = sample_from_target_chars(rng, 16000, op_type="bfs")

adj = {}
for u, v in s3.edges:
    adj.setdefault(u, []).append(v)

m = re.search(r"from node (n\d+) with depth (\d+)", s3.op_text)
start, depth = m.group(1), int(m.group(2))

# 真值 BFS 展开, 取每层"进入该层的树边"
visited, frontier = {start}, [start]
last_tree_edges = []
for d in range(1, depth + 1):
    nxt = []
    for u in frontier:
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v)
                nxt.append(v)
                last_tree_edges.append((u, v, d))
    frontier = nxt

edges_d = [(u, v) for u, v, d in last_tree_edges if d == depth]
state = ("Verified edges leading INTO depth {} of the BFS:\n".format(depth)
         + "\n".join(f"{u} -> {v}" for u, v in edges_d))
print(f"state: {len(edges_d)} 条边, {len(state)} 字符, 答案应为 {len(frontier)} 节点")

hint = ("The Final Answer is the set of DESTINATION (right-side) nodes "
        "of the verified edges above, deduplicated.")
user = (GRAPH_PREAMBLE + "\n\nYou did NOT see the full graph. A scan+verify "
        "pass extracted this verified evidence:\n" + state +
        "\n\n" + hint +
        "\n\nFull Operation:\n" + s3.op_text + "\n"
        "Return exactly one line: Final Answer: [node1, node2]. "
        "Use [] for the empty set.")

for think in (False, True):
    g = ollama_chat("openbmb/minicpm5-2b",
                    "You solve directed-graph algorithm problems.",
                    user, num_ctx=16384, think=think, temperature=0.7,
                    num_predict=4096)
    print(f"\n===== think={think} =====")
    print(f"content 长度={len(g.content)}  thinking 长度={len(g.thinking)}")
    print("content 最后 500 字:")
    print(repr(g.content[-500:]))
