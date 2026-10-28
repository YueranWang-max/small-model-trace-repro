# -*- coding: utf-8 -*-
"""离线自检: 验证 GraphWalks 生成器、评分、序列化、prompt 构造是否正确。
结果写入 logs/selftest.log (utf-8), 同时打印到控制台。"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

_LOG = open(os.path.join(HERE, "logs", "selftest.log"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    try:
        print(s)
    except Exception:
        pass
    _LOG.write(s + "\n")
    _LOG.flush()


from repro_lib import (  # noqa: E402
    build_prompt, extract_list, f1_score, em_score, generate_sample,
    sample_from_target_chars, serialize_traces, _bfs_frontier, _parents,
)

fail = 0

# 1) 评分函数
assert f1_score(["a", "b"], ["a", "b"]) == 1.0
assert em_score(["a", "b"], ["b", "a"]) == 1.0
assert abs(f1_score(["a"], ["a", "b"]) - 0.6666667) < 1e-6
assert extract_list("blah\nFinal Answer: [n1, n2]")[0] == ["n1", "n2"]
assert extract_list("Final Answer: []")[0] == []
assert extract_list("no answer here")[1] is True
p("[ok] 评分与抽取")

# 2) 生成器 + 答案正确性(独立重算)
rng = random.Random(7)
for _ in range(200):
    s = generate_sample(rng, n_nodes=200, n_edges=400, max_depth=4)
    adj, radj = {}, {}
    for u, v in s.edges:
        adj.setdefault(u, set()).add(v)
        radj.setdefault(v, set()).add(u)
    if s.op_type == "bfs":
        node = s.op_text.split("node ")[1].split(" ")[0]
        depth = int(s.op_text.split("depth ")[1].strip("."))
        if _bfs_frontier(adj, node, depth) != s.answer:
            fail += 1
    else:
        node = s.op_text.split("node ")[1].strip(".")
        if _parents(radj, node) != s.answer:
            fail += 1
    deg = {n for u, v in s.edges for n in (u, v)}
    assert len(deg) == 200, f"孤立节点: {200 - len(deg)}"
p(f"[ok] 生成器 200 样本, 答案重算不一致={fail}")

# 3) 序列化 + prompt 构造
s = sample_from_target_chars(random.Random(1), 8000, op_type="bfs")
T = serialize_traces(["思考一...", "思考二..."])
assert T.startswith("Below are selected reasoning")
assert "[Trace 1]" in T and "</first_run_reasoning_traces>" in T
p_state = build_prompt(s, T, "state")
p_append = build_prompt(s, T, "append")
# few-shot 里也含这些字样, 故取"最后一次出现"(即真正的图/操作块)
edge_pos_s = p_state.rindex("The graph has the following edges")
op_pos_s = p_state.rindex("Operation:\n")
edge_pos_a = p_append.rindex("The graph has the following edges")
op_pos_a = p_append.rindex("Operation:\n")
ti_s = p_state.index("Trace 1")
ti_a = p_append.index("Trace 1")
# state: 轨迹在真正图块之前
assert ti_s < edge_pos_s, f"state 前置失败 ti={ti_s} edge={edge_pos_s}"
# append: 轨迹在真正图块之后、且在所有 Operation 块之前的边缘(此处仅要求在图之后)
assert ti_a > edge_pos_a, f"append 后置失败 ti={ti_a} edge={edge_pos_a}"
p(f"[ok] 序列化与摆放; 样本图字符={s.graph_chars}, state prompt≈{len(p_state)}字符")

# 4) 不同规模字符数 + 答案规模分布
for tgt in (4000, 8000, 16000, 64000, 128000):
    sizes, chars = [], []
    for k in range(60):
        s = sample_from_target_chars(random.Random(1000 + k), tgt,
                                     op_type="bfs" if k % 2 == 0 else "parents")
        sizes.append(len(s.answer))
        chars.append(s.graph_chars)
    avg_sz = sum(sizes) / len(sizes)
    p(f"    目标 {tgt:>6} -> 图字符均值 {sum(chars)//len(chars):>7}, "
      f"答案size 均值 {avg_sz:5.1f} min/max {min(sizes)}/{max(sizes)}")

p("\n全部自检通过 ✅" if fail == 0 else f"\n有 {fail} 个失败 ❌")
_LOG.close()
