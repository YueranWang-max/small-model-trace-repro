# -*- coding: utf-8 -*-
"""
Trace as State 本地缩放复现 —— 核心库
复现对象: Zou & Tang, "Trace as State: Reasoning Traces as Conditional States
for Long-Context Transformers" (arXiv:2609.02702)

本文件提供:
  1) GraphWalks 风格图任务的本地生成器 (可控规模, 可放入小模型上下文窗口)
  2) 官方 GraphWalks 答案抽取 + EM/F1 评分
  3) 论文附录 C 的轨迹序列化器 (T)
  4) [T, x, q] (trace as state) 与 [x, T, q] (trace append) 两种 prompt 构造
  5) Ollama Chat API 客户端 (含 thinking 轨迹读取)
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Tuple

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# ---------------------------------------------------------------------------
# 1. GraphWalks 风格 prompt (严格对齐 OpenAI GraphWalks 的 3-shot 模板)
# ---------------------------------------------------------------------------

GRAPH_PREAMBLE = (
    "You will be given a graph as a list of directed edges. All nodes are at least "
    "degree 1. You will also get a description of an operation to perform on the graph. "
    "Your job is to execute the operation on the graph and return the set of nodes that "
    "the operation results in. If asked for a breadth-first search (BFS), only return "
    "the nodes that are reachable at that depth, do not return the starting node. "
    "If asked for the parents of a node, only return the nodes that have an edge leading "
    "to the given node, do not return the given node itself."
)

FEWSHOT = (
    "Here is an example:\n"
    "The graph has the following edges:\n"
    "uvwx -> alke\n"
    "abcd -> uvwx\n"
    "abcd -> efgh\n"
    "efgh -> uvwx\n"
    "Example 1:\n"
    "Operation:\n"
    "Perform a BFS from node abcd with depth 1.\n"
    "Final Answer: [uvwx, efgh]\n"
    "Example 2:\n"
    "Operation:\n"
    "Perform a BFS from node alke with depth 1.\n"
    "Final Answer: []\n"
    "Example 3:\n"
    "Operation:\n"
    "Find the parents of node uvwx.\n"
    "Final Answer: [abcd, efgh]\n"
    "Example 4:\n"
    "Operation:\n"
    "Find the parents of node abcd.\n"
    "Final Answer: []\n"
)

# 论文附录 C 给出的 GraphWalks 系统提示
GRAPH_SYSTEM = (
    "You solve directed-graph algorithm problems. Use only the graph and operation in "
    "the user message. Return exactly one visible line in this format: "
    "Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text "
    "before or after that line."
)

# 答案格式指令 (附加到用户消息末尾)
ANSWER_FORMAT = (
    "Return exactly one line in this format: Final Answer: [node1, node2]. "
    "Use [] for the empty set."
)

# 论文附录 C: GraphWalks 数据集专属轨迹块前言 P_D
TRACE_PREAMBLE_GRAPH = (
    "Below are selected reasoning traces or trace tail windows from independent first "
    "attempts on the same graph problem. They may contain mistakes. Use them only as "
    "scratchpad hints, and verify against the graph."
)


@dataclass
class GraphSample:
    edges: List[Tuple[str, str]]
    op_type: str            # "bfs" | "parents"
    op_text: str            # 操作描述文本
    answer: List[str]       # 标准答案节点列表
    graph_chars: int        # 仅边列表字符数

    def edges_block(self) -> str:
        return "".join(f"{u} -> {v}\n" for u, v in self.edges)


def _bfs_frontier(adj, start, depth):
    """标准 BFS: 返回恰好位于 depth 层的节点(不含起点), 已访问节点不重复入frontier。"""
    visited = {start}
    frontier = [start]
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):  # 去重排序保证确定性
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            return []
    return sorted(set(frontier))


def _parents(radj, node):
    return sorted(set(radj.get(node, ())))


def generate_sample(rng: random.Random, n_nodes: int, n_edges: int,
                    op_type: str = None, max_depth: int = 4) -> GraphSample:
    """生成一张 [n_nodes] 个节点的随机有向图 + 一个 BFS/Parents 操作任务。"""
    nodes = [f"n{i:04d}" for i in range(n_nodes)]
    edges = set()
    # 保证每个节点至少 degree 1 (入或出); 重试直到成功, 避免孤立节点
    for nd in nodes:
        for _try in range(64):
            other = rng.choice(nodes)
            if other == nd:
                continue
            if rng.random() < 0.5:
                edges.add((nd, other))
            else:
                edges.add((other, nd))
            break
    # 补足边数
    guard = 0
    while len(edges) < n_edges and guard < n_edges * 40:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v:
            edges.add((u, v))
        guard += 1

    adj, radj = {}, {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    for k in radj:
        radj[k] = sorted(set(radj[k]))

    edge_list = sorted(edges)

    if op_type is None:
        op_type = rng.choice(["bfs", "parents"])

    if op_type == "bfs":
        for _ in range(400):
            start = rng.choice(nodes)
            depth = rng.randint(2, 6)
            ans = _bfs_frontier(adj, start, depth)
            if 4 <= len(ans) <= 60:
                op_text = f"Perform a BFS from node {start} with depth {depth}."
                return GraphSample(edge_list, "bfs", op_text, ans,
                                   sum(len(f"{u} -> {v}\n") for u, v in edge_list))
        # 兜底
        op_text = f"Perform a BFS from node {start} with depth {depth}."
        return GraphSample(edge_list, "bfs", op_text, ans,
                           sum(len(f"{u} -> {v}\n") for u, v in edge_list))
    else:
        # 偏向高入度节点, 让 Parents 答案集不至于过小(避免天花板效应)
        by_indeg = sorted(nodes, key=lambda x: len(radj.get(x, [])), reverse=True)
        pool = by_indeg[: max(5, len(by_indeg) // 4)] + by_indeg
        for tgt in pool:
            ans = _parents(radj, tgt)
            if 3 <= len(ans) <= 60:
                op_text = f"Find the parents of node {tgt}."
                return GraphSample(edge_list, "parents", op_text, ans,
                                   sum(len(f"{u} -> {v}\n") for u, v in edge_list))
        tgt = by_indeg[0]
        op_text = f"Find the parents of node {tgt}."
        return GraphSample(edge_list, "parents", op_text, _parents(radj, tgt),
                           sum(len(f"{u} -> {v}\n") for u, v in edge_list))


def sample_from_target_chars(rng, target_chars, op_type=None, max_depth=4,
                             density=3.0):
    """按目标字符数生成样本: 约 600 字符是 fewshot+preamble 的固定开销。
    density = 平均出度, 越大则答案集越大、任务越难。"""
    n_edges = max(80, int((target_chars - 600) / 15))
    n_nodes = max(30, int(n_edges / density))
    return generate_sample(rng, n_nodes, n_edges, op_type=op_type, max_depth=max_depth)


# ---------------------------------------------------------------------------
# 2. 官方 GraphWalks 抽取与评分
# ---------------------------------------------------------------------------

def extract_list(response: str) -> Tuple[List[str], bool]:
    """官方 get_list: 取最后一行, 解析 'Final Answer: [...]'。返回 (列表, 是否格式错误)。"""
    lines = [ln for ln in response.split("\n") if ln.strip() != ""]
    if not lines:
        return [], True
    line = lines[-1]
    if "Final Answer:" not in line:
        # 退化: 在全文里找最后一次出现
        m_all = re.findall(r"Final Answer: ?\[.*?\]", response)
        if not m_all:
            return [], True
        line = m_all[-1]
    m = re.search(r"Final Answer: ?\[.*\]", line)
    if m:
        result = m.group(0).replace("Final Answer:", "").strip().strip("[]").split(",")
        result = [i.strip() for i in result if i.strip()]
        return result, False
    return [], True


def f1_score(sampled: List[str], truth: List[str]) -> float:
    s, t = set(sampled), set(truth)
    n_overlap = len(s & t)
    recall = n_overlap / len(t) if t else 0.0
    precision = n_overlap / len(s) if s else 0.0
    if recall + precision > 0:
        return 2 * recall * precision / (recall + precision)
    return 1.0 if not t and not s else 0.0


def em_score(sampled: List[str], truth: List[str]) -> float:
    return 1.0 if set(sampled) == set(truth) else 0.0


def extract_answer(content: str, thinking: str = "") -> Tuple[List[str], bool]:
    """优先从 content 抽取; 若 content 无 'Final Answer:' 则回退到 thinking 文本。
    返回 (答案列表, 是否格式错误)。"""
    ans, bad = extract_list(content)
    if bad and thinking:
        ans2, bad2 = extract_list(thinking)
        if not bad2:
            return ans2, False
    return ans, bad


# ---------------------------------------------------------------------------
# 3. 轨迹序列化 (论文附录 C, Table 4)
# ---------------------------------------------------------------------------

def serialize_traces(traces: List[str], truncate_chars: int = 50000) -> str:
    """T = P_D + <first_run_reasoning_traces> [Trace i] r_i ... </...>"""
    parts = [TRACE_PREAMBLE_GRAPH, "<first_run_reasoning_traces>"]
    for i, r in enumerate(traces, 1):
        r = (r or "").strip()
        if len(r) > truncate_chars:
            r = r[:truncate_chars] + "..."
        parts.append(f"[Trace {i}] {r}")
    parts.append("</first_run_reasoning_traces>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 4. 两种摆放条件的 prompt 构造
# ---------------------------------------------------------------------------

def build_graph_x(sample: GraphSample) -> Tuple[str, str]:
    """返回 (x_head, x_op): 把 x 拆成'图指令+边列表'与'Operation 块'两段,
    以便实现论文中 [x, T] 的插入边界(插在最后 Operation 块之前)。"""
    x_head = (GRAPH_PREAMBLE + "\n" + FEWSHOT +
              "The graph has the following edges:\n" + sample.edges_block())
    x_op = "Operation:\n" + sample.op_text + "\n" + ANSWER_FORMAT
    return x_head, x_op


def build_prompt(sample: GraphSample, trace_block: str, placement: str) -> str:
    """placement: 'state' -> [T, x] ; 'append' -> [x, T] ; 'none' -> [x]"""
    x_head, x_op = build_graph_x(sample)
    if placement == "state":
        return trace_block + "\n" + x_head + "\n" + x_op
    if placement == "append":
        return x_head + "\n" + trace_block + "\n" + x_op
    return x_head + "\n" + x_op


# ---------------------------------------------------------------------------
# 5. Ollama 客户端
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    thinking: str = ""
    content: str = ""
    raw: dict = field(default_factory=dict)


def ollama_chat(model: str, system: str, user: str, num_ctx: int = 32768,
                think: bool = True, temperature: float = 0.7,
                num_predict: int = 2048, timeout: int = 900) -> GenResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict, "top_p": 0.9},
    }
    data = json.dumps(payload).encode("utf-8")
    # 强制直连: 绕过系统代理(Windows 注册表), 避免代理客户端未启动时 502 劫持
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_exc = None
    for attempt in range(6):
        req = urllib.request.Request(
            OLLAMA_URL, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            msg = obj.get("message", {}) or {}
            return GenResult(thinking=msg.get("thinking", "") or "",
                             content=msg.get("content", "") or "", raw=obj)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as exc:
            status = getattr(exc, "code", None)
            # 4xx (除 429) 是请求本身的问题, 重试无意义
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_exc = exc
            wait = [10, 30, 60, 120, 240][min(attempt, 4)]
            print(f"[ollama_chat] {type(exc).__name__}{'' if status is None else f' HTTP {status}'}"
                  f" 第{attempt + 1}次失败, {wait}s 后重试", flush=True)
            time.sleep(wait)
    raise last_exc
