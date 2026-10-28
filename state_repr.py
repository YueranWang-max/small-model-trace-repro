# -*- coding: utf-8 -*-
"""
state_repr —— Budget-Matched Verified State 的状态构造。
所有解析/核验/压缩均为确定性程序(正则+集合成员), 零 LLM 调用。
协议见 protocol_budget_matched.md。
"""
import re
from typing import Dict, List, Tuple

# 首遍 prompt 尾巴: 要求在 Final Answer 之后输出 Evidence Ledger
LEDGER_SUFFIX = (
    "\nAfter the Final Answer line, output a state ledger between these exact "
    "markers:\n"
    "[STATE]\n"
    "EVIDENCE: <one verbatim edge per line, copied exactly from the graph, "
    "e.g. EVIDENCE: n0012 -> n0087>\n"
    "DERIVED: <one intermediate conclusion per line, each ending with "
    "(from: <comma-separated EVIDENCE line numbers>)>\n"
    "UNCERTAIN: <one unresolved item per line; write UNCERTAIN: NONE if none>\n"
    "[/STATE]\n"
    "EVIDENCE lines are implicitly numbered 1,2,3,... in order of appearance. "
    "Copy edges exactly as they appear in the graph; do not invent edges."
)

_STATE_RE = re.compile(r"\[STATE\](.*?)\[/STATE\]", re.S)
_EV_RE = re.compile(r"^\s*EVIDENCE:\s*(.+?)\s*$", re.M)
_DER_RE = re.compile(r"^\s*DERIVED:\s*(.+?)\s*$", re.M)
_UNC_RE = re.compile(r"^\s*UNCERTAIN:\s*(.+?)\s*$", re.M)
_REFS_RE = re.compile(r"\(from:\s*([0-9,\s]+)\)")


def parse_ledger(content: str) -> Dict:
    """从首遍 content 提取 ledger。返回:
    evidence: List[str]  (原文边串, 如 'n0012 -> n0087')
    derived:  List[Tuple[str, List[int]]]  (文本, 1-based EVIDENCE 编号)
    uncertain: List[str]
    has_ledger: bool
    """
    out = {"evidence": [], "derived": [], "uncertain": [],
           "has_ledger": False}
    text = content or ""
    m = _STATE_RE.search(text)
    if not m:
        # 宽容: 模型偶尔漏写闭合标记, 取 [STATE] 到文末
        i = text.find("[STATE]")
        if i < 0:
            return out
        body = text[i + len("[STATE]"):]
    else:
        body = m.group(1)
    out["has_ledger"] = True
    out["evidence"] = [x.strip() for x in _EV_RE.findall(body)]
    for d in _DER_RE.findall(body):
        rm = _REFS_RE.search(d)
        refs = []
        if rm:
            refs = [int(x) for x in rm.group(1).split(",") if x.strip().isdigit()]
        out["derived"].append((d.strip(), refs))
    out["uncertain"] = [u.strip() for u in _UNC_RE.findall(body)
                        if u.strip().upper() != "NONE"]
    return out


def _norm_edge(s: str) -> Tuple[str, str]:
    parts = re.split(r"->|→|;", s)
    if len(parts) != 2:
        return ("", "")
    return (parts[0].strip(), parts[1].strip())


def verify_ledger(led: Dict, edges: List[Tuple[str, str]]) -> Dict:
    """核验: EVIDENCE 必须逐字(规范化空白)存在于图边集;
    DERIVED/UNCERTAIN 仅当引用的 EVIDENCE 编号全部保留才保留。
    返回同结构 dict, 另含 n_ev_total/n_ev_kept。"""
    edge_set = {f"{u} -> {v}" for u, v in edges}
    ev = led["evidence"]
    kept_idx = [i for i, e in enumerate(ev) if _norm_edge(e) in
                {_norm_edge(x) for x in edge_set}]
    kept_set = set(kept_idx)
    new_der = [(t, r) for t, r in led["derived"]
               if r and all((x - 1) in kept_set for x in r)]
    new_unc = []
    for u in led["uncertain"]:
        rm = _REFS_RE.search(u)
        if rm:
            refs = [int(x) for x in rm.group(1).split(",") if x.strip().isdigit()]
            if all((x - 1) in kept_set for x in refs):
                new_unc.append(u)
        else:
            new_unc.append(u)
    return {"evidence": [ev[i] for i in kept_idx],
            "derived": new_der, "uncertain": new_unc,
            "has_ledger": led["has_ledger"],
            "n_ev_total": len(ev), "n_ev_kept": len(kept_idx)}


def format_ledger(ev: List[str], der: List[Tuple[str, List[int]]],
                  unc: List[str]) -> str:
    lines = ["[STATE]"]
    for e in ev:
        lines.append(f"EVIDENCE: {e}")
    for t, _ in der:
        lines.append(f"DERIVED: {t}")
    for u in unc:
        lines.append(f"UNCERTAIN: {u}")
    lines.append("[/STATE]")
    return "\n".join(lines)


def fit_budget(text: str, budget_chars: int) -> str:
    """按行粒度压缩到 budget_chars 内。保留优先级:
    EVIDENCE > DERIVED > UNCERTAIN; 超预算时先丢 UNCERTAIN, 再 DERIVED。
    返回 (text, dropped_kind) —— dropped_kind 记录被整类丢弃的最高优先级行。"""
    lines = text.splitlines()
    header = [l for l in lines if l.strip() in ("[STATE]", "[/STATE]")]
    ev = [l for l in lines if l.startswith("EVIDENCE:")]
    der = [l for l in lines if l.startswith("DERIVED:")]
    unc = [l for l in lines if l.startswith("UNCERTAIN:")]
    fixed = sum(len(l) + 1 for l in header)
    dropped = ""
    while fixed + sum(len(l) + 1 for l in ev + der + unc) > budget_chars:
        if unc:
            unc = []
            dropped = "UNCERTAIN"
        elif der:
            der = []
            dropped = "DERIVED"
        elif ev:
            ev = ev[:-1]
            dropped = "EVIDENCE"
        else:
            break
    out = header[:1] + ev + der + unc + header[1:]
    return "\n".join(out), dropped


def build_oracle(sample) -> str:
    """C8: 机械求解的 oracle 状态 (upper bound)。
    BFS: EVIDENCE=BFS 生成树边, DERIVED=frontier 结论;
    parents: EVIDENCE=指向目标的父边, DERIVED=parents 结论。"""
    import sys
    sys.path.insert(0, __file__.rsplit("\\", 1)[0])
    from repro_lib import _bfs_frontier, _parents
    adj, radj = {}, {}
    for u, v in sample.edges:
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)
    if sample.op_type == "parents":
        target = re.search(r"parents of node (\w+)", sample.op_text)
        tnode = target.group(1) if target else sample.op_text.split()[-1].rstrip(".")
        ev = [f"{u} -> {tnode}" for u in _parents(radj, tnode)]
        der = [(f"parents of {tnode} are heads of the EVIDENCE edges "
                f"(from: {','.join(str(i+1) for i in range(len(ev)))})",
                list(range(1, len(ev) + 1)))] if ev else []
    else:
        start = re.search(r"node (\w+)", sample.op_text).group(1)
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        # 逐层扩展记录树边
        visited = {start}
        frontier = [start]
        tree_edges = []
        for _ in range(depth):
            nxt = []
            for u in frontier:
                for v in adj.get(u, ()):
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
                        tree_edges.append(f"{u} -> {v}")
            frontier = nxt
            if not frontier:
                break
        ev = sorted(set(tree_edges))
        idx = {e: i + 1 for i, e in enumerate(ev)}
        # 只保留指向最后一层的 DERIVED 引用
        last = set(_bfs_frontier(adj, start, depth))
        tail = [e for e in ev if e.split(" -> ")[1] in last]
        der = [(f"nodes at depth {depth} are the tails of the final-layer "
                f"EVIDENCE edges (from: {','.join(str(idx[e]) for e in tail)})",
                [idx[e] for e in tail])] if tail else []
    return format_ledger(ev, der, [])
