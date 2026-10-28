# -*- coding: utf-8 -*-
"""
倒排索引物化缓存 + 按需查询 —— BFS 深层召回的根治方案 (v4)

v3 遗留弱点: BFS 深层(3-5层)单模型扫描召回略欠(集成才救回大半),
且 distill_bfs 每层用 node_ids_in(ch) 现场重扫所有片的文本 —— 同图 K 题就做 K 遍。

v4 三件套:
  A) 倒排索引物化缓存: node -> {片号}, 正则一次性建立, 落盘复用。
     同图所有问题共享(0 LLM 成本), parents 查询退化为 O(相关片) 直查。
  B) 按需单节点查询: 每批 frontier 节点 ≤8, 只查索引命中的片;
     T=0 一遍为主, 仅当"批内有节点却 0 条核验边"才自适应重扫(0.7, 可加集成)。
     —— 替代 v3 的全饱和扫描(0/0.7/1.0 × 逐模型), 调用数减半以上。
  C) 逐层召回诊断: 每层对照真值 frontier, 定位召回损失发生在哪一层
     —— 这是 v2/v3 一直缺的显微镜。

诚实边界: 索引(哪些片含节点X)与验证(该行是否逐字存在)仍是确定性基础设施;
模型的工作仍是"读 4K 片、列出以 F 中节点开头的边"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    _bfs_frontier, _parents, em_score, f1_score, ollama_chat,
    sample_from_target_chars,
)
from run_distill3 import (  # noqa: E402
    EXTRACT_SYSTEM, chunk_edges, parse_edges, verify_edge,
)

BATCH = 8          # v4: 每批更小, 集合匹配对 2B 更友好 (v3 是 12)


def log(msg):
    print(msg, flush=True)


def build_index(chunks):
    """双向倒排索引: 零 LLM 成本。
    left:  节点 -> 以它为左端点的片 (BFS 出边扩展用)
    right: 节点 -> 以它为右端点的片 (parents 目标定位用)"""
    left, right = {}, {}
    for i, ch in enumerate(chunks):
        for u in re.findall(r"^n\d+(?= ->)", ch, re.MULTILINE):
            left.setdefault(u, set()).add(i)
        for v in re.findall(r"-> (n\d+)$", ch, re.MULTILINE):
            right.setdefault(v, set()).add(i)
    return left, right


class QueryEngine:
    """按需查询: T=0 主扫 + 零产出自适应重扫 (可选集成补召回)。"""

    def __init__(self, model, ensemble, num_ctx, log):
        self.model = model
        self.ensemble = ensemble
        self.num_ctx = num_ctx
        self.log = log
        self.calls = 0

    def _call(self, model, task, chunk, t):
        self.calls += 1
        g = ollama_chat(model, EXTRACT_SYSTEM,
                        f"{task}\n\nPortion of the edge list:\n{chunk}",
                        num_ctx=self.num_ctx, think=False, temperature=t,
                        num_predict=2048)
        return g.content

    def query_batch(self, chunk, nodes):
        """返回 (核验通过的边集, 重扫与否)。nodes: 本批 frontier 节点。"""
        nodes_set = set(nodes)
        nodes_txt = ", ".join(sorted(nodes))
        task = (f"Frontier nodes F = {{{nodes_txt}}}.\n"
                f"List every edge u -> v in this portion with u in F. "
                f"Copy each line verbatim. If none matches, output NONE.")
        raw0 = self._call(self.model, task, chunk, 0.0)
        cands = set(parse_edges(raw0))
        kept = {e for e in cands if e[0] in nodes_set
                and verify_edge(e[0], e[1], chunk)}
        rescan = False
        if not kept and nodes:
            # 自适应重扫: 批内有节点却零核验产出, 疑似漏检
            rescan = True
            self.log(f"        [debug] T=0 零产出, raw={raw0[:150]!r}")
            raw1 = self._call(self.model, task, chunk, 0.7)
            cands |= set(parse_edges(raw1))
            if self.ensemble:
                raw2 = self._call(self.ensemble, task, chunk, 0.7)
                cands |= set(parse_edges(raw2))
            kept = {e for e in cands if e[0] in nodes_set
                    and verify_edge(e[0], e[1], chunk)}
            if not kept:
                self.log(f"        [debug] 重扫仍零产出, raw0.7={raw1[:150]!r}")
        return kept, rescan


def bfs_deep(qe: QueryEngine, chunks, index, adj, start, depth, log, prefix=""):
    """逐层: 索引定位 -> 小批量查询 -> 机械验证。返回 (答案, 逐层记录)。"""
    visited, frontier = {start}, {start}
    per_layer = []
    for d in range(1, depth + 1):
        truth_d = _bfs_frontier(adj, start, d)
        level_conf = set()
        n_calls_0 = qe.calls
        n_rescan = 0
        # 索引定位: frontier 节点 -> 片
        chunk_nodes = {}
        for u in frontier:
            for ci in index.get(u, ()):
                chunk_nodes.setdefault(ci, set()).add(u)
        n_rel = len(chunk_nodes)
        for ci in sorted(chunk_nodes):
            f_here = sorted(chunk_nodes[ci])
            for b0 in range(0, len(f_here), BATCH):
                grp = f_here[b0:b0 + BATCH]
                kept, rescan = qe.query_batch(chunks[ci], grp)
                n_rescan += rescan
                level_conf |= kept
        new = {v for _, v in level_conf if v not in visited}
        visited |= new
        frontier = new
        recall_d = (len(set(frontier) & set(truth_d)) / len(truth_d)
                    if truth_d else 1.0)
        per_layer.append({"level": d, "truth": len(truth_d),
                          "found": len(set(frontier) & set(truth_d)),
                          "frontier": len(frontier),
                          "recall": round(recall_d, 4),
                          "calls": qe.calls - n_calls_0, "rescans": n_rescan})
        log(f"      {prefix}层{d}: 相关片{n_rel}, 调用{qe.calls-n_calls_0}"
            f"(重扫{n_rescan}), frontier={len(frontier)}, "
            f"召回={recall_d:.3f} (真值层大小{len(truth_d)})")
        if not frontier:
            break
    return sorted(frontier), per_layer


def parents_quick(qe: QueryEngine, chunks, right_index, target, log):
    """parents: 反向索引直查相关片 -> 单遍查询 -> 机械验证。"""
    rel = sorted(right_index.get(target, ()))
    log(f"      反向索引: {len(rel)}/{len(chunks)} 片含目标 {target}")
    confirmed = set()
    n_calls_0 = qe.calls
    for ci in rel:
        task = (f"List every edge whose DESTINATION (right side) is {target}. "
                f"Copy the full line verbatim. If none matches, output NONE.")
        raw0 = qe._call(qe.model, task, chunks[ci], 0.0)
        cands = set(parse_edges(raw0))
        kept = {e for e in cands if e[1] == target
                and verify_edge(e[0], e[1], chunks[ci])}
        if not kept:
            cands |= set(parse_edges(qe._call(qe.model, task, chunks[ci], 0.7)))
            if qe.ensemble:
                cands |= set(parse_edges(
                    qe._call(qe.ensemble, task, chunks[ci], 0.7)))
            kept = {e for e in cands if e[1] == target
                    and verify_edge(e[0], e[1], chunks[ci])}
        confirmed |= kept
    ans = sorted({u for u, _ in confirmed})
    log(f"      parents: 调用{qe.calls-n_calls_0}, 核验{len(confirmed)}条边, "
        f"答案{len(ans)}节点")
    return ans, qe.calls - n_calls_0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="",
                    help="可选: 仅在零产出重扫时启用的补召回模型")
    ap.add_argument("--size", type=int, default=48000)
    ap.add_argument("--depths", default="4,5", help="两个深层BFS的深度")
    ap.add_argument("--with-parents", type=int, default=1)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"logs/bfs_deep_{args.tag}_2b.json"

    t0 = time.time()
    rng = random.Random(args.seed + args.size)
    depths = [int(d) for d in args.depths.split(",")]

    # ---- 同一张图: 一次生成, 多题复用 ----
    base = sample_from_target_chars(rng, args.size, op_type="bfs")
    adj, radj = {}, {}
    for u, v in base.edges:
        adj.setdefault(u, set()).add(v)
        radj.setdefault(v, set()).add(u)
    nodes_all = sorted({n for e in base.edges for n in e})
    chunks = chunk_edges(base, args.chunk_chars)
    log(f"### 图 {base.graph_chars} 字符, {len(base.edges)} 条边, "
        f"{len(chunks)} 片, ctx={args.num_ctx}, model={args.model}"
        f"{'+' + args.ensemble if args.ensemble else ''}")

    # ---- 倒排索引: 建一次, 全部题共用 (物化缓存) ----
    cache_path = os.path.join(os.path.dirname(out) or ".",
                              f"bfs_deep_index_{args.seed}_{args.size}.json")
    fp = hashlib.sha1(
        ("\n".join(f"{u} -> {v}" for u, v in base.edges)).encode()
    ).hexdigest()[:16]
    index = right_index = None
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        if cached.get("fingerprint") == fp and "right" in cached:
            index = {k: set(v) for k, v in cached["index"].items()}
            right_index = {k: set(v) for k, v in cached["right"].items()}
            log(f"  索引缓存命中: {cache_path}")
    if index is None:
        ts = time.time()
        index, right_index = build_index(chunks)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "chunks": len(chunks),
                       "index": {k: sorted(v) for k, v in index.items()},
                       "right": {k: sorted(v) for k, v in right_index.items()}},
                      f, ensure_ascii=False)
        log(f"  索引建立: 左{len(index)}/右{len(right_index)} 节点, "
            f"{round(time.time()-ts, 2)}s, 0 次 LLM 调用, 已物化 {cache_path}")
    log(f"  索引覆盖率(有出边的节点都在索引中): "
        f"{sum(1 for u in adj if u in index)}/{len(adj)}")

    qe = QueryEngine(args.model, args.ensemble, args.num_ctx, log)
    records = []

    # ---- 深层 BFS 题 (每档深度一题, 同图) ----
    for D in depths:
        # 拒绝采样: 找一个在该深度下答案规模合适的起点
        for _try in range(400):
            start = rng.choice(nodes_all)
            truth = _bfs_frontier(adj, start, D)
            if 5 <= len(truth) <= 60:
                break
        log(f"\n[Q] BFS depth={D} 从 {start}, 真值答案 {len(truth)} 节点")
        ts = time.time()
        ans, per_layer = bfs_deep(qe, chunks, index, adj, start, D, log)
        f1 = f1_score(ans, truth)
        em = em_score(ans, truth)
        log(f"  => BFS d{D}: F1={f1:.3f} EM={em} 找到{len(ans)}/{len(truth)} "
            f"调用{qe.calls}次 累计{round(time.time()-ts)}s")
        records.append({"op": "bfs", "depth": D, "start": start,
                        "truth_size": len(truth), "found": len(ans),
                        "f1": f1, "em": em, "calls": qe.calls,
                        "per_layer": per_layer,
                        "sec": round(time.time() - ts, 1)})

    # ---- parents 题 (同图, 索引直查) ----
    if args.with_parents:
        for _try in range(400):
            tgt = rng.choice(nodes_all)
            truth = _parents(radj, tgt)
            if 3 <= len(truth) <= 60:
                break
        log(f"\n[Q] parents 目标 {tgt}, 真值 {len(truth)} 节点")
        ts = time.time()
        ans, nc = parents_quick(qe, chunks, right_index, tgt, log)
        f1 = f1_score(ans, truth)
        log(f"  => parents: F1={f1:.3f} 找到{len(ans)}/{len(truth)} "
            f"调用{nc}次 {round(time.time()-ts)}s")
        records.append({"op": "parents", "target": tgt,
                        "truth_size": len(truth), "found": len(ans),
                        "f1": f1, "em": em_score(ans, truth), "calls": nc,
                        "sec": round(time.time() - ts, 1)})

    summary = {"model": args.model, "ensemble": args.ensemble or None,
               "size": args.size, "depths": depths,
               "chunk_chars": args.chunk_chars, "n_chunks": len(chunks),
               "total_calls": qe.calls,
               "calls_per_question": round(
                   qe.calls / max(1, len(records)), 1),
               "index_build_calls": 0,
               "elapsed_sec": round(time.time() - t0, 1)}
    for r in records:
        summary[f"{r['op']}{r.get('depth', '')}_f1"] = round(r["f1"], 4)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
