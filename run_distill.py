# -*- coding: utf-8 -*-
"""
突破上下文窗口: 分片状态蒸馏 (Chunked State Distillation)

思想: 端侧模型 + 外部状态循环 = 无限上下文
  - 把大图切成 ~8K 字符/片, 逐片做"任务条件化蒸馏"(只提取与操作相关的证据边)
  - 汇成极小证据状态 S (通常 < 2K 字符, 与图总大小无关)
  - 最终一遍只读 [S, 操作] 作答; parents 题再做逐节点证据核查(客观评价, 非投票)
  - 全程 num_ctx=16384 恒定, 图可以任意大

条件:
  A. direct      : 整图塞进 16K 窗口 (48K 档额外测 32K 窗口, 看加倍窗口能否追平)
  B. distill     : 分片蒸馏 -> 证据 -> 最终作答
  C. cleaned     : B 的答案做证据核查, 无证据支撑的节点剔除 (parents 题)
  另记 evidence_f1: 纯扫描聚合的确定性答案 (衡量"扫描引擎"本身的精度)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    GRAPH_PREAMBLE, build_graph_x, em_score, f1_score, generate_sample,
    ollama_chat, sample_from_target_chars,
)

EXTRACT_SYSTEM = (
    "You extract exact edges from a portion of a directed edge list. "
    "Output ONLY matching edge lines in the form 'u -> v', one per line. "
    "If no edge matches, output exactly: NONE"
)


def chunk_edges(sample, chunk_chars):
    lines = [f"{u} -> {v}" for u, v in sample.edges]
    chunks, cur, size = [], [], 0
    for ln in lines:
        if size + len(ln) + 1 > chunk_chars and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def parse_edges(text):
    return [(u, v) for u, v in re.findall(r"(n\d+)\s*->\s*(n\d+)", text or "")]


def extract_call(model, user, num_ctx):
    g = ollama_chat(model, EXTRACT_SYSTEM, user, num_ctx=num_ctx,
                    think=False, temperature=0.0, num_predict=1024)
    return parse_edges(g.content), len(g.content)


# ---------------------------------------------------------------------------
# 分片蒸馏
# ---------------------------------------------------------------------------

def distill(model, sample, chunks, num_ctx, chunk_chars, log):
    n = len(chunks)
    ev_pairs = set()
    calls = 0
    sent_chars = 0

    if sample.op_type == "parents":
        m = re.search(r"node (n\d+)", sample.op_text)
        target = m.group(1)
        for i, ch in enumerate(chunks, 1):
            user = (f"Portion {i}/{n} of the edge list:\n{ch}\n"
                    f"List every edge whose DESTINATION (right side) is {target}.")
            found, outlen = extract_call(model, user, num_ctx)
            calls += 1
            sent_chars += len(user)
            good = {(u, v) for u, v in found if v == target}
            ev_pairs |= good
            if good:
                log(f"      片{i}/{n}: +{len(good)}条证据边")
        evidence = "\n".join(f"{u} -> {t}" for u, t in sorted(ev_pairs)) or "NONE"
        state = (f"Evidence edges found pointing to {target} "
                 f"(scanned {n} portions):\n{evidence}")
        det_answer = sorted({u for u, t in ev_pairs})
        return state, det_answer, calls, sent_chars, ev_pairs

    # BFS: 逐层扩展, 每层逐片扫描
    m = re.search(r"from node (n\d+) with depth (\d+)", sample.op_text)
    start, depth = m.group(1), int(m.group(2))
    visited, frontier = {start}, {start}
    last_edges = set()
    for d in range(1, depth + 1):
        level_pairs = set()
        flist = ", ".join(sorted(frontier))
        for i, ch in enumerate(chunks, 1):
            user = (f"Portion {i}/{n} of the edge list:\n{ch}\n"
                    f"Frontier set F = {{{flist}}}.\n"
                    f"List every edge u -> v in this portion with u in F.")
            found, outlen = extract_call(model, user, num_ctx)
            calls += 1
            sent_chars += len(user)
            level_pairs |= {(u, v) for u, v in found if u in frontier}
        new = {v for u, v in level_pairs if v not in visited}
        visited |= new
        last_edges = level_pairs
        frontier = new
        log(f"      层{d}: +{len(level_pairs)}条边, 新frontier={len(new)}")
        if not frontier:
            break
    edges_txt = "\n".join(f"{u} -> {v}" for u, v in sorted(last_edges)) or "NONE"
    state = (f"Edges found leading INTO depth {depth} of the BFS "
             f"(scanned {n} portions per level):\n{edges_txt}")
    det_answer = sorted(frontier)
    return state, det_answer, calls, sent_chars, ev_pairs


def final_pass(model, sample, state, num_ctx, max_predict):
    user = (GRAPH_PREAMBLE + "\n\nYou did NOT see the full graph. Instead, a "
            "scanning pass extracted the following operation-relevant evidence "
            "from it:\n" + state +
            "\n\nFull Operation:\n" + sample.op_text + "\n" +
            "Return exactly one line: Final Answer: [node1, node2]. "
            "Use [] for the empty set.")
    g = ollama_chat(model, "You solve directed-graph algorithm problems.",
                    user, num_ctx=num_ctx, think=True, temperature=0.7,
                    num_predict=max_predict)
    from repro_lib import extract_answer
    ans, bad = extract_answer(g.content, g.thinking)
    return ans, len(user), g


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--sizes", default="16000,48000,128000")
    ap.add_argument("--questions-per-size", type=int, default=4)
    ap.add_argument("--chunk-chars", type=int, default=8000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="logs/distill_2b.json")
    args = ap.parse_args()

    def log(msg, **kwargs):
        print(msg, flush=True)

    sizes = [int(s) for s in args.sizes.split(",")]
    records = []
    t0 = time.time()

    for size in sizes:
        rng = random.Random(args.seed + size)
        log(f"\n{'='*60}\n### 图规模 ≈{size} 字符 (num_ctx 恒为 {args.num_ctx})")
        for qi in range(args.questions_per_size):
            op = "parents" if qi % 2 == 0 else "bfs"
            sample = sample_from_target_chars(rng, size, op_type=op)
            chunks = chunk_edges(sample, args.chunk_chars)
            log(f"\n[{size} | {qi+1}/{args.questions_per_size}] {op} "
                f"图实际{sample.graph_chars}字符 {len(sample.edges)}条边 "
                f"truth_size={len(sample.answer)} 分{len(chunks)}片", flush=True)

            rec = {"op_type": op, "graph_chars": sample.graph_chars,
                   "n_edges": len(sample.edges), "n_chunks": len(chunks),
                   "answer": sample.answer}

            # ---- A. direct 基线 ----
            x_head, x_op = build_graph_x(sample)
            x = x_head + "\n" + x_op
            direct_ctxs = [args.num_ctx]
            if size == 48000:
                direct_ctxs.append(32768)  # 加倍窗口基线
            for dctx in direct_ctxs:
                tag = f"direct@{dctx//1024}K"
                try:
                    g = ollama_chat(args.model,
                                    "You solve directed-graph algorithm problems.",
                                    x, num_ctx=dctx, think=True, temperature=0.7,
                                    num_predict=args.max_predict)
                    from repro_lib import extract_answer
                    ans, _ = extract_answer(g.content, g.thinking)
                    rec[tag] = {"f1": f1_score(ans, sample.answer),
                                "em": em_score(ans, sample.answer)}
                except Exception as e:
                    rec[tag] = {"f1": 0.0, "em": 0.0, "error": str(e)[:80]}
                log(f"      {tag}: f1={rec[tag]['f1']:.3f}", flush=True)

            # ---- B. 分片蒸馏 ----
            ts = time.time()
            state, det_answer, ncalls, sent, ev_pairs = distill(
                args.model, sample, chunks, args.num_ctx, args.chunk_chars, log)
            rec["evidence"] = {"f1": f1_score(det_answer, sample.answer),
                               "em": em_score(det_answer, sample.answer),
                               "n_evidence": len(ev_pairs)}
            ans, user_chars, g = final_pass(args.model, sample, state,
                                            args.num_ctx, args.max_predict)
            rec["distill"] = {"f1": f1_score(ans, sample.answer),
                              "em": em_score(ans, sample.answer),
                              "raw_answer": ans}
            rec["cost"] = {"extract_calls": ncalls, "sent_chars": sent + user_chars,
                           "state_chars": len(state),
                           "sec": round(time.time() - ts, 1)}
            log(f"      evidence_f1={rec['evidence']['f1']:.3f} "
                f"distill_f1={rec['distill']['f1']:.3f} "
                f"(state {len(state)}字符, {ncalls}次扫描+1次作答)", flush=True)

            # ---- C. 证据核查清洗 (parents) ----
            if op == "parents":
                m = re.search(r"node (n\d+)", sample.op_text)
                target = m.group(1)
                kept = [u for u in ans if (u, target) in ev_pairs]
                rec["cleaned"] = {"f1": f1_score(kept, sample.answer),
                                  "em": em_score(kept, sample.answer),
                                  "dropped": sorted(set(ans) - set(kept))}
                log(f"      cleaned_f1={rec['cleaned']['f1']:.3f} "
                    f"(剔除{len(rec['cleaned']['dropped'])}个无证据节点)", flush=True)

            records.append(rec)

    # ---- 汇总 ----
    summary = {"model": args.model, "sizes": sizes,
               "q_per_size": args.questions_per_size,
               "chunk_chars": args.chunk_chars, "num_ctx": args.num_ctx,
               "elapsed_sec": round(time.time() - t0, 1)}
    for size in sizes:
        rs = [r for r in records if abs(r["graph_chars"] - size) < size * 0.5]
        if not rs:
            continue
        conds = [k for k in rs[0].keys()
                 if k in ("direct@16K", "direct@32K", "evidence", "distill",
                          "cleaned")]
        for cond in conds:
            vals = [r[cond]["f1"] for r in rs if cond in r]
            ems = [r[cond]["em"] for r in rs if cond in r]
            summary[f"{size}_{cond}_f1"] = round(sum(vals) / len(vals), 4)
            summary[f"{size}_{cond}_em"] = round(sum(ems) / len(ems), 4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)

    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
