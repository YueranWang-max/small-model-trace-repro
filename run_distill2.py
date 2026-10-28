# -*- coding: utf-8 -*-
"""
分片状态蒸馏 v2 —— 双通道证据确认 (客观评价机制, 非投票)

v1 的两个失败模式:
  1) parents 扫描幻觉爆发: 单片冒出 128 条假证据边 (真值才 11 条)
  2) BFS frontier 复合膨胀: 层间假阳性滚雪球, evidence_f1 掉到 0.06

v2 修复:
  - 切片 8K -> 4K
  - 双通道: 第一遍"扫描"只产候选, 第二遍对同一片做独立"回读确认",
            候选边必须逐字再现才采信 (证据 ∩ 确认, 与多数投票无关)
  - BFS 每层扩展都走 扫描->确认 两通道, frontier 只从确认过的边生长
  - 输出指令强调 "逐字复制, 不确定不列"
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
    GRAPH_PREAMBLE, build_graph_x, em_score, extract_answer, f1_score,
    generate_sample, ollama_chat, sample_from_target_chars,
)

EXTRACT_SYSTEM = (
    "You extract exact edges from a portion of a directed edge list. "
    "Copy matching lines VERBATIM from the portion, one per line. "
    "Do not invent edges. If no edge matches, output exactly: NONE"
)
CONFIRM_SYSTEM = (
    "You verify whether given edges literally appear in an edge-list portion. "
    "Output ONLY the claimed edges that appear verbatim in the portion, "
    "one per line. If none appear, output exactly: NONE"
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


def call_extract(model, user, num_ctx, temperature=0.0):
    g = ollama_chat(model, EXTRACT_SYSTEM, user, num_ctx=num_ctx,
                    think=False, temperature=temperature, num_predict=1024)
    return parse_edges(g.content)


def call_confirm(model, user, num_ctx):
    g = ollama_chat(model, CONFIRM_SYSTEM, user, num_ctx=num_ctx,
                    think=False, temperature=0.0, num_predict=1024)
    return parse_edges(g.content)


def scan_chunk(model, chunk, idx, n, task_line, num_ctx, passes=1):
    """扫描: 默认温度0一遍; BFS 用 passes=2 (0.0+0.7) 取并集提召回。"""
    user = f"Portion {idx}/{n} of the edge list:\n{chunk}\n{task_line}"
    out = set(call_extract(model, user, num_ctx, temperature=0.0))
    if passes > 1:
        out |= set(call_extract(model, user, num_ctx, temperature=0.7))
    return out


def confirm_chunk(model, chunk, idx, n, candidates, num_ctx, batch=400):
    if not candidates:
        return set()
    cands = sorted(candidates)
    kept = set()
    for b0 in range(0, len(cands), batch):
        part = cands[b0:b0 + batch]
        cand_txt = "\n".join(f"{u} -> {v}" for u, v in part)
        user = (f"Portion {idx}/{n} of the edge list:\n{chunk}\n"
                f"Claimed edges to verify:\n{cand_txt}\n"
                f"Output only the claimed edges that appear verbatim above.")
        try:
            kept |= set(call_confirm(model, user, num_ctx))
        except Exception:
            continue  # 确认失败按"未确认"处理, 宁缺勿假
    return kept & set(candidates)


# ---------------------------------------------------------------------------
# parents: 扫描 -> 逐片确认 -> 交集
# ---------------------------------------------------------------------------

def distill_parents(model, sample, chunks, num_ctx, log):
    n = len(chunks)
    m = re.search(r"node (n\d+)", sample.op_text)
    target = m.group(1)
    task = f"List every edge whose DESTINATION (right side) is {target}."
    confirmed_pairs = set()
    n_scan_cand = 0
    for i, ch in enumerate(chunks, 1):
        cand = set(scan_chunk(model, ch, i, n, task, num_ctx))
        cand = {(u, v) for u, v in cand if v == target}
        n_scan_cand += len(cand)
        kept = confirm_chunk(model, ch, i, n, cand, num_ctx)
        confirmed_pairs |= kept
        if cand or kept:
            log(f"      片{i}/{n}: 候选{len(cand)} -> 确认{len(kept)}")
    evidence = "\n".join(f"{u} -> {t}" for u, t in sorted(confirmed_pairs)) or "NONE"
    state = (f"Confirmed evidence edges pointing to {target} "
             f"(scan+confirm over {n} portions):\n{evidence}")
    det = sorted({u for u, t in confirmed_pairs})
    return state, det, confirmed_pairs, n_scan_cand


# ---------------------------------------------------------------------------
# bfs: 每层 扫描 -> 确认 -> 新frontier
# ---------------------------------------------------------------------------

def distill_bfs(model, sample, chunks, num_ctx, log):
    n = len(chunks)
    m = re.search(r"from node (n\d+) with depth (\d+)", sample.op_text)
    start, depth = m.group(1), int(m.group(2))
    visited, frontier = {start}, {start}
    last_confirmed = set()
    total_cand = 0
    for d in range(1, depth + 1):
        level_cand = set()
        flist = ", ".join(sorted(frontier))
        task = (f"Frontier set F = {{{flist}}}.\n"
                f"List every edge u -> v in this portion with u in F.")
        for i, ch in enumerate(chunks, 1):
            cand = {(u, v) for u, v in
                    scan_chunk(model, ch, i, n, task, num_ctx, passes=2)
                    if u in frontier}
            level_cand |= cand
        # 确认: 按片回读 (候选按来源片分组, 简化为全量对全片)
        level_conf = set()
        for i, ch in enumerate(chunks, 1):
            level_conf |= confirm_chunk(model, ch, i, n, level_cand, num_ctx)
        level_conf = {(u, v) for u, v in level_conf if u in frontier}
        new = {v for u, v in level_conf if v not in visited}
        visited |= new
        last_confirmed = level_conf
        frontier = new
        total_cand += len(level_cand)
        log(f"      层{d}: 候选{len(level_cand)}条 -> 确认{len(level_conf)}条, "
            f"新frontier={len(new)}")
        if not frontier:
            break
    edges_txt = "\n".join(f"{u} -> {v}"
                          for u, v in sorted(last_confirmed)) or "NONE"
    state = (f"Confirmed edges leading INTO depth {depth} of the BFS:\n{edges_txt}")
    det = sorted(frontier)
    return state, det, last_confirmed, total_cand


def final_pass(model, sample, state, num_ctx, max_predict):
    user = (GRAPH_PREAMBLE + "\n\nYou did NOT see the full graph. A scan+confirm "
            "pass extracted this verified evidence:\n" + state +
            "\n\nFull Operation:\n" + sample.op_text + "\n"
            "Return exactly one line: Final Answer: [node1, node2]. "
            "Use [] for the empty set.")
    g = ollama_chat(model, "You solve directed-graph algorithm problems.",
                    user, num_ctx=num_ctx, think=True, temperature=0.7,
                    num_predict=max_predict)
    ans, _ = extract_answer(g.content, g.thinking)
    return ans, len(user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--sizes", default="16000,48000,128000")
    ap.add_argument("--parents-per-size", type=int, default=2)
    ap.add_argument("--bfs-per-size", type=int, default=1)
    ap.add_argument("--bfs-max-size", type=int, default=20000,
                    help="BFS 层级展开调用数 O(层×片×frontier), 超过此图规模跳过")
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="logs/distill2_2b.json")
    args = ap.parse_args()

    def log(msg, **kw):
        print(msg, flush=True)

    sizes = [int(s) for s in args.sizes.split(",")]
    records, t0 = [], time.time()

    for size in sizes:
        rng = random.Random(args.seed + size)
        log(f"\n{'='*60}\n### 图规模 ≈{size} 字符 (chunk={args.chunk_chars}, "
            f"num_ctx 恒 {args.num_ctx})")
        for qi in range(args.parents_per_size + args.bfs_per_size):
            op = "parents" if qi < args.parents_per_size else "bfs"
            if op == "bfs" and size > args.bfs_max_size:
                log(f"[{size}] bfs 跳过 (规模>{args.bfs_max_size}, "
                    f"层级展开成本失控——见报告的方法边界分析)")
                records.append({"op_type": "bfs", "graph_chars": size,
                                "skipped": True})
                continue
            sample = sample_from_target_chars(rng, size, op_type=op)
            chunks = chunk_edges(sample, args.chunk_chars)
            log(f"\n[{size}] {op} 图{sample.graph_chars}字符 "
                f"truth_size={len(sample.answer)} 分{len(chunks)}片", flush=True)
            rec = {"op_type": op, "graph_chars": sample.graph_chars,
                   "n_chunks": len(chunks), "answer": sample.answer}

            # direct 基线
            x_head, x_op = build_graph_x(sample)
            try:
                g = ollama_chat(args.model,
                                "You solve directed-graph algorithm problems.",
                                x_head + "\n" + x_op, num_ctx=args.num_ctx,
                                think=True, temperature=0.7,
                                num_predict=args.max_predict)
                ans, _ = extract_answer(g.content, g.thinking)
                rec["direct"] = {"f1": f1_score(ans, sample.answer),
                                 "em": em_score(ans, sample.answer)}
            except Exception as e:
                rec["direct"] = {"f1": 0.0, "em": 0.0, "error": str(e)[:60]}
            log(f"      direct: f1={rec['direct']['f1']:.3f}")

            # 蒸馏 (双通道)
            ts = time.time()
            try:
                if op == "parents":
                    state, det, ev, n_cand = distill_parents(
                        args.model, sample, chunks, args.num_ctx, log)
                else:
                    state, det, ev, n_cand = distill_bfs(
                        args.model, sample, chunks, args.num_ctx, log)
            except Exception as e:
                log(f"      !! 蒸馏失败, 跳过该题: {e}")
                rec["distill_error"] = str(e)[:80]
                records.append(rec)
                continue
            rec["evidence"] = {"f1": f1_score(det, sample.answer),
                               "em": em_score(det, sample.answer),
                               "n_confirmed": len(ev), "n_candidates": n_cand}
            ans, uchars = final_pass(args.model, sample, state,
                                     args.num_ctx, args.max_predict)
            rec["distill"] = {"f1": f1_score(ans, sample.answer),
                              "em": em_score(ans, sample.answer)}
            rec["cost"] = {"state_chars": len(state), "sec": round(time.time() - ts, 1)}
            log(f"      evidence_f1={rec['evidence']['f1']:.3f} "
                f"distill_f1={rec['distill']['f1']:.3f} "
                f"(state {len(state)}字符, {round(time.time()-ts)}s)")

            records.append(rec)

    # 先落盘再汇总 —— 任何汇总错误都不再丢数据
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": {}, "records": records}, f,
                  ensure_ascii=False, indent=2)

    summary = {"model": args.model, "sizes": sizes,
               "chunk_chars": args.chunk_chars, "num_ctx": args.num_ctx,
               "elapsed_sec": round(time.time() - t0, 1)}
    for size in sizes:
        rs = [r for r in records if abs(r["graph_chars"] - size) < size * 0.5
              and "direct" in r]
        if not rs:
            continue
        for cond in ("direct", "evidence", "distill"):
            vals = [r[cond]["f1"] for r in rs if cond in r]
            if vals:
                summary[f"{size}_{cond}_f1"] = round(sum(vals) / len(vals), 4)
        prs = [r for r in rs if r["op_type"] == "parents"]
        if prs:
            summary[f"{size}_parents_evidence_f1"] = round(
                sum(r["evidence"]["f1"] for r in prs) / len(prs), 4)
            summary[f"{size}_parents_distill_f1"] = round(
                sum(r["distill"]["f1"] for r in prs) / len(prs), 4)
        brs = [r for r in rs if r["op_type"] == "bfs" and "evidence" in r]
        if brs:
            summary[f"{size}_bfs_evidence_f1"] = round(
                sum(r["evidence"]["f1"] for r in brs) / len(brs), 4)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
