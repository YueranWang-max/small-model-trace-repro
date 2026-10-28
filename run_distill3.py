# -*- coding: utf-8 -*-
"""
分片状态蒸馏 v3 —— 机械验证 + 检索定位 + 饱和扫描

v2 BFS 失败根因 (diagnose_bfs.py 确诊):
  1) 召回塌陷: 让 2B 拿 11 节点集合在 250 行边表里逐行对照, 每层有效召回 ~0.5-0.6,
     跨层复合 r^3 -> frontier F1 崩到 0.19
  2) LLM 确认在大候选列表下自身幻觉 ~50% (623 候选放行 326, 真值树边仅 40)

v3 三个修复 (邪修三件套):
  A) 机械验证: 候选边用正则在来源片里做逐字核验 (^u -> v$), 精度恒 1.0,
     零幻觉零成本 —— 模型负责发现(召回), 机器负责核验(精度)
  B) 检索定位: 正则取出每片包含的节点 id (BM25 式检索基础设施, 不泄题),
     每片只查询它包含的 frontier 节点, 集合匹配从 11 个降到 1-4 个
  C) 饱和扫描: 温度 0 -> 0.7 -> 1.0 逐遍, 某遍不再贡献新候选即停 (穷尽原则)

诚实边界: 检索(哪些片提到节点X)与验证(该行是否逐字存在)是确定性基础设施,
模型的工作仍是"读 4K 片、列出匹配的边" —— 能力被测的部分不变。
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
    ollama_chat, sample_from_target_chars,
)

EXTRACT_SYSTEM = (
    "You extract exact edges from a portion of a directed edge list. "
    "Copy matching lines VERBATIM from the portion, one per line. "
    "Do not invent edges. If no edge matches, output exactly: NONE"
)

BATCH = 12          # 每次查询对照的 frontier 节点上限
MAX_PASSES = 3      # 饱和扫描最多遍数
CALL_BUDGET = 500   # 单题抽取调用预算护栏


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


def node_ids_in(text):
    return set(re.findall(r"n\d+", text or ""))


def verify_edge(u, v, chunk):
    """机械验证: 该边是否逐字存在于本片 (整行匹配)。精度恒 1.0。"""
    return re.search(rf"^{re.escape(u)} -> {re.escape(v)}$", chunk,
                     re.MULTILINE) is not None


def parse_edges(text):
    return [(u, v) for u, v in re.findall(r"(n\d+)\s*->\s*(n\d+)", text or "")]


class Extractor:
    """饱和扫描封装: 温度 0/0.7/1.0 逐遍, 无新候选即停。可叠加异构模型。"""

    def __init__(self, models, num_ctx, log, budget, num_predict=1024):
        self.models = models if isinstance(models, list) else [models]
        self.num_ctx = num_ctx
        self.log = log
        self.calls = 0
        self.budget = budget
        self.num_predict = num_predict  # 转录等整片抄写任务需要更大预算

    def scan(self, chunk, task, filt=lambda e: True):
        """返回 (候选集, 是否触及预算)。候选已过滤, 未做验证。"""
        cands = set()
        temps = [0.0, 0.7, 1.0]
        for mi, model in enumerate(self.models):
            # 集成补充模型只跑 T=0.7 一遍 (职责是补召回, 不做主力穷尽)
            use_temps = temps if mi == 0 else [0.7]
            for t in use_temps:
                if self.calls >= self.budget:
                    return cands, True
                self.calls += 1
                user = f"{task}\n\nPortion of the edge list:\n{chunk}"
                try:
                    g = ollama_chat(model, EXTRACT_SYSTEM, user,
                                    num_ctx=self.num_ctx, think=False,
                                    temperature=t, num_predict=self.num_predict)
                except Exception:
                    continue
                new = {e for e in parse_edges(g.content) if filt(e)}
                if new <= cands:
                    if model == self.models[0] and t == 0.0:
                        continue  # 第一遍空/重复属正常, 继续下一温度
                    break  # 该模型该温度无新贡献 -> 饱和, 换下一模型
                cands |= new
        return cands, False


# ---------------------------------------------------------------------------
# parents: 检索定位 -> 饱和扫描 -> 机械验证
# ---------------------------------------------------------------------------

def distill_parents(ex: Extractor, sample, chunks, log):
    m = re.search(r"node (n\d+)", sample.op_text)
    target = m.group(1)
    task = (f"List every edge whose DESTINATION (right side) is {target}. "
            f"Copy the full line verbatim.")
    confirmed = set()
    rel = [(i, ch) for i, ch in enumerate(chunks) if target in node_ids_in(ch)]
    log(f"      检索: {len(rel)}/{len(chunks)} 片包含目标 {target}")
    for i, ch in rel:
        cand, hit_budget = ex.scan(ch, task, filt=lambda e: e[1] == target)
        kept = {e for e in cand if verify_edge(e[0], e[1], ch)}
        confirmed |= kept
        log(f"      片{i}: 候选{len(cand)} -> 机械确认{len(kept)}")
    evidence = "\n".join(f"{u} -> {t}" for u, t in sorted(confirmed)) or "NONE"
    state = (f"Verified evidence edges pointing to {target} "
             f"(targeted scan + mechanical verification over {len(chunks)} portions):\n"
             f"{evidence}")
    det = sorted({u for u, t in confirmed})
    return state, det, confirmed, ex.calls


# ---------------------------------------------------------------------------
# bfs: 逐层 检索定位 -> 小批量扫描 -> 机械验证
# ---------------------------------------------------------------------------

def distill_bfs(ex: Extractor, sample, chunks, log):
    m = re.search(r"from node (n\d+) with depth (\d+)", sample.op_text)
    start, depth = m.group(1), int(m.group(2))
    visited, frontier = {start}, {start}
    last_conf = set()
    per_level = []
    for d in range(1, depth + 1):
        level_conf = set()
        n_calls_0 = ex.calls
        n_cand = 0
        rel_i = 0
        for i, ch in enumerate(chunks):
            f_here = sorted(frontier & node_ids_in(ch))
            if not f_here:
                continue
            rel_i += 1
            for b0 in range(0, len(f_here), BATCH):
                grp = f_here[b0:b0 + BATCH]
                task = (f"Frontier set F = {{{', '.join(grp)}}}.\n"
                        f"List every edge u -> v in this portion with u in F. "
                        f"Copy each line verbatim.")
                cand, hit_budget = ex.scan(ch, task, filt=lambda e: e[0] in set(grp))
                n_cand += len(cand)
                level_conf |= {e for e in cand if verify_edge(e[0], e[1], ch)}
                if hit_budget:
                    break
            if ex.calls >= ex.budget:
                break
        new = {v for u, v in level_conf if v not in visited}
        visited |= new
        last_conf = level_conf
        frontier = new
        per_level.append({"level": d, "frontier": len(new),
                          "calls": ex.calls - n_calls_0,
                          "candidates": n_cand, "confirmed": len(level_conf)})
        log(f"      层{d}: 相关片{rel_i}, 调用{ex.calls-n_calls_0}, "
            f"候选{n_cand} -> 机械确认{len(level_conf)}, 新frontier={len(new)}")
        if not frontier or ex.calls >= ex.budget:
            break
    edges_txt = "\n".join(f"{u} -> {v}" for u, v in sorted(last_conf)) or "NONE"
    state = (f"Verified edges leading INTO depth {depth} of the BFS:\n{edges_txt}")
    det = sorted(frontier)
    return state, det, last_conf, ex.calls, per_level


def final_pass(model, sample, state, num_ctx, max_predict):
    if sample.op_type == "bfs":
        hint = ("The Final Answer is the set of DESTINATION (right-side) nodes "
                "of the verified edges above, deduplicated.")
    else:
        hint = ("The Final Answer is the set of SOURCE (left-side) nodes "
                "of the verified edges above, deduplicated.")
    user = (GRAPH_PREAMBLE + "\n\nYou did NOT see the full graph. A scan+verify "
            "pass extracted this verified evidence:\n" + state +
            "\n\n" + hint +
            "\n\nFull Operation:\n" + sample.op_text + "\n"
            "Return exactly one line: Final Answer: [node1, node2]. "
            "Use [] for the empty set.")
    g = ollama_chat(model, "You solve directed-graph algorithm problems.",
                    user, num_ctx=num_ctx, think=False, temperature=0.7,
                    num_predict=max_predict)
    ans, _ = extract_answer(g.content, g.thinking)
    return ans, len(user)


def read_state(state, op):
    """机械读取: 从已核验的状态里直接取答案 (BFS=右端点, parents=左端点)。
    诊断结论: 2B 无法批量抄写 >~40 条 (会输出 '[list of all right-side nodes]'
    这样的占位符), 状态读取必须是机械的 —— 这本身就是一个设计发现。"""
    edges = parse_edges(state)
    if op == "bfs":
        return sorted({v for _, v in edges})
    return sorted({u for u, _ in edges})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="",
                    help="可选: 第二个抽取模型, 取并集提召回 (机械验证保精度)")
    ap.add_argument("--sizes", default="16000,48000,128000")
    ap.add_argument("--parents-per-size", type=int, default=2)
    ap.add_argument("--bfs-per-size", type=int, default=1)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"logs/distill3_{args.tag}_2b.json"

    def log(msg):
        print(msg, flush=True)

    models = [args.model] + ([args.ensemble] if args.ensemble else [])
    sizes = [int(s) for s in args.sizes.split(",")]
    records, t0 = [], time.time()

    for size in sizes:
        rng = random.Random(args.seed + size)
        log(f"\n{'='*60}\n### 图规模 ≈{size} (chunk={args.chunk_chars}, "
            f"ctx={args.num_ctx}, models={models})")
        for qi in range(args.parents_per_size + args.bfs_per_size):
            op = "parents" if qi < args.parents_per_size else "bfs"
            sample = sample_from_target_chars(rng, size, op_type=op)
            chunks = chunk_edges(sample, args.chunk_chars)
            log(f"\n[{size}] {op} 图{sample.graph_chars}字符 "
                f"truth={len(sample.answer)} 分{len(chunks)}片")
            rec = {"op_type": op, "graph_chars": sample.graph_chars,
                   "n_chunks": len(chunks), "answer": sample.answer}

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

            ts = time.time()
            ex = Extractor(models, args.num_ctx, log, CALL_BUDGET)
            try:
                if op == "parents":
                    state, det, ev, ncalls = distill_parents(
                        ex, sample, chunks, log)
                    per_level = []
                else:
                    state, det, ev, ncalls, per_level = distill_bfs(
                        ex, sample, chunks, log)
            except Exception as e:
                log(f"      !! 蒸馏异常: {e}")
                rec["distill_error"] = str(e)[:80]
                records.append(rec)
                continue
            rec["evidence"] = {"f1": f1_score(det, sample.answer),
                               "em": em_score(det, sample.answer),
                               "n_confirmed": len(ev), "n_calls": ncalls}
            ans_model, _ = final_pass(args.model, sample, state,
                                      args.num_ctx, args.max_predict)
            ans_mech = read_state(state, op)
            rec["distill"] = {"f1": f1_score(ans_mech, sample.answer),
                              "em": em_score(ans_mech, sample.answer),
                              "read": "mechanical"}
            rec["model_read"] = {"f1": f1_score(ans_model, sample.answer),
                                 "em": em_score(ans_model, sample.answer)}
            rec["cost"] = {"state_chars": len(state),
                           "sec": round(time.time() - ts, 1)}
            if per_level:
                rec["per_level"] = per_level
            log(f"      evidence_f1={rec['evidence']['f1']:.3f} "
                f"distill_f1={rec['distill']['f1']:.3f} "
                f"(state {len(state)}字符, {ncalls}次调用, "
                f"{round(time.time()-ts)}s)")

            records.append(rec)
            # 逐题落盘, 永不丢数据
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"records_so_far": records}, f,
                          ensure_ascii=False, indent=2)

    summary = {"models": models, "chunk_chars": args.chunk_chars,
               "num_ctx": args.num_ctx, "tag": args.tag,
               "elapsed_sec": round(time.time() - t0, 1)}
    for size in sizes:
        rs = [r for r in records if abs(r["graph_chars"] - size) < size * 0.5
              and "evidence" in r]
        for cond in ("direct", "evidence", "distill"):
            vals = [r[cond]["f1"] for r in rs if cond in r]
            if vals:
                summary[f"{size}_{cond}_f1"] = round(sum(vals) / len(vals), 4)
        for opt in ("parents", "bfs"):
            prs = [r for r in rs if r["op_type"] == opt]
            if prs:
                summary[f"{size}_{opt}_evidence_f1"] = round(
                    sum(r["evidence"]["f1"] for r in prs) / len(prs), 4)
                summary[f"{size}_{opt}_calls"] = round(
                    sum(r["evidence"]["n_calls"] for r in prs) / len(prs), 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
