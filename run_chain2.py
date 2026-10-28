# -*- coding: utf-8 -*-
"""
chain2 —— 攻克「证据在手也不会用」瓶颈 (T' 蒸馏轨迹 + 分层证据)

seed13 Q7 确诊的瓶颈: 五层深 BFS, E 的 det_f1=0.727 (证据质量够),
但四条件全部 0 分 —— 2B 拿着扁平证据也推不动深层多跳。

v3 老结论: 2B「读不出自己没推导过的状态」→ 推理侧干预两个抓手:
  A) E_layered 分层证据: distill_bfs 本来就算了逐层 frontier/确认边,
     但 E 只保留最后一层的扁平边表 —— 把逐层组织显式写进状态,
     最终 frontier 直接以节点列表给出 (绕开「从边表重推层结构」)。
  B) T' 蒸馏轨迹: 把蒸馏过程本身序列化成叙事 (每层查了什么/确认了几条/
     frontier 如何演化) —— 同题、已验证、压缩的「任务状态代理」,
     正是论文留白「trace 的选择与压缩」的实例。

题目: 精确重放 seed 13 的 24 题流 (与 run_chain.py 同 rng 消费序列),
取 12 道 BFS 题; 对照组 (trace/chain_et 等) 直接引用 chain_2b_3k_s13.json。
新条件 (全部 [状态块, x] 前置, 同模型同题):
  evidence_layered : [E_layered, x]
  chain_et_layered : [E_layered, x]  (header 强化版)
  chain_tp_flat    : [T', E_flat, x]
  chain_tp_layered : [T', E_layered, x]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    GRAPH_SYSTEM, build_graph_x, build_prompt, em_score, extract_answer,
    f1_score, ollama_chat, sample_from_target_chars,
)
from run_distill3 import (  # noqa: E402
    Extractor, chunk_edges, node_ids_in, parse_edges, verify_edge,
)

E_HEADER = (
    "Below is a VERIFIED evidence state extracted from the graph by a targeted "
    "scan + mechanical verification pass. Every edge listed below was confirmed "
    "to exist verbatim in the graph. Treat it as the primary source of truth, "
    "and cross-check against the graph."
)

E_LAYER_HEADER = (
    "Below is a VERIFIED, level-by-level evidence state for the BFS question, "
    "extracted by a targeted scan + mechanical verification pass. Every edge "
    "was confirmed to exist verbatim in the graph, and each level's frontier "
    "is the verified set of destination nodes of that level's edges. "
    "Treat it as the primary source of truth, and cross-check against the graph."
)

TP_HEADER = (
    "Below is a distillation trace from a mechanical scan+verify pass on the "
    "same graph problem. It records what was queried, what was confirmed, and "
    "how the search frontier evolved level by level. Use it as a scratchpad of "
    "the derivation; the evidence block that follows is the verified result."
)


def log(msg):
    print(msg, flush=True)


def distill_bfs_capture(models, sample, num_ctx, budget=500, num_predict=1024):
    """distill_bfs 的逐层捕获版: 返回 (逐层明细, 最终det, 总调用数)。"""
    import re
    m = re.search(r"from node (n\d+) with depth (\d+)", sample.op_text)
    start, depth = m.group(1), int(m.group(2))
    chunks = chunk_edges(sample, 4000)
    ex = Extractor(models, num_ctx, log, budget, num_predict)
    visited, frontier = {start}, {start}
    levels = []
    for d in range(1, depth + 1):
        level_conf = set()
        n_calls_0 = ex.calls
        n_cand = 0
        for i, ch in enumerate(chunks):
            f_here = sorted(frontier & node_ids_in(ch))
            if not f_here:
                continue
            for b0 in range(0, len(f_here), 12):
                grp = f_here[b0:b0 + 12]
                task = (f"Frontier set F = {{{', '.join(grp)}}}.\n"
                        f"List every edge u -> v in this portion with u in F. "
                        f"Copy each line verbatim.")
                cand, hit = ex.scan(ch, task, filt=lambda e: e[0] in set(grp))
                n_cand += len(cand)
                level_conf |= {e for e in cand
                               if verify_edge(e[0], e[1], ch)}
                if hit:
                    break
            if ex.calls >= ex.budget:
                break
        new = {v for _, v in level_conf if v not in visited}
        visited |= new
        levels.append({"level": d,
                       "queried_frontier": sorted(frontier),
                       "edges": sorted(level_conf),
                       "new_frontier": sorted(new),
                       "calls": ex.calls - n_calls_0,
                       "candidates": n_cand})
        log(f"      层{d}: 查询frontier={len(frontier)} 候选{n_cand} "
            f"确认{len(level_conf)} 新frontier={len(new)}")
        frontier = new
        if not frontier or ex.calls >= ex.budget:
            break
    det = sorted(frontier)
    return levels, det, ex.calls


def build_e_flat(levels, depth):
    """与 run_distill3/run_chain 完全一致的扁平 E (对照)。"""
    last = [lv for lv in levels if lv["level"] == depth]
    conf = last[0]["edges"] if last else (levels[-1]["edges"] if levels else [])
    txt = "\n".join(f"{u} -> {v}" for u, v in conf) or "NONE"
    return (f"Verified edges leading INTO depth {depth} of the BFS:\n{txt}")


def build_e_layered(levels, depth):
    parts = []
    for lv in levels:
        edges = "\n".join(f"{u} -> {v}" for u, v in lv["edges"]) or "NONE"
        parts.append(
            f"Level {lv['level']} (queried frontier: "
            f"{', '.join(lv['queried_frontier'])}):\n"
            f"Confirmed edges:\n{edges}\n"
            f"Level {lv['level']} frontier "
            f"({len(lv['new_frontier'])} nodes): "
            f"{', '.join(lv['new_frontier']) or 'NONE'}")
    final = levels[-1] if levels else {"new_frontier": []}
    parts.append(
        f"The Final Answer of the BFS with depth {depth} is the "
        f"Level-{depth} frontier listed above "
        f"({len(final['new_frontier'])} nodes), deduplicated.")
    return "\n\n".join(parts)


def build_tprime(levels):
    parts = []
    for lv in levels:
        parts.append(
            f"[Distill Level {lv['level']}] Queried portions containing "
            f"frontier {{{', '.join(lv['queried_frontier'])}}} with "
            f"{lv['calls']} scan call(s), {lv['candidates']} candidate edge(s) "
            f"found, {len(lv['edges'])} mechanically confirmed. "
            f"New frontier: {{{', '.join(lv['new_frontier']) or 'empty'}}}.")
    return "\n".join(parts)


def second_pass(model, sample, state_block, num_ctx, max_predict):
    prompt = build_prompt(sample, state_block, "state")
    g = ollama_chat(model, GRAPH_SYSTEM, prompt, num_ctx=num_ctx,
                    think=True, temperature=0.7, num_predict=max_predict)
    ans, bad = extract_answer(g.content, g.thinking)
    return ans, {"f1": f1_score(ans, sample.answer),
                 "em": em_score(ans, sample.answer),
                 "malformed": bad, "prompt_chars": len(prompt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="qwen3.5:4b")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default="logs/chain_2b_3k_s13.json")
    ap.add_argument("--out", default="logs/chain2_hard_2b.json")
    args = ap.parse_args()

    distill_models = [args.model] + ([args.ensemble] if args.ensemble else [])

    # ---- 精确重放 seed 13 的 24 题流, 校验后取 BFS 题 ----
    ref = json.load(open(args.ref, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)
    bfs_qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs":
            j = len(bfs_qs)
            assert sample.answer == ref[i]["answer"], \
                f"重放校验失败: 题{i} 答案不一致"
            bfs_qs.append((i, sample))
    log(f"重放校验通过: 12 道 BFS 题与 {args.ref} 完全一致")

    records, t0 = [], time.time()
    for qi, (idx, sample) in enumerate(bfs_qs):
        log(f"\n[{qi+1}/12] (原题#{idx+1}) bfs 图({sample.graph_chars}字符) "
            f"depth见op truth={len(sample.answer)}")
        rec = {"orig_index": idx, "op_text": sample.op_text,
               "answer": sample.answer, "graph_chars": sample.graph_chars,
               "ref_conditions": ref[idx]["conditions"],
               "ref_first_mean": sum(ref[idx]["first_f1s"]) / 3}

        # ---- 逐层蒸馏捕获 ----
        ts = time.time()
        try:
            levels, det, ncalls = distill_bfs_capture(
                distill_models, sample, args.num_ctx)
        except Exception as e:
            log(f"      !! 蒸馏失败: {e}")
            records.append(rec)
            _save(args.out, records)
            continue
        depth = int(sample.op_text.split("depth ")[1].rstrip("."))
        e_flat = build_e_flat(levels, depth)
        e_layered = build_e_layered(levels, depth)
        tp = build_tprime(levels)
        rec["distill"] = {"det_f1": f1_score(det, sample.answer),
                          "n_calls": ncalls,
                          "n_levels": len(levels),
                          "e_flat_chars": len(e_flat),
                          "e_layered_chars": len(e_layered),
                          "tprime_chars": len(tp)}
        log(f"      蒸馏: det_f1={rec['distill']['det_f1']:.3f} "
            f"{ncalls}次调用 E_flat={len(e_flat)} "
            f"E_layered={len(e_layered)} T'={len(tp)}字符 "
            f"{round(time.time()-ts)}s")

        # ---- 四个新条件 ----
        conditions = {
            "evidence_layered": E_LAYER_HEADER + "\n" + e_layered,
            "chain_tp_flat": TP_HEADER + "\n" + tp + "\n\n"
                             + E_HEADER + "\n" + e_flat,
            "chain_tp_layered": TP_HEADER + "\n" + tp + "\n\n"
                                + E_LAYER_HEADER + "\n" + e_layered,
        }
        rec["conditions"] = {}
        for name, block in conditions.items():
            try:
                ans, det2 = second_pass(args.model, sample, block,
                                        args.num_ctx, args.max_predict)
            except Exception as e:
                det2 = {"f1": 0.0, "em": 0.0, "error": str(e)[:80]}
            rec["conditions"][name] = det2
            log(f"      {name}: f1={det2['f1']:.3f} "
                f"(prompt {det2.get('prompt_chars', '?')}字符)")

        records.append(rec)
        _save(args.out, records)

    # ---- 汇总: 新条件 vs 引用的旧条件 ----
    ok = [r for r in records if "conditions" in r]
    n = len(ok)
    summary = {"model": args.model, "n_questions": len(bfs_qs), "n_ok": n,
               "elapsed_sec": round(time.time() - t0, 1)}
    all_conds = ["ref_first", "ref_trace_state", "ref_chain_et"] + \
        list(ok[0]["conditions"].keys()) if n else []
    for c in all_conds:
        if c.startswith("ref_"):
            key = c[4:]
            vals = [r["ref_conditions"][key]["f1"] for r in ok
                    if key in r["ref_conditions"]]
        else:
            vals = [r["conditions"][c]["f1"] for r in ok]
        if vals:
            summary[f"{c}_f1"] = round(sum(vals) / len(vals), 4)
    # 胜负: 新条件 vs 引用 chain_et
    for c in [k for k in all_conds if not k.startswith("ref_")]:
        wins = sum(1 for r in ok
                   if r["conditions"][c]["f1"]
                   > r["ref_conditions"]["chain_et"]["f1"])
        losses = sum(1 for r in ok
                     if r["conditions"][c]["f1"]
                     < r["ref_conditions"]["chain_et"]["f1"])
        summary[f"{c}_vs_chain_et"] = f"{wins}胜{losses}负"
    _save(args.out, records, summary)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


def _save(out, records, summary=None):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = {"records_so_far": records} if summary is None \
        else {"summary": summary, "records": records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
