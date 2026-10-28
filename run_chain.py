# -*- coding: utf-8 -*-
"""
蒸馏状态 × 轨迹状态 串联实验 —— 两种「状态」能否叠加?

背景:
  唐老师 Trace as State: 第一遍 thinking 轨迹序列化成 T, 前置重读 (状态=模型自己的推理史)
  本项目蒸馏:  分片扫描+机械核验出的证据状态 E (状态=外部世界的按需蒸馏事实)

两者正交: T 装「怎么想的」, E 装「事实是什么」。串联假设: [T, E, x, q] ≥ max(T, E)。

条件 (全部前置 placement, 同题同模型):
  first         : 无状态基线 (3 次首遍平均, 免费)
  trace_state   : [T, x, q]        —— 论文原法 (对照 3K 基线 0.731)
  evidence_state: [E, x, q]        —— 纯蒸馏状态
  chain_te      : [T, E, x, q]     —— 轨迹在前, 证据紧贴图 (recency 假设)
  chain_et      : [E, T, x, q]     —— 证据在前, 轨迹其次

题目: 与 run_pilot/run_diverse 完全同题 (seed 7, 3K, 12 题, bfs/parents 交替)
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
    f1_score, ollama_chat, sample_from_target_chars, serialize_traces,
)
from run_distill3 import (  # noqa: E402
    Extractor, chunk_edges, distill_bfs, distill_parents,
)

E_HEADER = (
    "Below is a VERIFIED evidence state extracted from the graph by a targeted "
    "scan + mechanical verification pass. Every edge listed below was confirmed "
    "to exist verbatim in the graph. Treat it as the primary source of truth, "
    "and cross-check against the graph."
)

N_TR = 3  # 首遍采样次数 (与 3K 基线一致)


def log(msg):
    print(msg, flush=True)


def distill_evidence(models, sample, num_ctx, budget=80):
    """用 v3 已验证的蒸馏器构造证据状态 E。返回 (E块文本, 明细)。"""
    chunks = chunk_edges(sample, 4000)
    ex = Extractor(models, num_ctx, log, budget)
    if sample.op_type == "parents":
        state, det, ev, ncalls = distill_parents(ex, sample, chunks, log)
        per_level = []
    else:
        state, det, ev, ncalls, per_level = distill_bfs(ex, sample, chunks, log)
    return state, {"det_f1": f1_score(det, sample.answer),
                   "n_confirmed": len(ev), "calls": ncalls,
                   "state_chars": len(state), "per_level": per_level}


def second_pass(model, sample, state_block, num_ctx, max_predict):
    """state_block 前置 + 完整图 + 操作 -> 答案。"""
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
    ap.add_argument("--ensemble", default="qwen3.5:4b",
                    help="蒸馏 E 时的补召回模型 (只用于抽取, 不用于作答)")
    ap.add_argument("--no-chain-et", action="store_true",
                    help="跳过 [E,T,x] 条件省时间")
    ap.add_argument("--target-chars", type=int, default=3000)
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="logs/chain_2b_3k.json")
    args = ap.parse_args()
    out = args.out

    distill_models = [args.model] + ([args.ensemble] if args.ensemble else [])
    rng = random.Random(args.seed)
    log(f"模型={args.model} (蒸馏集成={distill_models}) "
        f"题数={args.n_questions} 目标字符≈{args.target_chars} seed={args.seed}")

    records = []
    t0 = time.time()
    for i in range(args.n_questions):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, args.target_chars, op_type=op)
        log(f"\n[{i+1}/{args.n_questions}] {op} 图({sample.graph_chars}字符) "
            f"truth_size={len(sample.answer)}")
        rec = {"op_type": op, "op_text": sample.op_text,
               "answer": sample.answer, "graph_chars": sample.graph_chars}
        x_head, x_op = build_graph_x(sample)
        x = x_head + "\n" + x_op

        # ---- 第一遍 ×3: 收集轨迹与答案 ----
        attempts = []
        for k in range(N_TR):
            g = ollama_chat(args.model, GRAPH_SYSTEM, x, num_ctx=args.num_ctx,
                            think=True, temperature=0.7,
                            num_predict=args.max_predict)
            ans, bad = extract_answer(g.content, g.thinking)
            attempts.append({"thinking": g.thinking, "answers": ans,
                             "malformed": bad})
            log(f"      first{k+1}: f1={f1_score(ans, sample.answer):.3f}")
        rec["first_f1s"] = [f1_score(a["answers"], sample.answer)
                            for a in attempts]

        # ---- 蒸馏证据状态 E ----
        try:
            e_block, edet = distill_evidence(distill_models, sample,
                                             args.num_ctx)
        except Exception as e:
            log(f"      !! 蒸馏失败: {e}")
            e_block, edet = "NONE", {"error": str(e)[:80]}
        rec["distill"] = edet
        log(f"      E: det_f1={edet.get('det_f1', -1):.3f} "
            f"{edet.get('calls', '?')}次调用 {edet.get('state_chars', '?')}字符")

        # ---- 轨迹状态 T (论文原法) ----
        T = serialize_traces([a["thinking"] for a in attempts])

        # ---- 四个前置条件 ----
        conditions = {
            "trace_state": T,
            "evidence_state": E_HEADER + "\n" + e_block,
            "chain_te": T + "\n\n" + E_HEADER + "\n" + e_block,
        }
        if not args.no_chain_et:
            conditions["chain_et"] = E_HEADER + "\n" + e_block + "\n\n" + T

        rec["conditions"] = {}
        for name, block in conditions.items():
            try:
                ans, det = second_pass(args.model, sample, block,
                                       args.num_ctx, args.max_predict)
            except Exception as e:
                det = {"f1": 0.0, "em": 0.0, "error": str(e)[:80]}
            rec["conditions"][name] = det
            log(f"      {name}: f1={det['f1']:.3f} "
                f"(prompt {det.get('prompt_chars', '?')}字符)")

        # ---- 逐题落盘 ----
        records.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"records_so_far": records}, f, ensure_ascii=False,
                      indent=2)

    # ---- 汇总 ----
    ok = [r for r in records if "conditions" in r]
    n = len(ok)
    conds = ["first(mean)"] + list(ok[0]["conditions"].keys()) if n else []
    summary = {"model": args.model, "distill_ensemble": distill_models,
               "target_chars": args.target_chars, "n_questions": args.n_questions,
               "n_ok": n, "elapsed_sec": round(time.time() - t0, 1)}
    if n:
        first_mean = sum(sum(r["first_f1s"]) / len(r["first_f1s"])
                         for r in ok) / n
        summary["first_mean_f1"] = round(first_mean, 4)
        for c in ok[0]["conditions"]:
            summary[f"{c}_f1"] = round(
                sum(r["conditions"][c]["f1"] for r in ok) / n, 4)
            summary[f"{c}_em"] = round(
                sum(r["conditions"][c]["em"] for r in ok) / n, 4)
        # 逐题胜负 (对论文原法)
        for c in ok[0]["conditions"]:
            if c == "trace_state":
                continue
            summary[f"{c}_beats_trace"] = sum(
                1 for r in ok if r["conditions"][c]["f1"]
                > r["conditions"]["trace_state"]["f1"])
            summary[f"{c}_loses_trace"] = sum(
                1 for r in ok if r["conditions"][c]["f1"]
                < r["conditions"]["trace_state"]["f1"])
        # 分操作类型
        for opt in ("bfs", "parents"):
            rs = [r for r in ok if r["op_type"] == opt]
            if rs:
                summary[f"{opt}_n"] = len(rs)
                for c in ok[0]["conditions"]:
                    summary[f"{opt}_{c}_f1"] = round(
                        sum(r["conditions"][c]["f1"] for r in rs) / len(rs), 4)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
