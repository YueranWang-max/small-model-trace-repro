# -*- coding: utf-8 -*-
"""
Trace as State 缩放复现 —— 实验驱动

流程 (对齐论文 §2):
  第一遍: 同一问题跑 n_tr 次, 收集 thinking 轨迹 r_j 与可见答案 a_j
  序列化: T = pi(r_1..r_n)
  第二遍:
     trace as state  ->  [T, x, q]
     trace append    ->  [x, T, q]
  评分: 对照 first-pass 基线 / trace append / trace as state 的 EM 与 set-F1

用法示例:
  python run_pilot.py --model openbmb/minicpm5-2b --target-chars 8000 \
      --n-questions 20 --n-tr 3 --num-ctx 16384 --out logs/pilot_8k.json
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
    f1_score, generate_sample, ollama_chat, sample_from_target_chars,
    serialize_traces,
)


def run_one(model, sample, rng, n_tr, num_ctx, max_predict, verbose=False):
    x_head, x_op = build_graph_x(sample)
    x = x_head + "\n" + x_op

    # ---- 第一遍: 收集轨迹 ----
    first = []
    for j in range(n_tr):
        try:
            g = ollama_chat(model, GRAPH_SYSTEM, x, num_ctx=num_ctx,
                            think=True, temperature=0.7, num_predict=max_predict)
        except Exception as e:
            return {"error": f"first_pass_t{j}: {e}"}
        ans, bad = extract_answer(g.content, g.thinking)
        first.append({"thinking": g.thinking, "answers": ans, "malformed": bad,
                      "raw_content": g.content[:600]})

    # 基线: n_tr 次平均
    base_em = sum(em_score(f["answers"], sample.answer) for f in first) / n_tr
    base_f1 = sum(f1_score(f["answers"], sample.answer) for f in first) / n_tr

    # ---- 序列化 T ----
    T = serialize_traces([f["thinking"] for f in first])

    # ---- 第二遍: 两种摆放 ----
    out = {}
    for placement in ("state", "append"):
        prompt = build_prompt(sample, T, placement)
        try:
            g = ollama_chat(model, GRAPH_SYSTEM, prompt, num_ctx=num_ctx,
                            think=True, temperature=0.7, num_predict=max_predict)
        except Exception as e:
            return {"error": f"second_pass_{placement}: {e}"}
        ans, bad = extract_answer(g.content, g.thinking)
        out[placement] = {
            "answers": ans,
            "malformed": bad,
            "em": em_score(ans, sample.answer),
            "f1": f1_score(ans, sample.answer),
            "raw_content": g.content[:600],
        }

    rec = {
        "op_type": sample.op_type,
        "op_text": sample.op_text,
        "answer": sample.answer,
        "graph_chars": sample.graph_chars,
        "baseline": {"em": base_em, "f1": base_f1,
                     "first_answers": [f["answers"] for f in first]},
        "append": {"em": out["append"]["em"], "f1": out["append"]["f1"],
                   "answers": out["append"]["answers"]},
        "state": {"em": out["state"]["em"], "f1": out["state"]["f1"],
                  "answers": out["state"]["answers"]},
        "trace_chars": len(T),
    }
    if verbose:
        print(f"  [{sample.op_type}] ans={sorted(sample.answer)[:6]} "
              f"base_f1={base_f1:.2f} app_f1={out['append']['f1']:.2f} "
              f"state_f1={out['state']['f1']:.2f}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--target-chars", type=int, default=8000)
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--n-tr", type=int, default=3)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="logs/pilot.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"模型={args.model} 目标字符≈{args.target_chars} 题数={args.n_questions} "
          f"n_tr={args.n_tr} num_ctx={args.num_ctx}")

    records = []
    t0 = time.time()
    for i in range(args.n_questions):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, args.target_chars, op_type=op)
        print(f"[{i+1}/{args.n_questions}] 生成 {op} 图({sample.graph_chars}字符) 开始…")
        rec = run_one(args.model, sample, rng, args.n_tr, args.num_ctx,
                      args.max_predict, verbose=True)
        records.append(rec)
        if "error" in rec:
            print("   !! 错误:", rec["error"])

    ok = [r for r in records if "error" not in r]
    n = len(ok)
    summary = {
        "model": args.model,
        "target_chars": args.target_chars,
        "n_questions": args.n_questions,
        "n_tr": args.n_tr,
        "num_ctx": args.num_ctx,
        "n_ok": n,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if n:
        for cond in ("baseline", "append", "state"):
            summary[f"{cond}_f1"] = round(sum(r[cond]["f1"] for r in ok) / n, 4)
            summary[f"{cond}_em"] = round(sum(r[cond]["em"] for r in ok) / n, 4)
        # 逐题胜负 (state vs append)
        wins = sum(1 for r in ok if r["state"]["f1"] > r["append"]["f1"])
        ties = sum(1 for r in ok if r["state"]["f1"] == r["append"]["f1"])
        summary["state_beats_append_f1"] = wins
        summary["state_ties_append_f1"] = ties
        summary["state_loss_append_f1"] = n - wins - ties

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
