# -*- coding: utf-8 -*-
"""
消融实验: 把 Best-of-N / 自洽选择 借鉴进 Trace as State

条件设计 (每题):
  第一遍: n_tr 条轨迹 + 各自答案 (与 run_pilot 相同)
  ---- 免费条件 (不加推理调用) ----
  A. baseline      : n_tr 次首遍平均 (对照)
  B. bestofN_medoid: 从首遍答案中选与其它答案 F1 总和最大的 medoid (答案级自洽选择)
  ---- 第二遍条件 (各 1 次调用) ----
  C. state_all     : 序列化全部轨迹, 摆放=state  (论文原方法)
  D. state_consens : 只序列化与共识答案一致的轨迹子集, 摆放=state (借鉴: 内容筛选)
  ---- 堆叠条件 (3 次调用) ----
  E. state_consens_vote3: 条件 D 的 prompt 跑 3 次, 对 3 个输出答案再取 medoid
                           (完整复刻 "多次思考挑最好" 且作用在最强条件上)

选择器: 无验证器, 用答案间两两 F1 一致性 (agreement medoid) —— 端侧免费可得。
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


def medoid_answer(answers):
    """返回 (medoid答案, 它的索引)。平票取第一个。"""
    if not answers:
        return [], -1
    best_i, best_s = 0, -1.0
    for i, a in enumerate(answers):
        s = sum(f1_score(a, b) for j, b in enumerate(answers) if j != i)
        if s > best_s:
            best_s, best_i = s, i
    return answers[best_i], best_i


def consensus_trace_indices(answers, threshold=0.5):
    """与 medoid 答案 F1 >= threshold 的轨迹下标; 若共识簇只有 medoid 自己, 仍返回它。"""
    med, mi = medoid_answer(answers)
    if mi < 0:
        return list(range(len(answers)))
    keep = [i for i, a in enumerate(answers) if f1_score(a, med) >= threshold]
    return keep or [mi]


def run_question(model, sample, rng, n_tr, num_ctx, max_predict, vote_n):
    x_head, x_op = build_graph_x(sample)
    x = x_head + "\n" + x_op

    # ---- 第一遍 ----
    first = []
    for j in range(n_tr):
        g = ollama_chat(model, GRAPH_SYSTEM, x, num_ctx=num_ctx,
                        think=True, temperature=0.7, num_predict=max_predict)
        ans, bad = extract_answer(g.content, g.thinking)
        first.append({"thinking": g.thinking, "answers": ans, "malformed": bad})

    first_answers = [f["answers"] for f in first]
    base_f1 = sum(f1_score(a, sample.answer) for a in first_answers) / n_tr
    base_em = sum(em_score(a, sample.answer) for a in first_answers) / n_tr

    # ---- 免费条件: medoid 选择 ----
    med, med_i = medoid_answer(first_answers)
    bo_f1 = f1_score(med, sample.answer)
    bo_em = em_score(med, sample.answer)

    # ---- 轨迹序列化: 全量 vs 共识子集 ----
    keep = consensus_trace_indices(first_answers)
    T_all = serialize_traces([f["thinking"] for f in first])
    T_cons = serialize_traces([first[i]["thinking"] for i in keep])

    def second_pass(T, placement, tag):
        prompt = build_prompt(sample, T, placement)
        g = ollama_chat(model, GRAPH_SYSTEM, prompt, num_ctx=num_ctx,
                        think=True, temperature=0.7, num_predict=max_predict)
        ans, bad = extract_answer(g.content, g.thinking)
        print(f"      {tag}: f1={f1_score(ans, sample.answer):.3f} "
              f"len(T)={len(T)} think={len(g.thinking)}", flush=True)
        return {"answers": ans, "f1": f1_score(ans, sample.answer),
                "em": em_score(ans, sample.answer), "thinking_len": len(g.thinking)}

    c = {}
    c["state_all"] = second_pass(T_all, "state", "C state_all")
    c["state_consens"] = second_pass(T_cons, "state", "D state_consens")

    # ---- 堆叠: D 的 prompt 跑 vote_n 次, 答案级 medoid ----
    vote_out = []
    for v in range(vote_n):
        prompt = build_prompt(sample, T_cons, "state")
        g = ollama_chat(model, GRAPH_SYSTEM, prompt, num_ctx=num_ctx,
                        think=True, temperature=0.7, num_predict=max_predict)
        ans, bad = extract_answer(g.content, g.thinking)
        vote_out.append(ans)
        print(f"      E vote{v+1}: f1={f1_score(ans, sample.answer):.3f}", flush=True)
    vmed, _ = medoid_answer(vote_out)
    c["state_vote3"] = {"answers": vmed, "f1": f1_score(vmed, sample.answer),
                        "em": em_score(vmed, sample.answer)}

    rec = {
        "op_type": sample.op_type, "op_text": sample.op_text,
        "answer": sample.answer, "graph_chars": sample.graph_chars,
        "first_answers": first_answers,
        "first_f1s": [f1_score(a, sample.answer) for a in first_answers],
        "keep_indices": keep,
        "baseline": {"f1": base_f1, "em": base_em},
        "bestofN_medoid": {"f1": bo_f1, "em": bo_em, "picked": med_i},
        "state_all": c["state_all"],
        "state_consens": c["state_consens"],
        "state_vote3": c["state_vote3"],
        "vote_f1s": [f1_score(a, sample.answer) for a in vote_out],
    }
    print(f"  => base={base_f1:.2f} boN={bo_f1:.2f} all={c['state_all']['f1']:.2f} "
          f"cons={c['state_consens']['f1']:.2f} vote={c['state_vote3']['f1']:.2f} "
          f"(truth={sorted(sample.answer)[:5]}{'…' if len(sample.answer)>5 else ''})",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--target-chars", type=int, default=3000)
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--n-tr", type=int, default=3)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vote-n", type=int, default=3)
    ap.add_argument("--out", default="logs/consensus_2b_3k.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"模型={args.model} 目标字符≈{args.target_chars} 题数={args.n_questions} "
          f"n_tr={args.n_tr} vote_n={args.vote_n} num_ctx={args.num_ctx}", flush=True)

    records = []
    t0 = time.time()
    for i in range(args.n_questions):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, args.target_chars, op_type=op)
        print(f"\n[{i+1}/{args.n_questions}] {op} 图({sample.graph_chars}字符) "
              f"truth_size={len(sample.answer)}", flush=True)
        try:
            rec = run_question(args.model, sample, rng, args.n_tr, args.num_ctx,
                               args.max_predict, args.vote_n)
        except Exception as e:
            rec = {"error": str(e), "op_type": op}
            print("   !! 错误:", e, flush=True)
        records.append(rec)

    ok = [r for r in records if "error" not in r]
    n = len(ok)
    summary = {"model": args.model, "target_chars": args.target_chars,
               "n_questions": args.n_questions, "n_tr": args.n_tr,
               "vote_n": args.vote_n, "num_ctx": args.num_ctx, "n_ok": n,
               "elapsed_sec": round(time.time() - t0, 1)}
    if n:
        conds = ("baseline", "bestofN_medoid", "state_all", "state_consens",
                 "state_vote3")
        for cond in conds:
            summary[f"{cond}_f1"] = round(sum(r[cond]["f1"] for r in ok) / n, 4)
            summary[f"{cond}_em"] = round(sum(r[cond]["em"] for r in ok) / n, 4)
        # 逐题胜负: 每个新条件 vs state_all 与 vs baseline
        for cond in ("bestofN_medoid", "state_consens", "state_vote3"):
            summary[f"{cond}_beats_stateall"] = sum(
                1 for r in ok if r[cond]["f1"] > r["state_all"]["f1"])
            summary[f"{cond}_ties_stateall"] = sum(
                1 for r in ok if r[cond]["f1"] == r["state_all"]["f1"])
            summary[f"{cond}_beats_baseline"] = sum(
                1 for r in ok if r[cond]["f1"] > r["baseline"]["f1"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n结果已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
