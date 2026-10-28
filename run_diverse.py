# -*- coding: utf-8 -*-
"""
序贯多样化搜索实验: "第 k 次必须避开前 k-1 次的路线"

与 run_consensus.py 的独立采样对照:
  那边: 同一 prompt 采样 n 次, 路线可能高度重复 (实测 Q1 三次投票答案几乎相同)
  这里: 串行执行, 第 k 次 prompt 里注入前 k-1 次的路线摘要+答案,
        明确指令 "换一条路, 不要重复, 除非你能严格验证".

条件:
  A. diverse_seq  : 序贯 k 次带规避的尝试, 收集全部答案与轨迹
                    - 提前停止: 新答案与上一答案相同 (规避后仍收敛 = 饱和信号)
                    - 最优选取: 答案间 medoid (免费)
  B. state_diverse: 把 k 条互不相同的轨迹序列化后前置 (trace as state), 1 次调用
                    —— 检验 "五条不同的路线" 作为状态是否优于 "五条重复的路线"

对照组数据: logs/consensus_2b_3k.json (独立采样 + vote3)
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

AVOID_HEADER = (
    "\n\n---\nPrevious attempts on this exact operation (they may be wrong):\n"
)
AVOID_INSTR = (
    "\nYou MUST take a DIFFERENT route this time: do not repeat the reasoning "
    "paths listed above, and do not return an identical answer unless you can "
    "rigorously verify it node-by-node against the edge list. Re-derive the "
    "result from scratch with a different strategy (for example: walk the "
    "frontier level by level, or check every edge touching the relevant node), "
    "then give your own Final Answer in the required format."
)


def tail(text: str, n: int = 500) -> str:
    t = (text or "").strip()
    return ("…" + t[-n:]) if len(t) > n else t


def medoid_answer(answers):
    if not answers:
        return [], -1
    best_i, best_s = 0, -1.0
    for i, a in enumerate(answers):
        s = sum(f1_score(a, b) for j, b in enumerate(answers) if j != i)
        if s > best_s:
            best_s, best_i = s, i
    return answers[best_i], best_i


def run_question(model, sample, n_att, num_ctx, max_predict):
    x_head, x_op = build_graph_x(sample)
    x = x_head + "\n" + x_op

    attempts = []   # {thinking, answers}
    for k in range(n_att):
        if k == 0:
            user = x
        else:
            blocks = []
            for j, at in enumerate(attempts, 1):
                blocks.append(f"[Attempt {j}] route tail: {tail(at['thinking'])}\n"
                              f"Its answer: [{', '.join(at['answers']) or '(empty)'}]")
            user = x + AVOID_HEADER + "\n".join(blocks) + AVOID_INSTR
        g = ollama_chat(model, GRAPH_SYSTEM, user, num_ctx=num_ctx,
                        think=True, temperature=0.7, num_predict=max_predict)
        ans, bad = extract_answer(g.content, g.thinking)
        prev = attempts[-1]["answers"] if attempts else None
        attempts.append({"thinking": g.thinking, "answers": ans, "malformed": bad})
        f1k = f1_score(ans, sample.answer)
        print(f"      attempt{k+1}: f1={f1k:.3f} ans={sorted(ans)[:6]} "
              f"think={len(g.thinking)}", flush=True)
        if prev is not None and ans == prev:
            print(f"      (规避后仍与上次相同 -> 提前停止, 共{k+1}次)", flush=True)
            break

    k_used = len(attempts)
    answers = [a["answers"] for a in attempts]
    med, med_i = medoid_answer(answers)
    seq_f1, seq_em = f1_score(med, sample.answer), em_score(med, sample.answer)
    best_any = max(f1_score(a, sample.answer) for a in answers)

    # ---- state_diverse: 全部不同路线轨迹前置 ----
    # 论文序列化器允许 "traces or trace tail windows": 轨迹总长会撑爆上下文,
    # 故每条轨迹取末尾 tail_chars 字符 (尾部包含路径结论, 信息密度最高)。
    def state_call(traces):
        T = serialize_traces([tail(t, tail_chars) for t in traces])
        prompt = build_prompt(sample, T, "state")
        g = ollama_chat(model, GRAPH_SYSTEM, prompt, num_ctx=num_ctx,
                        think=True, temperature=0.7, num_predict=max_predict)
        ans, bad = extract_answer(g.content, g.thinking)
        return T, {"answers": ans, "f1": f1_score(ans, sample.answer),
                   "em": em_score(ans, sample.answer), "T_chars": len(T)}

    tail_chars = 3500
    try:
        T, st = state_call([a["thinking"] for a in attempts])
    except Exception as e:
        print(f"      state_diverse 全文失败({e}), 降级为尾窗 {tail_chars} 字符",
              flush=True)
        T, st = state_call([a["thinking"] for a in attempts])
    print(f"      state_diverse: f1={st['f1']:.3f} T={len(T)}", flush=True)

    f1s = [f1_score(a, sample.answer) for a in answers]
    print(f"  => attempts={k_used} f1s={[round(x,2) for x in f1s]} "
          f"medoid={seq_f1:.2f} best_any={best_any:.2f} state_div={st['f1']:.2f} "
          f"(truth={sorted(sample.answer)[:5]}{'…' if len(sample.answer)>5 else ''})",
          flush=True)
    return {
        "op_type": sample.op_type, "op_text": sample.op_text,
        "answer": sample.answer, "graph_chars": sample.graph_chars,
        "attempts_used": k_used,
        "attempt_f1s": f1s,
        "diverse_seq": {"f1": seq_f1, "em": seq_em, "picked": med_i},
        "best_any_attempt": {"f1": best_any,
                             "em": max(em_score(a, sample.answer)
                                       for a in answers)},
        "state_diverse": st,
        "trace_chars": len(T),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--target-chars", type=int, default=3000)
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--n-att", type=int, default=5)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="logs/diverse_2b_3k.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"模型={args.model} 目标字符≈{args.target_chars} 题数={args.n_questions} "
          f"最多尝试={args.n_att} (序贯带规避, 收敛提前停)", flush=True)

    records = []
    t0 = time.time()
    for i in range(args.n_questions):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, args.target_chars, op_type=op)
        print(f"\n[{i+1}/{args.n_questions}] {op} 图({sample.graph_chars}字符) "
              f"truth_size={len(sample.answer)}", flush=True)
        try:
            rec = run_question(args.model, sample, args.n_att, args.num_ctx,
                               args.max_predict)
        except Exception as e:
            rec = {"error": str(e), "op_type": op}
            print("   !! 错误:", e, flush=True)
        records.append(rec)

    ok = [r for r in records if "error" not in r]
    n = len(ok)
    summary = {"model": args.model, "target_chars": args.target_chars,
               "n_questions": args.n_questions, "n_att_max": args.n_att,
               "num_ctx": args.num_ctx, "n_ok": n,
               "elapsed_sec": round(time.time() - t0, 1)}
    if n:
        summary["avg_attempts_used"] = round(
            sum(r["attempts_used"] for r in ok) / n, 2)
        summary["early_stopped"] = sum(1 for r in ok if r["attempts_used"] < args.n_att)
        for cond in ("diverse_seq", "best_any_attempt", "state_diverse"):
            summary[f"{cond}_f1"] = round(sum(r[cond]["f1"] for r in ok) / n, 4)
            summary[f"{cond}_em"] = round(sum(r[cond]["em"] for r in ok) / n, 4)
        # 与独立采样对照组同题对比
        ctrl_path = os.path.join(os.path.dirname(args.out), "consensus_2b_3k.json")
        if os.path.exists(ctrl_path):
            ctrl = json.load(open(ctrl_path, encoding="utf-8"))["records"]
            pairs = [(r, c) for r, c in zip(ok, ctrl) if "error" not in c]
            if pairs:
                for name, key in (("diverse_seq", "diverse_seq"),
                                  ("state_diverse", "state_diverse")):
                    pass  # 对照组键不同, 下面手工比
                summary["vs_ctrl_diverseq_beats_vote3"] = sum(
                    1 for r, c in pairs
                    if r["diverse_seq"]["f1"] > c["state_vote3"]["f1"])
                summary["vs_ctrl_statediv_beats_stateall"] = sum(
                    1 for r, c in pairs
                    if r["state_diverse"]["f1"] > c["state_all"]["f1"])

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
