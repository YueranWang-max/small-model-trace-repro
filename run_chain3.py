# -*- coding: utf-8 -*-
"""
chain3 —— d≥4 执行墙的四路突破尝试 (持续改进, 尝试各种可能)

6.7 改判: 瓶颈不是状态设计, 是 2B 二过时的一次性多跳大读出 (d≥4 全条件≈0,
即使 det=1.0)。四条路, 共用一次逐层蒸馏捕获:
  A) 多轮小问: 逐层一问一答, 每轮只读一层的 frontier (14 节点内, 在 2B
     ~40 条抄写上限内) —— 绕开「一次性大读出」。visited 用模型自己的累积。
  B) 纯证据无图单过 (对照论文 Trace Only 消融): [E_layered, q] 无 x,
     排除「全图干扰」假设。
  C) 自一致性 Best-of-5: 同一纯证据 prompt 采 5 次 T=0.7, 集合多数投票
     (+union) —— 试图用采样次数换执行稳定性。
  D) 4B 能力检查: qwen3.5:4b 跑 [E_flat,x] 与 [T',E_layered,x] (仅 d≥4)
     —— 区分「2B 特有」还是「小模型共性」。
题目: 精确重放 seed 13 的 12 道 BFS 题 (与 run_chain2 同一校验)。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    GRAPH_SYSTEM, GenResult, build_graph_x, build_prompt, em_score,
    extract_answer, extract_list, f1_score, ollama_chat,
    sample_from_target_chars,
)
from run_chain2 import (  # noqa: E402
    E_HEADER, E_LAYER_HEADER, TP_HEADER, build_e_flat, build_e_layered,
    build_tprime, distill_bfs_capture,
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def log(msg):
    print(msg, flush=True)


def chat_mt(model, messages, num_ctx, temperature=0.7, num_predict=512,
            think=False, timeout=600):
    """多轮版 chat: messages 为完整历史。"""
    payload = {"model": model, "messages": messages, "stream": False,
               "think": think,
               "options": {"temperature": temperature, "num_ctx": num_ctx,
                           "num_predict": num_predict, "top_p": 0.9}}
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    msg = obj.get("message", {}) or {}
    return GenResult(thinking=msg.get("thinking", "") or "",
                     content=msg.get("content", "") or "", raw=obj)


def parse_frontier(text, level):
    m = re.findall(rf"Level {level} frontier: ?\[(.*?)\]", text)
    if m:
        return [x.strip() for x in m[-1].split(",") if x.strip()]
    ans, _ = extract_list(text)
    return ans


def mode_a_multiturn(model, sample, levels, num_ctx):
    """A: 逐层小问。返回 (最终frontier, 逐层记录)。"""
    messages = [{"role": "system", "content":
                 "You execute one step of a breadth-first search at a time, "
                 "using only the verified evidence given. Be exact."}]
    visited = set()
    start = levels[0]["queried_frontier"][0] if levels else ""
    visited.add(start)
    per_level = []
    for lv in levels:
        d = lv["level"]
        edges = "\n".join(f"{u} -> {v}" for u, v in lv["edges"]) or "NONE"
        user = (f"Level {d} evidence (all edges mechanically verified):\n"
                f"Queried frontier: {{{', '.join(lv['queried_frontier'])}}}\n"
                f"Confirmed edges:\n{edges}\n\n"
                f"Task: list the Level {d} frontier — the destination nodes "
                f"of the confirmed edges above, excluding any node already "
                f"listed in previous levels or the start node.\n"
                f"Answer with exactly one line: "
                f"Level {d} frontier: [node1, node2]")
        messages.append({"role": "user", "content": user})
        g = chat_mt(model, messages, num_ctx, temperature=0.0, num_predict=512)
        messages.append({"role": "assistant", "content": g.content})
        fr = parse_frontier(g.content, d)
        new = [n for n in fr if n not in visited]
        visited.update(fr)
        per_level.append({"level": d, "model_frontier": fr,
                          "deduped_new": new})
    final = per_level[-1]["model_frontier"] if per_level else []
    return final, per_level


def e_only_prompt(sample, e_layered):
    return (E_LAYER_HEADER + "\n" + e_layered +
            "\n\nFull Operation:\n" + sample.op_text + "\n"
            "Return exactly one line in this format: "
            "Final Answer: [node1, node2]. Use [] for the empty set.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="qwen3.5:4b")
    ap.add_argument("--big-model", default="qwen3.5:4b")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default="logs/chain_2b_3k_s13.json")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--out", default="logs/chain3_2b.json")
    args = ap.parse_args()

    distill_models = [args.model] + ([args.ensemble] if args.ensemble else [])

    # ---- 重放 seed 13 的 12 道 BFS 题 ----
    ref = json.load(open(args.ref, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)
    bfs_qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs":
            assert sample.answer == ref[i]["answer"], f"重放校验失败: 题{i}"
            bfs_qs.append((i, sample))
    log(f"重放校验通过: 12 道 BFS 题; 四路模式 A/B/C/D")

    records, t0 = [], time.time()
    for qi, (idx, sample) in enumerate(bfs_qs):
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        hard = depth >= 4
        log(f"\n[{qi+1}/12] (原题#{idx+1}) bfs d={depth} "
            f"truth={len(sample.answer)} {'[HARD]' if hard else ''}")
        rec = {"orig_index": idx, "depth": depth, "hard": hard,
               "op_text": sample.op_text, "answer": sample.answer}

        # ---- 共用蒸馏 ----
        ts = time.time()
        levels, det, ncalls = distill_bfs_capture(distill_models, sample,
                                                  args.num_ctx)
        e_flat = build_e_flat(levels, depth)
        e_layered = build_e_layered(levels, depth)
        tp = build_tprime(levels)
        rec["distill"] = {"det_f1": f1_score(det, sample.answer),
                          "n_calls": ncalls}
        log(f"      蒸馏: det={rec['distill']['det_f1']:.3f} "
            f"{ncalls}调用 {round(time.time()-ts)}s")

        # ---- A: 多轮小问 ----
        try:
            final, per_level = mode_a_multiturn(args.model, sample, levels,
                                                args.num_ctx)
            rec["A_multiturn"] = {
                "f1": f1_score(final, sample.answer),
                "em": em_score(final, sample.answer),
                "per_level": [{"level": p["level"],
                               "f1": f1_score(p["model_frontier"],
                                              [lv for lv in levels
                                               if lv["level"] == p["level"]]
                                              [0]["new_frontier"])
                               if any(l["level"] == p["level"]
                                      for l in levels) else None,
                               "n_model": len(p["model_frontier"])}
                              for p in per_level]}
        except Exception as e:
            rec["A_multiturn"] = {"error": str(e)[:80], "f1": 0.0}
        log(f"      A多轮: f1={rec['A_multiturn']['f1']:.3f}")

        # ---- B: 纯证据无图单过 ----
        try:
            g = ollama_chat(args.model,
                            "You solve directed-graph algorithm problems.",
                            e_only_prompt(sample, e_layered),
                            num_ctx=args.num_ctx, think=True, temperature=0.7,
                            num_predict=args.max_predict)
            ans, _ = extract_answer(g.content, g.thinking)
            rec["B_eonly"] = {"f1": f1_score(ans, sample.answer),
                              "em": em_score(ans, sample.answer)}
        except Exception as e:
            rec["B_eonly"] = {"error": str(e)[:80], "f1": 0.0}
        log(f"      B纯证据: f1={rec['B_eonly']['f1']:.3f}")

        # ---- C: 自一致性 Best-of-5 (仅 hard) ----
        if hard:
            votes = []
            for k in range(args.n_samples):
                try:
                    g = ollama_chat(args.model,
                                    "You solve directed-graph algorithm "
                                    "problems.",
                                    e_only_prompt(sample, e_layered),
                                    num_ctx=args.num_ctx, think=False,
                                    temperature=0.7, num_predict=512)
                    ans, _ = extract_answer(g.content, g.thinking)
                    votes.append(tuple(sorted(set(ans))))
                except Exception:
                    continue
            if votes:
                cnt = Counter(votes)
                maj = list(cnt.most_common(1)[0][0])
                union = sorted({n for v in votes for n in v})
                rec["C_bestof5"] = {
                    "majority_f1": f1_score(maj, sample.answer),
                    "union_f1": f1_score(union, sample.answer),
                    "n_votes": len(votes),
                    "any_correct": max(f1_score(list(v), sample.answer)
                                       for v in votes)}
                log(f"      C投票: majority={rec['C_bestof5']['majority_f1']:.3f} "
                    f"union={rec['C_bestof5']['union_f1']:.3f}")
            else:
                rec["C_bestof5"] = {"error": "no votes", "majority_f1": 0.0}

        # ---- D: 4B 能力检查 (仅 hard) ----
        if hard:
            rec["D_4b"] = {}
            x_head, x_op = build_graph_x(sample)
            x = x_head + "\n" + x_op
            for name, block in {
                "evidence_flat": E_HEADER + "\n" + e_flat,
                "chain_tp_layered": TP_HEADER + "\n" + tp + "\n\n"
                                    + E_LAYER_HEADER + "\n" + e_layered,
            }.items():
                try:
                    prompt = block + "\n" + x + "\n" + x_op
                    g = ollama_chat(args.big_model, GRAPH_SYSTEM, prompt,
                                    num_ctx=args.num_ctx, think=True,
                                    temperature=0.7,
                                    num_predict=args.max_predict)
                    ans, _ = extract_answer(g.content, g.thinking)
                    rec["D_4b"][name] = {"f1": f1_score(ans, sample.answer),
                                         "em": em_score(ans, sample.answer)}
                except Exception as e:
                    rec["D_4b"][name] = {"error": str(e)[:80], "f1": 0.0}
                log(f"      D 4B {name}: "
                    f"f1={rec['D_4b'][name].get('f1', 0):.3f}")

        records.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"records_so_far": records}, f, ensure_ascii=False,
                      indent=2)

    # ---- 汇总 ----
    n = len(records)
    summary = {"model": args.model, "n_questions": n,
               "elapsed_sec": round(time.time() - t0, 1)}
    for key in ("A_multiturn", "B_eonly"):
        summary[f"{key}_f1"] = round(
            sum(r[key]["f1"] for r in records) / n, 4)
    hard = [r for r in records if r["hard"]]
    if hard:
        summary["hard_n"] = len(hard)
        summary["hard_A_f1"] = round(
            sum(r["A_multiturn"]["f1"] for r in hard) / len(hard), 4)
        summary["hard_B_f1"] = round(
            sum(r["B_eonly"]["f1"] for r in hard) / len(hard), 4)
        if all("C_bestof5" in r for r in hard):
            summary["hard_C_majority_f1"] = round(
                sum(r["C_bestof5"]["majority_f1"] for r in hard)
                / len(hard), 4)
            summary["hard_C_union_f1"] = round(
                sum(r["C_bestof5"]["union_f1"] for r in hard) / len(hard), 4)
            summary["hard_C_any_f1"] = round(
                sum(r["C_bestof5"]["any_correct"] for r in hard)
                / len(hard), 4)
        for name in ("evidence_flat", "chain_tp_layered"):
            vals = [r["D_4b"][name]["f1"] for r in hard
                    if "D_4b" in r and name in r.get("D_4b", {})]
            if vals:
                summary[f"hard_D4b_{name}_f1"] = round(
                    sum(vals) / len(vals), 4)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
