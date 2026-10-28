# -*- coding: utf-8 -*-
"""
run_scale —— 模型缩放轴: 用 qwen3.5:4b 重放 seed13 的 12 道 BFS 题, 三条件:
  S1) eonly_4b     [E_layered, q]        纯证据无图读出 (4B)
  S2) state_ctx_4b [E_layered, x, q]     状态+全图 (4B) —— 检验执行墙是否持续
  S3) paper_4b     [T4B, x, q]           论文原法 (4B 首遍 3 轨迹序列化前置)
目的: 补齐「状态-上下文反转是否随模型规模持续」的关键证据。
蒸馏与 chain3 完全一致 (2B+4B 集成, 逐层捕获), 保证 E 可比。
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
    GRAPH_SYSTEM, build_graph_x, build_prompt, em_score, extract_answer,
    f1_score, ollama_chat, sample_from_target_chars, serialize_traces,
)
from run_chain2 import (  # noqa: E402
    E_LAYER_HEADER, build_e_layered, distill_bfs_capture,
)

MODEL_BIG = "qwen3.5:4b"


def log(msg):
    print(msg, flush=True)


def e_only_prompt(sample, e_layered):
    return (E_LAYER_HEADER + "\n" + e_layered +
            "\n\nFull Operation:\n" + sample.op_text + "\n"
            "Return exactly one line in this format: "
            "Final Answer: [node1, node2]. Use [] for the empty set.")


def state_ctx_prompt(sample, e_layered):
    x_head, x_op = build_graph_x(sample)
    x = x_head + "\n" + x_op
    return (E_LAYER_HEADER + "\n" + e_layered + "\n\n" + x + "\n" + x_op)


def ask(model, system, prompt, num_ctx, max_predict, think=True):
    g = ollama_chat(model, system, prompt, num_ctx=num_ctx, think=think,
                    temperature=0.7, num_predict=max_predict)
    ans, _ = extract_answer(g.content, g.thinking)
    return ans, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="qwen3.5:4b")
    ap.add_argument("--big-model", default=MODEL_BIG)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-predict", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default="logs/chain_2b_3k_s13.json")
    ap.add_argument("--out", default="logs/scale_4b.json")
    args = ap.parse_args()

    distill_models = [args.model] + ([args.ensemble] if args.ensemble else [])

    ref = json.load(open(args.ref, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)
    bfs_qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs":
            assert sample.answer == ref[i]["answer"], f"重放校验失败: 题{i}"
            bfs_qs.append((i, sample))
    log(f"重放校验通过: 12 道 BFS 题; 缩放模型={args.big_model}")

    records, t0 = [], time.time()
    for qi, (idx, sample) in enumerate(bfs_qs):
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        hard = depth >= 4
        log(f"\n[{qi+1}/12] (原题#{idx+1}) bfs d={depth} "
            f"truth={len(sample.answer)} {'[HARD]' if hard else ''}")
        rec = {"orig_index": idx, "depth": depth, "hard": hard,
               "op_text": sample.op_text, "answer": sample.answer}

        # ---- 共用蒸馏 (与 chain3 同一套) ----
        ts = time.time()
        levels, det, ncalls = distill_bfs_capture(distill_models, sample,
                                                  args.num_ctx)
        e_layered = build_e_layered(levels, depth)
        rec["distill"] = {"det_f1": f1_score(det, sample.answer),
                          "n_calls": ncalls}
        log(f"      蒸馏: det={rec['distill']['det_f1']:.3f} "
            f"{ncalls}调用 {round(time.time()-ts)}s")

        # ---- S1: 纯证据无图 (4B) ----
        try:
            ans, _ = ask(args.big_model,
                         "You solve directed-graph algorithm problems.",
                         e_only_prompt(sample, e_layered),
                         args.num_ctx, args.max_predict)
            rec["eonly_4b"] = {"f1": f1_score(ans, sample.answer),
                               "em": em_score(ans, sample.answer)}
        except Exception as e:
            rec["eonly_4b"] = {"error": str(e)[:80], "f1": 0.0}
        log(f"      S1 纯证据4B: f1={rec['eonly_4b']['f1']:.3f}")

        # ---- S2: 状态+全图 (4B) ----
        try:
            ans, _ = ask(args.big_model, GRAPH_SYSTEM,
                         state_ctx_prompt(sample, e_layered),
                         args.num_ctx, args.max_predict)
            rec["state_ctx_4b"] = {"f1": f1_score(ans, sample.answer),
                                   "em": em_score(ans, sample.answer)}
        except Exception as e:
            rec["state_ctx_4b"] = {"error": str(e)[:80], "f1": 0.0}
        log(f"      S2 状态+全图4B: f1={rec['state_ctx_4b']['f1']:.3f}")

        # ---- S3: 论文原法 (4B 首遍 3 轨迹) ----
        try:
            ts = time.time()
            x_head, x_op = build_graph_x(sample)
            x = x_head + "\n" + x_op
            first = []
            for j in range(3):
                g = ollama_chat(args.big_model, GRAPH_SYSTEM, x,
                                num_ctx=args.num_ctx, think=True,
                                temperature=0.7,
                                num_predict=args.max_predict)
                first.append(g.thinking)
            T = serialize_traces(first)
            prompt = build_prompt(sample, T, "state")
            ans, _ = ask(args.big_model, GRAPH_SYSTEM, prompt,
                         args.num_ctx, args.max_predict)
            rec["paper_4b"] = {"f1": f1_score(ans, sample.answer),
                               "em": em_score(ans, sample.answer)}
            log(f"      S3 论文原法4B: f1={rec['paper_4b']['f1']:.3f} "
                f"({round(time.time()-ts)}s)")
        except Exception as e:
            rec["paper_4b"] = {"error": str(e)[:80], "f1": 0.0}
            log(f"      S3 论文原法4B: ERROR {rec['paper_4b']['error']}")

        records.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"records_so_far": records}, f, ensure_ascii=False,
                      indent=2)

    # ---- 汇总 ----
    n = len(records)
    summary = {"model": args.big_model, "n_questions": n,
               "elapsed_sec": round(time.time() - t0, 1)}
    for key in ("eonly_4b", "state_ctx_4b", "paper_4b"):
        summary[f"{key}_f1"] = round(
            sum(r[key]["f1"] for r in records) / n, 4)
        summary[f"{key}_em"] = round(
            sum(r[key]["em"] for r in records
                if "em" in r[key]) / max(1, sum(1 for r in records
                                                if "em" in r[key])), 4)
    hard = [r for r in records if r["hard"]]
    for key in ("eonly_4b", "state_ctx_4b", "paper_4b"):
        if hard:
            summary[f"hard_{key}_f1"] = round(
                sum(r[key]["f1"] for r in hard) / len(hard), 4)
    summary["hard_n"] = len(hard)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
