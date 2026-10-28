# -*- coding: utf-8 -*-
"""
run_scale8b —— 8B 缩放探针 (双模型)。
目标: 收窄「轨迹消费能力涌现区间」。
  Qwen 轴:  qwen3.5:4b (0.917) -> qwen3-vl:8b
  MiniCPM 轴: minicpm5-2b (0.326) -> minicpm-o4.5 (8.2B)
题目: seed 13 重放 12 道 BFS 题中 depth>=4 的 4 道难题 (与 4B run 的 hard 子集同构)。
条件: S1 纯证据 [E,q] / S2 状态+全图 [E,x,q] / S3 论文原法 [T,x,q]。
蒸馏共享 (minicpm5-2b + qwen3.5:4b 集成), 与 4B run 同源。
伪影护栏: num_predict=12288; 空正文/占位符 => 预算加倍重试一次。
逐条件断点续跑。
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
    GRAPH_SYSTEM, build_prompt, build_graph_x, em_score, extract_answer,
    f1_score, ollama_chat, sample_from_target_chars, serialize_traces,
)
from run_chain2 import (  # noqa: E402
    E_LAYER_HEADER, build_e_layered, distill_bfs_capture,
)

MODELS = {
    # 32768: 图仅 ~3K 字符, 65K ctx 的 KV cache 挤爆 12GB 显存致生成 <5 tok/s 反复超时
    "qwen3-vl:8b": {"num_ctx": 32768, "num_predict": 12288},
    "openbmb/minicpm-o4.5:latest": {"num_ctx": 40960, "num_predict": 12288},
    "qwen2.5:7b": {"num_ctx": 16384, "num_predict": 4096, "think": False},
    # 同代际参数扫描 (Qwen3 家族, 16K 图仅 3K 字符, 16K ctx 足够)
    "qwen3:1.7b": {"num_ctx": 16384, "num_predict": 12288},
    "qwen3:4b": {"num_ctx": 16384, "num_predict": 12288},
    "qwen3:8b": {"num_ctx": 32768, "num_predict": 12288},
}


def log(m):
    print(m, flush=True)


def chat_with_guard(model, system, prompt, cfg, temperature=0.7):
    """带伪影自愈的调用: 空正文/占位符 => 输出预算加倍重试一次。"""
    for attempt in range(2):
        g = ollama_chat(model, system, prompt, num_ctx=cfg["num_ctx"],
                        think=cfg.get("think", True), temperature=temperature,
                        num_predict=cfg["num_predict"] * (2 ** attempt))
        ans, bad = extract_answer(g.content, g.thinking)
        placeholder = ans == ["node1", "node2"]
        meta = {"content_ch": len(g.content), "thinking_ch": len(g.thinking),
                "placeholder": placeholder, "malformed": bad,
                "attempt": attempt + 1}
        if not placeholder and len(g.content) > 0:
            return ans, meta
        log(f"      [guard] attempt{attempt+1} content={len(g.content)}ch "
            f"placeholder={placeholder} => 预算加倍重试")
    return ans, meta


COT_PREFIX = ("Think step by step. First write your step-by-step reasoning, "
              "then end with a line 'Final Answer: [node1, node2]'. "
              "Use [] for the empty set.\n\n")


def first_pass(model, x, cfg):
    """S3 首遍单次采样: thinking 模型取 thinking 通道;
    非思考模型 (think=False) 走 CoT-in-content, 轨迹=去掉 Final Answer 行的正文。"""
    if cfg.get("think", True):
        g = ollama_chat(model, GRAPH_SYSTEM, x, num_ctx=cfg["num_ctx"],
                        think=True, temperature=0.7,
                        num_predict=cfg["num_predict"], timeout=4800)
        a, _ = extract_answer(g.content, g.thinking)
        return a, g.thinking, {"content_ch": len(g.content),
                               "thinking_ch": len(g.thinking)}
    g = ollama_chat(model, GRAPH_SYSTEM, COT_PREFIX + x,
                    num_ctx=cfg["num_ctx"], think=False, temperature=0.7,
                    num_predict=cfg["num_predict"], timeout=4800)
    a, _ = extract_answer(g.content, "")
    trace = re.sub(r"Final Answer:.*$", "", g.content or "",
                   flags=re.S | re.I).strip()
    return a, trace, {"content_ch": len(g.content), "thinking_ch": 0,
                      "cot_in_content": True}


def state_ctx_prompt(sample, e_layered):
    x_head, x_op = build_graph_x(sample)
    return (E_LAYER_HEADER + "\n" + e_layered + "\n\n"
            + x_head + "\n" + x_op)


def e_only_prompt(sample, e_layered):
    return (E_LAYER_HEADER + "\n" + e_layered +
            "\n\nFull Operation:\n" + sample.op_text + "\n"
            "Return exactly one line in this format: "
            "Final Answer: [node1, node2]. Use [] for the empty set.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default="logs/chain_2b_3k_s13.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--n-hard", type=int, default=4)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    BIG = args.model

    ref = json.load(open(args.ref, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)
    bfs_qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs":
            assert sample.answer == ref[i]["answer"], f"重放校验失败: 题{i}"
            depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
            if depth >= 4:
                bfs_qs.append((i, sample))
    bfs_qs = bfs_qs[: args.n_hard]
    log(f"重放校验通过; 读出模型={BIG} ctx={cfg['num_ctx']} "
        f"pred={cfg['num_predict']}; 难题 {len(bfs_qs)} 道 "
        f"(d>=4): {[i+1 for i, _ in bfs_qs]}")

    done = {}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            for r in old.get("records_so_far", []):
                done[r["orig_index"]] = r
            log(f"断点续跑: 已有 {len(done)} 题的部分条件")
        except Exception:
            pass

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"records_so_far": records}, f, ensure_ascii=False,
                      indent=2)

    records, t0 = [], time.time()
    for qi, (idx, sample) in enumerate(bfs_qs):
        prev = done.get(idx)
        if prev and all(k in prev for k in ("S1", "S2", "S3")):
            records.append(prev)
            log(f"[{qi+1}/{len(bfs_qs)}] 题#{idx+1} 完整, 跳过")
            continue
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        log(f"\n[{qi+1}/{len(bfs_qs)}] (原题#{idx+1}) bfs d={depth} "
            f"truth={len(sample.answer)}")
        rec = prev if prev else {"orig_index": idx, "depth": depth,
                                 "hard": True, "op_text": sample.op_text,
                                 "answer": sample.answer}
        if rec not in records:
            records.append(rec)

        if "_e_layered" not in rec:
            ts = time.time()
            levels, det, ncalls = distill_bfs_capture(
                ["openbmb/minicpm5-2b", "qwen3.5:4b"], sample, args.num_ctx)
            rec["_e_layered"] = build_e_layered(levels, depth)
            rec["distill"] = {"det_f1": f1_score(det, sample.answer),
                              "n_calls": ncalls}
            log(f"      蒸馏: det={rec['distill']['det_f1']:.3f} "
                f"{ncalls}调用 {round(time.time()-ts)}s")
            save()
        e_layered = rec["_e_layered"]

        if "S1" not in rec:
            ans, meta = chat_with_guard(
                BIG, GRAPH_SYSTEM, e_only_prompt(sample, e_layered), cfg)
            rec["S1"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer), **meta}
            log(f"      S1 纯证据: f1={rec['S1']['f1']:.3f} {meta}")
            save()

        if "S2" not in rec:
            ans, meta = chat_with_guard(
                BIG, GRAPH_SYSTEM, state_ctx_prompt(sample, e_layered), cfg)
            rec["S2"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer), **meta}
            log(f"      S2 状态+全图: f1={rec['S2']['f1']:.3f} {meta}")
            save()

        if "S3" not in rec:
            x_head, x_op = build_graph_x(sample)
            x = x_head + "\n" + x_op
            traces, first_f1s = [], []
            for j in range(3):
                a, tr, _ = first_pass(BIG, x, cfg)
                traces.append(tr)
                first_f1s.append(f1_score(a, sample.answer))
            T = serialize_traces(traces)
            ans, meta = chat_with_guard(
                BIG, GRAPH_SYSTEM, build_prompt(sample, T, "state"), cfg)
            rec["S3"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer),
                         "first_mean_f1": round(sum(first_f1s) / 3, 4),
                         "T_chars": len(T), **meta}
            log(f"      S3 论文原法: f1={rec['S3']['f1']:.3f} "
                f"首遍直接={rec['S3']['first_mean_f1']:.3f} {meta}")
            save()

    n = len(records)
    summary = {"model": BIG, "num_ctx": cfg["num_ctx"],
               "num_predict": cfg["num_predict"],
               "think": cfg.get("think", True), "n_questions": n,
               "question_indices": [i for i, _ in bfs_qs],
               "elapsed_sec": round(time.time() - t0, 1)}
    for key in ("S1", "S2", "S3"):
        summary[f"{key}_f1"] = round(sum(r[key]["f1"] for r in records) / n, 4)
        summary[f"{key}_em"] = round(sum(r[key]["em"] for r in records) / n, 4)
    ph = sum(1 for r in records for k in ("S1", "S2", "S3")
             if r[k].get("placeholder"))
    summary["placeholder_responses"] = ph
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
