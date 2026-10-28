# -*- coding: utf-8 -*-
"""
rerun_scale —— 4B 缩放轴重跑 (修正输出预算伪影)。
诊断结论: qwen3.5:4b thinking 需 12-15K 字符 (~4-5K token), 4096 num_predict
会把 thinking 掐断在半途, content 恒空, 提取器抓到 prompt 占位符 => 假 0 分。
修正: num_predict=12288, num_ctx=65536 (qwen3.5:4b 支持 256K)。
条件 (读出模型均为 qwen3.5:4b):
  S1 纯证据无图 [E_layered, q]
  S2 状态+全图 [E_layered, x, q]
  S3 论文原法 3 轨迹前置 [T, x, q] (顺带记录首遍直接作答 F1 = 4B 直接基线)
每题落盘断点续跑。
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
    GRAPH_SYSTEM, build_prompt, em_score, extract_answer, f1_score,
    ollama_chat, sample_from_target_chars, serialize_traces,
)
from run_chain2 import (  # noqa: E402
    E_LAYER_HEADER, build_e_layered, distill_bfs_capture,
)

BIG = "qwen3.5:4b"
BIG_CTX = 65536
BIG_PRED = 12288


def log(m):
    print(m, flush=True)


def big_chat(sample_or_prompt, system=GRAPH_SYSTEM, temperature=0.7):
    prompt = sample_or_prompt
    g = ollama_chat(BIG, system, prompt, num_ctx=BIG_CTX, think=True,
                    temperature=temperature, num_predict=BIG_PRED)
    ans, bad = extract_answer(g.content, g.thinking)
    placeholder = ans == ["node1", "node2"]
    return ans, {"content_ch": len(g.content), "thinking_ch": len(g.thinking),
                 "placeholder": placeholder, "malformed": bad}


def state_ctx_prompt(sample, e_layered):
    from repro_lib import build_graph_x
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
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default="logs/chain_2b_3k_s13.json")
    ap.add_argument("--out", default="logs/scale_4b_v2.json")
    ap.add_argument("--num-ctx", type=int, default=16384)
    args = ap.parse_args()

    ref = json.load(open(args.ref, encoding="utf-8"))["records"]
    rng = random.Random(args.seed)
    bfs_qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs":
            assert sample.answer == ref[i]["answer"], f"重放校验失败: 题{i}"
            bfs_qs.append((i, sample))
    log(f"重放校验通过: 12 道 BFS 题; 读出模型={BIG} "
        f"num_ctx={BIG_CTX} num_predict={BIG_PRED}")

    done = {}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            for r in old.get("records_so_far", []):
                # 残缺记录也保留: 只跳过已完成的条件
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
            log(f"[{qi+1}/12] 题#{idx+1} 已有完整结果, 跳过")
            continue
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        hard = depth >= 4
        log(f"\n[{qi+1}/12] (原题#{idx+1}) bfs d={depth} "
            f"truth={len(sample.answer)} {'[HARD]' if hard else ''}")
        rec = prev if prev else {"orig_index": idx, "depth": depth,
                                 "hard": hard, "op_text": sample.op_text,
                                 "answer": sample.answer}
        if rec not in records:
            records.append(rec)

        if "distill" not in rec:
            ts = time.time()
            levels, det, ncalls = distill_bfs_capture(
                ["openbmb/minicpm5-2b", "qwen3.5:4b"], sample, args.num_ctx)
            e_layered = build_e_layered(levels, depth)
            rec["distill"] = {"det_f1": f1_score(det, sample.answer),
                              "n_calls": ncalls}
            rec["_e_layered"] = e_layered
            log(f"      蒸馏: det={rec['distill']['det_f1']:.3f} "
                f"{ncalls}调用 {round(time.time()-ts)}s")
        e_layered = rec.get("_e_layered")
        if e_layered is None:
            # 断点恢复时 e_layered 不落盘(太大), 重蒸馏一次(确定性 T=0)
            ts = time.time()
            levels, det, ncalls = distill_bfs_capture(
                ["openbmb/minicpm5-2b", "qwen3.5:4b"], sample, args.num_ctx)
            e_layered = build_e_layered(levels, depth)
            log(f"      重蒸馏(恢复): det={f1_score(det, sample.answer):.3f} "
                f"{round(time.time()-ts)}s")
        rec["_e_layered"] = e_layered

        # ---- S1: 纯证据无图 ----
        if "S1" not in rec:
            ans, meta = big_chat(e_only_prompt(sample, e_layered))
            rec["S1"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer), **meta}
            log(f"      S1 纯证据: f1={rec['S1']['f1']:.3f} {meta}")
            save()

        # ---- S2: 状态+全图 ----
        if "S2" not in rec:
            ans, meta = big_chat(state_ctx_prompt(sample, e_layered))
            rec["S2"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer), **meta}
            log(f"      S2 状态+全图: f1={rec['S2']['f1']:.3f} {meta}")
            save()

        # ---- S3: 论文原法 (4B 首遍 3 轨迹 + [T,x] 二过) ----
        if "S3" not in rec:
            from repro_lib import build_graph_x
            x_head, x_op = build_graph_x(sample)
            x = x_head + "\n" + x_op
            traces, first_f1s = [], []
            for j in range(3):
                g = ollama_chat(BIG, GRAPH_SYSTEM, x, num_ctx=BIG_CTX,
                                think=True, temperature=0.7,
                                num_predict=BIG_PRED)
                a, _ = extract_answer(g.content, g.thinking)
                traces.append(g.thinking)
                first_f1s.append(f1_score(a, sample.answer))
            T = serialize_traces(traces)
            ans, meta = big_chat(build_prompt(sample, T, "state"))
            rec["S3"] = {"f1": f1_score(ans, sample.answer),
                         "em": em_score(ans, sample.answer),
                         "first_mean_f1": round(sum(first_f1s) / 3, 4),
                         "T_chars": len(T), **meta}
            log(f"      S3 论文原法: f1={rec['S3']['f1']:.3f} "
                f"首遍直接={rec['S3']['first_mean_f1']:.3f} {meta}")
            save()
        rec.pop("_e_layered", None)
        save()

    n = len(records)
    summary = {"model": BIG, "num_ctx": BIG_CTX, "num_predict": BIG_PRED,
               "n_questions": n, "elapsed_sec": round(time.time() - t0, 1)}
    for key in ("S1", "S2", "S3"):
        summary[f"{key}_f1"] = round(sum(r[key]["f1"] for r in records) / n, 4)
        summary[f"{key}_em"] = round(sum(r[key]["em"] for r in records) / n, 4)
    hard = [r for r in records if r["hard"]]
    for key in ("S1", "S2", "S3"):
        summary[f"hard_{key}_f1"] = round(
            sum(r[key]["f1"] for r in hard) / len(hard), 4)
    summary["hard_n"] = len(hard)
    ph = sum(1 for r in records for k in ("S1", "S2", "S3") if r[k]["placeholder"])
    summary["placeholder_responses"] = ph
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
