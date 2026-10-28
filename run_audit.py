# -*- coding: utf-8 -*-
"""
run_audit —— S2-hard thinking 审计。
问题: 4B 的 S2(状态+全图) 在 d>=3 的 7 道题上 F1=0.000(其中 3 题 placeholder)。
零分是否低估了真实推导? T=0 重跑 S2, 保存完整 thinking/content,
多模式提取(bracket 列表扫描 + Final Answer 行 + 提取器), 对比 truth。
蒸馏用同样的重放+捕获, e_layered 从检查点恢复(若有 _e_layered)否则重新蒸馏。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    GRAPH_SYSTEM, extract_answer, f1_score, ollama_chat,
    sample_from_target_chars,
)
from run_chain2 import (  # noqa: E402
    E_LAYER_HEADER, build_e_layered, distill_bfs_capture,
)
from rerun_scale import BIG, BIG_CTX, BIG_PRED, state_ctx_prompt  # noqa: E402


def scan_thinking_answers(thinking):
    """在 thinking 里扫描所有 [..] 列表与 Final Answer 行, 返回候选集合."""
    cands = []
    for m in re.finditer(r"Final Answer[:：]\s*(\[[^\]]*\])", thinking):
        cands.append(m.group(1))
    # 尾部 3000 字符内的 bracket 列表(常见: 推导到最后给出集合)
    tail = thinking[-3000:]
    for m in re.finditer(r"\[([a-zA-Z0-9_,\s\u4e00-\u9fff]*)\]", tail):
        s = m.group(1).strip()
        if s:
            cands.append("[" + s + "]")
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ckpt", default="logs/scale_4b_v2.json")
    ap.add_argument("--out", default="logs/audit_s2_hard.json")
    ap.add_argument("--num-ctx", type=int, default=16384)
    args = ap.parse_args()
    ref_path = args.ref or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logs", "chain_2b_3k_s13.json")

    ref = json.load(open(ref_path, encoding="utf-8"))["records"]
    ck = json.load(open(args.ckpt, encoding="utf-8"))["records"]
    ck_by_idx = {r["orig_index"]: r for r in ck}

    rng = random.Random(args.seed)
    targets = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        sample = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs" and ck_by_idx[i]["S2"]["f1"] < 0.5 and ck_by_idx[i]["depth"] >= 3:
            assert sample.answer == ref[i]["answer"], f"重放失败: {i}"
            targets.append((i, sample))
    print(f"审计目标: {len(targets)} 道 S2<0.5 的 d>=3 题", flush=True)

    out = []
    for idx, sample in targets:
        depth = ck_by_idx[idx]["depth"]
        print(f"\n=== #{idx+1} d={depth} truth={sample.answer} ===", flush=True)
        # e_layered: 优先从检查点恢复, 否则重蒸馏
        e_layered = ck_by_idx[idx].get("_e_layered")
        if not e_layered:
            t0 = time.time()
            levels, det, _ = distill_bfs_capture(
                ["openbmb/minicpm5-2b", "qwen3.5:4b"], sample, args.num_ctx)
            e_layered = build_e_layered(levels, depth)
            print(f"  重蒸馏 det={f1_score(det, sample.answer):.3f} "
                  f"({round(time.time()-t0)}s)", flush=True)
        prompt = state_ctx_prompt(sample, e_layered)

        t0 = time.time()
        g = ollama_chat(BIG, GRAPH_SYSTEM, prompt, num_ctx=BIG_CTX,
                        think=True, temperature=0.0, num_predict=BIG_PRED)
        ans, bad = extract_answer(g.content, g.thinking)
        cands = scan_thinking_answers(g.thinking + "\n" + g.content)
        rec = {
            "orig_index": idx, "depth": depth, "answer": sample.answer,
            "f1_extractor": f1_score(ans, sample.answer),
            "content": g.content, "thinking": g.thinking,
            "candidates": cands[:20],
            "elapsed": round(time.time() - t0, 1),
        }
        # 候选里最好的 F1 (答案是否藏在 thinking 中)
        best = 0.0
        best_c = None
        for c in cands:
            try:
                f = f1_score(c, sample.answer)
            except Exception:
                continue
            if f > best:
                best, best_c = f, c
        rec["best_candidate_f1"] = round(best, 4)
        rec["best_candidate"] = best_c
        out.append(rec)
        print(f"  T=0: extractor_f1={rec['f1_extractor']:.3f} "
              f"best_in_thinking={best:.3f} ({rec['elapsed']}s) "
              f"content={len(g.content)}ch thinking={len(g.thinking)}ch",
              flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== 审计汇总 =====", flush=True)
    n_recover = sum(1 for r in out if r["best_candidate_f1"] >= 0.9
                    and r["f1_extractor"] < 0.5)
    n_true0 = sum(1 for r in out if r["best_candidate_f1"] < 0.5)
    print(f"  格式掩蔽(答案在 thinking 里, 提取器漏): {n_recover}/{len(out)}",
          flush=True)
    print(f"  真零分(候选也到不了 0.5): {n_true0}/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
