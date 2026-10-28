# -*- coding: utf-8 -*-
"""
run_scale_probe —— 诊断 4B 读出的零分是真能力结果还是伪影。
对 seed13 题 #1 (d2) 与 #3 (d4) 各跑一次 S2 (状态+全图), 打印:
  - 生成 content / thinking 的长度与尾部 (看是否被 ctx 截断)
  - num_ctx 放大到 32768 再跑一次对比
"""
from __future__ import annotations
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    GRAPH_SYSTEM, extract_answer, f1_score, ollama_chat,
    sample_from_target_chars,
)
from run_chain2 import (  # noqa: E402
    E_LAYER_HEADER, build_e_layered, distill_bfs_capture,
)
from run_scale import state_ctx_prompt  # noqa: E402

REF = "logs/chain_2b_3k_s13.json"
BIG = "qwen3.5:4b"


def log(m):
    print(m, flush=True)


def probe(sample, e_layered, num_ctx, label):
    prompt = state_ctx_prompt(sample, e_layered)
    approx_tokens = len(prompt) // 3
    g = ollama_chat(BIG, GRAPH_SYSTEM, prompt, num_ctx=num_ctx, think=True,
                    temperature=0.7, num_predict=4096)
    ans, _ = extract_answer(g.content, g.thinking)
    log(f"  [{label}] ctx={num_ctx} prompt_chars={len(prompt)} "
        f"(~{approx_tokens} tok) content={len(g.content)}ch "
        f"thinking={len(g.thinking)}ch ans={ans[:60]}")
    log(f"    content_tail: {g.content[-200:]!r}")
    log(f"    F1 = {f1_score(ans, sample.answer):.3f}")
    return f1_score(ans, sample.answer)


def main():
    ref = json.load(open(REF, encoding="utf-8"))["records"]
    rng = random.Random(13)
    qs = []
    for i in range(24):
        op = "bfs" if i % 2 == 0 else "parents"
        s = sample_from_target_chars(rng, 3000, op_type=op)
        if op == "bfs" and i in (0, 2):
            assert s.answer == ref[i]["answer"]
            qs.append((i, s))

    for idx, sample in qs:
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        log(f"\n=== 题#{idx+1} d={depth} truth={len(sample.answer)} ===")
        # 只用 2B 蒸馏快速拿 E (不必等集成, 探针只看读出伪影)
        levels, det, ncalls = distill_bfs_capture(["openbmb/minicpm5-2b"],
                                                  sample, 16384)
        e_layered = build_e_layered(levels, depth)
        log(f"  蒸馏 det={f1_score(det, sample.answer):.3f}")
        probe(sample, e_layered, 16384, "ctx16k")
        probe(sample, e_layered, 32768, "ctx32k")


if __name__ == "__main__":
    main()
