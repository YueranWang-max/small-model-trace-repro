# -*- coding: utf-8 -*-
"""
塌方片诊断: 精确重建 256K 片17 与 512K 片83 的抽取输入,
独立重跑每个 (模型×温度) 组合多次, 记录原始输出 —— 判断:
  A) 内容触发(确定性) vs 随机瞬态(可重跑修复)
  B) 两种塌方模式的机理 (抽空 vs 幻觉输出)
另取邻近正常片作对照组。
"""
import json
import random
import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import sample_from_target_chars, ollama_chat
from run_distill3 import chunk_edges, parse_edges, EXTRACT_SYSTEM

OUT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logs", "diag_chunks.txt")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", "diag_chunks.json")

TASK = ("List EVERY edge line in this portion, verbatim, one per line. "
        "Copy all of them. Do not skip any line.")
NUM_CTX = 16384
NUM_PREDICT = 3072
REPEATS = 3

# (tag, size, chunk_idx_1based, 原始记录)
TARGETS = [
    ("256K_c17", 256000, 17, {"cands": 6, "kept": 6, "lines": 266}),
    ("256K_c16", 256000, 16, {"cands": 266, "kept": 266, "lines": 266}),   # 对照
    ("512K_c83", 512000, 83, {"cands": 384, "kept": 1, "lines": 264}),
    ("512K_c82", 512000, 82, None),  # 对照 (现场记录: 正常)
]


def verify_edge(u, v, chunk):
    return re.search(rf"^{re.escape(u)} -> {re.escape(v)}$", chunk,
                     re.MULTILINE) is not None


def main():
    log_f = open(OUT_LOG, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    results = []
    # 缓存已生成的图, 避免重复生成大图
    graph_cache = {}

    def get_chunks(size):
        if size not in graph_cache:
            rng = random.Random(11 + size)
            sample = sample_from_target_chars(rng, size, op_type="bfs")
            graph_cache[size] = chunk_edges(sample, 4000)
        return graph_cache[size]

    for tag, size, cidx, orig in TARGETS:
        chunks = get_chunks(size)
        ch = chunks[cidx - 1]
        n_lines = ch.count("\n") + 1
        log(f"\n{'='*64}\n[{tag}] size={size} 片{cidx} 行数={n_lines} "
            f"原始记录={orig}")
        # 片内容特征: 前两行 + 节点统计
        lines = ch.split("\n")
        log(f"  首2行: {lines[:2]}")
        nodes = set(re.findall(r"n\d+", ch))
        log(f"  片内不同节点数: {len(nodes)}")

        for model, temp in [("openbmb/minicpm5-2b", 0.0),
                            ("openbmb/minicpm5-2b", 0.7),
                            ("openbmb/minicpm5-2b", 1.0),
                            ("qwen3.5:4b", 0.7)]:
            for rep in range(REPEATS):
                t0 = time.time()
                try:
                    g = ollama_chat(model, EXTRACT_SYSTEM,
                                    f"{TASK}\n\nPortion of the edge list:\n{ch}",
                                    num_ctx=NUM_CTX, think=False,
                                    temperature=temp, num_predict=NUM_PREDICT)
                except Exception as e:
                    log(f"  {model} T={temp} r{rep}: 调用失败 {e}")
                    continue
                dt = time.time() - t0
                cands = parse_edges(g.content)
                kept = {e for e in cands if verify_edge(e[0], e[1], ch)}
                n_out_lines = g.content.count("\n") + 1 if g.content else 0
                # 输出形态特征
                head = g.content[:160].replace("\n", " | ")
                dup = len(cands) - len(set(cands))
                log(f"  {model} T={temp} r{rep}: {dt:5.1f}s "
                    f"输出{n_out_lines}行/{len(g.content)}字 候选{len(cands)}"
                    f"(重复{dup}) 核验{len(kept)}")
                log(f"      输出头: {head!r}")
                results.append({
                    "tag": tag, "model": model, "temp": temp, "rep": rep,
                    "sec": round(dt, 1), "out_chars": len(g.content),
                    "n_cands": len(cands), "n_unique": len(set(cands)),
                    "n_kept": len(kept), "head": head,
                })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\n诊断完成, 共 {len(results)} 次调用, 结果已存 {OUT_JSON}")
    log_f.close()


if __name__ == "__main__":
    main()
