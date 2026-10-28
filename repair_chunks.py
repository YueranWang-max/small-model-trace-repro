# -*- coding: utf-8 -*-
"""
塌方片定向修复:
  - 256K 片17: 重扫并合并进 transcribe_256k_db.json, 重算召回率
  - 512K 片83: 重扫, 统计能补回多少条漏检边 -> 修正召回率估计 (原边库未落盘, 无法全库重建)
要点: 不用饱和提前停(原始事故的共犯: T=0 幻觉大集合让后续扫描被判无新贡献),
每个 (模型×温度) 独立扫描, 机械核验后取并集。
"""
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import sample_from_target_chars, ollama_chat
from run_distill3 import chunk_edges, parse_edges, EXTRACT_SYSTEM

BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")
TASK = ("List EVERY edge line in this portion, verbatim, one per line. "
        "Copy all of them. Do not skip any line.")
NUM_CTX, NUM_PREDICT = 16384, 3072
# 修复用组合: 每个都独立、不带提前停; minicpm 加一道 T=1.0 保险
COMBOS = [("openbmb/minicpm5-2b", 0.0),
          ("openbmb/minicpm5-2b", 0.7),
          ("openbmb/minicpm5-2b", 1.0),
          ("qwen3.5:4b", 0.7)]


def verify_edge(u, v, chunk):
    return re.search(rf"^{re.escape(u)} -> {re.escape(v)}$", chunk,
                     re.MULTILINE) is not None


def repair_chunk(size, cidx, log):
    """重扫单个塌方片, 返回 (kept_union, per_call 明细, 重复率)"""
    rng = random.Random(11 + size)
    sample = sample_from_target_chars(rng, size, op_type="bfs")
    ch = chunk_edges(sample, 4000)[cidx - 1]
    n_lines = ch.count("\n") + 1
    truth_chunk = set()
    for ln in ch.split("\n"):
        m = re.match(r"^(n\d+) -> (n\d+)$", ln)
        if m:
            truth_chunk.add((m.group(1), m.group(2)))
    log(f"片{cidx}: {n_lines} 行, 片内真值边 {len(truth_chunk)}")
    kept_union, detail, calls = set(), [], 0
    for model, temp in COMBOS:
        # 每个组合跑 2 遍取并集 (诊断显示失败率与调用历史相关, 重复提升稳健性)
        for rep in range(2):
            calls += 1
            t0 = time.time()
            try:
                g = ollama_chat(model, EXTRACT_SYSTEM,
                                f"{TASK}\n\nPortion of the edge list:\n{ch}",
                                num_ctx=NUM_CTX, think=False,
                                temperature=temp, num_predict=NUM_PREDICT)
            except Exception as e:
                log(f"  {model} T={temp} r{rep}: 调用失败 {e}")
                continue
            cands = parse_edges(g.content)
            kept = {e for e in cands if verify_edge(e[0], e[1], ch)}
            kept_union |= kept
            ok_full = len(kept) >= n_lines - 2
            detail.append({"model": model, "temp": temp, "rep": rep,
                           "sec": round(time.time() - t0, 1),
                           "cands": len(cands), "kept": len(kept),
                           "full_copy": ok_full})
            log(f"  {model} T={temp} r{rep}: {time.time()-t0:5.1f}s "
                f"候选{len(cands)} 核验{len(kept)}{' ✓全抄' if ok_full else ''}")
            if len(kept_union) >= n_lines:   # 已全收, 无需更多
                break
        if len(kept_union) >= n_lines:
            break
    missing_in_chunk = truth_chunk - kept_union
    log(f"  片内真值覆盖: {len(kept_union & truth_chunk)}/{len(truth_chunk)}"
        f"  仍缺 {len(missing_in_chunk)}  调用 {calls} 次")
    return kept_union, detail, truth_chunk, missing_in_chunk


def main():
    lf = open(os.path.join(LOGS, "repair_chunks.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    # ---------- 256K 片17: 真修复 (边库在) ----------
    log("=" * 60 + "\n[256K] 修复片17")
    kept17, det17, truth17, miss17 = repair_chunk(256000, 17, log)
    db_path = os.path.join(LOGS, "transcribe_256k_db.json")
    db = set(map(tuple, json.load(open(db_path, encoding="utf-8"))["db"]))
    rng = random.Random(11 + 256000)
    sample = sample_from_target_chars(rng, 256000, op_type="bfs")
    truth_edges = set(sample.edges)
    before = len(db & truth_edges) / len(truth_edges)
    db |= kept17
    after = len(db & truth_edges) / len(truth_edges)
    json.dump({"db": sorted(db)}, open(db_path, "w", encoding="utf-8"),
              ensure_ascii=False)
    log(f"  256K 召回率: {before:.4f} -> {after:.4f} "
        f"(补回 {len((kept17 & truth_edges) - set())} 条中的净增量 "
        f"{after - before:+.4f}), 边库现 {len(db)} 条, 已写回 {db_path}")

    out256 = {"size": 256000, "chunk": 17, "detail": det17,
              "chunk_truth": len(truth17), "chunk_missing_after": len(miss17),
              "recall_before": round(before, 4),
              "recall_after": round(after, 4), "db_size": len(db)}
    json.dump(out256, open(os.path.join(LOGS, "repair_256k_c17.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- 512K 片83: 修正估计 (原边库未落盘) ----------
    log("=" * 60 + "\n[512K] 修复片83 (仅估计修正)")
    kept83, det83, truth83, miss83 = repair_chunk(512000, 83, log)
    # 原始: n_missing=378, 其中 ~263 来自片83 (塌方只留 1 条)
    recovered = len(kept83 & truth83)
    n_missing_orig, from_c83 = 378, 263
    corrected_missing = n_missing_orig - min(recovered, from_c83)
    truth_total = 34093
    recall_orig = 0.9889
    recall_corr = (truth_total - corrected_missing) / truth_total
    log(f"  片83 真值边 {len(truth83)}, 重扫补回 {recovered}, "
        f"片内仍缺 {len(miss83)}")
    log(f"  512K 召回率估计: {recall_orig:.4f} -> ~{recall_corr:.4f} "
        f"(修正漏检 {corrected_missing})")
    out512 = {"size": 512000, "chunk": 83, "detail": det83,
              "chunk_truth": len(truth83), "recovered": recovered,
              "chunk_missing_after": len(miss83),
              "recall_original": recall_orig,
              "recall_corrected_est": round(recall_corr, 4),
              "note": "原边库未落盘(PermissionError), 此为召回率修正估计, 非全库重建"}
    json.dump(out512, open(os.path.join(LOGS, "repair_512k_c83.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    log("\n修复完成")
    lf.close()


if __name__ == "__main__":
    main()
