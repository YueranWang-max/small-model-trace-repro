# -*- coding: utf-8 -*-
"""
run_recall —— 蒸馏召回上界攻坚: 倒排索引逐节点零产出报警 + 定点重扫

B 读出 = det (6.8), 管线的上界就是蒸馏召回。召回损失模式反常
(d2 Q21 det=0.57 比 d5 还低) => 逐片扫描随机漏检, 不是深度问题。

报警器: 倒排索引(零 LLM)精确知道「u 在片 c 里是某条边的左端点」。
扫描后 u 在该片零确认边 = 必然漏检 (图真实存在该边) => 定点重扫:
  1) 单节点批重扫 T=0.7
  2) 扰动任务语 T=0
  3) 集成模型 T=0.7
  4) 扰动任务语 T=1.0
不泄题: 报警器只说「至少有一条 u 开头的边在这」, 不说边是什么。

题目: 精确重放 seed 7 与 seed 13 的全部 BFS 题 (共 24 道, 双重校验)。
配对指标: 每层修复前/后 frontier 召回, det_pre vs det_post。
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
from repro_lib import em_score, f1_score, sample_from_target_chars  # noqa: E402
from run_distill3 import (  # noqa: E402
    Extractor, chunk_edges, node_ids_in, parse_edges, verify_edge,
)
from run_chain2 import distill_bfs_capture  # noqa: E402

BATCH = 12


def log(msg):
    print(msg, flush=True)


def build_left_index(chunks):
    """左端点倒排: 节点 -> 含其出边的片集合 (零 LLM, 精确)。"""
    idx = {}
    for i, ch in enumerate(chunks):
        for u in re.findall(r"^n\d+(?= ->)", ch, re.MULTILINE):
            idx.setdefault(u, set()).add(i)
    return idx


RESCAN_TASKS = [
    ("t0.7", 0.7, None,
     "Frontier node F = {u}.\nList every edge u2 -> v in this portion with "
     "u2 in F. Copy each line verbatim. If none matches, output NONE."),
    ("perturb-t0", 0.0, None,
     "Find every line in this portion that starts with {u} followed by "
     "\" -> \". Output each such line verbatim, one per line. Do not stop "
     "early. If none, output NONE."),
    ("ens-t0.7", 0.7, "ens",
     "Frontier node F = {u}.\nList every edge u2 -> v in this portion with "
     "u2 in F. Copy each line verbatim. If none matches, output NONE."),
    ("perturb-t1.0", 1.0, None,
     "Locate all edges in this portion whose LEFT endpoint (the node before "
     "\" -> \") is {u}. Copy the complete lines verbatim. If none, output "
     "NONE."),
]


def scan_level_with_alarm(ex, chunks, left_idx, frontier, visited, qlog):
    """一层: 小批量扫描 -> 机械验证 -> 逐节点报警重扫。
    返回 (level_conf_pre, level_conf_post, alarms, repaired, unrepaired)。"""
    pre, post = set(), set()
    alarms = repaired = unrepaired = 0
    for i, ch in enumerate(chunks):
        f_here = sorted(frontier & node_ids_in(ch))
        if not f_here:
            continue
        for b0 in range(0, len(f_here), BATCH):
            grp = f_here[b0:b0 + BATCH]
            task = (f"Frontier set F = {{{', '.join(grp)}}}.\n"
                    f"List every edge u -> v in this portion with u in F. "
                    f"Copy each line verbatim.")
            cand, _ = ex.scan(ch, task, filt=lambda e: e[0] in set(grp))
            ok = {e for e in cand if verify_edge(e[0], e[1], ch)}
            pre |= ok
            post |= ok
            # ---- 逐节点报警 ----
            for u in grp:
                if i not in left_idx.get(u, ()):
                    continue  # 索引说本片没有 u 的出边, 无从报警
                if any(e[0] == u for e in ok):
                    continue  # 已有确认边, 合格
                alarms += 1
                fixed = set()
                for name, t, who, tpl in RESCAN_TASKS:
                    if fixed:
                        break
                    model = ex.models[1] if who == "ens" and len(
                        ex.models) > 1 else ex.models[0]
                    task1 = tpl.replace("{u}", u)
                    g = ex._raw(model, ch, task1, t)
                    cands = {e for e in parse_edges(g)
                             if e[0] == u and verify_edge(e[0], e[1], ch)}
                    if cands:
                        fixed = cands
                        qlog(f"        [ALARM修复] {u}@片{i} via {name}: "
                             f"+{len(cands)}边")
                if fixed:
                    repaired += 1
                    post |= fixed
                else:
                    unrepaired += 1
                    qlog(f"        [ALARM未修复] {u}@片{i} (4次尝试)")
    return pre, post, alarms, repaired, unrepaired


def distill_with_alarm(models, sample, num_ctx, budget=800):
    import re as _re
    m = _re.search(r"from node (n\d+) with depth (\d+)", sample.op_text)
    start, depth = m.group(1), int(m.group(2))
    chunks = chunk_edges(sample, 4000)
    left_idx = build_left_index(chunks)
    ex = Extractor(models, num_ctx, log, budget)
    # 给 Extractor 挂一个原始调用口 (不经过饱和逻辑)
    def raw(model, chunk, task, t):
        ex.calls += 1
        user = f"{task}\n\nPortion of the edge list:\n{chunk}"
        try:
            g = ex._ollama(model, user, t)
        except Exception:
            return ""
        return g
    ex._raw = raw
    import run_distill3 as rd3
    if not hasattr(Extractor, "_ollama"):
        def _ollama(self, model, user, t):
            from repro_lib import ollama_chat
            g = ollama_chat(model, rd3.EXTRACT_SYSTEM, user,
                            num_ctx=self.num_ctx, think=False,
                            temperature=t, num_predict=self.num_predict)
            return g.content
        Extractor._ollama = _ollama

    visited, frontier = {start}, {start}
    per_level = []
    tot = {"alarms": 0, "repaired": 0, "unrepaired": 0}
    for d in range(1, depth + 1):
        pre, post, al, rep, unrep = scan_level_with_alarm(
            ex, chunks, left_idx, frontier, visited, log)
        new_pre = {v for _, v in pre if v not in visited}
        new_post = {v for _, v in post if v not in visited}
        visited |= new_post
        per_level.append({"level": d,
                          "frontier_pre": sorted(new_pre),
                          "frontier_post": sorted(new_post),
                          "calls": ex.calls})
        tot["alarms"] += al
        tot["repaired"] += rep
        tot["unrepaired"] += unrep
        log(f"      层{d}: pre={len(new_pre)} post={len(new_post)} "
            f"报警{al} 修复{rep} 未修{unrep}")
        frontier = new_post
        if not frontier:
            break
    det_pre = sorted({v for lv in per_level for v in []})  # 占位, 见下
    return per_level, depth, tot, ex.calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="qwen3.5:4b")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--out", default="logs/recall_alarm_2b.json")
    args = ap.parse_args()
    models = [args.model] + ([args.ensemble] if args.ensemble else [])

    # ---- 双 seed 重放全部 BFS 题 ----
    qs = []
    for seed, ref_path, n in ((7, "logs/chain_2b_3k.json", 12),
                              (13, "logs/chain_2b_3k_s13.json", 24)):
        ref = json.load(open(ref_path, encoding="utf-8"))["records"]
        rng = random.Random(seed)
        for i in range(n):
            op = "bfs" if i % 2 == 0 else "parents"
            sample = sample_from_target_chars(rng, 3000, op_type=op)
            if op == "bfs":
                assert sample.answer == ref[i]["answer"], \
                    f"重放校验失败 seed{seed} 题{i}"
                qs.append((seed, i, sample))
    log(f"双重放校验通过: 共 {len(qs)} 道 BFS 题")

    records, t0 = [], time.time()
    for qi, (seed, idx, sample) in enumerate(qs):
        depth = int(re.search(r"depth (\d+)", sample.op_text).group(1))
        log(f"\n[{qi+1}/{len(qs)}] seed{seed} 题#{idx+1} d={depth} "
            f"truth={len(sample.answer)}")
        rec = {"seed": seed, "orig_index": idx, "depth": depth,
               "answer": sample.answer}
        ts = time.time()
        per_level, _, tot, ncalls = distill_with_alarm(
            models, sample, args.num_ctx)
        # 配对评分: pre 用各层 pre_frontier 串成的 BFS 前缀展开不可行
        # (pre 层间不独立), 改用近似: det_pre = 最后一层的 pre 召回
        last = per_level[-1]
        det_pre = last["frontier_pre"]
        det_post = last["frontier_post"]
        rec.update({
            "det_pre_f1": f1_score(det_pre, sample.answer),
            "det_post_f1": f1_score(det_post, sample.answer),
            "em_post": em_score(det_post, sample.answer),
            "alarms": tot["alarms"], "repaired": tot["repaired"],
            "unrepaired": tot["unrepaired"], "n_calls": ncalls,
            "per_level": [{"level": lv["level"],
                           "pre_n": len(lv["frontier_pre"]),
                           "post_n": len(lv["frontier_post"])}
                          for lv in per_level],
            "sec": round(time.time() - ts, 1)})
        log(f"      => det_pre={rec['det_pre_f1']:.3f} "
            f"det_post={rec['det_post_f1']:.3f} "
            f"EM={rec['em_post']} ({ncalls}调用, {rec['sec']}s)")
        records.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"records_so_far": records}, f, ensure_ascii=False,
                      indent=2)

    n = len(records)
    summary = {"models": models, "n_questions": n,
               "elapsed_sec": round(time.time() - t0, 1),
               "det_pre_f1": round(sum(r["det_pre_f1"] for r in records) / n, 4),
               "det_post_f1": round(sum(r["det_post_f1"] for r in records) / n, 4),
               "em_post": round(sum(r["em_post"] for r in records) / n, 4),
               "total_alarms": sum(r["alarms"] for r in records),
               "total_repaired": sum(r["repaired"] for r in records),
               "total_unrepaired": sum(r["unrepaired"] for r in records)}
    for tag in ("pre", "post"):
        hard = [r for r in records if r["depth"] >= 4]
        if hard:
            summary[f"hard_d4_{tag}_f1"] = round(
                sum(r[f"det_{tag}_f1"] for r in hard) / len(hard), 4)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    log("\n===== 汇总 =====")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
