# -*- coding: utf-8 -*-
"""
「先建库后问答」—— 全图一次性转录实验 (v4 / 野路子终极形态)

思路: 与其每题蒸馏一次, 不如把整张图用模型转录成"验证过的边库",
之后同图的任何问题(parents/BFS/任意)都由边库机械作答, 边际成本为零。
这正是 LLM 版 ETL: 摄入时一次昂贵抽取(集成+饱和+机械核验),
查询时纯机械推理。

测量:
  1) 转录召回率: |边库 ∩ 真值边集| / |真值边集| —— 本方案唯一的天花板
     (精度由机械核验恒等于 1.0)
  2) 边库作答 5 题 (3 parents + 2 BFS) 的 F1 与耗时
  3) 对比逐题蒸馏的成本 (v3 实测: parents 12-30 次/题, BFS 62-135 次/题)
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
    f1_score, generate_sample, sample_from_target_chars,
)
from run_distill3 import Extractor, chunk_edges, parse_edges  # noqa: E402


def verify_edge(u, v, chunk):
    return re.search(rf"^{re.escape(u)} -> {re.escape(v)}$", chunk,
                     re.MULTILINE) is not None


def save_json(path, obj, retries=5, delay=10):
    """带重试的落盘: 防杀软瞬时文件锁 (512K 事故教训)。
    先写临时文件再原子替换, 重试间退避。"""
    import time as _t
    tmp = path + ".tmp"
    for i in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return
        except PermissionError as e:
            print(f"  [warn] 落盘被锁 ({e}), {delay}s 后重试 {i+1}/{retries}",
                  flush=True)
            _t.sleep(delay)
    raise RuntimeError(f"落盘失败(重试{retries}次): {path}")


class Checkpoint:
    """逐片增量检查点: 每片追加一行 JSONL, 写失败退避重试,
    最终兜底写备用路径/系统临时目录 —— 任何时刻最多丢一片。"""

    def __init__(self, base_path):
        import time as _t
        self.base = base_path
        self.fallback = base_path.replace(
            ".jsonl", f"_fb{_t.strftime('%H%M%S')}.jsonl")
        self.wrote_fallback = False
        self.n_retry_total = 0

    def append(self, obj):
        import time as _t
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        targets = [self.base] if not self.wrote_fallback else [self.fallback]
        delay = 5
        for attempt in range(8):
            for tgt in targets:
                try:
                    with open(tgt, "a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                        os.fsync(f.fileno())
                    return tgt
                except PermissionError:
                    pass
            self.n_retry_total += 1
            print(f"  [warn] 检查点写入被锁 (第{attempt+1}次), {delay}s 后重试",
                  flush=True)
            _t.sleep(delay)
            delay = min(delay * 2, 60)
            if attempt == 3 and not self.wrote_fallback:
                # 主路径持续被锁 -> 切换备用文件名
                self.wrote_fallback = True
                targets = [self.fallback]
                print(f"  [warn] 切换备用检查点 {self.fallback}", flush=True)
        # 最终兜底: 系统临时目录
        import tempfile
        tgt = os.path.join(tempfile.gettempdir(), os.path.basename(self.base))
        with open(tgt, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"  [warn!!] 主/备路径全锁, 该片已写入系统临时目录 {tgt}",
              flush=True)
        return tgt


def bfs_frontier(adj, start, depth):
    visited, frontier = {start}, [start]
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        frontier = nxt
    return sorted(set(frontier))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/minicpm5-2b")
    ap.add_argument("--ensemble", default="qwen3.5:4b")
    ap.add_argument("--size", type=int, default=128000)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--num-predict", type=int, default=3072,
                    help="转录是整片抄写, 输出比问答长, 预算要大")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-parents", type=int, default=3)
    ap.add_argument("--n-bfs", type=int, default=2)
    ap.add_argument("--out", default="logs/transcribe_128k.json")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    rng = random.Random(args.seed + args.size)
    sample = sample_from_target_chars(rng, args.size, op_type="bfs")
    chunks = chunk_edges(sample, args.chunk_chars)
    truth_edges = set(sample.edges)
    log(f"### 全图转录: {sample.graph_chars} 字符, {len(truth_edges)} 条真值边, "
        f"{len(chunks)} 片, models={[args.model, args.ensemble] if args.ensemble else [args.model]}")

    ex = Extractor([args.model] + ([args.ensemble] if args.ensemble else []),
                   args.num_ctx, log, budget=10000,
                   num_predict=args.num_predict)

    task = ("List EVERY edge line in this portion, verbatim, one per line. "
            "Copy all of them. Do not skip any line.")
    # 重扫用的扰动任务语: 打破 "跳到片尾/格式幻觉" 退化吸引子
    TASK_VARIANTS = [
        task,
        "Copy the portion from its FIRST line to its LAST line, verbatim, "
        "one per line. Start your output with the first edge of the portion.",
        "Transcribe every line of the portion exactly as written. "
        "Double-check you did not skip the beginning.",
    ]
    db = set()
    t0 = time.time()
    per_chunk = []
    repaired = []
    # 逐片增量检查点: 任何写盘失败最多丢一片 (两次 512K 落盘事故的根治)
    ckpt = Checkpoint(args.out.replace(".json", "_ckpt.jsonl"))
    log(f"  检查点: {ckpt.base}")

    # 断点续跑: 读取已有检查点 (含备用文件), 跳过已完成片
    done = {}
    import glob as _glob
    for cpf in sorted(_glob.glob(args.out.replace(".json", "_ckpt*.jsonl"))):
        try:
            with open(cpf, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                        done[rec["chunk"]] = rec
                    except json.JSONDecodeError:
                        continue  # 半行(崩溃残留), 忽略
        except PermissionError:
            log(f"  [warn] 检查点 {cpf} 暂不可读, 跳过")
    if done:
        log(f"  断点续跑: 检查点已有 {len(done)}/{len(chunks)} 片, 跳过重扫")

    def verify_chunk(ch, task_text):
        """一次完整扫描(含饱和), 返回机械核验后的边集。"""
        cands, _ = ex.scan(ch, task_text)
        return {e for e in cands if verify_edge(e[0], e[1], ch)}

    for i, ch in enumerate(chunks, 1):
        n_lines = ch.count("\n") + 1
        if i in done:  # 断点续跑: 该片已完成
            rec = done[i]
            kept = {tuple(e) for e in rec["edges"]}
            db |= kept
            per_chunk.append({"chunk": i, "kept": len(kept), "lines": n_lines,
                              "rescans": rec.get("rescans", 0),
                              "resumed": True,
                              "unrepaired": rec.get("unrepaired", False)})
            continue
        kept = verify_chunk(ch, TASK_VARIANTS[0])
        # 自愈护栏: 塌方检测 (核验数 << 行数) -> 扰动任务语重扫, 最多 2 次
        tries = 0
        while len(kept) < n_lines * 0.9 and tries < 2:
            tries += 1
            log(f"  片{i}: 塌方 detected (核验{len(kept)}/{n_lines}), 重扫 #{tries}")
            kept |= verify_chunk(ch, TASK_VARIANTS[tries % len(TASK_VARIANTS)])
        if tries:
            repaired.append({"chunk": i, "lines": n_lines,
                             "kept_after_repair": len(kept), "rescans": tries})
            if len(kept) < n_lines * 0.9:
                log(f"  !! 片{i}: 重扫{tries}次后仍 {len(kept)}/{n_lines}，"
                    f"标记为未修复塌方，汇总时需人工复查")
                per_chunk.append({"chunk": i, "kept": len(kept),
                                  "lines": n_lines, "rescans": tries,
                                  "unrepaired": True})
                db |= kept
                ckpt.append({"chunk": i, "edges": sorted(kept),
                             "unrepaired": True})
                continue
        db |= kept
        per_chunk.append({"chunk": i, "cands": len(kept), "kept": len(kept),
                          "lines": n_lines, "rescans": tries})
        ckpt.append({"chunk": i, "edges": sorted(kept)})
        log(f"  片{i}/{len(chunks)}: 核验{len(kept)} "
            f"(片内行数{n_lines}{' , 重扫' + str(tries) if tries else ''})")
    elapsed = time.time() - t0
    log(f"  检查点写满, 重试计数 {ckpt.n_retry_total}, "
        f"备用路径启用: {ckpt.wrote_fallback}")

    recall = len(db & truth_edges) / len(truth_edges)
    missing = sorted(truth_edges - db)
    log(f"\n转录完成: 边库 {len(db)} 条 (精度恒1.0, 全部经机械核验)")
    log(f"  转录召回率 = {recall:.4f}  漏检 {len(missing)} 条")
    log(f"  调用 {ex.calls} 次, 耗时 {elapsed:.0f}s")

    # ---- 边库机械作答: 3 parents + 2 bfs (同图多题, 边际成本 0) ----
    adj, radj = {}, {}
    for u, v in db:
        adj.setdefault(u, set()).add(v)
        radj.setdefault(v, set()).add(u)
    nodes = sorted({u for u, _ in db} | {v for _, v in db})

    qas = []
    # parents: 选高入度目标
    by_indeg = sorted(nodes, key=lambda x: len(radj.get(x, ())), reverse=True)
    for tgt in by_indeg[:args.n_parents * 3]:
        truth = sorted(radj.get(tgt, ()))
        if 3 <= len(truth) <= 60:
            qas.append(("parents", tgt, truth))
        if len(qas) >= args.n_parents:
            break
    # bfs: 随机起点+深度, 答案 5-60
    tries = 0
    while sum(1 for q in qas if q[0] == "bfs") < args.n_bfs and tries < 500:
        tries += 1
        start = rng.choice(nodes)
        depth = rng.randint(3, 4)
        truth = bfs_frontier(adj, start, depth)
        if 5 <= len(truth) <= 60:
            qas.append(("bfs", f"{start}@{depth}", truth))
    # 真值邻接 (答案用真值算, 避免边库缺边污染 truth)
    tadj = {}    # 出边邻接 (BFS 用)
    tradj = {}   # 入边邻接 (parents 真值用)
    for u, v in truth_edges:
        tadj.setdefault(u, set()).add(v)
        tradj.setdefault(v, set()).add(u)
    # 边库落盘, 供后续任意问题复用 (先建库后问答)
    # 非致命: 检查点 JSONL 已是事实源头, 最终合并失败只报警
    db_path = args.out.replace(".json", "_db.json")
    try:
        save_json(db_path, {"db": sorted(db)}, retries=8, delay=15)
        log(f"  边库已存 {db_path}")
    except RuntimeError as e:
        log(f"  [warn!!] 边库最终合并写失败 ({e}); "
            f"完整数据在检查点 {ckpt.base}, 可由其重建")

    results = []
    for op, q, truth in qas:
        if op == "parents":
            tgt = q
            # 真值用真值入边邻接
            truth_full = sorted(tradj.get(tgt, ()))
            # 边库作答 = 边库中指向 tgt 的边 (机械)
            db_adj_r = {}
            for u, v in db:
                db_adj_r.setdefault(v, set()).add(u)
            got = sorted(db_adj_r.get(tgt, ()))
            f1 = f1_score(got, truth_full)
            qas_desc = f"parents({tgt})"
        else:
            start, depth = q.split("@")
            depth = int(depth)
            # 真值: 用真值邻接算
            tv, tf = {start}, [start]
            for _ in range(depth):
                nxt = []
                for u in tf:
                    for v in tadj.get(u, ()):
                        if v not in tv:
                            tv.add(v)
                            nxt.append(v)
                tf = nxt
            truth_full = sorted(set(tf))
            # 边库作答: 边库邻接机械 BFS
            got = bfs_frontier(adj, start, depth)
            f1 = f1_score(got, truth_full)
            qas_desc = f"bfs({start}, d={depth})"
        results.append({"q": qas_desc, "truth_size": len(truth_full),
                        "got_size": len(got), "f1": f1})
        log(f"  [{qas_desc}] truth={len(truth_full)} db答={len(got)} F1={f1:.3f}")

    out = {
        "size": args.size, "n_chunks": len(chunks), "models": [args.model, args.ensemble],
        "transcribe_calls": ex.calls, "elapsed_sec": round(elapsed, 1),
        "db_size": len(db), "truth_edges": len(truth_edges),
        "recall": round(recall, 4), "n_missing": len(missing),
        "missing_edges": missing[:20],
        "repaired_chunks": repaired,
        "questions": results,
        "per_chunk": per_chunk,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        save_json(args.out, out, retries=8, delay=15)
        log(f"\n结果已写入 {args.out}")
    except RuntimeError as e:
        log(f"\n[warn!!] 结果 JSON 写失败 ({e}); 关键数值已全部在日志中, "
            f"边库在检查点 {ckpt.base}")


if __name__ == "__main__":
    main()
