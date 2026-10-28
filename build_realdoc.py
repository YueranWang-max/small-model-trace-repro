# -*- coding: utf-8 -*-
"""
build_realdoc —— 把 MuSiQue dev 变成长上下文多跳 QA 实例。

做法:
  1. 选 answerable、>=2 跳的题;
  2. 长文档 = 本题 20 段(含 2-4 金标支撑段) + 从其他题均匀抽干扰段填充到
     target_chars(默认 32000) —— 干扰段来自别的题目, 保证与本题无信息交互;
  3. 按段落边界拼接(带标题), 再切成 4000 字符片(与图管线同规格);
  4. 记录每段所在片号 -> support_chunk_ids(金标支撑片, 用于蒸馏召回评估);
  5. 输出 JSONL: id/question/answer/answer_aliases/doc/chunks/support 跳数等。

确定性: seed 7 采样, 可复现。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def log(m):
    print(m, flush=True)


def build_chunks(doc, chunk_size=4000):
    """按 4000 字符硬切(与图管线同规格), 返回 [(start, end)]。"""
    return [(i, min(i + chunk_size, len(doc)))
            for i in range(0, len(doc), chunk_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--out", default="data/realdoc_musique_32k.jsonl")
    ap.add_argument("--target-chars", type=int, default=32000)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base, args.src)
    out = os.path.join(base, args.out)
    rng = random.Random(args.seed)

    recs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    log(f"加载 {len(recs)} 题")

    # 全库干扰段池 (题目间共享, 打乱后取用)
    all_paras = []
    for r in recs:
        for p in r["paragraphs"]:
            all_paras.append((p["title"], p["paragraph_text"]))
    rng.shuffle(all_paras)
    log(f"干扰段池: {len(all_paras)} 段")

    # 候选: answerable & >=2 跳 & 答案非空, 按 hops 分桶
    by_hops = {2: [], 3: [], 4: []}
    for r in recs:
        if not r.get("answerable", True):
            continue
        ans = (r.get("answer") or "").strip()
        hops = len(r["question_decomposition"])
        if not ans or hops < 2 or hops > 4:
            continue
        by_hops[hops].append(r)
    log(f"候选分桶: " + " ".join(f"{h}跳={len(v)}" for h, v in by_hops.items()))
    # 分层采样配额 (50 实例: 2/3/4跳 = 20/20/10)
    quota = {2: 20, 3: 20, 4: 10}
    if args.n_instances != 50:
        total = sum(quota.values())
        quota = {h: max(1, round(args.n_instances * q / total))
                 for h, q in quota.items()}
    for h in by_hops:
        rng.shuffle(by_hops[h])
    queue = []
    hop_cycle = [h for h in (2, 3, 4) for _ in range(quota[h])]
    idxs = {h: 0 for h in by_hops}
    for h in hop_cycle:
        if idxs[h] < len(by_hops[h]):
            queue.append(by_hops[h][idxs[h]])
            idxs[h] += 1
    log(f"分层队列: {len(queue)} 题 "
        f"({ {h: quota[h] for h in quota} })")

    pool = list(all_paras)
    ptr = 0
    written, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as fo:
        for r in queue:
            if written >= args.n_instances:
                break
            paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
            support_idx = {p["idx"] for p in paras if p["is_supporting"]}
            # 基础文档: 标题 + 正文
            parts = [f"{p['title']}: {p['paragraph_text']}" for p in paras]
            n_part_chars = [len(s) for s in parts]
            base_len = sum(n_part_chars) + 2 * len(parts)
            if base_len > args.target_chars - 2000:
                skipped += 1
                continue
            # 填充干扰段到目标长度
            filler = []
            fill_len = 0
            while base_len + fill_len < args.target_chars - 800:
                if ptr >= len(pool):
                    rng.shuffle(pool)
                    ptr = 0
                t, x = pool[ptr]
                ptr += 1
                s = f"{t}: {x}"
                if len(s) > 3000:  # 防超长段
                    continue
                filler.append(s)
                fill_len += len(s) + 2
            # 干扰段等距散布在支撑段之间(避免聚堆)
            doc_parts = list(parts)
            if filler:
                step = max(1, len(doc_parts) // (len(filler) + 1))
                new_parts, fi = [], 0
                for i, part in enumerate(doc_parts):
                    new_parts.append(part)
                    if fi < len(filler) and (i + 1) % step == 0:
                        new_parts.append(filler[fi])
                        fi += 1
                new_parts.extend(filler[fi:])
                doc_parts = new_parts
            rng.shuffle(doc_parts)  # 段序打乱: 消除 "支撑段总是相邻" 的捷径
            # 重新记录支撑段位置: 用段落正文头部指纹匹配
            # (段落可能跨片边界, 整段 in chunk 会漏标; 150 字符头部几乎必在单片内)
            support_texts = [paras[i]["paragraph_text"][:150]
                             for i, s in enumerate(parts)
                             if paras[i]["idx"] in support_idx]
            doc = "\n\n".join(doc_parts)
            chunks = build_chunks(doc, args.chunk_chars)
            chunk_texts = [doc[a:b] for a, b in chunks]
            support_chunk_ids = sorted({
                ci for ci, ct in enumerate(chunk_texts)
                if any(sp in ct for sp in support_texts)
            })
            hops = len(r["question_decomposition"])
            inst = {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_aliases": r.get("answer_aliases", []),
                "hops": hops,
                "doc_chars": len(doc),
                "n_chunks": len(chunk_texts),
                "chunks": chunk_texts,
                "support_chunk_ids": support_chunk_ids,
            }
            fo.write(json.dumps(inst, ensure_ascii=False) + "\n")
            written += 1
    log(f"完成: 写出 {written} 实例, 跳过(基础过长) {skipped}")
    log(f"输出: {out}")


if __name__ == "__main__":
    main()
