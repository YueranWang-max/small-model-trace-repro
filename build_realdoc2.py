# -*- coding: utf-8 -*-
"""
build_realdoc2 —— 把 HotpotQA dev (distractor) 变成长上下文多跳 QA 实例。
与 build_realdoc.py (MuSiQue) 同构, 换第二文档族验证表示排序翻转的稳健性。

  1. 选 bridge/comparison 两类 (均 2 跳), 分层采样;
  2. 长文档 = 本题 10 段(含 2 金标支撑段) + 其他题段落填充到 target_chars;
  3. 4000 字符切片, 记录支撑片 id;
  4. 输出与 realdoc_musique_32k.jsonl 同 schema。
"""
from __future__ import annotations

import argparse
import json
import os
import random


def log(m):
    print(m, flush=True)


def build_chunks(doc, chunk_size=4000):
    return [(i, min(i + chunk_size, len(doc)))
            for i in range(0, len(doc), chunk_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hotpot_dev_distractor.parquet")
    ap.add_argument("--out", default="data/realdoc_hotpot_32k.jsonl")
    ap.add_argument("--target-chars", type=int, default=32000)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    base = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base, args.src)
    out = os.path.join(base, args.out)
    rng = random.Random(args.seed)

    t = pq.read_table(src)
    cols = {c: t[c].to_pylist() for c in t.column_names}
    n = len(cols["id"])
    log(f"加载 {n} 题; 列: {t.column_names}")

    # 段落还原: context.title[i] -> join(context.sentences[i])
    def para_texts(i):
        titles = cols["context"][i]["title"]
        sents = cols["context"][i]["sentences"]
        return [(ti, "".join(sj)) for ti, sj in zip(titles, sents)]

    # 干扰段池 (题目间共享)
    all_paras = []
    for i in range(n):
        for ti, tx in para_texts(i):
            all_paras.append((ti, tx))
    rng.shuffle(all_paras)
    log(f"干扰段池: {len(all_paras)} 段")

    # 候选分层: type bridge/comparison
    by_type = {"bridge": [], "comparison": []}
    for i in range(n):
        ans = (cols["answer"][i] or "").strip()
        if not ans:
            continue
        by_type.setdefault(cols["type"][i], []).append(i)
    for v in by_type.values():
        rng.shuffle(v)
    log("候选: " + " ".join(f"{k}={len(v)}" for k, v in by_type.items()))

    quota = {"bridge": 30, "comparison": 20}
    queue = []
    for k, q in quota.items():
        queue.extend(by_type[k][:q])
    rng.shuffle(queue)

    pool = list(all_paras)
    ptr = 0
    written, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as fo:
        for i in queue:
            if written >= args.n_instances:
                break
            paras = para_texts(i)
            sup_titles = set(cols["supporting_facts"][i]["title"])
            parts = [f"{ti}: {tx}" for ti, tx in paras]
            base_len = sum(len(s) for s in parts) + 2 * len(parts)
            if base_len > args.target_chars - 2000:
                skipped += 1
                continue
            filler, fill_len = [], 0
            while base_len + fill_len < args.target_chars - 800:
                if ptr >= len(pool):
                    rng.shuffle(pool)
                    ptr = 0
                ti, tx = pool[ptr]
                ptr += 1
                s = f"{ti}: {tx}"
                if len(s) > 3000:
                    continue
                filler.append(s)
                fill_len += len(s) + 2
            doc_parts = list(parts)
            if filler:
                step = max(1, len(doc_parts) // (len(filler) + 1))
                new_parts, fi = [], 0
                for j, part in enumerate(doc_parts):
                    new_parts.append(part)
                    if fi < len(filler) and (j + 1) % step == 0:
                        new_parts.append(filler[fi])
                        fi += 1
                new_parts.extend(filler[fi:])
                doc_parts = new_parts
            rng.shuffle(doc_parts)
            support_texts = [tx[:150] for ti, tx in paras
                             if ti in sup_titles]
            doc = "\n\n".join(doc_parts)
            chunks = build_chunks(doc, args.chunk_chars)
            chunk_texts = [doc[a:b] for a, b in chunks]
            support_chunk_ids = sorted({
                ci for ci, ct in enumerate(chunk_texts)
                if any(sp in ct for sp in support_texts)
            })
            inst = {
                "id": f"hotpot__{cols['id'][i]}",
                "question": cols["question"][i],
                "answer": cols["answer"][i],
                "answer_aliases": [],
                "hops": 2,
                "type": cols["type"][i],
                "doc_chars": len(doc),
                "n_chunks": len(chunk_texts),
                "chunks": chunk_texts,
                "support_chunk_ids": support_chunk_ids,
            }
            fo.write(json.dumps(inst, ensure_ascii=False) + "\n")
            written += 1
    log(f"完成: 写出 {written} 实例, 跳过 {skipped}")
    log(f"输出: {out}")


if __name__ == "__main__":
    main()
