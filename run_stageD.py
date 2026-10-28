# -*- coding: utf-8 -*-
"""
run_stageD —— 匹配式上下文移除实验（protocol_stageDE.md, 冻结 2026-09-21）。
条件: eq / exq / tq / txq / bm25_top4 / dense_top4(若 embedding 可用)
模型: qwen3:1.7b; 数据: realdoc_hotpot_32k 50 题 + realdoc_musique_32k 50 题。
同一题内 E/T 严格共享(配对状态); checkpoint 逐题 JSONL + fsync; 断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_realdoc as R  # noqa: E402
from repro_lib import ollama_chat, serialize_traces  # noqa: E402

MODEL = "qwen3:1.7b"
NUM_PREDICT = 12288
SHUFFLE_SEED = 20260921

R.CTX = 32768
R.PRED_READ = NUM_PREDICT

BASE = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(m, flush=True)


def answer_once(material, question, tag):
    prompt = f"{material}\n\nQuestion: {question}\n"
    t0 = time.time()
    ans, meta = R.readout(MODEL, prompt)
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    meta["n_calls"] = 1
    meta["prompt_ch"] = len(prompt)
    meta["cond"] = tag
    return ans, meta


# ---------------- 条件实现 ----------------

def make_conditions(inst, E, T, bm25_fn, dense_fn):
    """返回 {cond: result-dict}。E/T 对四个状态条件严格共享。"""
    q = inst["question"]
    x = "\n\n".join(inst["chunks"])
    out = {}

    # eq = [E, q]
    ans, meta = answer_once(f"Verified evidence:\n{E}", q, "eq")
    f1, em = R.score(ans, inst)
    out["eq"] = {"f1": f1, "em": em, "raw": ans, **meta, "E_chars": len(E)}

    # exq = [E, x, q]
    ans, meta = answer_once(f"Verified evidence:\n{E}\n\nFull material:\n{x}", q, "exq")
    f1, em = R.score(ans, inst)
    out["exq"] = {"f1": f1, "em": em, "raw": ans, **meta, "E_chars": len(E)}

    # tq = [T, q]
    ans, meta = answer_once(T, q, "tq")
    f1, em = R.score(ans, inst)
    out["tq"] = {"f1": f1, "em": em, "raw": ans, **meta, "T_chars": len(T)}

    # txq = [T, x, q]
    ans, meta = answer_once(f"{T}\n\nFull material:\n{x}", q, "txq")
    f1, em = R.score(ans, inst)
    out["txq"] = {"f1": f1, "em": em, "raw": ans, **meta, "T_chars": len(T)}

    # bm25_top4 / dense_top4
    for name, fn in [("bm25_top4", bm25_fn), ("dense_top4", dense_fn)]:
        if fn is None:
            continue
        out[name] = fn(inst)

    return out


def gen_trace_T(inst):
    """3 次首遍 thinking 调用 -> serialize -> 尾窗截断 (与 run_realdoc S3 相同)。"""
    x = "\n\n".join(inst["chunks"])
    traces, first_f1s, calls = [], [], 0
    for _ in range(3):
        g = ollama_chat(MODEL, R.QA_SYSTEM,
                        f"{x}\n\nQuestion: {inst['question']}\n"
                        "Answer with only the final answer itself. "
                        "End your reply with exactly this line:\n"
                        "Final Answer: <answer>",
                        num_ctx=R.CTX, think=True, temperature=0.7,
                        num_predict=NUM_PREDICT)
        a, _ = R.extract_answer(g.content, g.thinking)
        if not a and (g.content or "").strip():
            a = [g.content.strip()[:300]]
        traces.append(g.thinking)
        ff1, _ = R.score(a, inst)
        first_f1s.append(ff1)
        calls += 1
    T_raw = serialize_traces(traces)
    T = R.truncate_tails(T_raw)
    return T, calls, first_f1s


# ---------------- 检索基线 ----------------

def build_bm25_fn():
    sys.path.insert(0, BASE)
    from run_stageC import BM25

    def fn(inst):
        chunks = inst["chunks"]
        idx = BM25(chunks)
        gold = set(inst["support_chunk_ids"])
        sel = idx.search(inst["question"], 4)
        mat = "\n\n".join(chunks[i] for i in sel)
        ans, meta = answer_once(mat, inst["question"], "bm25_top4")
        f1, em = R.score(ans, inst)
        return {"f1": f1, "em": em, "raw": ans, **meta,
                "sel_chunks": sel,
                "gold_recall": round(len(gold & set(sel)) / max(1, len(gold)), 4)}
    return fn


def build_dense_fn():
    """稠密检索 top4 (nomic-embed-text); 模型不可用返回 None。"""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": "nomic-embed-text", "prompt": "ping"}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as ex:
        log(f"dense 检索不可用 ({ex}); dense_top4 取消 (protocol §2 预注册条件)")
        return None

    import math as _m
    import re as _re

    def _embed(text):
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["embedding"]

    def _cos(a, b):
        dot = sum(u * v for u, v in zip(a, b))
        na = _m.sqrt(sum(u * u for u in a))
        nb = _m.sqrt(sum(v * v for v in b))
        return dot / (na * nb) if na and nb else 0.0

    def fn(inst):
        chunks = inst["chunks"]
        gold = set(inst["support_chunk_ids"])
        qv = _embed(inst["question"])
        sims = [(_cos(qv, _embed(c)), i) for i, c in enumerate(chunks)]
        sims.sort(key=lambda t: -t[0])
        sel = [i for _, i in sims[:4]]
        mat = "\n\n".join(chunks[i] for i in sel)
        ans, meta = answer_once(mat, inst["question"], "dense_top4")
        f1, em = R.score(ans, inst)
        return {"f1": f1, "em": em, "raw": ans, **meta,
                "sel_chunks": sel,
                "gold_recall": round(len(gold & set(sel)) / max(1, len(gold)), 4)}
    return fn


# ---------------- 蒸馏缓存 (与 Stage C 共享) ----------------

DCACHE = {}


def STAGE_CACHE(inst):
    key = f"{inst['id']}::{MODEL}"
    if key in DCACHE:
        fbc = {int(kk): v for kk, v in DCACHE[key]["fbc"].items()}
        return fbc, DCACHE[key]["nc"], True
    fbc, nc = R.distill_doc(inst, [MODEL])
    DCACHE[key] = {"fbc": {str(kk): v for kk, v in fbc.items()}, "nc": nc}
    # 并发安全: 写前重读磁盘缓存并合并, 避免两个 Stage D 实例互相丢更新
    cp = os.path.join(BASE, "logs/stageC_distill_cache.json")
    try:
        with open(cp, encoding="utf-8") as f:
            disk = json.load(f)
        for k, v in disk.items():
            DCACHE.setdefault(k, v)
    except Exception:
        pass
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(DCACHE, f, ensure_ascii=False)
    return fbc, nc, False


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-instances", type=int, default=50)
    args = ap.parse_args()

    global DCACHE
    cp = os.path.join(BASE, "logs/stageC_distill_cache.json")
    if os.path.exists(cp):
        DCACHE = json.load(open(cp, encoding="utf-8"))
        log(f"蒸馏缓存: {len(DCACHE)} 条")

    bm25_fn = build_bm25_fn()
    dense_fn = build_dense_fn()

    insts = [json.loads(l) for l in
             open(os.path.join(BASE, args.data), encoding="utf-8") if l.strip()]
    insts = insts[: args.n_instances]
    import random
    random.Random(SHUFFLE_SEED).shuffle(insts)
    log(f"实例 {len(insts)} (洗牌 seed={SHUFFLE_SEED}); "
        f"dense={'可用' if dense_fn else '取消'}")

    out_path = os.path.join(BASE, args.out)
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done[json.loads(line)["id"]] = True
            except Exception:
                pass
        log(f"断点续跑: 已有 {len(done)} 题")

    with open(out_path, "a", encoding="utf-8") as fo:
        for k, inst in enumerate(insts):
            if inst["id"] in done:
                continue
            ts = time.time()
            fbc, nc, hit = STAGE_CACHE(inst)
            E = R.build_e(fbc)
            T, tcalls, first_f1s = gen_trace_T(inst)
            conds = make_conditions(inst, E, T, bm25_fn, dense_fn)
            rec = {"id": inst["id"], "hops": inst["hops"],
                   "answer": inst["answer"],
                   "distill": {"n_calls": nc, "cache_hit": hit,
                               "E_chars": len(E),
                               "support_hit": round(
                                   len([c for c in inst["support_chunk_ids"]
                                        if c in fbc]) /
                                   max(1, len(inst["support_chunk_ids"])), 4)},
                   "trace": {"n_calls": tcalls, "T_chars": len(T),
                             "first_mean_f1": round(sum(first_f1s) / 3, 4)},
                   "conds": conds,
                   "elapsed_sec": round(time.time() - ts, 1)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            cs = " ".join(f"{c}={conds[c]['f1']:.2f}" for c in
                          ["eq", "exq", "tq", "txq", "bm25_top4", "dense_top4"]
                          if c in conds)
            log(f"[{k+1}/{len(insts)}] {inst['id']} h={inst['hops']} "
                f"E={len(E)}ch T={len(T)}ch {cs} ({rec['elapsed_sec']}s)")

    # ---- 汇总 ----
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    n = len(recs)
    log(f"\n===== 汇总 (n={n}) =====")
    for c in ["eq", "exq", "tq", "txq", "bm25_top4", "dense_top4"]:
        v = [r["conds"][c]["f1"] for r in recs if c in r["conds"]]
        if v:
            em = sum(r["conds"][c]["em"] for r in recs if c in r["conds"]) / len(v)
            log(f"  {c}: F1={sum(v)/len(v):.4f} EM={em:.4f} (n={len(v)})")


if __name__ == "__main__":
    main()
