# -*- coding: utf-8 -*-
"""
run_stageC —— 标准基线补全（protocol_stageC.md, 冻结 2026-09-20）。
条件: direct_full / bm25_top2 / bm25_top4 / sliding12k / sdr_state
模型: qwen3:1.7b; 数据: data/realdoc_hotpot_32k.jsonl 50 题。
checkpoint 逐题 JSONL + fsync; 断点续跑; 蒸馏按文档缓存。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_realdoc as R  # noqa: E402  复用蒸馏/核验/读出/评分
from repro_lib import ollama_chat  # noqa: E402

MODEL = "qwen3:1.7b"
NUM_PREDICT = 12288          # thinking 家族预算律 (protocol §2)
SHUFFLE_SEED = 20260921
BOOT_SEED = 20260921

# 覆盖被复用模块的全局: 全条件统一 32K ctx / 12288 输出预算
R.CTX = 32768
R.PRED_READ = NUM_PREDICT

BASE = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(m, flush=True)


# ---------------- BM25 (自实现, 无外部依赖; protocol §3.2) ----------------

def _tokens(s):
    return re.findall(r"[a-z0-9']+", (s or "").lower())


class BM25:
    def __init__(self, docs):
        self.docs = docs
        self.dls = []
        self.tfs = []
        df = {}
        for d in docs:
            toks = _tokens(d)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)
            self.dls.append(len(toks))
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.N = len(docs)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self.avgdl = sum(self.dls) / max(1, self.N)

    def search(self, query, k):
        qt = _tokens(query)
        scores = []
        for i in range(self.N):
            s = 0.0
            for t in qt:
                f = self.tfs[i].get(t, 0)
                if not f:
                    continue
                s += self.idf.get(t, 0) * f * (2.2) / (
                    f + 1.5 * (1 - 0.75 + 0.75 * self.dls[i] / self.avgdl))
            scores.append((s, i))
        scores.sort(key=lambda x: -x[0])
        return [i for _, i in scores[:k]]


# ---------------- 滑窗 medoid 聚合 (protocol §3.4) ----------------

def medoid(cands):
    """候选间两两 token-F1, 取平均相似度最高者; 无金标参与。"""
    if len(cands) == 1:
        return cands[0]
    best, best_s = cands[0], -1.0
    for a in cands:
        ta = set(R._norm_tokens(a))
        s = 0.0
        for b in cands:
            if a is b:
                continue
            tb = set(R._norm_tokens(b))
            ov = len(ta & tb)
            f1 = (2 * ov / (len(ta) + len(tb))) if (ta or tb) else 1.0
            s += f1
        if s > best_s:
            best, best_s = a, s
    return best


# ---------------- 单次回答 ----------------

def answer_once(material, question):
    prompt = (f"{material}\n\nQuestion: {question}\n")
    t0 = time.time()
    ans, meta = R.readout(MODEL, prompt)
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    meta["n_calls"] = 1
    return ans, meta


def cond_direct_full(inst):
    chunks = inst["chunks"]
    ans, meta = answer_once("\n\n".join(chunks), inst["question"])
    f1, em = R.score(ans, inst)
    return {"f1": f1, "em": em, "raw": ans, **meta}


def _bm25(inst, k, name):
    chunks = inst["chunks"]
    idx = BM25(chunks)
    gold = set(inst["support_chunk_ids"])
    sel = idx.search(inst["question"], k)
    mat = "\n\n".join(chunks[i] for i in sel)
    ans, meta = answer_once(mat, inst["question"])
    f1, em = R.score(ans, inst)
    return {"f1": f1, "em": em, "raw": ans, **meta,
            "sel_chunks": sel,
            "gold_recall": round(len(gold & set(sel)) / max(1, len(gold)), 4)}


def cond_bm25_top2(inst):
    return _bm25(inst, 2, "bm25_top2")


def cond_bm25_top4(inst):
    return _bm25(inst, 4, "bm25_top4")


def cond_sliding12k(inst):
    q = inst["question"]
    full = "\n\n".join(inst["chunks"])
    W = 12000
    blocks = [full[i:i + W] for i in range(0, len(full), W)]
    cands, metas, calls = [], [], 0
    for b in blocks:
        ans, meta = answer_once(b, q)
        calls += meta["n_calls"]
        cands.append(ans[0] if ans else "")
        metas.append(meta)
    winner = medoid(cands)
    ans = [winner] if winner else []
    f1, em = R.score(ans, inst)
    return {"f1": f1, "em": em, "raw": ans,
            "content_ch": sum(m["content_ch"] for m in metas),
            "thinking_ch": sum(m["thinking_ch"] for m in metas),
            "malformed": any(m["malformed"] for m in metas),
            "elapsed_sec": round(sum(m["elapsed_sec"] for m in metas), 1),
            "n_calls": calls,
            "cands": cands}


def cond_sdr_state(inst):
    q = inst["question"]
    t0 = time.time()
    fbc, nc, hit = STAGE_CACHE(inst)
    e = R.build_e(fbc)
    p1 = f"Verified evidence:\n{e}\n\nQuestion: {q}\n"
    ans, meta = R.readout(MODEL, p1)
    f1, em = R.score(ans, inst)
    return {"f1": f1, "em": em, "raw": ans, **meta,
            "n_calls": meta.get("n_calls", 1) + nc,
            "distill_calls": nc, "cache_hit": hit,
            "E_chars": len(e),
            "support_hit": round(
                len([c for c in inst["support_chunk_ids"] if c in fbc]) /
                max(1, len(inst["support_chunk_ids"])), 4),
            "total_sec": round(time.time() - t0, 1)}


COND_FUNCS = {
    "direct_full": cond_direct_full,
    "bm25_top2": cond_bm25_top2,
    "bm25_top4": cond_bm25_top4,
    "sliding12k": cond_sliding12k,
    "sdr_state": cond_sdr_state,
}


# ---------------- 蒸馏缓存 ----------------

DCACHE = {}


def STAGE_CACHE(inst):
    key = f"{inst['id']}::{MODEL}"
    if key in DCACHE:
        fbc = {int(kk): v for kk, v in DCACHE[key]["fbc"].items()}
        return fbc, DCACHE[key]["nc"], True
    fbc, nc = R.distill_doc(inst, [MODEL])
    DCACHE[key] = {"fbc": {str(kk): v for kk, v in fbc.items()}, "nc": nc}
    with open(os.path.join(BASE, "logs/stageC_distill_cache.json"), "w",
              encoding="utf-8") as f:
        json.dump(DCACHE, f, ensure_ascii=False)
    return fbc, nc, False


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/realdoc_hotpot_32k.jsonl")
    ap.add_argument("--out", default="logs/stageC_baselines.jsonl")
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--conds", default="direct_full,bm25_top2,bm25_top4,sliding12k,sdr_state",
                    help="冒烟时可只跑部分条件")
    args = ap.parse_args()
    conds = [c for c in args.conds.split(",") if c]

    global DCACHE
    cp = os.path.join(BASE, "logs/stageC_distill_cache.json")
    if os.path.exists(cp):
        DCACHE = json.load(open(cp, encoding="utf-8"))
        log(f"蒸馏缓存: {len(DCACHE)} 条")

    insts = [json.loads(l) for l in
             open(os.path.join(BASE, args.data), encoding="utf-8") if l.strip()]
    insts = insts[: args.n_instances]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(insts)  # protocol §2 题目顺序随机化
    log(f"实例 {len(insts)} (洗牌 seed={SHUFFLE_SEED}); 条件顺序: {conds}")

    out_path = os.path.join(BASE, args.out)
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except Exception:
                pass
        log(f"断点续跑: 已有 {len(done)} 实例")

    fo = open(out_path, "a", encoding="utf-8")
    for k, inst in enumerate(insts):
        if inst["id"] in done:
            continue
        ts = time.time()
        rec = {"id": inst["id"], "hops": inst["hops"], "type": inst.get("type", ""),
               "answer": inst["answer"], "conds": {}}
        for c in conds:
            sub = COND_FUNCS[c](inst)
            rec["conds"][c] = sub
            log(f"  [{c}] f1={sub['f1']:.3f} em={sub['em']:.3f} "
                f"calls={sub.get('n_calls','?')}")
        rec["elapsed_sec"] = round(time.time() - ts, 1)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        os.fsync(fo.fileno())
        m = rec["conds"]
        log(f"[{k+1}/{len(insts)}] {inst['id']} h={inst['hops']} "
            + " ".join(f"{c}={m[c]['f1']:.2f}" for c in m)
            + f" ({rec['elapsed_sec']}s)")

    # 汇总
    recs = list(done.values()) + [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    uniq = {r["id"]: r for r in recs}
    n = len(uniq)
    if n:
        log(f"\n===== 汇总 (n={n}) =====")
        for c in conds:
            v = [r["conds"][c]["f1"] for r in uniq.values() if c in r["conds"]]
            if v:
                log(f"  {c}: F1={sum(v)/len(v):.4f} (n={len(v)})")


if __name__ == "__main__":
    main()
