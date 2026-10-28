# -*- coding: utf-8 -*-
"""
run_stageE —— 跨模型家族验证 (protocol_stageDE.md, 冻结 2026-09-21)。
模型: llama3.2:3b (旧代家族对照); 数据: realdoc_hotpot_32k 同 50 题同洗牌。
共享状态设计: E 直接复用 qwen3:1.7b 蒸馏缓存 (同状态喂不同模型, 检验消费)。
条件: direct_full / bm25_top4 / eq / exq。checkpoint + 断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_realdoc as R  # noqa: E402

MODEL = "llama3.2:3b"
NUM_PREDICT = 12288
SHUFFLE_SEED = 20260921

R.CTX = 32768
R.PRED_READ = NUM_PREDICT

BASE = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(m, flush=True)


def answer_once(material, question, tag):
    prompt = (f"{material}\n\nQuestion: {question}\n".rstrip()
              + "\n\nAnswer with only the final answer itself. "
                "End your reply with exactly this line:\n"
                "Final Answer: <answer>")
    t0 = time.time()
    # llama3.2:3b 不支持 thinking -> think=False (protocol_stageDE.md 勘误 1)
    g = R.ollama_chat(MODEL, R.QA_SYSTEM, prompt, num_ctx=R.CTX, think=False,
                      temperature=0.7, num_predict=NUM_PREDICT)
    ans, bad = R.extract_answer(g.content, "")
    if bad and (g.content or "").strip():
        ans = [g.content.strip()[:300]]
    f1_meta = {"content_ch": len(g.content or ""),
               "thinking_ch": 0,
               "malformed": bad,
               "elapsed_sec": round(time.time() - t0, 1),
               "n_calls": 1,
               "prompt_ch": len(prompt)}
    return ans, f1_meta


def build_bm25_fn():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/realdoc_hotpot_32k.jsonl")
    ap.add_argument("--out", default="logs/stageE_llama.jsonl")
    ap.add_argument("--n-instances", type=int, default=50)
    args = ap.parse_args()

    dcache = {}
    cp = os.path.join(BASE, "logs/stageC_distill_cache.json")
    if os.path.exists(cp):
        dcache = json.load(open(cp, encoding="utf-8"))
        log(f"共享蒸馏缓存: {len(dcache)} 条 (qwen3:1.7b 抽取, llama 消费)")

    bm25_fn = build_bm25_fn()

    insts = [json.loads(l) for l in
             open(os.path.join(BASE, args.data), encoding="utf-8") if l.strip()]
    insts = insts[: args.n_instances]
    import random
    random.Random(SHUFFLE_SEED).shuffle(insts)
    log(f"实例 {len(insts)} (洗牌 seed={SHUFFLE_SEED}); 模型 {MODEL}")

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
            key = f"{inst['id']}::qwen3:1.7b"
            if key not in dcache:
                log(f"  警告 {inst['id']} 无共享状态, 现场蒸馏")
                fbc, nc = R.distill_doc(inst, ["qwen3:1.7b"])
                dcache[key] = {"fbc": {str(kk): v for kk, v in fbc.items()},
                               "nc": nc}
                with open(cp, "w", encoding="utf-8") as f:
                    json.dump(dcache, f, ensure_ascii=False)
            else:
                fbc = {int(kk): v for kk, v in dcache[key]["fbc"].items()}
                nc = dcache[key]["nc"]
            E = R.build_e(fbc)
            x = "\n\n".join(inst["chunks"])
            q = inst["question"]

            conds = {}
            for tag, mat in [
                ("direct_full", x),
                ("eq", f"Verified evidence:\n{E}"),
                ("exq", f"Verified evidence:\n{E}\n\nFull material:\n{x}"),
            ]:
                ans, meta = answer_once(mat, q, tag)
                f1, em = R.score(ans, inst)
                conds[tag] = {"f1": f1, "em": em, "raw": ans, **meta}
            conds["bm25_top4"] = bm25_fn(inst)

            rec = {"id": inst["id"], "hops": inst["hops"],
                   "answer": inst["answer"],
                   "distill": {"n_calls": nc, "shared_state": True,
                               "E_chars": len(E)},
                   "conds": conds,
                   "elapsed_sec": round(time.time() - ts, 1)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            cs = " ".join(f"{c}={conds[c]['f1']:.2f}" for c in
                          ["direct_full", "eq", "exq", "bm25_top4"])
            log(f"[{k+1}/{len(insts)}] {inst['id']} h={inst['hops']} {cs} "
                f"({rec['elapsed_sec']}s)")

    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    n = len(recs)
    log(f"\n===== 汇总 (n={n}) =====")
    for c in ["direct_full", "eq", "exq", "bm25_top4"]:
        v = [r["conds"][c]["f1"] for r in recs]
        em = sum(r["conds"][c]["em"] for r in recs) / n if n else 0
        log(f"  {c}: F1={sum(v)/len(v):.4f} EM={em:.4f} (n={len(v)})")


if __name__ == "__main__":
    main()
