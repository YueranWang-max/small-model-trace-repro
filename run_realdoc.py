# -*- coding: utf-8 -*-
"""
run_realdoc —— 真实文档多跳 QA 实验 (MuSiQue-32K)。

管线 (与图实验同构, 三处替换):
  蒸馏: 任务条件化逐片扫描 —— 问题+片 -> 候选事实三元组 (s | r | o)
  核验: span 校验 (s 与 o 须原文出现, 大小写不敏感) —— 替代图的正则邻接
  报警: 实体索引 —— 问题实体所在片零确认事实 => 定点重扫
  读出: S1 纯证据 [E,q] / S2 状态+全图 [E,x,q] / S3 论文原法 [T,x,q]
评分: F1/EM (接受 answer_aliases)。
逐实例 JSONL 检查点, 断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_lib import (  # noqa: E402
    em_score, extract_answer, f1_score, ollama_chat, serialize_traces,
)

MODEL = "openbmb/minicpm5-2b"
ENSEMBLE = ["openbmb/minicpm5-2b", "qwen3.5:4b"]
CTX = 16384
PRED_SCAN = 6144  # thinking+content 共享预算: thinking ~2.5K token 后还需余量吐 QUOTE 行
PRED_READ = 4096

QA_SYSTEM = ("You answer questions using the provided material. "
             "Be concise and factual.")

SCAN_PROMPT = """Material (chunk {ci}/{n}):
{chunk}

Quote VERBATIM up to 8 sentences from this material that contain concrete
information: names of people, works, dates, places, numbers, memberships,
or events. One per line, in this format:
QUOTE: <copy the sentence exactly as written>
Do not paraphrase and do not answer any question. If there are no such
sentences, write nothing."""

RESCAN_PROMPT = """Material (chunk {ci}/{n}):
{chunk}

This material is known to mention items related to: {hints}

Quote VERBATIM every sentence that mentions any of these items (or any name,
date, title, place, number). One per line, in this format:
QUOTE: <copy the sentence exactly as written>
Do not paraphrase. If truly none, write nothing."""


def log(m):
    print(m, flush=True)


def scan_chat(prompt, model, num_predict=PRED_SCAN):
    g = ollama_chat(model, QA_SYSTEM, prompt, num_ctx=CTX, think=True,
                    temperature=0.0, num_predict=num_predict)
    ans, bad = extract_answer(g.content, g.thinking)
    return g, ans, bad


def parse_quotes(text):
    quotes = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*QUOTE:\s*(.+?)\s*$", line)
        if m:
            q = m.group(1).strip()
            if 20 < len(q) < 500:
                quotes.append(q)
    return quotes


def span_verify(chunk_text, quote):
    """引句必须逐字出现在片内 (大小写不敏感, 空白折叠)。"""
    norm = lambda s: " ".join(s.lower().split())
    return norm(quote) in norm(chunk_text)


def question_entities(q):
    """粗粒度问题实体: 去停用词后的词面。"""
    stop = {"who", "what", "which", "when", "where", "the", "a", "an", "of",
            "is", "are", "was", "were", "in", "on", "at", "to", "for", "by",
            "and", "or", "does", "did", "do", "from", "that", "this"}
    return [w for w in re.findall(r"[A-Za-z0-9']+", q)
            if w.lower() not in stop and len(w) > 2]


def distill_doc(inst, model_list, chunk_chars=None):
    """逐片逐字摘引 -> 机械校验(逐字+实体覆盖) -> 报警重扫。
    返回 (quotes_by_chunk, n_calls)。"""
    chunks = inst["chunks"]
    ents = [w.lower() for w in question_entities(inst["question"])]
    quotes_by_chunk = {}
    n_calls = 0
    for ci, ct in enumerate(chunks):
        got = []
        for attempt in range(3):
            model = model_list[min(attempt, len(model_list) - 1)]
            prompt = (RESCAN_PROMPT.format(ci=ci + 1, n=len(chunks),
                                           chunk=ct,
                                           hints=", ".join(ents[:8]))
                      if attempt > 0 else
                      SCAN_PROMPT.format(ci=ci + 1, n=len(chunks),
                                         chunk=ct))
            g, _, _ = scan_chat(prompt, model)
            n_calls += 1
            raw = parse_quotes(g.content or g.thinking)
            got = [q for q in raw if span_verify(ct, q)]
            if got:
                break
        # 机械过滤: 保留与问题实体有词面重叠的引句; 全无则保留原文引句
        hit = [q for q in got
               if any(e in q.lower() for e in ents)]
        if hit:
            got = hit
        if got:
            quotes_by_chunk[ci] = got[:8]
    return quotes_by_chunk, n_calls


def build_e(quotes_by_chunk):
    lines = []
    for ci in sorted(quotes_by_chunk):
        lines.append(f"[chunk {ci}]")
        for q in quotes_by_chunk[ci]:
            lines.append(f"- {q}")
    return "\n".join(lines) if lines else "(no verified evidence)"


_STOP_TOKENS = {"the", "a", "an", "of", "in", "and", "is", "was", "were",
                "to", "for", "by", "on", "at", "it", "its"}


def _norm_tokens(s):
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in toks if t not in _STOP_TOKENS} or set(toks)


def score(ans_list, inst):
    """文本答案评分: 预测与金标/别名做 token 集 F1 / EM。"""
    cands = [inst["answer"]] + list(inst.get("answer_aliases") or [])
    pred = " ".join(ans_list) if ans_list else ""
    pt = _norm_tokens(pred)
    best_f1 = best_em = 0.0
    for c in cands:
        ct = _norm_tokens(c)
        if not pt and not ct:
            return 1.0, 1.0
        ov = len(pt & ct)
        prec = ov / len(pt) if pt else 0.0
        rec = ov / len(ct) if ct else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        em = 1.0 if pt == ct else 0.0
        best_f1, best_em = max(best_f1, f1), max(best_em, em)
    return round(best_f1, 4), round(best_em, 4)


def readout(model, prompt):
    import urllib.error
    prompt = (prompt.rstrip()
              + "\n\nAnswer with only the final answer itself. "
                "End your reply with exactly this line:\n"
                "Final Answer: <answer>")
    try:
        g = ollama_chat(model, QA_SYSTEM, prompt, num_ctx=CTX, think=True,
                        temperature=0.7, num_predict=PRED_READ)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"ollama HTTP {e.code}: {body}") from e
    ans, bad = extract_answer(g.content, g.thinking)
    if bad and (g.content or "").strip():
        # 兜底: 正文有裸文本答案时直接采用 (去 think 残留)
        txt = re.sub(r"</?think>", "", g.content, flags=re.I).strip()
        txt = re.sub(r"Final Answer:\s*", "", txt, flags=re.I).strip()
        if txt:
            ans, bad = [txt[:300]], False
    return ans, {"content_ch": len(g.content), "thinking_ch": len(g.thinking),
                 "malformed": bad}


# T 尾窗预算: 2B 16K ctx 下, 31.5K 文档 ~8K token, T 只能占 ~3K token
T_BUDGET_CHARS = 12000


def truncate_tails(T):
    if len(T) <= T_BUDGET_CHARS:
        return T
    return "...[earlier traces truncated]\n" + T[-T_BUDGET_CHARS:]


def main():
    global PRED_READ
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/realdoc_musique_32k.jsonl")
    ap.add_argument("--out", default="logs/realdoc_2b.jsonl")
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--ensemble", action="store_true",
                    help="蒸馏用双模型集成 (默认单 2B)")
    ap.add_argument("--pred-read", type=int, default=4096,
                    help="读出+首遍输出预算 (thinking 家族须 >=12288)")
    ap.add_argument("--ids", default="",
                    help="逗号分隔实例 id 子集 (空=全部)")
    ap.add_argument("--distill-cache", default="",
                    help="蒸馏缓存 json (id -> quotes_by_chunk), 命中则跳过蒸馏")
    args = ap.parse_args()
    PRED_READ = args.pred_read
    log(f"预算: PRED_SCAN={PRED_SCAN} PRED_READ={PRED_READ}")

    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, args.data)
    out = os.path.join(base, args.out)
    models = ENSEMBLE if args.ensemble else [MODEL]

    insts = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            insts.append(json.loads(line))
    insts = insts[: args.n_instances]
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        insts = [i for i in insts if i["id"] in want]
    log(f"实例 {len(insts)}; 蒸馏模型 {models}")

    # ---- 蒸馏缓存 ----
    cache_path = os.path.join(base, args.distill_cache) if args.distill_cache else None
    dcache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            dcache = json.load(f)
        log(f"蒸馏缓存: {len(dcache)} 条")

    def cached_distill(inst, model_list):
        key = f"{inst['id']}::{'+'.join(model_list)}"
        if key in dcache:
            fbc = {int(k): v for k, v in dcache[key]["fbc"].items()}
            return fbc, dcache[key]["nc"], True
        fbc, nc = distill_doc(inst, model_list)
        dcache[key] = {"fbc": {str(k): v for k, v in fbc.items()}, "nc": nc}
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(dcache, f, ensure_ascii=False)
        return fbc, nc, False

    done = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except Exception:
                    pass
        log(f"断点续跑: 已有 {len(done)} 实例")

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fo:
        for k, inst in enumerate(insts):
            if inst["id"] in done:
                continue
            ts = time.time()
            rec = {"id": inst["id"], "hops": inst["hops"],
                   "answer": inst["answer"]}
            # ---- 蒸馏 + 核验 + 报警 ----
            fbc, nc, hit = cached_distill(inst, models)
            e = build_e(fbc)
            if hit:
                log(f"    蒸馏缓存命中 {inst['id']} (E={len(e)}ch)")
            rec["distill"] = {"n_calls": nc, "n_facts": sum(len(v) for v in
                                                            fbc.values()),
                              "n_chunks_hit": len(fbc),
                              "E_chars": len(e),
                              "support_hit": round(
                len([c for c in inst["support_chunk_ids"] if c in fbc])
                / max(1, len(inst["support_chunk_ids"])), 4)}
            # ---- S1 纯证据 ----
            p1 = (f"Verified evidence:\n{e}\n\n"
                  f"Question: {inst['question']}\n")
            ans, meta = readout(models[0], p1)
            f1, em = score(ans, inst)
            rec["S1"] = {"f1": f1, "em": em, "raw": ans, **meta}
            # ---- S2 状态+全图 ----
            p2 = (f"Verified evidence:\n{e}\n\nFull material:\n"
                  f"{inst['chunks'] and chr(10).join(inst['chunks'])}\n\n"
                  f"Question: {inst['question']}\n")
            ans, meta = readout(models[0], p2)
            f1, em = score(ans, inst)
            rec["S2"] = {"f1": f1, "em": em, "raw": ans, **meta}
            # ---- S3 论文原法 ----
            x = "\n\n".join(inst["chunks"])
            traces, first_f1s = [], []
            for j in range(3):
                g = ollama_chat(models[0], QA_SYSTEM,
                                f"{x}\n\nQuestion: {inst['question']}\n"
                                "Answer with only the final answer itself. "
                                "End your reply with exactly this line:\n"
                                "Final Answer: <answer>",
                                num_ctx=CTX, think=True, temperature=0.7,
                                num_predict=PRED_READ)
                a, _ = extract_answer(g.content, g.thinking)
                if not a and (g.content or "").strip():
                    a = [g.content.strip()[:300]]
                traces.append(g.thinking)
                ff1, _ = score(a, inst)
                first_f1s.append(ff1)
            T = serialize_traces(traces)
            T_raw = len(T)
            T = truncate_tails(T)
            log(f"      T: {T_raw} -> {len(T)} 字符 (尾窗截断)")
            ans, meta = readout(models[0],
                                f"{T}\n\nFull material:\n{x}\n\n"
                                f"Question: {inst['question']}\n")
            f1, em = score(ans, inst)
            rec["S3"] = {"f1": f1, "em": em, "raw": ans,
                         "first_mean_f1": round(sum(first_f1s) / 3, 4),
                         **meta}
            rec["elapsed_sec"] = round(time.time() - ts, 1)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            log(f"[{k+1}/{len(insts)}] {inst['id']} h={inst['hops']} "
                f"E={rec['distill']['E_chars']}ch "
                f"sup={rec['distill']['support_hit']} "
                f"S1={rec['S1']['f1']:.2f} S2={rec['S2']['f1']:.2f} "
                f"S3={rec['S3']['f1']:.2f} ({rec['elapsed_sec']}s)")

    # ---- 汇总 ----
    recs = []
    with open(out, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    n = len(recs)
    if n:
        summary = {"n": n, "elapsed_sec": round(time.time() - t0, 1),
                   "support_hit_mean": round(sum(
                       r["distill"]["support_hit"] for r in recs) / n, 4)}
        for key in ("S1", "S2", "S3"):
            summary[f"{key}_f1"] = round(sum(r[key]["f1"] for r in recs) / n, 4)
            summary[f"{key}_em"] = round(sum(r[key]["em"] for r in recs) / n, 4)
        for h in (2, 3, 4):
            rs = [r for r in recs if r["hops"] == h]
            if rs:
                summary[f"h{h}_n"] = len(rs)
                for key in ("S1", "S3"):
                    summary[f"h{h}_{key}_f1"] = round(
                        sum(r[key]["f1"] for r in rs) / len(rs), 4)
        log("\n===== 汇总 =====")
        for kk, v in summary.items():
            log(f"  {kk}: {v}")


if __name__ == "__main__":
    main()
