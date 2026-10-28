# -*- coding: utf-8 -*-
"""
run_budget_matched —— Budget-Matched Verified State 实验驱动 (Stage A)。
协议: protocol_budget_matched.md。条件 C0-C8 共享同一 3 条首遍 (等预算归因)。
断点: 级 [题][条件] 落盘; 失败分类独立字段; 严格串行。
"""
import argparse
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import random
from repro_lib import (GRAPH_SYSTEM, GraphSample, build_prompt,
                       em_score, extract_answer, f1_score, ollama_chat,
                       sample_from_target_chars, serialize_traces)
import token_budget as tb
from state_repr import (LEDGER_SUFFIX, build_oracle, fit_budget, format_ledger,
                        parse_ledger, verify_ledger)

STAGE_A = {
    "model": "qwen3:1.7b",
    "num_ctx": 16384,
    "num_predict": 12288,  # thinking 通道预算: 4096 会被思考烧穿致 content 恒空 (4B 同款伪影)
    "budget_tokens": 512,
    "seed_replay": 13,
    "seed_new": 20260918,
    "n_replay": 12,
    "n_new": 8,
    "target_chars": 3000,
    "temperature": 0.7,
}

STAGE_B = {
    # 确认性复跑 (protocol_stageB.md 冻结): 100 道全新题, 无重放, 预注册三主比较
    "model": "qwen3:1.7b",
    "num_ctx": 16384,
    "num_predict": 12288,
    "budget_tokens": 512,
    "seed_replay": 13,
    "seed_new": 20260920,
    "n_replay": 0,
    "n_new": 100,
    "target_chars": 3000,
    "temperature": 0.7,
}

CONDITIONS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]  # C0/C1 无调用

# 首遍专用: GRAPH_SYSTEM 的「只准一行」与 LEDGER_SUFFIX 冲突, 1.7B 会服从系统提示词
# 而丢弃 STATE 块 (Gate1 提取 0/20 的根因)。首遍允许多行输出, 其余条件不变。
FIRSTPASS_SYSTEM = (
    "You solve directed-graph algorithm problems. Use only the graph and operation in "
    "the user message. Every response MUST contain two parts and is INVALID without "
    "both: (1) the line 'Final Answer: [node1, node2]' (use [] for the empty set); "
    "(2) the [STATE]...[/STATE] ledger block exactly as specified at the end of the "
    "user message. Never omit the [STATE] block.")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_questions(cfg):
    """重放 seed13 的 n_replay 道混合题 + seed_new 的 n_new 道新题 (Stage B: 全新 100 道)。"""
    qs = []
    if cfg["n_replay"] > 0:
        rng = random.Random(cfg["seed_replay"])
        for i in range(24):
            op = "bfs" if i % 2 == 0 else "parents"
            s = sample_from_target_chars(rng, cfg["target_chars"], op_type=op)
            if i < cfg["n_replay"]:
                qs.append({"qid": f"rp{i:02d}", "sample": s, "replay": True})
    rng2 = random.Random(cfg["seed_new"])
    for j in range(cfg["n_new"]):
        op = "bfs" if j % 2 == 0 else "parents"
        s = sample_from_target_chars(rng2, cfg["target_chars"], op_type=op)
        qs.append({"qid": f"B{j:03d}", "sample": s, "replay": False})
    return qs


def classify_failure(ans_list, malformed, g, ledger) -> str:
    if g is not None and not (g.content or "").strip() and not (g.thinking or "").strip():
        return "empty_content"
    if malformed:
        return "malformed_answer"
    if ledger["has_ledger"] and not ledger["evidence"] and not ledger["derived"]:
        return "ledger_parse_partial"
    return ""


def first_pass(model, sample, cfg):
    """3 条首遍: 答案 + T(thinking/CoT) + ledger(content)。缓存于返回值。"""
    x = build_prompt(sample, "", "none") + LEDGER_SUFFIX
    traces, ledgers, answers, telem = [], [], [], []
    for j in range(3):
        t0 = time.time()
        for attempt in range(3):  # 无 ledger 重试 (1.7B 指令依从弱)
            g = ollama_chat(model, FIRSTPASS_SYSTEM, x, num_ctx=cfg["num_ctx"],
                            think=True, temperature=cfg["temperature"],
                            num_predict=cfg["num_predict"], timeout=2400)
            a, bad = extract_answer(g.content, g.thinking)
            led = parse_ledger(g.content)
            if led["has_ledger"] and (led["evidence"] or led["derived"]):
                break
        tb.update_from_response(model, len(x), g.raw)
        # T = thinking; 若无 thinking 通道则用去 ledger 的 CoT 正文
        if (g.thinking or "").strip():
            T = g.thinking
        else:
            T = (g.content or "").split("[STATE]")[0]
        fail = classify_failure(a, bad, g, led)
        traces.append(T)
        ledgers.append(led)
        answers.append(a)
        telem.append({"eval": g.raw.get("eval_count"),
                      "prompt_eval": g.raw.get("prompt_eval_count"),
                      "sec": round(time.time() - t0, 1), "fail": fail})
        log(f"    首遍{j+1}: f1={f1_score(a, sample.answer):.3f} "
            f"ledger={'Y' if led['has_ledger'] else 'N'} "
            f"ev={len(led['evidence'])} {telem[-1]}")
    return {"traces": traces, "ledgers": ledgers, "answers": answers,
            "telem": telem}


def majority_vote(ans_lists):
    """C1: 三首遍答案的多集合多数投票 (按节点计票, 取票数>1.5 的节点)。"""
    from collections import Counter
    c = Counter()
    for a in ans_lists:
        c.update(set(a))
    return sorted([n for n, k in c.items() if k >= 2])


def second_pass(model, sample, user, cfg):
    t0 = time.time()
    g = ollama_chat(model, GRAPH_SYSTEM, user, num_ctx=cfg["num_ctx"],
                    think=True, temperature=cfg["temperature"],
                    num_predict=cfg["num_predict"], timeout=2400)
    tb.update_from_response(model, len(user), g.raw)
    a, bad = extract_answer(g.content, g.thinking)
    return a, bad, g, {"eval": g.raw.get("eval_count"),
                       "prompt_eval": g.raw.get("prompt_eval_count"),
                       "sec": round(time.time() - t0, 1)}


def run_conditions(model, sample, fp, cfg, prev):
    """对一道题执行 C0-C8, 返回条件记录 dict。prev=断点。"""
    x = build_prompt(sample, "", "none")
    bch = tb.budget_chars(model, cfg["budget_tokens"])
    T = serialize_traces([fp["traces"][0]])  # 与论文一致: 用单条代表轨迹
    # —— 无调用条件 ——
    c0 = prev.get("C0") or {
        "f1": f1_score(fp["answers"][0], sample.answer),
        "em": em_score(fp["answers"][0], sample.answer),
        "raw": fp["answers"][0][:200], "calls": 0}
    c1 = prev.get("C1") or {
        "f1": f1_score(majority_vote(fp["answers"]), sample.answer),
        "em": em_score(majority_vote(fp["answers"]), sample.answer),
        "raw": majority_vote(fp["answers"])[:50], "calls": 0}
    out = {"C0": c0, "C1": c1}
    # —— ledger 衍生状态 ——
    led = fp["ledgers"][0]
    ver = verify_ledger(led, sample.edges)
    led_txt = format_ledger(led["evidence"], led["derived"], led["uncertain"])
    ver_txt = format_ledger(ver["evidence"], ver["derived"], ver["uncertain"])
    led_txt, _d1 = fit_budget(led_txt, bch)
    ver_txt, _d2 = fit_budget(ver_txt, bch)
    T_tail = "Below are a reasoning trace tail window from a first attempt.\n" + \
        fp["traces"][0][-bch:]
    jobs = {
        "C2": build_prompt(sample, T, "append"),
        "C3": build_prompt(sample, T, "state"),
        "C4": build_prompt(sample, T_tail, "state"),
        "C5": build_prompt(sample, led_txt, "state"),
        "C6": build_prompt(sample, ver_txt, "state"),
        "C7": ver_txt + "\n\nOperation:\n" + sample.op_text,
        "C8": build_prompt(sample, build_oracle(sample), "state"),
    }
    for cond in CONDITIONS:
        if prev.get(cond) and prev[cond].get("f1") is not None:
            out[cond] = prev[cond]
            continue
        a, bad, g, tel = second_pass(model, sample, jobs[cond], cfg)
        rec = {"f1": f1_score(a, sample.answer), "em": em_score(a, sample.answer),
               "raw": a[:200], "calls": 1, **tel}
        fail = classify_failure(a, bad, g, parse_ledger(""))
        if fail:
            rec["fail"] = fail
        if cond == "C6" and ver["has_ledger"] and ver["n_ev_total"] and not ver["n_ev_kept"]:
            rec["fail"] = rec.get("fail") or "verification_removed_all"
        out[cond] = rec
        log(f"    {cond}: f1={rec['f1']:.3f} em={rec['em']:.3f} {tel}")
    # 元信息
    out["_meta"] = {
        "n_ev_total": ver["n_ev_total"], "n_ev_kept": ver["n_ev_kept"],
        "ledger_chars": len(led_txt), "verified_chars": len(ver_txt),
        "trace_chars": len(T), "budget_chars": bch,
        "drop_ledger": _d1, "drop_ver": _d2,
        "fp_fail": [t.get("fail", "") for t in fp["telem"]],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/budget_matched_stageA.json")
    ap.add_argument("--stage", default="A")
    args = ap.parse_args()
    cfg = STAGE_B if args.stage == "B" else STAGE_A
    model = cfg["model"]
    qs = load_questions(cfg)
    log(f"Stage {args.stage}: {model} ctx={cfg['num_ctx']} "
        f"budget={cfg['budget_tokens']}tok 题数={len(qs)} "
        f"cpt={tb.chars_per_token(model)}")

    done = {}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out, encoding="utf-8"))
            for r in old.get("records", []):
                done[r["qid"]] = r
            log(f"断点: 已有 {len(done)} 题")
        except Exception:
            pass

    records = []
    for qi, q in enumerate(qs):
        sample = q["sample"]
        prev = done.get(q["qid"], {})
        fp_cached = prev.get("_firstpass")
        if fp_cached:
            fp = fp_cached
            log(f"[{qi+1}/{len(qs)}] {q['qid']} op={sample.op_type} "
                f"truth={len(sample.answer)} (首遍缓存)")
        else:
            log(f"[{qi+1}/{len(qs)}] {q['qid']} op={sample.op_type} "
                f"truth={len(sample.answer)} edges={len(sample.edges)}")
            fp = first_pass(model, sample, cfg)
        _m = __import__("re").search(r"depth (\d+)", sample.op_text)
        rec = {"qid": q["qid"], "replay": q["replay"],
               "op": sample.op_type, "answer": sample.answer,
               "depth": int(_m.group(1)) if _m else None,
               "_firstpass": fp}
        full_prev = dict(prev)
        full_prev["_firstpass"] = fp
        conds = run_conditions(model, sample, fp, cfg, full_prev)
        rec.update(conds)
        records.append(rec)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"stage": args.stage, "cfg": cfg,
                       "records": records}, f, ensure_ascii=False, indent=2)
        # 汇总
        acc = {c: sum(r[c]["f1"] for r in records) / len(records)
               for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
               if all(c in r for r in records)}
        log(f"  累计均分: " + " ".join(f"{k}={v:.3f}" for k, v in acc.items()))
    log(f"Stage {args.stage} 完成: {args.out}")


if __name__ == "__main__":
    main()
