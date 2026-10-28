# -*- coding: utf-8 -*-
"""
analyze_stageD —— Stage D 预注册分析 (protocol_stageDE.md §2)。
主比较 m=4 (六轮审查后修正 2026-09-21): exq-eq, txq-tq, bm25-eq, dense-eq;
每族分别检验; Holm 步降。原冻结版 m=3 (best_rag-eq) 的输出存档于
logs/stageDE_final_analysis.txt; 本修正仅重分析, 未重跑实验 (MuSiQue 除外,
见干净复跑 stageD_musique_rerun.jsonl)。
bootstrap B=1e4 (seed 20260922) 只用于 CI; MC sign-flip 1e6 (seed 20260922, add-one) 用于 p。
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
BOOT_B = 10_000
BOOT_SEED = 20260922
MC = 1_000_000
MC_SEED = 20260922


def load(path):
    if not os.path.exists(path):
        return []
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    # 同 id 去重（双进程并发期可能重复追加同一题），保留最后一条
    uniq = {}
    for r in recs:
        uniq[r.get("id", f"_idx{len(uniq)}")] = r
    return list(uniq.values())


def conds_f1(recs, c):
    return [r["conds"][c]["f1"] if c in r["conds"] else None for r in recs]


def paired(a, b):
    """配对差 (只取两条件都有值的题), 返回 (diffs, n_used)。"""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    return d, len(d)


def boot_ci(diffs, b=BOOT_B, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(b):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(s) / n)
    stats.sort()
    return stats[int(0.025 * b)], stats[int(0.975 * b)]


def signflip_p(diffs, direction=None, mc=MC, seed=MC_SEED):
    """双侧 sign-flip MC p (add-one); direction='neg'/'pos' 给单侧。"""
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    ge = le = 0
    for _ in range(mc):
        s = sum(diffs[i] if rng.random() < 0.5 else -diffs[i] for i in range(n))
        m = s / n
        if m >= obs - 1e-12:
            ge += 1
        if m <= obs + 1e-12:
            le += 1
    if direction == "neg":
        p = (le + 1) / (mc + 1)
    elif direction == "pos":
        p = (ge + 1) / (mc + 1)
    else:
        p = (min(le, ge) * 2 + 1) / (mc + 1)
    se = math.sqrt((min(le, ge) / mc) * (1 - min(le, ge) / mc) / mc)
    return obs, p, se


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def analyze_family(recs, tag):
    print(f"\n########## {tag} (n={len(recs)}) ##########")
    avail = [c for c in ["eq", "exq", "tq", "txq", "bm25_top4", "dense_top4"]
             if recs and c in recs[0]["conds"]]
    means = {}
    for c in avail:
        v = [r["conds"][c]["f1"] for r in recs]
        em = sum(r["conds"][c]["em"] for r in recs) / len(recs)
        lat = sum(r["conds"][c].get("elapsed_sec", 0) for r in recs) / len(recs)
        means[c] = sum(v) / len(v)
        print(f"  {c}: F1={means[c]:.4f} EM={em:.4f} lat={lat:.1f}s")

    # 六轮审查修正 (2026-09-21, post-hoc analysis revision): 取消数据依赖的
    # best_rag 选择, bm25-eq 与 dense-eq 都作为主比较, Holm 族扩为 m=4。
    # 事件记录: 原冻结规则为 best_rag-eq (m=3); 本修正不重跑实验, 仅重分析。
    primaries = [("exq", "eq"), ("txq", "tq"),
                 ("bm25_top4", "eq"), ("dense_top4", "eq")]
    diffs_list, labels = [], []
    for hi, lo in primaries:
        if hi in means and lo in means:
            d, n = paired(conds_f1(recs, hi), conds_f1(recs, lo))
            diffs_list.append((d, n))
            labels.append(f"{hi}-{lo}")

    ps, ses, cis, deltas = [], [], [], []
    for d, n in diffs_list:
        obs, p, se = signflip_p(d)
        lo_ci, hi_ci = boot_ci(d)
        ps.append(p)
        ses.append(se)
        cis.append((lo_ci, hi_ci))
        deltas.append(obs)
    adj = holm(ps)
    print("\n  --- 主比较 m=4 (审查后修正: RAG 两条件均入族; MC sign-flip 1e6 add-one; bootstrap CI 1e4) ---")
    for lab, dv, ci, p, a, se in zip(labels, deltas, cis, ps, adj, ses):
        verdict = "PASS Holm<0.05" if a < 0.05 else "not significant"
        print(f"  {lab}: delta={dv:+.4f} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"p={p:.6f} (se={se:.1e}) adj_p={a:.4f} -> {verdict}")

    # 次要: exq - 各 RAG 基线 (不校正)
    for rag in ["bm25_top4", "dense_top4"]:
        if rag in means and "exq" in means:
            d, n = paired(conds_f1(recs, "exq"), conds_f1(recs, rag))
            obs, p, se = signflip_p(d)
            lo_ci, hi_ci = boot_ci(d)
            print(f"  [次要] exq-{rag}: delta={obs:+.4f} "
                  f"CI95=[{lo_ci:+.4f},{hi_ci:+.4f}] p={p:.6f}")

    # 按 hops 分层 (exploratory)
    hops_set = sorted({r["hops"] for r in recs if "exq" in r["conds"]})
    for h in hops_set:
        rs = [r for r in recs if r["hops"] == h and "exq" in r["conds"]
              and "eq" in r["conds"]]
        if len(rs) >= 5:
            d = [r["conds"]["exq"]["f1"] - r["conds"]["eq"]["f1"] for r in rs]
            print(f"  [exploratory] h{h} (n={len(rs)}): "
                  f"exq-eq={sum(d)/len(d):+.4f}")
    return {lab: {"delta": dv, "adj_p": a}
            for lab, dv, a in zip(labels, deltas, adj)}


if __name__ == "__main__":
    res = {}
    res["hotpot"] = analyze_family(load(os.path.join(BASE, "logs/stageD_hotpot.jsonl")), "HotpotQA-32K 主族")
    mus_path = os.path.join(BASE, "logs/stageD_musique_rerun.jsonl")
    if not os.path.exists(mus_path):
        mus_path = os.path.join(BASE, "logs/stageD_musique.jsonl")
    mus_tag = "MuSiQue-32K 复制族 (干净复跑)" if "rerun" in mus_path else "MuSiQue-32K 复制族"
    res["musique"] = analyze_family(load(mus_path), mus_tag)
    print("\n===== 跨族方向一致性 =====")
    for lab in ["exq-eq", "txq-tq", "bm25_top4-eq", "dense_top4-eq"]:
        ds = []
        for fam in ["hotpot", "musique"]:
            r = res.get(fam, {}).get(lab)
            ds.append(f"{fam}={r['delta']:+.4f}(adj {r['adj_p']:.3f})" if r else f"{fam}=N/A")
        print(f"  {lab}: " + " | ".join(ds))
