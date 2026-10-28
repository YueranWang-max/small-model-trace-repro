# -*- coding: utf-8 -*-
"""Stage E 事后解释性补充比较 (ChatGPT 六轮审查 critical 1):
exq-direct_full, exq-bm25_top4, eq-direct_full
配对 delta + MC sign-flip p (1e6, add-one, 双侧) + W/L
输出: logs/stageE_extra_comparisons.txt
"""
import os
import json, random, statistics, io

BASE = os.path.dirname(os.path.abspath(__file__))
recs = {}
for l in open(BASE + r"\logs\stageE_llama.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        recs[r["id"]] = r  # 同 id 保留最后一条
rows = list(recs.values())

def f1(r, c):
    return r["conds"][c]["f1"] if c in r["conds"] else None

def signflip_p(diffs, n=1_000_000, seed=20260924):
    obs = abs(statistics.mean(diffs))
    rng = random.Random(seed)
    ge = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(s) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (n + 1)

out = io.StringIO()
out.write("Stage E (llama3.2:3b, n=%d) post-hoc explanatory comparisons\n" % len(rows))
out.write("MC sign-flip 1e6 add-one two-sided seed=20260924\n\n")
for a, b in [("exq", "direct_full"), ("exq", "bm25_top4"), ("eq", "direct_full")]:
    diffs = [f1(r, a) - f1(r, b) for r in rows if f1(r, a) is not None and f1(r, b) is not None]
    w = sum(1 for d in diffs if d > 0)
    lo = sum(1 for d in diffs if d < 0)
    p = signflip_p(diffs)
    out.write("%s - %s: delta=%+.5f  p=%.6f  W=%d L=%d (n=%d)\n" % (a, b, statistics.mean(diffs), p, w, lo, len(diffs)))

with open(BASE + r"\logs\stageE_extra_comparisons.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("done")
