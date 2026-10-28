# -*- coding: utf-8 -*-
"""
bootstrap_ci —— 论文头条对比的配对 bootstrap 置信区间 (10000 次重采样)。
对比组:
  1) chain36: chain_et vs trace_state / evidence_state (36 题, seed7+seed13)
  2) wall12:  state-only(B_eonly, chain3) vs 带图 chain_et (chain2 引用, 12 题配对)
  3) recall18: det_post vs det_pre (18 题, 报警修复前后)
输出: logs/bootstrap_ci.json + 控制台表。p 值 = 2*min(P(diff<=0), P(diff>=0))。
"""
from __future__ import annotations
import io
import json
import os
import random

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
N_BOOT = 10000
rng = random.Random(7)


def load(name):
    return json.load(io.open(f"{BASE}\\{name}", encoding="utf-8"))


def paired_boot(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    boots = []
    for _ in range(N_BOOT):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    p_le = sum(1 for b in boots if b <= 0) / N_BOOT
    p_ge = sum(1 for b in boots if b >= 0) / N_BOOT
    p = max(2 * min(p_le, p_ge), 1.0 / N_BOOT)
    return {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_value": round(p, 5), "n": n,
            "n_pos": sum(1 for d in diffs if d > 0),
            "n_neg": sum(1 for d in diffs if d < 0),
            "n_zero": sum(1 for d in diffs if d == 0)}


results = {}

# ---- 1) chain 36 题 ----
recs = (load("chain_2b_3k.json")["records"]
        + load("chain_2b_3k_s13.json")["records"])
for cond in ("chain_et", "chain_te", "evidence_state"):
    diffs = [r["conditions"][cond]["f1"]
             - r["conditions"]["trace_state"]["f1"] for r in recs]
    results[f"chain36_{cond}_vs_trace"] = paired_boot(diffs)

# ---- 2) 执行墙 12 题: state-only vs 带图 ----
c3 = load("chain3_2b.json")["records"]
c2 = load("chain2_hard_2b.json")["records"]
# 配对: chain3 的 orig_index 顺序 == chain2 records 顺序 (同一次重放)
assert [r["orig_index"] for r in c3] == [r["orig_index"] for r in c2]
diffs = [a["B_eonly"]["f1"] - b["ref_conditions"]["chain_et"]["f1"]
         for a, b in zip(c3, c2)]
results["wall12_eonly_vs_chain_et"] = paired_boot(diffs)

# ---- 3) 召回报警 18 题 ----
recs = load("recall_alarm_2b.json")["records"]
diffs = [r["det_post_f1"] - r["det_pre_f1"] for r in recs]
results["recall18_post_vs_pre"] = paired_boot(diffs)

out = f"{BASE}\\bootstrap_ci.json"
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("===== Bootstrap 95% CI (paired, 10000 resamples) =====")
for k, v in results.items():
    print(f"{k}: diff={v['mean_diff']:+.3f} "
          f"CI=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
          f"p={v['p_value']:.4f} (+{v['n_pos']}/-{v['n_neg']}/={v['n_zero']})")
