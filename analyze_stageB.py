# -*- coding: utf-8 -*-
"""Stage B 终局分析 —— 严格按 protocol_stageB.md 附录 A 预注册算法，只读 json。"""
import os
import json, random, sys
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "budget_matched_stageB.json")
B, SEED = 10000, 20260921

data = json.load(open(DATA, encoding="utf-8"))
recs = data["records"]
assert len(recs) == 100, f"records={len(recs)} != 100"
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# ---------- 总表 (ITT) ----------
print("=" * 72)
print("A. 九条件总表 (n=100, intention-to-run)")
print(f"{'cond':6} {'F1':>7} {'EM':>7} {'valid_n':>8} {'fail_n':>7} {'latency':>8}")
valid_f1, fail_dist, lat = {}, {}, {}
for c in CONDS:
    f1s = [r[c]["f1"] for r in recs]
    ems = [r[c]["em"] for r in recs]
    fails = [r[c].get("fail", "") for r in recs]
    ok = [r[c]["f1"] for r in recs if not r[c].get("fail")]
    nfail = sum(1 for f in fails if f)
    sec = sum(r[c].get("sec", 0) for r in recs) / 100
    valid_f1[c] = sum(ok) / len(ok) if ok else 0.0
    fail_dist[c] = Counter(f for f in fails if f)
    lat[c] = sec
    print(f"{c:6} {sum(f1s)/100:7.3f} {sum(ems)/100:7.3f} {len(ok):>8} {nfail:>7} {sec:7.1f}s")

print("\n失败分类明细 (仅非零):")
for c in CONDS:
    if fail_dist[c]:
        print(f"  {c}: {dict(fail_dist[c])}")

# ---------- 预注册主比较 ----------
def paired_analysis(a_key, b_key):
    """δ_i = F1_i(a) - F1_i(b); 附录 A 算法。"""
    d = [r[a_key]["f1"] - r[b_key]["f1"] for r in recs]
    n = len(d)
    mean = sum(d) / n
    rng = random.Random(SEED)
    boots = []
    for _ in range(B):
        s = sum(d[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    n_le = sum(1 for b in boots if b <= 0)
    n_ge = sum(1 for b in boots if b >= 0)
    p = min(1.0, 2 * min(n_le + 1, n_ge + 1) / (B + 1))
    wins = sum(1 for x in d if x > 0); losses = sum(1 for x in d if x < 0)
    return {"mean": mean, "lo": lo, "hi": hi, "p": p, "wins": wins, "losses": losses, "d": d}

print("\n" + "=" * 72)
print("B. 三个预注册主比较 (B=10000, seed 20260921, add-one p)")
results = {
    "C2-C0": paired_analysis("C2", "C0"),
    "C3-C0": paired_analysis("C3", "C0"),
    "C6-C3": paired_analysis("C6", "C3"),
}
for k, v in results.items():
    print(f"  {k}: delta={v['mean']:+.3f}  CI=[{v['lo']:+.3f},{v['hi']:+.3f}]  "
          f"p={v['p']:.4f}  ({v['wins']}W-{v['losses']}L)")

# Holm 步降 (m=3, alpha=0.05)
print("\nC. Holm 步降校正 (alpha=0.05)")
order = sorted(results.items(), key=lambda kv: kv[1]["p"])
m = 3
prev_adj = 0.0
verdicts = {}
for i, (k, v) in enumerate(order):
    thresh = 0.05 / (m - i)
    adj = min(1.0, max(prev_adj, (m - i) * v["p"]))
    prev_adj = max(prev_adj, adj)
    reject = v["p"] <= thresh
    print(f"  rank{i+1} {k}: p={v['p']:.4f} vs alpha/{m-i}={thresh:.4f} -> "
          f"{'拒绝H0' if reject else '不拒绝'}  adj_p={adj:.4f}")
    verdicts[k] = {"reject": reject, "adj_p": adj}

print("\nD. 判定 (确认 = adj_p<0.05 且 CI 排除零 且 方向符合预注册)")
direction = {"C2-C0": "<0", "C3-C0": ">0", "C6-C3": "!=0"}
for k, v in results.items():
    ci_excl = (v["lo"] > 0 or v["hi"] < 0)
    if k == "C2-C0":
        dir_ok = v["mean"] < 0 and ci_excl
    elif k == "C3-C0":
        dir_ok = v["mean"] > 0 and ci_excl
    else:
        dir_ok = ci_excl
    confirmed = verdicts[k]["reject"] and ci_excl and dir_ok
    print(f"  {k}: adj_p={verdicts[k]['adj_p']:.4f}  CI排除零={ci_excl}  "
          f"方向({direction[k]})符合={dir_ok}  => {'** 确认 **' if confirmed else '未确认'}")

# ---------- 分层稳定性 ----------
print("\n" + "=" * 72)
print("E. 分层稳定性")
def strat(filt, label):
    sub = [r for r in recs if filt(r)]
    if not sub:
        return
    row = f"  {label} (n={len(sub)}): "
    for a, b in [("C2", "C0"), ("C3", "C0"), ("C6", "C3")]:
        dd = [r[a]["f1"] - r[b]["f1"] for r in sub]
        row += f"{a}-{b}={sum(dd)/len(dd):+.3f}  "
    print(row)
strat(lambda r: r["op"] == "bfs", "BFS")
strat(lambda r: r["op"] == "parents", "parents")
for dep in range(2, 7):
    strat(lambda r, dep=dep: r.get("depth") == dep, f"depth {dep}")

# ---------- pilot 对比 ----------
print("\n" + "=" * 72)
print("F. Stage A pilot (n=20) 对比")
pilot = {"C0": 0.487, "C1": 0.509, "C2": 0.399, "C3": 0.550, "C4": 0.466,
         "C5": 0.476, "C6": 0.485, "C7": 0.238, "C8": 0.582}
for c in CONDS:
    cur = sum(r[c]["f1"] for r in recs) / 100
    print(f"  {c}: pilot {pilot[c]:.3f} -> StageB {cur:.3f}  ({cur - pilot[c]:+.3f})")
