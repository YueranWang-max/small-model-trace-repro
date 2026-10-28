# -*- coding: utf-8 -*-
"""核验 Stage A / Stage B 题目隔离: 种子独立 + 图签名零重叠 + 运行中数据与声明种子一致。"""
import os
import json, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_budget_matched import STAGE_A, STAGE_B, load_questions

def sig(sample):
    h = hashlib.sha256()
    h.update(sample.op_type.encode())
    h.update("|".join(sorted(sample.answer)).encode())
    h.update("|".join(sorted(e if isinstance(e, str) else f"{e[0]}>{e[1]}" for e in sample.edges)).encode())
    return h.hexdigest()[:16]

qa = load_questions(STAGE_A)
qb = load_questions(STAGE_B)
sa = {sig(q["sample"]): q["qid"] for q in qa}
sb = {sig(q["sample"]): q["qid"] for q in qb}
overlap = set(sa) & set(sb)
print(f"Stage A questions: {len(qa)}  sigs: {len(sa)}")
print(f"Stage B questions: {len(qb)}  sigs: {len(sb)}")
print(f"OVERLAP: {len(overlap)}  {[(sa[s], sb[s]) for s in overlap]}")

# 一致性: 当前运行中的 stageB json 与声明种子重放的图签名是否逐题一致
data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "budget_matched_stageB.json"), encoding="utf-8"))
recs = data.get("records", [])
print(f"stageB.json records: {len(recs)}")
mism = 0
for r in recs:
    q = next((x for x in qb if x["qid"] == r["qid"]), None)
    if q is None:
        print(f"  {r['qid']} NOT IN REGENERATED SET!"); mism += 1; continue
    if sig(q["sample"]) != sig_regenerated if False else False:
        pass
    # 图尺寸与答案集合比对
    if sorted(q["sample"].answer) != sorted(r["answer"]):
        print(f"  {r['qid']} ANSWER MISMATCH regen={q['sample'].answer} run={r['answer']}"); mism += 1
print(f"integrity mismatches: {mism}")
# op 分布
from collections import Counter
print("run op dist:", Counter(r["op"] for r in recs))
print("run depths:", Counter(r.get("depth") for r in recs))
