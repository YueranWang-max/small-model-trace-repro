"""Frozen paired answer and evidence analysis, clustering by paper."""
from __future__ import annotations
import json
import random
from analyze_qasper_evidence_gaps import norm
from run_qasper_genre_structure import candidates
from run_qasper_genre_structure_readout import ARMS, OUT
from run_qasper_prefix_vs_bm25 import DEV

def mean(xs):
    return sum(xs) / len(xs)

def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    source = json.loads(DEV.read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == len(plan["papers"]) == 30
    assert all(set(r["arms"]) == set(ARMS) for g in groups for r in g)
    assert sum(len(g) for g in groups) == sum(len(p["questions"]) for p in plan["papers"])
    rows = [r for g in groups for r in g]
    summary = {"papers": 30, "questions": len(rows), "arms": {}, "paired": {}, "evidence": {}}
    preprocess = sum(p.get("construction_and_selection_sec", 0) for p in plan["papers"])
    summary["prototype_preprocessing_sec"] = preprocess
    for a in ARMS:
        xs = [r["arms"][a] for r in rows]
        summary["arms"][a] = {"f1": mean([x["f1"] for x in xs]), "em": mean([x["em"] for x in xs]),
            "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in xs),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in xs),
            "api_sec": sum((x.get("api_total_ns") or 0) / 1e9 for x in xs),
            "wall_sec": sum(x["wall_sec"] for x in xs), "errors": sum("error" in x for x in xs)}
    diffs = [r["arms"]["structure8"]["f1"] - r["arms"]["flat8"]["f1"] for r in rows]
    rng = random.Random(20261003)
    boot = []
    for _ in range(10000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(mean([r["arms"]["structure8"]["f1"] - r["arms"]["flat8"]["f1"] for g in sampled for r in g]))
    boot.sort()
    summary["paired"] = {"f1_difference": mean(diffs), "paper_ci95": [boot[250], boot[9750]],
        "wins": sum(x > 1e-9 for x in diffs), "losses": sum(x < -1e-9 for x in diffs),
        "ties": sum(abs(x) <= 1e-9 for x in diffs), "answer_gate_pass": boot[250] > 0}
    recalls = {a: [] for a in ARMS}
    hits = {a: 0 for a in ARMS}
    alls = {a: 0 for a in ARMS}
    changes = 0
    for p in plan["papers"]:
        for q in p["questions"]:
            changes += q["flat8"] != q["structure8"]
            original = next(x for x in source[p["paper_id"]]["qas"] if x["question_id"] == q["id"])
            spans = candidates(original)
            eids = {i for i, v in enumerate(p["paragraphs"]) if any(s in norm(v["text"]) for s in spans)}
            if not eids:
                continue
            for a in ARMS:
                n = len(eids & set(q[a]))
                recalls[a].append(n / len(eids))
                hits[a] += n > 0
                alls[a] += n == len(eids)
    summary["changed_selections"] = changes
    for a in ARMS:
        summary["evidence"][a] = {"eligible": len(recalls[a]), "recall": mean(recalls[a]),
                                   "any_hit": hits[a], "all_hit": alls[a]}
    summary["caveat"] = "Development split only; one reader; eight-paragraph budget is not equal token budget; local sequential API timings; prototype preprocessing includes both arms and unused summary creation."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
