"""Paper-clustered analysis of question versus generic instruction prefixes."""
from __future__ import annotations

import json
import random

from run_qasper_question_first_controls import ARMS, OUT


def mean(x):
    return sum(x) / len(x)


def sec(x, key):
    return (x.get(key) or 0) / 1e9


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == 8 and all(len(g) == 3 and all(set(q["arms"]) == set(ARMS) for q in g) for g in groups)
    rows = [q for g in groups for q in g]
    summary = {"status": "exploratory_dev_question_first_controls", "n_papers": 8,
               "n_questions": len(rows), "arms": {}, "paired": {}, "per_paper": []}
    for arm in ARMS:
        x = [q["arms"][arm] for q in rows]
        summary["arms"][arm] = {"f1": mean([v["f1"] for v in x]),
            "em": mean([v["em"] for v in x]),
            "prompt_tokens": sum(v.get("prompt_tokens") or 0 for v in x),
            "generation_tokens": sum(v.get("generation_tokens") or 0 for v in x),
            "api_total_sec": sum(sec(v, "api_total_ns") for v in x),
            "wall_sec": sum(v["wall_sec"] for v in x),
            "errors": sum("error" in v for v in x)}
    for a, b in (("question_first", "reread"), ("combined", "reread"),
                 ("combined", "question_first"), ("instruction_first", "reread")):
        diffs = [q["arms"][a]["f1"] - q["arms"][b]["f1"] for q in rows]
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sample = [groups[rng.randrange(len(groups))] for _ in groups]
            boot.append(mean([q["arms"][a]["f1"] - q["arms"][b]["f1"] for g in sample for q in g]))
        boot.sort()
        summary["paired"][a + "_minus_" + b] = {"f1_difference": mean(diffs),
            "paper_ci95": [boot[250], boot[9750]],
            "wins": sum(v > 1e-9 for v in diffs),
            "losses": sum(v < -1e-9 for v in diffs),
            "ties": sum(abs(v) <= 1e-9 for v in diffs)}
    for p in data["papers"]:
        summary["per_paper"].append({"paper_id": p["paper_id"], "arm_order": p["arm_order"],
            "f1": {a: mean([q["arms"][a]["f1"] for q in p["records"]]) for a in ARMS}})
    d = summary["paired"]["question_first_minus_reread"]
    summary["question_first_gate_pass"] = d["f1_difference"] >= 0.03 and d["paper_ci95"][0] > 0
    summary["caveat"] = "Eight Qasper dev papers, one local Qwen reader, full-source only; prompt order and generic instruction tested separately; sequential timing."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
