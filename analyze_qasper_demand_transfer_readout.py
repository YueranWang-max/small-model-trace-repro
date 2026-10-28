"""Paired official-test answer analysis with paper-clustered uncertainty."""
from __future__ import annotations
import json
import random
from run_qasper_demand_transfer import OUT, ARMS
from run_qasper_prefix_vs_bm25 import sha

def mean(xs):
    return sum(xs) / len(xs)

def main():
    data = json.loads((OUT / "reader_results.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == 40 and set(p["paper_id"] for p in data["papers"]) == set(frozen["reader_paper_ids"])
    expected = sum(len(p["questions"]) for p in frozen["papers"] if p["id"] in frozen["reader_paper_ids"])
    rows = [r for g in groups for r in g]
    assert len(rows) == expected and all(set(r["arms"]) == set(ARMS) for r in rows)
    summary = {"papers": 40, "questions": len(rows), "arms": {}, "paired": {},
               "reader_results_sha256": sha((OUT / "reader_results.json").read_bytes())}
    for a in ARMS:
        xs = [r["arms"][a] for r in rows]
        summary["arms"][a] = {"f1": mean([x["f1"] for x in xs]), "em": mean([x["em"] for x in xs]),
            "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in xs),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in xs),
            "api_sec": sum((x.get("api_total_ns") or 0)/1e9 for x in xs),
            "wall_sec": sum(x["wall_sec"] for x in xs),
            "model_load_sec": sum((x.get("load_ns") or 0)/1e9 for x in xs),
            "errors": sum("error" in x for x in xs),
            "missing_final_line": sum(not x.get("answer") for x in xs)}
    diffs = [r["arms"]["global"]["f1"] - r["arms"]["bm25"]["f1"] for r in rows]
    rng = random.Random(20261003)
    boot = []
    for _ in range(10000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(mean([r["arms"]["global"]["f1"] - r["arms"]["bm25"]["f1"] for g in sampled for r in g]))
    boot.sort()
    summary["paired"] = {"f1_difference": mean(diffs), "paper_ci95": [boot[250], boot[9750]],
        "wins": sum(d > 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs),
        "ties": sum(abs(d) <= 1e-9 for d in diffs), "reader_gate_pass": boot[250] > 0}
    evid = json.loads((OUT / "evidence_results.json").read_text(encoding="utf-8"))
    erows = [q for p in evid if p["paper_id"] in frozen["reader_paper_ids"] for q in p["records"]]
    summary["reader_cohort_evidence"] = {}
    for a in ARMS:
        eligible = [r["arms"][a] for r in erows if r["arms"][a]["recall"] is not None]
        summary["reader_cohort_evidence"][a] = {"eligible": len(eligible), "recall": mean([v["recall"] for v in eligible]),
            "any_hit": sum(v["any"] for v in eligible), "all_hit": sum(v["all"] for v in eligible),
            "chars": sum(r["arms"][a]["chars"] for r in erows)}
    summary["caveat"] = "One frozen Qwen reader on40 ID-selected official-test papers, all questions; paragraph rather than token matching; local sequential timings."
    (OUT / "reader_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
