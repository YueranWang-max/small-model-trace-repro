"""Paper-clustered paired analysis of a common evidence prefix."""
from __future__ import annotations

import json
import random

from run_qasper_intersection_prefix import OUT


def mean(x):
    return sum(x) / len(x)


def seconds(row, key):
    return (row.get(key) or 0) / 1e9


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == 9 and all(len(g) == 4 and all(set(q["arms"]) == {"common_prefix", "bm25"} for q in g) for g in groups)
    rows = [q for g in groups for q in g]
    summary = {"status": "exploratory_dev_common_evidence_prefix_vs_bm25_order",
               "n_papers": len(groups), "n_questions": len(rows), "arms": {}, "per_paper": []}
    for arm in ("common_prefix", "bm25"):
        x = [q["arms"][arm] for q in rows]
        summary["arms"][arm] = {
            "f1": mean([r["f1"] for r in x]), "em": mean([r["em"] for r in x]),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in x),
            "generation_tokens": sum(r.get("generation_tokens") or 0 for r in x),
            "prompt_eval_sec": sum(seconds(r, "prompt_eval_duration_ns") for r in x),
            "eval_sec": sum(seconds(r, "eval_duration_ns") for r in x),
            "api_total_sec": sum(seconds(r, "total_duration_ns") for r in x),
            "load_sec": sum(seconds(r, "load_duration_ns") for r in x),
            "wall_sec": sum(r["wall_sec"] for r in x),
            "errors": sum("error" in r for r in x)}
    for p, fp in zip(data["papers"], plan["papers"]):
        assert p["paper_id"] == fp["paper_id"]
        entry = {"paper_id": p["paper_id"], "arm_order": p["arm_order"],
                 "common_paragraphs": len(fp["common_order"]),
                 "index_sec": fp["bm25_index_sec"],
                 "search_sec": sum(q["search_sec"] for q in fp["questions"])}
        for arm in ("common_prefix", "bm25"):
            x = [q["arms"][arm] for q in p["records"]]
            entry[arm] = {"f1": mean([r["f1"] for r in x]),
                          "prompt_eval_sec": sum(seconds(r, "prompt_eval_duration_ns") for r in x),
                          "first_prefill_sec": seconds(x[0], "prompt_eval_duration_ns"),
                          "later_mean_prefill_sec": mean([seconds(r, "prompt_eval_duration_ns") for r in x[1:]]),
                          "api_total_sec": sum(seconds(r, "total_duration_ns") for r in x)}
        summary["per_paper"].append(entry)
    diffs = [q["arms"]["common_prefix"]["f1"] - q["arms"]["bm25"]["f1"] for q in rows]
    rng = random.Random(20261003)
    boot = []
    for _ in range(10000):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(mean([q["arms"]["common_prefix"]["f1"] - q["arms"]["bm25"]["f1"] for g in sample for q in g]))
    boot.sort()
    summary["paired"] = {"f1_difference": mean(diffs), "paper_ci95": [boot[250], boot[9750]],
                         "wins": sum(d > 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs),
                         "ties": sum(abs(d) <= 1e-9 for d in diffs)}
    summary["first_later_prefill"] = {arm: {
        "first_mean_sec": mean([p[arm]["first_prefill_sec"] for p in summary["per_paper"]]),
        "later_mean_sec": mean([p[arm]["later_mean_prefill_sec"] for p in summary["per_paper"]])}
        for arm in ("common_prefix", "bm25")}
    summary["bm25_index_search_sec"] = sum(p["index_sec"] + p["search_sec"] for p in summary["per_paper"])
    nonempty = [p for p in summary["per_paper"] if p["common_paragraphs"] > 0]
    summary["nonempty_common_subset"] = {
        "n_papers": len(nonempty), "n_questions": 4 * len(nonempty),
        "common_prefix_f1": mean([p["common_prefix"]["f1"] for p in nonempty]),
        "bm25_f1": mean([p["bm25"]["f1"] for p in nonempty]),
        "common_prefix_prompt_eval_sec": sum(p["common_prefix"]["prompt_eval_sec"] for p in nonempty),
        "bm25_prompt_eval_sec": sum(p["bm25"]["prompt_eval_sec"] for p in nonempty),
        "common_prefix_api_total_sec": sum(p["common_prefix"]["api_total_sec"] for p in nonempty),
        "bm25_api_total_sec": sum(p["bm25"]["api_total_sec"] for p in nonempty)}
    summary["evidence_paragraph_sets_identical"] = all(
        set(q["selected"]) == set(q["prefix_order"]) for p in plan["papers"] for q in p["questions"])
    summary["caveat"] = "Sequential local Ollama timing; cache state, arm order, and output length affect timing. This is an exploratory development cohort."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
