"""Paper-clustered analysis of bounded visible-trace placement."""
from __future__ import annotations

import json
import random

from run_qasper_visible_trace_replication import ARMS, OUT


def mean(x):
    return sum(x) / len(x)


def sec(x, key):
    return (x.get(key) or 0) / 1e9


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == 6 and all(len(g) >= 3 and all("first" in q and set(q["arms"]) == set(ARMS) for q in g) for g in groups)
    rows = [q for g in groups for q in g]
    summary = {"status": "dev_visible_trace_placement_frozen_replication", "n_papers": 6,
               "n_questions": len(rows), "arms": {}, "paired": {}, "per_paper": []}
    for arm in ("first",) + ARMS:
        x = [q["first"] if arm == "first" else q["arms"][arm] for q in rows]
        summary["arms"][arm] = {"f1": mean([v["f1"] for v in x]),
            "em": mean([v["em"] for v in x]),
            "prompt_tokens": sum(v.get("prompt_tokens") or 0 for v in x),
            "generation_tokens": sum(v.get("generation_tokens") or 0 for v in x),
            "api_total_sec": sum(sec(v, "api_total_ns") for v in x),
            "wall_sec": sum(v["wall_sec"] for v in x),
            "errors": sum("error" in v for v in x)}
    for a, b in (("trace_pre", "trace_post"), ("trace_pre", "reread"),
                 ("quote_pre", "reread"), ("trace_pre", "first")):
        def val(q, arm):
            return q["first"]["f1"] if arm == "first" else q["arms"][arm]["f1"]
        diffs = [val(q, a) - val(q, b) for q in rows]
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sample = [groups[rng.randrange(len(groups))] for _ in groups]
            boot.append(mean([val(q, a) - val(q, b) for g in sample for q in g]))
        boot.sort()
        summary["paired"][a + "_minus_" + b] = {"f1_difference": mean(diffs),
            "paper_ci95": [boot[250], boot[9750]],
            "wins": sum(v > 1e-9 for v in diffs),
            "losses": sum(v < -1e-9 for v in diffs),
            "ties": sum(abs(v) <= 1e-9 for v in diffs)}
    summary["trace_nonempty"] = sum(bool(q["feedback"]["trace"]) for q in rows)
    summary["trace_truncated"] = sum(q["feedback"]["trace_truncated"] for q in rows)
    summary["verified_quote_count"] = sum(bool(q["feedback"]["verified_quote"]) for q in rows)
    summary["two_pass_api_total_sec"] = {a: summary["arms"]["first"]["api_total_sec"] +
        summary["arms"][a]["api_total_sec"] for a in ARMS}
    for p in data["papers"]:
        summary["per_paper"].append({"paper_id": p["paper_id"], "arm_order": p["arm_order"],
            "f1": {a: mean([q["first"]["f1"] if a == "first" else q["arms"][a]["f1"]
                           for q in p["records"]]) for a in ("first",) + ARMS}})
    pre = summary["paired"]["trace_pre_minus_trace_post"]
    reread = summary["paired"]["trace_pre_minus_reread"]
    summary["placement_gate_pass"] = (pre["f1_difference"] >= 0.03 and pre["paper_ci95"][0] > 0 and
                                       reread["f1_difference"] >= 0.03 and reread["paper_ci95"][0] > 0)
    summary["caveat"] = "Frozen bounded visible-trace parser and method from prior exploratory cohort; not provider reasoning field; only exact quotes verified; six dev papers; sequential timing."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
