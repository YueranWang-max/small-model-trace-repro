"""Summarize paper-transfer retrieval and answer outcomes, including cost."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_holdout"
ARMS = ("bm25_16", "fused_16", "fused_budget")


def bootstrap(diffs):
    rng = random.Random(20261002)
    draws = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
    return [draws[249], draws[9749]]


def main():
    frozen_path, selected_path = OUT / "frozen.json", OUT / "selections.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    reads = json.loads((OUT / "readout.json").read_text(encoding="utf-8"))
    assert reads["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert reads["selection_sha256"] == hashlib.sha256(selected_path.read_bytes()).hexdigest()
    source = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    qrows = []
    for fg, sg, rg in zip(frozen["groups"], selected["groups"], reads["groups"]):
        assert fg["group"] == sg["group"] == rg["group"]
        assert len(fg["questions"]) == len(sg["records"]) == len(rg["records"])
        for q, s, r in zip(fg["questions"], sg["records"], rg["records"]):
            assert q["id"] == s["id"] == r["id"] and all(a in r["arms"] for a in ARMS)
            original = next(x for x in source[q["paper_id"]]["qas"] if x["question_id"] == q["id"])
            evidence = set(evidence_paragraph_ids(original, fg["paragraphs"]))
            qrows.append({"id": q["id"], "paper_id": q["paper_id"], "evidence_mappable": bool(evidence),
                          "f1": {a: r["arms"][a]["f1"] for a in ARMS},
                          "prompt_tokens": {a: r["arms"][a].get("prompt_tokens") or 0 for a in ARMS},
                          "reader_sec": {a: r["arms"][a]["wall_sec"] for a in ARMS},
                          "evidence_hit": {a: bool(evidence & set(s[a])) if evidence else None for a in ARMS},
                          "selected_n": {a: len(s[a]) for a in ARMS}})
    paper_ids = [pid for g in frozen["groups"] for pid in g["paper_ids"]]
    per_paper = []
    for pid in paper_ids:
        rows = [r for r in qrows if r["paper_id"] == pid]
        per_paper.append({"paper_id": pid, "n_questions": len(rows),
                          "f1": {a: statistics.mean(r["f1"][a] for r in rows) for a in ARMS},
                          "prompt_tokens": {a: sum(r["prompt_tokens"][a] for r in rows) for a in ARMS},
                          "mappable": sum(r["evidence_mappable"] for r in rows),
                          "evidence_hit": {a: sum(r["evidence_hit"][a] is True for r in rows) for a in ARMS}})
    total = {"n_papers": len(per_paper), "n_questions": len(qrows),
             "n_mappable": sum(r["evidence_mappable"] for r in qrows),
             "f1": {a: statistics.mean(r["f1"][a] for r in qrows) for a in ARMS},
             "prompt_tokens": {a: sum(r["prompt_tokens"][a] for r in qrows) for a in ARMS},
             "reader_sec": {a: round(sum(r["reader_sec"][a] for r in qrows), 3) for a in ARMS},
             "evidence_hit": {a: sum(r["evidence_hit"][a] is True for r in qrows) for a in ARMS},
             "mean_selected_n": {a: statistics.mean(r["selected_n"][a] for r in qrows) for a in ARMS}}
    total["model_load_sec"] = selected["model_load_sec"]
    total["index_query_sec"] = round(sum(g["bm25_index_sec"] + g["dense_index_sec"] + sum(r["search_sec"] for r in g["records"])
                                          for g in selected["groups"]), 3)
    total["train_sec"] = json.loads((ROOT / "logs/qasper_multiq/hardnegative_training.json").read_text(encoding="utf-8"))["train_sec"]
    total["paired_differences"] = {}
    for arm in ("fused_16", "fused_budget"):
        diff = [r["f1"][arm] - r["f1"]["bm25_16"] for r in qrows]
        pdiff = [p["f1"][arm] - p["f1"]["bm25_16"] for p in per_paper]
        total["paired_differences"][arm] = {"question_mean": statistics.mean(diff),
                                             "question_bootstrap_95_ci": bootstrap(diff),
                                             "paper_mean": statistics.mean(pdiff),
                                             "paper_bootstrap_95_ci": bootstrap(pdiff),
                                             "wins_questions": sum(d > 1e-9 for d in diff),
                                             "losses_questions": sum(d < -1e-9 for d in diff)}
    summary = {"status": "exploratory_paper_level_transfer_not_official_test", "selection_rule": frozen["selection_rule"],
               "per_paper": per_paper, "pooled": total, "bootstrap_seed": 20261002}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
