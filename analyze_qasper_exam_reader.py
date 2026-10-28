"""Paired outcome and lifecycle summary for the exam-style answer workflow."""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_exam_reader"
ARMS = ("baseline", "exam")


def boot(values):
    rng = random.Random(20261002)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def main():
    path = OUT / "frozen.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert result["frozen_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for g, rg in zip(frozen["groups"], result["groups"]):
        assert g["group"] == rg["group"] and len(g["questions"]) == len(rg["records"])
        for q, r in zip(g["questions"], rg["records"]):
            assert q["id"] == r["id"] and all(a in r["arms"] for a in ARMS)
            rows.append({"paper_id": q["paper_id"], "id": q["id"], "n_selected": len(r["selected"]), "arms": r["arms"]})
    per_paper = []
    for pid in [pid for g in frozen["groups"] for pid in g["paper_ids"]]:
        p = [r for r in rows if r["paper_id"] == pid]
        per_paper.append({"paper_id": pid, "n_questions": len(p),
                          "f1": {a: statistics.mean(r["arms"][a]["f1"] for r in p) for a in ARMS},
                          "em": {a: statistics.mean(r["arms"][a]["em"] for r in p) for a in ARMS}})
    diff = [r["arms"]["exam"]["f1"] - r["arms"]["baseline"]["f1"] for r in rows]
    pdiff = [p["f1"]["exam"] - p["f1"]["baseline"] for p in per_paper]
    evidence_present = [r for r in rows if r["arms"]["exam"]["evidence_line"]]
    evidence_valid = sum(all(1 <= int(i) <= r["n_selected"] for i in re.findall(r"\bP(\d+)\b", r["arms"]["exam"]["evidence_line"], re.I))
                         for r in evidence_present)
    pooled = {"n_papers": len(per_paper), "n_questions": len(rows),
              "f1": {a: statistics.mean(r["arms"][a]["f1"] for r in rows) for a in ARMS},
              "em": {a: statistics.mean(r["arms"][a]["em"] for r in rows) for a in ARMS},
              "prompt_tokens": {a: sum(r["arms"][a].get("prompt_tokens") or 0 for r in rows) for a in ARMS},
              "generation_tokens": {a: sum(r["arms"][a].get("generation_tokens") or 0 for r in rows) for a in ARMS},
              "reader_sec": {a: round(sum(r["arms"][a]["wall_sec"] for r in rows), 3) for a in ARMS},
              "malformed": {a: sum(r["arms"][a]["malformed"] for r in rows) for a in ARMS},
              "unknown_answers": {a: sum(r["arms"][a]["answer"].strip().lower() == "unknown" for r in rows) for a in ARMS},
              "exam_evidence_line_present": len(evidence_present),
              "exam_evidence_id_syntax_valid_when_present": evidence_valid,
              "paired_f1_difference": statistics.mean(diff), "question_bootstrap_95_ci": boot(diff),
              "paper_mean_difference": statistics.mean(pdiff), "paper_bootstrap_95_ci": boot(pdiff),
              "exam_wins": sum(d > 1e-9 for d in diff), "exam_losses": sum(d < -1e-9 for d in diff),
              "ties": sum(abs(d) <= 1e-9 for d in diff)}
    pooled["bm25_index_sec"] = round(sum(g["bm25_build_sec"] + sum(r["search_sec"] for r in g["records"])
                                           for g in result["groups"]), 3)
    summary = {"status": "exploratory_exam_workflow_not_state_experiment", "selection_rule": frozen["selection_rule"],
               "per_paper": per_paper, "pooled": pooled, "bootstrap_seed": 20261002}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
