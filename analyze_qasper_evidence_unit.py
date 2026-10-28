"""Paper-clustered readout for the frozen ninth-paragraph evidence-unit pilot."""
from __future__ import annotations

import json
import random

from run_qasper_evidence_unit import ARMS, OUT
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV


def mean(values):
    return sum(values) / len(values)


def sec(row):
    return (row.get("api_total_ns") or 0) / 1e9


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    groups = [p["records"] for p in data["papers"]]
    assert len(groups) == 6 and all(len(g) == 3 for g in groups)
    rows = [r for g in groups for r in g]
    assert all(set(r["arms"]) == set(ARMS) for r in rows)
    summary = {"status": data["status"], "papers": len(groups), "questions": len(rows),
               "valid_quotes": sum(r["extraction"]["valid"] for r in rows),
               "arms": {}, "paired": {}}
    summary["quote_validity"] = summary["valid_quotes"] / len(rows)
    # Post hoc diagnostic only: Qasper highlighted evidence is never used in selection/prompts.
    source = json.loads(DEV.read_text(encoding="utf-8"))
    evidence_rank = {"exact_match_questions": 0, "top8": 0, "rank9": 0,
                     "outside_top9": 0, "unmatched": 0}
    for p in data["papers"]:
        paragraphs = build_paper(p["paper_id"], source[p["paper_id"]])["paragraphs"]
        for r in p["records"]:
            original = next(q for q in source[p["paper_id"]]["qas"] if q["question_id"] == r["id"])
            spans = [v.strip() for a in original["answers"]
                     for v in a["answer"].get("highlighted_evidence", []) if len(v.strip()) >= 20]
            matched = {i for i, para in enumerate(paragraphs) if any(v in para for v in spans)}
            if not matched:
                evidence_rank["unmatched"] += 1
                continue
            evidence_rank["exact_match_questions"] += 1
            if any(i in matched for i in r["top9"][:8]):
                evidence_rank["top8"] += 1
            elif r["top9"][8] in matched:
                evidence_rank["rank9"] += 1
            else:
                evidence_rank["outside_top9"] += 1
    summary["posthoc_highlighted_evidence_rank"] = evidence_rank
    extraction_sec = sum(sec(r["extraction"]) for r in rows)
    summary["extraction_api_sec"] = extraction_sec
    summary["extraction_prompt_tokens"] = sum(r["extraction"].get("prompt_tokens") or 0 for r in rows)
    summary["extraction_generation_tokens"] = sum(r["extraction"].get("generation_tokens") or 0 for r in rows)
    summary["index_sec"] = sum(p["index_sec"] for p in data["papers"])
    for arm in ARMS:
        a = [r["arms"][arm] for r in rows]
        summary["arms"][arm] = {"f1": mean([x["f1"] for x in a]),
            "em": mean([x["em"] for x in a]),
            "api_sec": sum(sec(x) for x in a),
            "lifecycle_api_sec": sum(sec(x) for x in a) + (extraction_sec if arm in ("extract", "sentence") else 0),
            "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in a),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in a),
            "errors": sum("error" in x for x in a)}
    for a, b in (("extract", "sentence"), ("extract", "base8"),
                 ("extract", "full9"), ("sentence", "base8"), ("full9", "base8")):
        diffs = [r["arms"][a]["f1"] - r["arms"][b]["f1"] for r in rows]
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sampled = [groups[rng.randrange(len(groups))] for _ in groups]
            boot.append(mean([r["arms"][a]["f1"] - r["arms"][b]["f1"]
                              for group in sampled for r in group]))
        boot.sort()
        summary["paired"][a + "_minus_" + b] = {"difference": mean(diffs),
            "paper_ci95": [boot[250], boot[9750]],
            "wins": sum(x > 1e-9 for x in diffs),
            "losses": sum(x < -1e-9 for x in diffs),
            "ties": sum(abs(x) <= 1e-9 for x in diffs)}
    primary = summary["paired"]["extract_minus_sentence"]
    summary["typed_record_progression_gate"] = (primary["paper_ci95"][0] > 0
        and summary["quote_validity"] >= 0.8
        and summary["arms"]["extract"]["lifecycle_api_sec"] <= summary["arms"]["full9"]["api_sec"])
    summary["caveat"] = ("Exact-substring quote validation is syntactic only. Sentence arm depends on extract "
                         "length and is charged the same extraction cost. API time is sequential local inference.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
