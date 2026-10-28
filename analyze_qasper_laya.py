"""Apply the predeclared evidence-coverage gate to local Laya reranking."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def main():
    source = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    laya = json.loads((OUT / "results_laya_top12.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((OUT / "evidence_gap_diagnostic.json").read_text(encoding="utf-8"))
    assert len(laya["groups"]) == len(frozen["groups"]) == 3
    groups = []
    eligible = hits = complete = 0
    prompt_chars = input_tokens = decision_sec = 0
    all_scores = []
    for g, lg in zip(frozen["groups"], laya["groups"]):
        assert g["group"] == lg["group"] and len(lg["records"]) == g["n_questions"]
        ge = gh = ga = 0
        for q, r in zip(g["questions"], lg["records"]):
            assert q["id"] == r["id"]
            assert len(r["candidate_ids"]) == 24 and len(r["selected"]) == 12
            original = next(x for x in source[q["paper_id"]]["qas"] if x["question_id"] == q["id"])
            evidence = set(evidence_paragraph_ids(original, g["paragraphs"]))
            if evidence:
                ge += 1
                overlap = evidence & set(r["selected"])
                gh += bool(overlap)
                ga += overlap == evidence
            prompt_chars += len("Retrieved original source paragraphs:\n" + "\n\n".join(g["paragraphs"][i] for i in r["selected"]))
            input_tokens += r["laya_input_tokens"]
            decision_sec += r["laya_wall_sec"]
            all_scores.extend(r["yes_scores"])
        groups.append({"group": g["group"], "eligible": ge, "any_hit": gh, "all_hit": ga})
        eligible += ge; hits += gh; complete += ga
    control = diagnostic["pooled"]["paragraph12"]
    assert eligible == control["eligible"] == 73
    gate = hits >= control["any_hit"] + 5
    summary = {"status": "post_hoc_exploratory_method_search",
               "laya_version": laya["laya_version"], "model": laya["model"],
               "groups": groups, "eligible": eligible, "any_hit": hits, "all_hit": complete,
               "bm25_paragraph12_any_hit": control["any_hit"],
               "any_hit_gain": hits - control["any_hit"],
               "gate_threshold_gain": 5, "coverage_gate_passed": gate,
               "selected_material_chars": prompt_chars,
               "laya_input_tokens": input_tokens,
               "laya_load_sec": laya["model_load_sec"],
               "laya_decision_sec": round(decision_sec, 3),
               "yes_score_min": min(all_scores), "yes_score_median": statistics.median(all_scores),
               "yes_score_max": max(all_scores)}
    (OUT / "summary_laya_top12.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
