"""Frozen evidence-coverage gate for the within-paper hard-negative encoder."""
from __future__ import annotations

import json
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def main():
    source = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "results_hardnegative16.json").read_text(encoding="utf-8"))
    training = json.loads((OUT / "hardnegative_training.json").read_text(encoding="utf-8"))
    assert result["trained_weights_sha256"] == training["weights_sha256"]
    groups = []
    total = {"eligible": 0, "fused_any_hit": 0, "dense_any_hit": 0, "bm25_any_hit": 0,
             "selected_material_chars": 0}
    for g, rg in zip(frozen["groups"], result["groups"]):
        assert g["group"] == rg["group"] and len(g["questions"]) == len(rg["records"])
        local = {"group": g["group"], "eligible": 0, "fused_any_hit": 0,
                 "dense_any_hit": 0, "bm25_any_hit": 0,
                 "selected_material_chars": 0,
                 "dense_index_sec": rg["dense_index_sec"],
                 "bm25_index_sec": rg["bm25_index_sec"],
                 "query_sec": round(sum(r["search_sec"] for r in rg["records"]), 3)}
        for q, r in zip(g["questions"], rg["records"]):
            assert q["id"] == r["id"]
            local["selected_material_chars"] += r["selected_material_chars"]
            original = next(x for x in source[q["paper_id"]]["qas"] if x["question_id"] == q["id"])
            evidence = set(evidence_paragraph_ids(original, g["paragraphs"]))
            if evidence:
                local["eligible"] += 1
                local["fused_any_hit"] += bool(evidence & set(r["selected"]))
                local["dense_any_hit"] += bool(evidence & set(r["dense_top16"]))
                local["bm25_any_hit"] += bool(evidence & set(r["bm25_top16"]))
        groups.append(local)
        for key in total:
            total[key] += local[key]
    assert total["eligible"] == 73 and total["bm25_any_hit"] == 54
    summary = {"status": "post_hoc_within_paper_hard_negative",
               "n_questions": 82, "n_train_triples": training["n_train_triples"],
               "train_sec": training["train_sec"],
               "model_load_sec": result["model_load_sec"], "groups": groups,
               "pooled": total,
               "coverage_gate_passed": total["fused_any_hit"] >= 59,
               "total_index_query_sec_excluding_load": round(sum(g["dense_index_sec"] + g["bm25_index_sec"] + g["query_sec"] for g in groups), 3)}
    (OUT / "summary_hardnegative16.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


