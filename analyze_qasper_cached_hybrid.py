"""Gold-only coverage gate for the cached bi-encoder BM25 hybrid."""
from __future__ import annotations

import json
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def main():
    source = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    hybrid = json.loads((OUT / "results_cached_hybrid16.json").read_text(encoding="utf-8"))
    eligible = hybrid_hits = bm25_hits = dense_hits = 0
    groups = []
    for g, hg in zip(frozen["groups"], hybrid["groups"]):
        assert g["group"] == hg["group"] and len(g["questions"]) == len(hg["records"])
        gh = gb = gd = ge = 0
        for q, r in zip(g["questions"], hg["records"]):
            assert q["id"] == r["id"]
            original = next(x for x in source[q["paper_id"]]["qas"] if x["question_id"] == q["id"])
            evidence = set(evidence_paragraph_ids(original, g["paragraphs"]))
            if not evidence:
                continue
            ge += 1
            gh += bool(evidence & set(r["selected"]))
            gb += bool(evidence & set(r["bm25_top16"]))
            gd += bool(evidence & set(r["dense_top16"]))
        groups.append({"group": g["group"], "eligible": ge,
                       "hybrid_any_hit": gh, "bm25_any_hit": gb, "dense_any_hit": gd,
                       "dense_index_sec": hg["dense_index_sec"],
                       "bm25_index_sec": hg["bm25_index_sec"],
                       "query_sec": round(sum(r["search_sec"] for r in hg["records"]), 3),
                       "selected_material_chars": sum(r["selected_material_chars"] for r in hg["records"])})
        eligible += ge; hybrid_hits += gh; bm25_hits += gb; dense_hits += gd
    assert eligible == 73 and bm25_hits == 54
    summary = {"status": "post_hoc_clm_inspired_surrogate", "model": hybrid["model"],
               "model_revision": hybrid["model_revision"], "n_questions": 82,
               "eligible": eligible, "hybrid_any_hit": hybrid_hits,
               "bm25_top16_any_hit": bm25_hits, "dense_top16_any_hit": dense_hits,
               "coverage_gate_passed": hybrid_hits >= 59,
               "model_load_sec": hybrid["model_load_sec"], "groups": groups,
               "selected_material_chars": sum(g["selected_material_chars"] for g in groups),
               "total_index_query_sec_excluding_load": round(sum(g["dense_index_sec"] + g["bm25_index_sec"] + g["query_sec"] for g in groups), 3)}
    (OUT / "summary_cached_hybrid16.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
