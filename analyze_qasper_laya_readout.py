"""Complete exploratory Laya reranking lifecycle and paired answer comparison."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "qasper_multiq"


def main():
    reads = json.loads((OUT / "results_laya_top12_readout.json").read_text(encoding="utf-8"))
    scores = json.loads((OUT / "results_laya_top12.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    paragraph24 = json.loads((OUT / "results_paragraph24.json").read_text(encoding="utf-8"))
    assert reads["selection_sha256"] == __import__("hashlib").sha256((OUT / "results_laya_top12.json").read_bytes()).hexdigest()
    group_results = []
    rows = []
    index_search_sec = 0.0
    for rg, sg, bg, pg in zip(reads["groups"], scores["groups"], base["groups"], paragraph24["groups"]):
        assert rg["group"] == sg["group"] == bg["group"] == pg["group"]
        assert len(rg["records"]) == len(sg["records"]) == len(bg["records"]) == len(pg["records"])
        group_pairs = []
        for rr, sr, br, pr in zip(rg["records"], sg["records"], bg["records"], pg["records"]):
            assert rr["id"] == sr["id"] == br["id"] == pr["id"]
            assert rr["selected"] == sr["selected"]
            group_pairs.append((rr["arm"], br["arms"]["chunk4"], br["arms"]["paragraph12"], pr["arm"]))
        rows.extend(group_pairs)
        gp = {
            "group": rg["group"], "n_questions": len(group_pairs),
            "f1_laya12": statistics.mean(x[0]["f1"] for x in group_pairs),
            "f1_chunk4": statistics.mean(x[1]["f1"] for x in group_pairs),
            "f1_paragraph12": statistics.mean(x[2]["f1"] for x in group_pairs),
            "f1_paragraph24": statistics.mean(x[3]["f1"] for x in group_pairs),
            "laya_reader_prompt_tokens": sum(x[0].get("prompt_tokens") or 0 for x in group_pairs),
            "laya_reader_sec": round(sum(x[0]["wall_sec"] for x in group_pairs), 3),
            "laya_decision_sec": round(sum(r["laya_wall_sec"] for r in sg["records"]), 3),
            "malformed": sum(x[0]["malformed"] for x in group_pairs),
        }
        index_search_sec += pg["paragraph_index_sec"] + sum(r["search_sec"] for r in pg["records"])
        group_results.append(gp)
    pooled = {
        "f1_laya12": statistics.mean(x[0]["f1"] for x in rows),
        "f1_chunk4": statistics.mean(x[1]["f1"] for x in rows),
        "f1_paragraph12": statistics.mean(x[2]["f1"] for x in rows),
        "f1_paragraph24": statistics.mean(x[3]["f1"] for x in rows),
        "laya_reader_prompt_tokens": sum(x[0].get("prompt_tokens") or 0 for x in rows),
        "laya_model_input_tokens": sum(r["laya_input_tokens"] for g in scores["groups"] for r in g["records"]),
        "bm25_index_search_sec": round(index_search_sec, 3),
        "laya_model_load_sec": scores["model_load_sec"],
        "laya_decision_sec": round(sum(g["laya_decision_sec"] for g in group_results), 3),
        "reader_sec": round(sum(g["laya_reader_sec"] for g in group_results), 3),
        "wins_vs_chunk4": sum(x[0]["f1"] > x[1]["f1"] + 1e-9 for x in rows),
        "losses_vs_chunk4": sum(x[0]["f1"] < x[1]["f1"] - 1e-9 for x in rows),
        "ties_vs_chunk4": sum(abs(x[0]["f1"] - x[1]["f1"]) <= 1e-9 for x in rows),
    }
    pooled["total_lifecycle_sec"] = round(sum(pooled[k] for k in ("bm25_index_search_sec", "laya_model_load_sec", "laya_decision_sec", "reader_sec")), 3)
    summary = {"status": "post_hoc_exploratory_laya_gate_passing_readout", "n_corpora": 3,
               "n_questions": len(rows), "groups": group_results, "pooled": pooled}
    (OUT / "summary_laya_top12_readout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
