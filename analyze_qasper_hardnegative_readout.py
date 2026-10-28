"""Compare hard-negative retriever with frozen Qasper reader controls."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs/qasper_multiq"


def main():
    selection_path = OUT / "results_hardnegative16.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reads = json.loads((OUT / "results_hardnegative16_readout.json").read_text(encoding="utf-8"))
    assert reads["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    bm25 = json.loads((OUT / "results_paragraph16_control.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    coverage = json.loads((OUT / "summary_hardnegative16.json").read_text(encoding="utf-8"))
    assert coverage["coverage_gate_passed"]
    rows = []
    for hg, sg, bg, cg in zip(reads["groups"], selection["groups"], bm25["groups"], base["groups"]):
        assert hg["group"] == sg["group"] == bg["group"] == cg["group"]
        assert len(hg["records"]) == len(sg["records"]) == len(bg["records"]) == len(cg["records"])
        for h, s, b, c in zip(hg["records"], sg["records"], bg["records"], cg["records"]):
            assert h["id"] == s["id"] == b["id"] == c["id"]
            assert h["selected"] == s["selected"]
            rows.append({"hardnegative": h["arm"], "bm25": b["arm"], "chunk4": c["arms"]["chunk4"]})
    summary = {"status": "post_hoc_exploratory_hardnegative_readout", "n_questions": len(rows),
               "f1_hardnegative16": statistics.mean(x["hardnegative"]["f1"] for x in rows),
               "f1_bm25_16": statistics.mean(x["bm25"]["f1"] for x in rows),
               "f1_chunk4": statistics.mean(x["chunk4"]["f1"] for x in rows),
               "prompt_tokens_hardnegative16": sum(x["hardnegative"].get("prompt_tokens") or 0 for x in rows),
               "prompt_tokens_bm25_16": sum(x["bm25"].get("prompt_tokens") or 0 for x in rows),
               "reader_sec": round(sum(x["hardnegative"]["wall_sec"] for x in rows), 3),
               "train_sec": coverage["train_sec"], "model_load_sec": coverage["model_load_sec"],
               "index_query_sec": coverage["total_index_query_sec_excluding_load"],
               "wins_vs_bm25": sum(x["hardnegative"]["f1"] > x["bm25"]["f1"] + 1e-9 for x in rows),
               "losses_vs_bm25": sum(x["hardnegative"]["f1"] < x["bm25"]["f1"] - 1e-9 for x in rows),
               "ties_vs_bm25": sum(abs(x["hardnegative"]["f1"] - x["bm25"]["f1"]) <= 1e-9 for x in rows),
               "wins_vs_chunk4": sum(x["hardnegative"]["f1"] > x["chunk4"]["f1"] + 1e-9 for x in rows),
               "losses_vs_chunk4": sum(x["hardnegative"]["f1"] < x["chunk4"]["f1"] - 1e-9 for x in rows),
               "ties_vs_chunk4": sum(abs(x["hardnegative"]["f1"] - x["chunk4"]["f1"]) <= 1e-9 for x in rows),
               "evidence_any_hit_hardnegative16": coverage["pooled"]["fused_any_hit"],
               "evidence_any_hit_bm25_16": coverage["pooled"]["bm25_any_hit"]}
    summary["total_lifecycle_sec"] = round(sum(summary[k] for k in ("reader_sec", "train_sec", "model_load_sec", "index_query_sec")), 3)
    differences = [x["hardnegative"]["f1"] - x["bm25"]["f1"] for x in rows]
    summary["paired_f1_difference"] = statistics.mean(differences)
    rng = random.Random(20261002)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    summary["paired_bootstrap_95_ci"] = [draws[249], draws[9749]]
    summary["bootstrap_seed"] = 20261002
    (OUT / "summary_hardnegative16_readout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
