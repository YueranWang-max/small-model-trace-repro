"""Paired analysis of identical paragraph evidence in two presentation orders."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "qasper_multiq"


def main():
    old = json.loads((OUT / "results_paragraph24.json").read_text(encoding="utf-8"))
    new = json.loads((OUT / "results_paragraph24_source_order.json").read_text(encoding="utf-8"))
    assert old["frozen_sha256"] == new["frozen_sha256"]
    groups = []
    pairs = []
    for g0, g1 in zip(old["groups"], new["groups"]):
        assert g0["group"] == g1["group"] and len(g0["records"]) == len(g1["records"])
        local = []
        for a, b in zip(g0["records"], g1["records"]):
            assert a["id"] == b["id"]
            assert sorted(a["selected"]) == b["same_selected_ids_source_order"]
            local.append((a["arm"], b["arm"]))
        pairs.extend(local)
        groups.append({"group": g0["group"], "n_questions": len(local),
                       "mean_f1_relevance_order": statistics.mean(a["f1"] for a, b in local),
                       "mean_f1_source_order": statistics.mean(b["f1"] for a, b in local),
                       "mean_paired_difference": statistics.mean(b["f1"] - a["f1"] for a, b in local),
                       "prompt_tokens_relevance_order": sum(a.get("prompt_tokens") or 0 for a, b in local),
                       "prompt_tokens_source_order": sum(b.get("prompt_tokens") or 0 for a, b in local)})
    delta = [b["f1"] - a["f1"] for a, b in pairs]
    summary = {"status": "post_hoc_exploratory_order_test", "n_questions": len(pairs),
               "groups": groups,
               "pooled": {"mean_f1_relevance_order": statistics.mean(a["f1"] for a, b in pairs),
                          "mean_f1_source_order": statistics.mean(b["f1"] for a, b in pairs),
                          "mean_paired_difference": statistics.mean(delta),
                          "wins": sum(x > 1e-9 for x in delta),
                          "losses": sum(x < -1e-9 for x in delta),
                          "ties": sum(abs(x) <= 1e-9 for x in delta),
                          "prompt_tokens_relevance_order": sum(a.get("prompt_tokens") or 0 for a, b in pairs),
                          "prompt_tokens_source_order": sum(b.get("prompt_tokens") or 0 for a, b in pairs)}}
    (OUT / "summary_order.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
