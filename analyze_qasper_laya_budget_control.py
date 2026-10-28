"""Post hoc BM25 top-16 budget control versus Laya-selected top-12."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "qasper_multiq"


def main():
    bm = json.loads((OUT / "results_paragraph16_control.json").read_text(encoding="utf-8"))
    la = json.loads((OUT / "results_laya_top12_readout.json").read_text(encoding="utf-8"))
    assert bm["frozen_sha256"] == la["frozen_sha256"]
    groups = []
    pairs = []
    for bg, lg in zip(bm["groups"], la["groups"]):
        assert bg["group"] == lg["group"] and len(bg["records"]) == len(lg["records"])
        local = []
        for br, lr in zip(bg["records"], lg["records"]):
            assert br["id"] == lr["id"]
            local.append((br["arm"], lr["arm"]))
        pairs.extend(local)
        groups.append({"group": bg["group"], "n_questions": len(local),
                       "f1_bm25_top16": statistics.mean(b["f1"] for b, l in local),
                       "f1_laya_top12": statistics.mean(l["f1"] for b, l in local),
                       "prompt_tokens_bm25_top16": sum(b.get("prompt_tokens") or 0 for b, l in local),
                       "prompt_tokens_laya_top12": sum(l.get("prompt_tokens") or 0 for b, l in local),
                       "bm25_reader_sec": round(sum(b["wall_sec"] for b, l in local), 3),
                       "bm25_index_search_sec": round(bg["index_sec"] + sum(r["search_sec"] for r in bg["records"]), 3)})
    d = [l["f1"] - b["f1"] for b, l in pairs]
    summary = {"status": "post_hoc_laya_budget_control", "n_questions": len(pairs),
               "groups": groups,
               "pooled": {"f1_bm25_top16": statistics.mean(b["f1"] for b, l in pairs),
                          "f1_laya_top12": statistics.mean(l["f1"] for b, l in pairs),
                          "paired_difference": statistics.mean(d),
                          "wins_laya": sum(x > 1e-9 for x in d),
                          "losses_laya": sum(x < -1e-9 for x in d),
                          "ties": sum(abs(x) <= 1e-9 for x in d),
                          "prompt_tokens_bm25_top16": sum(b.get("prompt_tokens") or 0 for b, l in pairs),
                          "prompt_tokens_laya_top12": sum(l.get("prompt_tokens") or 0 for b, l in pairs),
                          "bm25_index_search_reader_sec": round(sum(g["bm25_reader_sec"] + g["bm25_index_search_sec"] for g in groups), 3)}}
    (OUT / "summary_laya_budget_control.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
