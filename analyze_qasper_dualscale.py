"""Summarize the fixed dual-scale exploratory arm against archived controls."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "qasper_multiq"


def main():
    data = json.loads((OUT / "results_dualscale16.json").read_text(encoding="utf-8"))
    baseline = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert len(data["groups"]) == 3
    groups = []
    all_arms = []
    for g in data["groups"]:
        assert len(g["records"]) == g["n_questions"]
        arms = [r["arm"] for r in g["records"]]
        all_arms.extend(arms)
        total_sec = g["both_index_sec"] + sum(r["search_sec"] + r["arm"]["wall_sec"] for r in g["records"])
        groups.append({"group": g["group"], "n_questions": len(arms),
                       "mean_f1": statistics.mean(a["f1"] for a in arms),
                       "em": statistics.mean(a["em"] for a in arms),
                       "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in arms),
                       "total_lifecycle_sec": round(total_sec, 3),
                       "malformed": sum(a["malformed"] for a in arms)})
    pooled = {"mean_f1": statistics.mean(a["f1"] for a in all_arms),
              "em": statistics.mean(a["em"] for a in all_arms),
              "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in all_arms),
              "total_lifecycle_sec": sum(g["total_lifecycle_sec"] for g in groups)}
    ctrl = baseline["pooled_question_weighted"]["chunk4"]
    summary = {"status": "post_hoc_exploratory_method_search", "n_corpora": 3,
               "n_questions": len(all_arms), "groups": groups, "pooled": pooled,
               "versus_chunk4": {"f1_difference": pooled["mean_f1"] - ctrl["mean_f1"],
                                 "prompt_token_ratio": pooled["prompt_tokens"] / ctrl["prompt_tokens"]}}
    (OUT / "summary_dualscale16.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
