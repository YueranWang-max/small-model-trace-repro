"""Summarize the frozen 16-instance natural-document readout pilot."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "realdoc_hybrid_pilot"
ARMS = ("state", "state_source2", "bm25_top6")


def main():
    d = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    rows = d["records"]
    assert len(rows) == 16
    control = json.loads((OUT / "results_top2_control.json").read_text(encoding="utf-8"))["records"]
    assert [x["id"] for x in rows] == [x["id"] for x in control]
    control_by_id = {x["id"]: x["bm25_top2"] for x in control}
    paragraphs = json.loads((OUT / "results_paragraph12.json").read_text(encoding="utf-8"))["records"]
    assert [x["id"] for x in rows] == [x["id"] for x in paragraphs]
    paragraph_by_id = {x["id"]: x for x in paragraphs}
    summary = {"n": len(rows), "model": d["model"], "think": d["think"],
               "num_predict": d["num_predict"],
               "mean_top2_support_recall": statistics.mean(x["top2_support_recall"] for x in rows),
               "mean_top6_support_recall": statistics.mean(x["top6_support_recall"] for x in rows)}
    for dataset_name in ("all", "hotpot", "musique"):
        subset = rows if dataset_name == "all" else [x for x in rows if dataset_name in x["dataset"]]
        summary[dataset_name] = {"n": len(subset)}
        for arm in ARMS:
            rs = [x["arms"][arm] for x in subset]
            summary[dataset_name][arm] = {
                "mean_f1": statistics.mean(x["f1"] for x in rs),
                "em": statistics.mean(x["em"] for x in rs),
                "malformed": sum(x["malformed"] for x in rs),
                "mean_prompt_chars": statistics.mean(x["prompt_chars"] for x in rs),
                "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in rs),
                "generation_tokens": sum(x.get("generation_tokens") or 0 for x in rs),
                "reader_wall_sec": sum(x["wall_sec"] for x in rs),
            }
        top2 = [control_by_id[x["id"]] for x in subset]
        summary[dataset_name]["bm25_top2_posthoc"] = {
            "mean_f1": statistics.mean(x["f1"] for x in top2),
            "em": statistics.mean(x["em"] for x in top2),
            "malformed": sum(x["malformed"] for x in top2),
            "mean_prompt_chars": statistics.mean(x["prompt_chars"] for x in top2),
            "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in top2),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in top2),
            "reader_wall_sec": sum(x["wall_sec"] for x in top2),
        }
        para_rows = [paragraph_by_id[x["id"]] for x in subset]
        para = [x["paragraph12"] for x in para_rows]
        summary[dataset_name]["paragraph_bm25_top12_posthoc"] = {
            "mean_f1": statistics.mean(x["f1"] for x in para),
            "em": statistics.mean(x["em"] for x in para),
            "malformed": sum(x["malformed"] for x in para),
            "mean_support_chunk_recall": statistics.mean(x["support_chunk_recall"] for x in para_rows),
            "mean_prompt_chars": statistics.mean(x["prompt_chars"] for x in para),
            "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in para),
            "generation_tokens": sum(x.get("generation_tokens") or 0 for x in para),
            "reader_wall_sec": sum(x["wall_sec"] for x in para),
            "cpu_index_search_sec": sum(x["index_sec"] + x["search_sec"] for x in para_rows),
        }
    diff = [x["arms"]["state_source2"]["f1"] - control_by_id[x["id"]]["f1"] for x in rows]
    rng = random.Random(20261001)
    draws = sorted(statistics.mean(diff[rng.randrange(len(diff))] for _ in diff) for _ in range(10000))
    summary["hybrid_minus_top2_exploratory"] = {
        "mean": statistics.mean(diff), "ci95": [draws[249], draws[9749]],
        "wins": sum(x > 0 for x in diff), "losses": sum(x < 0 for x in diff),
        "ties": sum(x == 0 for x in diff),
    }
    diff_para = [paragraph_by_id[x["id"]]["paragraph12"]["f1"] - x["arms"]["bm25_top6"]["f1"] for x in rows]
    rng = random.Random(20261001)
    draws = sorted(statistics.mean(diff_para[rng.randrange(len(diff_para))] for _ in diff_para) for _ in range(10000))
    summary["paragraph12_minus_chunk6_exploratory"] = {
        "mean": statistics.mean(diff_para), "ci95": [draws[249], draws[9749]],
        "wins": sum(x > 0 for x in diff_para), "losses": sum(x < 0 for x in diff_para),
        "ties": sum(x == 0 for x in diff_para),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
