"""Analyze the frozen Qasper repeated-query lifecycle experiment."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "logs" / "qasper_multiq"
ARMS = ("chunk4", "paragraph12")


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    assert len(data["groups"]) == len(frozen["groups"]) == 3
    summary = {"n_corpora": 3, "n_questions": 0, "groups": []}
    all_rows = []
    for group in data["groups"]:
        rows = group["records"]
        assert len(rows) == group["n_questions"]
        all_rows.extend(rows)
        item = {"group": group["group"], "corpus_chars": group["corpus_chars"],
                "n_questions": len(rows), "arms": {}}
        summary["n_questions"] += len(rows)
        for arm in ARMS:
            rs = [r["arms"][arm] for r in rows]
            index_sec = group[arm.split("12")[0] + "_index_sec"] if arm == "paragraph12" else group["chunk_index_sec"]
            search_sec = sum(r["paragraph_search_sec" if arm == "paragraph12" else "chunk_search_sec"] for r in rows)
            read_sec = sum(r["wall_sec"] for r in rs)
            cumulative = {}
            for n in (1,5,10,20,len(rows)):
                if n <= len(rows):
                    cumulative[str(n)] = round(index_sec + sum(rows[j]["paragraph_search_sec" if arm == "paragraph12" else "chunk_search_sec"] + rows[j]["arms"][arm]["wall_sec"] for j in range(n)),3)
            item["arms"][arm] = {"mean_f1": statistics.mean(r["f1"] for r in rs),
                                 "em": statistics.mean(r["em"] for r in rs),
                                 "malformed": sum(r["malformed"] for r in rs),
                                 "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rs),
                                 "generation_tokens": sum(r.get("generation_tokens") or 0 for r in rs),
                                 "index_sec": index_sec, "search_sec": round(search_sec,3),
                                 "readout_sec": round(read_sec,3),
                                 "total_lifecycle_sec": round(index_sec+search_sec+read_sec,3),
                                 "cumulative_lifecycle_sec": cumulative}
        summary["groups"].append(item)
    summary["pooled_question_weighted"] = {}
    for arm in ARMS:
        rs=[r["arms"][arm] for r in all_rows]
        summary["pooled_question_weighted"][arm] = {
            "mean_f1": statistics.mean(r["f1"] for r in rs),
            "em": statistics.mean(r["em"] for r in rs),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rs),
            "generation_tokens": sum(r.get("generation_tokens") or 0 for r in rs),
            "total_lifecycle_sec": sum(x["arms"][arm]["total_lifecycle_sec"] for x in summary["groups"])}
    sensitivity_path = OUT / "results_paragraph24.json"
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if len(sensitivity["groups"]) == len(frozen["groups"]) and all(
            len(g["records"]) == g["n_questions"] for g in sensitivity["groups"]
        ):
            assert sensitivity["frozen_sha256"] == data["frozen_sha256"]
            by_group = {g["group"]: g for g in sensitivity["groups"]}
            sensitivity_groups = []
            all_sensitivity = []
            for base_group in data["groups"]:
                sg = by_group[base_group["group"]]
                assert sg["corpus_sha256"] == base_group["corpus_sha256"]
                assert [r["id"] for r in sg["records"]] == [r["id"] for r in base_group["records"]]
                sr = [r["arm"] for r in sg["records"]]
                all_sensitivity.extend(sr)
                search_sec = sum(r["search_sec"] for r in sg["records"])
                read_sec = sum(r["wall_sec"] for r in sr)
                sensitivity_groups.append({
                    "group": sg["group"], "mean_f1": statistics.mean(r["f1"] for r in sr),
                    "em": statistics.mean(r["em"] for r in sr),
                    "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in sr),
                    "generation_tokens": sum(r.get("generation_tokens") or 0 for r in sr),
                    "index_sec": sg["paragraph_index_sec"],
                    "search_sec": round(search_sec, 3), "readout_sec": round(read_sec, 3),
                    "total_lifecycle_sec": round(sg["paragraph_index_sec"] + search_sec + read_sec, 3),
                })
            summary["post_hoc_paragraph24"] = {
                "groups": sensitivity_groups,
                "pooled_question_weighted": {
                    "mean_f1": statistics.mean(r["f1"] for r in all_sensitivity),
                    "em": statistics.mean(r["em"] for r in all_sensitivity),
                    "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in all_sensitivity),
                    "generation_tokens": sum(r.get("generation_tokens") or 0 for r in all_sensitivity),
                    "total_lifecycle_sec": sum(g["total_lifecycle_sec"] for g in sensitivity_groups),
                },
            }
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
