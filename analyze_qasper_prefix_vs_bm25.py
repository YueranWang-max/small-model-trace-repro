"""Paper-clustered answer and cost analysis of prefix reuse versus BM25."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"logs/qasper_prefix_vs_bm25"


def mean(values):
    return sum(values)/len(values) if values else None


def ci(values):
    s=sorted(values)
    return [s[int(.025*len(s))],s[int(.975*len(s))]]


def main():
    data=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    assert len(data["papers"])==9
    groups=[p["records"] for p in data["papers"]]
    assert all(len(g)>=4 and all(set(q["arms"])=={"source_prefix","bm25"} for q in g) for g in groups)
    rows=[q for g in groups for q in g]
    summary={"status":"exploratory_dev_repeated_document_prefix_vs_bm25",
             "n_papers":len(groups),"n_questions":len(rows),"arms":{},"per_paper":[]}
    for arm in ("source_prefix","bm25"):
        x=[q["arms"][arm] for q in rows]
        summary["arms"][arm]={"f1":mean([v["f1"] for v in x]),"em":mean([v["em"] for v in x]),
            "prompt_tokens":sum(v.get("prompt_tokens") or 0 for v in x),
            "generation_tokens":sum(v.get("generation_tokens") or 0 for v in x),
            "prompt_eval_sec":sum((v.get("prompt_eval_duration_ns") or 0)/1e9 for v in x),
            "eval_sec":sum((v.get("eval_duration_ns") or 0)/1e9 for v in x),
            "api_total_sec":sum((v.get("total_duration_ns") or 0)/1e9 for v in x),
            "load_sec":sum((v.get("load_duration_ns") or 0)/1e9 for v in x),
            "wall_sec":sum(v["wall_sec"] for v in x),
            "errors":sum("error" in v for v in x)}
    for p in data["papers"]:
        entry={"paper_id":p["paper_id"],"n_questions":len(p["records"]),"source_chars":p["source_chars"],
               "arm_order":p["arm_order"],"bm25_index_sec":p["bm25_index_sec"],
               "bm25_search_sec":sum(q.get("bm25_search_sec",0) for q in p["records"])}
        for arm in ("source_prefix","bm25"):
            x=[q["arms"][arm] for q in p["records"]]
            entry[arm]={"f1":mean([z["f1"] for z in x]),
                        "prompt_eval_sec":sum((z.get("prompt_eval_duration_ns") or 0)/1e9 for z in x),
                        "first_prefill_sec":(x[0].get("prompt_eval_duration_ns") or 0)/1e9,
                        "later_mean_prefill_sec":mean([(z.get("prompt_eval_duration_ns") or 0)/1e9 for z in x[1:]]),
                        "api_total_sec":sum((z.get("total_duration_ns") or 0)/1e9 for z in x)}
        summary["per_paper"].append(entry)
    diffs=[q["arms"]["source_prefix"]["f1"]-q["arms"]["bm25"]["f1"] for q in rows]
    rng=random.Random(20261003)
    boot=[]
    for _ in range(10000):
        sample=[groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(mean([q["arms"]["source_prefix"]["f1"]-q["arms"]["bm25"]["f1"] for g in sample for q in g]))
    summary["paired"]={"f1_difference":mean(diffs),"paper_ci95":ci(boot),
                       "wins":sum(x>1e-9 for x in diffs),"losses":sum(x< -1e-9 for x in diffs),
                       "ties":sum(abs(x)<=1e-9 for x in diffs)}
    summary["source_prefix_prefill"]={"first_mean_sec":mean([p["source_prefix"]["first_prefill_sec"] for p in summary["per_paper"]]),
                                      "later_mean_sec":mean([p["source_prefix"]["later_mean_prefill_sec"] for p in summary["per_paper"]])}
    summary["bm25_index_search_sec"]=sum(p["bm25_index_sec"]+p["bm25_search_sec"] for p in summary["per_paper"])
    summary["caveat"]="Sequential local Ollama timing with model/cache state and block order; answer metrics and token counters are per-question. No energy or universal serving claim."
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
