"""Paired reader readout on frozen learned-fact evidence selections."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from run_qasper_outline_route import build_paper
from run_realdoc_hybrid_pilot import read

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_fact_ranker"
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.input
    frozen = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    source = json.loads(DEV.read_text(encoding="utf-8"))
    path = out_dir / "readout.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_paired_same_question_reader", "source_results": "results.json",
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024, "papers": []}
    global_q = 0
    for fp in frozen["papers"]:
        pid = fp["paper_id"]
        paper = build_paper(pid, source[pid])
        rp = next((x for x in result["papers"] if x["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "records": []}
            result["papers"].append(rp)
        by_id = {q["id"]: q for q in paper["questions"]}
        for fq in fp["questions"]:
            q = by_id[fq["id"]]
            rr = next((x for x in rp["records"] if x["id"] == q["id"]), None)
            if rr is None:
                rr = {"id": q["id"], "arms": {},
                      "order": ["baseline", "learned"] if global_q % 2 == 0 else ["learned", "baseline"]}
                rp["records"].append(rr)
            question = "Paper: " + paper["title"] + ". Question: " + q["question"]
            inst = {"question": question, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            for arm in rr["order"]:
                if arm in rr["arms"]:
                    continue
                material = "Retrieved evidence:\n" + "\n\n".join(paper["paragraphs"][i] for i in fq[arm])
                rr["arms"][arm] = read(material, inst)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, q["id"], arm, rr["arms"][arm]["f1"], flush=True)
            global_q += 1


if __name__ == "__main__":
    main()
