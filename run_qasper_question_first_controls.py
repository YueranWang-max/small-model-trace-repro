"""Frozen question-first versus instruction-first full-source controls."""
from __future__ import annotations

import json

from prepare_qasper_multiq import gold_answers
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, sha
from run_qasper_visible_trace_replication import call, query_text

OUT = ROOT / "logs/qasper_question_first_controls"
ARMS = ("reread", "question_first", "instruction_first", "combined")
INSTRUCTION = "Find original evidence, check scope and answer briefly.\n"


def freeze(source):
    ranked = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))
    ids = ranked[162:170]
    papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        text = "\n\n".join(paper["paragraphs"])
        assert len(source[pid]["qas"]) == 3 and 10000 <= len(text) <= 40000
        papers.append({"paper_id": pid, "title": paper["title"], "source_sha256": sha(text.encode("utf-8")),
                       "source_chars": len(text),
                       "question_ids": [q["question_id"] for q in source[pid]["qas"]]})
    return {"dataset_sha256": sha(DEV.read_bytes()),
            "selection": "Qasper dev ranks 163-170, all three questions per paper",
            "model": MODEL, "arms": list(ARMS), "papers": papers}


def make_prompt(arm, source, query):
    if arm == "question_first":
        prefix = "Task focus: " + query + "\n"
    elif arm == "instruction_first":
        prefix = INSTRUCTION
    elif arm == "combined":
        prefix = "Task focus: " + query + "\n" + INSTRUCTION
    else:
        prefix = ""
    return (prefix + "Original paper:\n" + source + "\n\nQuestion: " + query +
            "\nEnd with exactly one line: Final Answer: <short answer>")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen == freeze(source)
    else:
        frozen = freeze(source)
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_question_first_controls",
        "frozen_sha256": sha(frozen_path.read_bytes()), "papers": []}
    assert results["frozen_sha256"] == sha(frozen_path.read_bytes())
    for pi, fp in enumerate(frozen["papers"]):
        pid = fp["paper_id"]
        paper = build_paper(pid, source[pid])
        text = "\n\n".join(paper["paragraphs"])
        assert sha(text.encode("utf-8")) == fp["source_sha256"]
        rp = next((p for p in results["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "arm_order": list(ARMS[pi % len(ARMS):] + ARMS[:pi % len(ARMS)]),
                  "records": []}
            results["papers"].append(rp)
        for arm in rp["arm_order"]:
            for original in source[pid]["qas"]:
                rr = next((r for r in rp["records"] if r["id"] == original["question_id"]), None)
                if rr is None:
                    rr = {"id": original["question_id"], "arms": {}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                query = query_text(paper["title"], original["question"])
                gold = gold_answers(original)
                inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = call(make_prompt(arm, text, query), inst, think=False, cap=1024)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, rr["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
