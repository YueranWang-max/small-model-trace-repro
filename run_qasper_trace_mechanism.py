"""Frozen controls for source-aware information in a bounded visible trace."""
from __future__ import annotations

import json

from prepare_qasper_multiq import gold_answers
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, sha
from run_qasper_visible_trace_replication import call, extract_quote, extract_trace, query_text
from run_realdoc import score

OUT = ROOT / "logs/qasper_trace_mechanism"
ARMS = ("reread", "question_pre", "answer_pre", "trace_pre", "trace_post", "quote_pre")


def freeze(source):
    ranked = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))
    papers = []
    for pid in ranked[146:]:
        if len(source[pid]["qas"]) != 3:
            continue
        paper = build_paper(pid, source[pid])
        text = "\n\n".join(paper["paragraphs"])
        if not 10000 <= len(text) <= 40000:
            continue
        papers.append({"paper_id": pid, "title": paper["title"],
                       "source_sha256": sha(text.encode("utf-8")),
                       "source_chars": len(text),
                       "question_ids": [q["question_id"] for q in source[pid]["qas"]]})
        if len(papers) == 8:
            break
    assert len(papers) == 8 and sum(len(p["question_ids"]) for p in papers) == 24
    return {"dataset_sha256": sha(DEV.read_bytes()),
            "selection": "first eight rank>=147 papers with three questions and 10000<=joined paragraph chars<=40000",
            "model": MODEL, "arms": list(ARMS), "papers": papers}


def second_prompt(arm, source, query, feedback, first_answer):
    warning = "First-pass notes may be wrong. Verify against the paper.\n"
    trace = "<first_run_reasoning_trace>\n" + (feedback["trace"] or "[missing trace]") + "\n</first_run_reasoning_trace>\n"
    quote = "<source_verified_quote>\n" + (feedback["verified_quote"] or "[no verified quote]") + "\n</source_verified_quote>\n"
    answer = "<tentative_first_answer>\n" + (first_answer or "[no first answer]") + "\n</tentative_first_answer>\n"
    question = "Task focus: " + query + "\nFind original evidence, check scope and answer briefly.\n"
    original = "Original paper:\n" + source + "\n"
    if arm == "question_pre":
        body = question + original
    elif arm == "answer_pre":
        body = warning + answer + original
    elif arm == "trace_pre":
        body = warning + trace + original
    elif arm == "trace_post":
        body = original + warning + trace
    elif arm == "quote_pre":
        body = quote + original
    else:
        body = original
    return body + "\nQuestion: " + query + "\nEnd with exactly one line: Final Answer: <short answer>"


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
        "status": "exploratory_dev_visible_trace_mechanism",
        "frozen_sha256": sha(frozen_path.read_bytes()), "papers": []}
    assert results["frozen_sha256"] == sha(frozen_path.read_bytes())
    for pi, fp in enumerate(frozen["papers"]):
        pid = fp["paper_id"]
        paper = build_paper(pid, source[pid])
        paragraphs = paper["paragraphs"]
        original_text = "\n\n".join(paragraphs)
        assert sha(original_text.encode("utf-8")) == fp["source_sha256"]
        rp = next((p for p in results["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "arm_order": list(ARMS[pi % len(ARMS):] + ARMS[:pi % len(ARMS)]), "records": []}
            results["papers"].append(rp)
        for original in source[pid]["qas"]:
            qid = original["question_id"]
            rr = next((r for r in rp["records"] if r["id"] == qid), None)
            if rr is None:
                rr = {"id": qid, "arms": {}}
                rp["records"].append(rr)
            if "first" in rr:
                continue
            query = query_text(paper["title"], original["question"])
            gold = gold_answers(original)
            inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
            prompt = ("Original paper:\n" + original_text + "\nQuestion: " + query +
                      "\nRespond with exactly three short lines. Trace: <what fact or relation is needed and where to look>; "
                      "Evidence Quote: <one exact original-source sentence>; Final Answer: <short answer>. "
                      "Keep the trace under 80 words. Do not add other lines.")
            first = call(prompt, inst, think=False, cap=1024)
            lines = [line.strip() for line in first.get("content", "").splitlines() if line.strip()]
            if len(lines) == 3 and not first.get("answer"):
                first["answer"] = lines[-1]
                first["f1"], first["em"] = score([lines[-1]], inst)
            quote, quote_pid = extract_quote(first.get("content", ""), paragraphs)
            trace, truncated = extract_trace(first.get("content", ""))
            rr["first"] = first
            rr["feedback"] = {"trace": trace, "trace_truncated": truncated,
                              "verified_quote": quote, "quote_paragraph_id": quote_pid}
            path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(pid, qid, "first", round(first["f1"], 3), "trace", len(trace), "quote", bool(quote), flush=True)
        for arm in rp["arm_order"]:
            for original in source[pid]["qas"]:
                rr = next(r for r in rp["records"] if r["id"] == original["question_id"])
                if arm in rr["arms"]:
                    continue
                query = query_text(paper["title"], original["question"])
                gold = gold_answers(original)
                inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
                prompt = second_prompt(arm, original_text, query, rr["feedback"], rr["first"]["answer"])
                rr["arms"][arm] = call(prompt, inst, think=False, cap=1024)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, rr["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
