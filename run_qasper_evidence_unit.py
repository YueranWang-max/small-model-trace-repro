"""Frozen ninth-paragraph evidence-unit pilot; see protocol_qasper_evidence_unit_20261003.md."""
from __future__ import annotations

import json
import re
import time

from prepare_qasper_multiq import gold_answers
from repro_lib import ollama_chat
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, sha
from run_qasper_visible_trace_replication import call, query_text
from run_stageC import BM25

OUT = ROOT / "logs/qasper_evidence_unit"
ARMS = ("base8", "extract", "sentence", "full9")
SYSTEM = "Copy an exact evidence quote from the supplied paragraph. Do not answer the question."


def freeze(source):
    ranked = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))
    papers = []
    for pid in ranked[170:]:
        paper = build_paper(pid, source[pid])
        if len(source[pid]["qas"]) != 3 or len(paper["paragraphs"]) < 9:
            continue
        index = BM25(paper["paragraphs"])
        questions = []
        for original in source[pid]["qas"]:
            query = query_text(paper["title"], original["question"])
            questions.append({"id": original["question_id"], "query": query,
                              "top9": index.search(query, 9)})
        papers.append({"paper_id": pid, "title": paper["title"],
                       "paragraph_sha256": [sha(p.encode("utf-8")) for p in paper["paragraphs"]],
                       "questions": questions})
        if len(papers) == 6:
            break
    assert len(papers) == 6 and all(len(p["questions"]) == 3 for p in papers)
    return {"dataset_sha256": sha(DEV.read_bytes()), "model": MODEL,
            "selection": "first six ranked>=171 with three questions and >=9 paragraphs",
            "arms": list(ARMS), "papers": papers}


def extract(query, paragraph):
    prompt = ("Question: " + query + "\nParagraph:\n" + paragraph +
              "\nCopy one contiguous verbatim passage from this paragraph, at most 300 characters, "
              "most useful for answering the question. Output one line beginning Quote: . "
              "If irrelevant, output Quote: NONE.")
    t0 = time.perf_counter()
    try:
        g = ollama_chat(MODEL, SYSTEM, prompt, num_ctx=32768, think=False,
                        temperature=0.0, num_predict=512)
        raw = g.raw
        lines = re.findall(r"(?im)^Quote:\s*(.*?)\s*$", g.content)
        proposal = lines[-1].strip().strip('"“”') if lines else ""
        valid = 20 <= len(proposal) <= 300 and proposal in paragraph
        return {"proposal": proposal, "quote": proposal if valid else "",
                "valid": valid, "content": g.content,
                "prompt_sha256": sha(prompt.encode("utf-8")),
                "prompt_tokens": raw.get("prompt_eval_count"),
                "generation_tokens": raw.get("eval_count"),
                "api_total_ns": raw.get("total_duration"),
                "wall_sec": round(time.perf_counter() - t0, 3),
                "response_sha256": sha(json.dumps(raw, sort_keys=True).encode("utf-8"))}
    except Exception as exc:
        return {"proposal": "", "quote": "", "valid": False, "error": repr(exc),
                "wall_sec": round(time.perf_counter() - t0, 3)}


def sentence_control(paragraph, query, n_chars):
    if n_chars <= 0:
        return ""
    units = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", paragraph) if s.strip()]
    if not units:
        return paragraph[:n_chars]
    ranked = BM25(units).search(query, 1)
    return units[ranked[0]][:n_chars]


def prompt(arm, paragraphs, q, extraction, sentence):
    top = q["top9"]
    base = "\n\n".join(f"[P{i}] {paragraphs[i]}" for i in top[:8])
    if arm == "base8":
        addition = ""
    elif arm == "extract":
        addition = extraction["quote"] or "[no verified quote]"
    elif arm == "sentence":
        addition = sentence or "[no verified quote]"
    else:
        addition = paragraphs[top[8]]
    if arm != "base8":
        base += f"\n\n[P{top[8]}] " + addition
    return ("Answer using only the evidence below. If insufficient, say unknown.\nEvidence:\n" +
            base + "\n\nQuestion: " + q["query"] +
            "\nEnd with exactly one line: Final Answer: <short answer>")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    fp = OUT / "frozen.json"
    plan = freeze(source)
    if fp.exists():
        assert json.loads(fp.read_text(encoding="utf-8")) == plan
    else:
        fp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_ninth_paragraph_evidence_unit",
        "frozen_sha256": sha(fp.read_bytes()), "papers": []}
    assert data["frozen_sha256"] == sha(fp.read_bytes())
    for pi, paper_plan in enumerate(plan["papers"]):
        pid = paper_plan["paper_id"]
        paragraphs = build_paper(pid, source[pid])["paragraphs"]
        assert [sha(p.encode("utf-8")) for p in paragraphs] == paper_plan["paragraph_sha256"]
        index_start = time.perf_counter()
        BM25(paragraphs)
        index_sec = time.perf_counter() - index_start
        rp = next((p for p in data["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "index_sec": index_sec,
                  "arm_order": list(ARMS[pi % 4:] + ARMS[:pi % 4]), "records": []}
            data["papers"].append(rp)
        for q in paper_plan["questions"]:
            rr = next((r for r in rp["records"] if r["id"] == q["id"]), None)
            if rr is None:
                rr = {"id": q["id"], "top9": q["top9"], "arms": {}}
                rp["records"].append(rr)
            if "extraction" not in rr:
                rr["extraction"] = extract(q["query"], paragraphs[q["top9"][8]])
                rr["sentence"] = sentence_control(paragraphs[q["top9"][8]], q["query"],
                                                   len(rr["extraction"]["quote"]))
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, q["id"], "extract", rr["extraction"]["valid"], flush=True)
        for arm in rp["arm_order"]:
            for q in paper_plan["questions"]:
                rr = next(r for r in rp["records"] if r["id"] == q["id"])
                if arm in rr["arms"]:
                    continue
                original = next(x for x in source[pid]["qas"] if x["question_id"] == q["id"])
                gold = gold_answers(original)
                inst = {"question": q["query"], "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = call(prompt(arm, paragraphs, q, rr["extraction"], rr["sentence"]),
                                        inst, think=False, cap=1024)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, q["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
