"""Question-independent extractive paper outlines and frozen type routing."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from prepare_qasper_multiq import gold_answers
from run_realdoc_hybrid_pilot import read
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_outline_route"
SOURCE = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
CUES = ("propose", "method", "result", "perform", "achieve", "experiment", "dataset",
        "evaluat", "compar", "improv", "limit", "conclude")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def build_paper(pid, paper):
    start = time.perf_counter()
    prefix = f"Paper {pid}: {paper['title']}"
    paragraphs = [prefix + "\nAbstract: " + paper["abstract"]]
    sections = []
    for section in paper["full_text"]:
        name = section["section_name"] or ""
        ids = []
        for p in section["paragraphs"]:
            if p.strip():
                ids.append(len(paragraphs))
                paragraphs.append(prefix + "\nSection: " + name + "\n" + p.strip())
        sections.append((name, ids))
    cards = [{"source_id": 0, "section": "Abstract", "quote": paper["abstract"],
              "source_sha256": hashlib.sha256(paragraphs[0].encode()).hexdigest()}]
    state = f"{prefix}\n[Source P0 | Abstract] {paper['abstract']}"
    assert len(state) <= 7000
    for name, ids in sections:
        if not ids:
            continue
        first = sentences(paragraphs[ids[0]])[0]
        candidates = [(i, s) for i in ids for s in sentences(paragraphs[i])]
        cue = next(((i, s) for i, s in candidates if any(t in s.lower() for t in CUES) and s != first), None)
        second = cue or next(((i, s) for i, s in reversed(candidates) if s != first), None)
        for i, quote in [(ids[0], first)] + ([second] if second else []):
            assert quote in paragraphs[i]
            line = f"\n[Source P{i} | Section: {name}] {quote}"
            if len(state) + len(line) <= 7000:
                state += line
                cards.append({"source_id": i, "section": name, "quote": quote,
                              "source_sha256": hashlib.sha256(paragraphs[i].encode()).hexdigest()})
    assert len(state) <= 7000
    return {"paper_id": pid, "title": paper["title"], "paragraphs": paragraphs, "state": state,
            "state_chars": len(state), "source_chars": len("\n\n".join(paragraphs)),
            "cards": cards, "construction_sec": round(time.perf_counter()-start, 6),
            "questions": [{"id": q["question_id"], "question": q["question"],
                           "gold_answers": gold_answers(q)} for q in paper["qas"]]}


def qtype(question):
    s = question.lower()
    if re.search(r"\bhow (many|much)\b|\b(percentage|percent|difference|compare|compared|higher|lower|better|worse)\b", s):
        return "numeric_comparison"
    if re.search(r"\b(main|overall|contribution|goal|aim|purpose|approach)\b|propos", s):
        return "overview"
    return "detail"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    else:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        ids = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))[27:36]
        assert len(ids) == 9
        papers = [build_paper(pid, source[pid]) for pid in ids]
        frozen = {"source_sha256": sha(SOURCE), "selection_rule": "dev paper ranks 28-36, descending question count then ID",
                  "state_rule": "abstract plus up to two exact sentences per section, <=7000 chars per paper",
                  "papers": papers}
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": sha(frozen_path), "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "papers": []}
    assert result["frozen_sha256"] == sha(frozen_path)
    global_index = 0
    for paper in frozen["papers"]:
        t0 = time.perf_counter()
        bm25 = BM25(paper["paragraphs"])
        build_sec = time.perf_counter()-t0
        rp = next((x for x in result["papers"] if x["paper_id"] == paper["paper_id"]), None)
        if rp is None:
            rp = {"paper_id": paper["paper_id"], "bm25_build_sec": round(build_sec, 6), "records": []}
            result["papers"].append(rp)
        for qi, q in enumerate(paper["questions"]):
            rr = next((x for x in rp["records"] if x["id"] == q["id"]), None)
            if rr is None:
                query = "Paper: " + paper["title"] + ". Question: " + q["question"]
                t0 = time.perf_counter()
                selected = bm25.search(query, 16)
                rr = {"id": q["id"], "type": qtype(q["question"]), "selected": selected,
                      "bm25_search_sec": round(time.perf_counter()-t0, 6), "arms": {}}
                rp["records"].append(rr)
            else:
                assert rr["type"] == qtype(q["question"])
            query = "Paper: " + paper["title"] + ". Question: " + q["question"]
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            materials = {"bm25": "Retrieved evidence:\n" + "\n\n".join(paper["paragraphs"][i] for i in rr["selected"]),
                         "outline": "Retrieved evidence:\n" + paper["state"]}
            order = ("bm25", "outline") if global_index % 2 == 0 else ("outline", "bm25")
            rr["order"] = list(order)
            for arm in order:
                if arm in rr["arms"]:
                    continue
                rr["arms"][arm] = read(materials[arm], inst)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"paper {paper['paper_id']} q{qi+1}/{len(paper['questions'])} {arm} f1={rr['arms'][arm]['f1']:.3f}", flush=True)
            global_index += 1


if __name__ == "__main__":
    main()
