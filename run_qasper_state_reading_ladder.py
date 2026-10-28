"""Frozen Qasper bridge: source-verified reusable state with decreasing source text."""
from __future__ import annotations

import json
import time

from prepare_qasper_multiq import gold_answers
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, ask, sha
from run_stageC import BM25

OUT = ROOT / "logs/qasper_state_reading_ladder"
ARMS = ("full", "state_full", "bm25_16", "state_16", "bm25_8", "state_8", "state_only")


def make_plan(source):
    ranked = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))
    ids = ranked[124:130]
    assert len(ids) == 6 and all(len(source[pid]["qas"]) == 4 for pid in ids)
    papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        for card in paper["cards"]:
            paragraph = paper["paragraphs"][card["source_id"]]
            assert card["quote"] in paragraph
            assert card["source_sha256"] == sha(paragraph.encode("utf-8"))
        t0 = time.perf_counter()
        index = BM25(paper["paragraphs"])
        index_sec = time.perf_counter() - t0
        qs = []
        for original in source[pid]["qas"]:
            query = "Paper: " + paper["title"] + ". Question: " + original["question"]
            t0 = time.perf_counter()
            selected = index.search(query, 16)
            search_sec = time.perf_counter() - t0
            assert len(selected) == 16
            qs.append({"id": original["question_id"], "question": original["question"],
                       "query": query, "selected_16": selected,
                       "selected_8": selected[:8], "search_sec": round(search_sec, 6)})
        papers.append({"paper_id": pid, "title": paper["title"], "state": paper["state"],
                       "state_sha256": sha(paper["state"].encode("utf-8")),
                       "state_chars": len(paper["state"]), "state_construction_sec": paper["construction_sec"],
                       "source_chars": paper["source_chars"],
                       "paragraph_sha256": [sha(p.encode("utf-8")) for p in paper["paragraphs"]],
                       "cards": paper["cards"], "index_sec": round(index_sec, 6), "questions": qs})
    return {"source_sha256": sha(DEV.read_bytes()), "selection": "Qasper dev ranks 125-130 inclusive; all questions",
            "state_rule": "run_qasper_outline_route.build_paper, exact quotes and per-paragraph SHA256",
            "model": MODEL, "arms": list(ARMS), "papers": papers}


def material(arm, state, paragraphs, q):
    if arm == "state_only":
        return "Verified state:\n" + state
    if arm in ("full", "state_full"):
        selected = list(range(len(paragraphs)))
    elif arm in ("bm25_16", "state_16"):
        selected = q["selected_16"]
    else:
        selected = q["selected_8"]
    source = "\n\n".join(paragraphs[i] for i in selected)
    if arm.startswith("state_"):
        return "Verified state:\n" + state + "\n\nOriginal source:\n" + source
    return "Original source:\n" + source


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen["source_sha256"] == sha(DEV.read_bytes())
    else:
        frozen = make_plan(source)
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_verified_state_reading_ladder", "model": MODEL,
        "frozen_sha256": sha(frozen_path.read_bytes()), "papers": []}
    assert results["frozen_sha256"] == sha(frozen_path.read_bytes())
    for pi, fp in enumerate(frozen["papers"]):
        pid = fp["paper_id"]
        paragraphs = build_paper(pid, source[pid])["paragraphs"]
        assert [sha(p.encode("utf-8")) for p in paragraphs] == fp["paragraph_sha256"]
        rp = next((p for p in results["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            order = list(ARMS[pi:] + ARMS[:pi])
            rp = {"paper_id": pid, "arm_order": order, "records": []}
            results["papers"].append(rp)
        for arm in rp["arm_order"]:
            for fq in fp["questions"]:
                rr = next((r for r in rp["records"] if r["id"] == fq["id"]), None)
                if rr is None:
                    rr = {"id": fq["id"], "arms": {}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                original = next(q for q in source[pid]["qas"] if q["question_id"] == fq["id"])
                gold = gold_answers(original)
                inst = {"question": fq["query"], "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = ask(material(arm, fp["state"], paragraphs, fq), fq["query"], inst)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, fq["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
