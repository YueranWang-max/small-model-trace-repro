"""Frozen, paired Qasper readout of BM25 order versus common-prefix order."""
from __future__ import annotations

import json
import time

from analyze_qasper_pointer_oracle import serialize_paper
from prepare_qasper_multiq import gold_answers
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, ask, sha
from run_stageC import BM25

OUT = ROOT / "logs/qasper_intersection_prefix"


def freeze(source):
    ranked = sorted(source, key=lambda p: (-len(source[p]["qas"]), p))
    ids = ranked[115:124]
    assert len(ids) == 9 and all(len(source[p]["qas"]) == 4 for p in ids)
    papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        t0 = time.perf_counter()
        index = BM25(paper["paragraphs"])
        index_sec = time.perf_counter() - t0
        questions = []
        for original in source[pid]["qas"]:
            query = "Paper: " + paper["title"] + ". Question: " + original["question"]
            t0 = time.perf_counter()
            selected = index.search(query, 16)
            search_sec = time.perf_counter() - t0
            questions.append({"id": original["question_id"], "question": original["question"],
                              "query": query, "selected": selected,
                              "search_sec": round(search_sec, 6)})
        common = set(questions[0]["selected"])
        for q in questions[1:]:
            common.intersection_update(q["selected"])
        common_order = sorted(common, key=lambda i: (sum(q["selected"].index(i) for q in questions) / len(questions), i))
        for q in questions:
            q["prefix_order"] = common_order + [i for i in q["selected"] if i not in common]
            assert set(q["selected"]) == set(q["prefix_order"])
        source_bytes = serialize_paper(source[pid])
        papers.append({"paper_id": pid, "source_sha256": sha(source_bytes),
                       "source_chars": len(source_bytes.decode("utf-8")),
                       "paragraph_sha256": [sha(p.encode("utf-8")) for p in paper["paragraphs"]],
                       "bm25_index_sec": round(index_sec, 6),
                       "common_order": common_order, "questions": questions})
    return {"source_sha256": sha(DEV.read_bytes()), "model": MODEL,
            "selection": "Qasper dev paper ranks 116-124; all four questions per paper",
            "papers": papers}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    computed = freeze(source)
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        # Timers do not enter the plan comparison.
        for p in frozen["papers"]:
            p.pop("bm25_index_sec", None)
            for q in p["questions"]:
                q.pop("search_sec", None)
        for p in computed["papers"]:
            p.pop("bm25_index_sec", None)
            for q in p["questions"]:
                q.pop("search_sec", None)
        assert frozen == computed, "Frozen retrieval plan differs from current source/index"
    else:
        frozen = computed
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_common_evidence_prefix_vs_bm25_order",
        "frozen_sha256": sha(frozen_path.read_bytes()), "model": MODEL, "papers": []}
    assert results["frozen_sha256"] == sha(frozen_path.read_bytes())
    for pi, fp in enumerate(frozen["papers"]):
        pid = fp["paper_id"]
        paragraphs = build_paper(pid, source[pid])["paragraphs"]
        rp = next((p for p in results["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "arm_order": ["common_prefix", "bm25"] if pi % 2 == 0 else ["bm25", "common_prefix"],
                  "records": []}
            results["papers"].append(rp)
        for arm in rp["arm_order"]:
            for fq in fp["questions"]:
                qid = fq["id"]
                rr = next((r for r in rp["records"] if r["id"] == qid), None)
                if rr is None:
                    rr = {"id": qid, "arms": {}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                order = fq["prefix_order"] if arm == "common_prefix" else fq["selected"]
                material = "\n\n".join(paragraphs[i] for i in order)
                original = next(q for q in source[pid]["qas"] if q["question_id"] == qid)
                gold = gold_answers(original)
                inst = {"question": fq["query"], "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = ask(material, fq["query"], inst)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, qid, arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
