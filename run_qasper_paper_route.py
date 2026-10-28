"""Run a metadata-routed, source-only BM25 comparator on all frozen Qasper questions."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from run_realdoc_hybrid_pilot import read
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def main():
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    path = OUT / "results_paper_route3.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "analysis_status": "post_hoc_exploratory_method_search", "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for group in frozen["groups"]:
        paper_data = {}
        total_build = 0.0
        for pid in group["paper_ids"]:
            paragraphs = [p for p in group["paragraphs"] if p.startswith(f"Paper {pid}:")]
            corpus = "\n\n".join(paragraphs)
            chunks = [corpus[i:i + 4000] for i in range(0, len(corpus), 4000)]
            t0 = time.perf_counter()
            index = BM25(chunks)
            total_build += time.perf_counter() - t0
            paper_data[pid] = (chunks, index)
        existing = next((x for x in result["groups"] if x["group"] == group["group"]), None)
        if existing is None:
            existing = {"group": group["group"], "corpus_sha256": group["corpus_sha256"],
                        "paper_ids": group["paper_ids"], "n_questions": group["n_questions"],
                        "paper_index_sec": round(total_build, 6), "records": []}
            result["groups"].append(existing)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        assert existing["corpus_sha256"] == group["corpus_sha256"]
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            chunks, index = paper_data[q["paper_id"]]
            t0 = time.perf_counter()
            selected = index.search(query, min(3, len(chunks)))
            search_sec = time.perf_counter() - t0
            material = "Retrieved original source chunks:\n" + "\n\n".join(chunks[i] for i in selected)
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            arm = read(material, inst)
            done.append({"id": q["id"], "paper_id": q["paper_id"], "selected": selected,
                         "search_sec": round(search_sec, 6), "arm": arm})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} q{qi + 1}/{group['n_questions']} paper_route3={arm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
