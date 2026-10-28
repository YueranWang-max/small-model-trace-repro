"""Exploratory dual-scale, source-only retrieval on the frozen Qasper corpora."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from run_realdoc_hybrid_pilot import read
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def parent_chunks(paragraphs):
    parents = []
    offset = 0
    for paragraph in paragraphs:
        parents.append(offset // 4000)
        offset += len(paragraph) + 2
    return parents


def main():
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    path = OUT / "results_dualscale16.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "analysis_status": "post_hoc_exploratory_method_search", "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for group in frozen["groups"]:
        gi = group["group"]
        corpus = "\n\n".join(group["paragraphs"])
        assert hashlib.sha256(corpus.encode()).hexdigest() == group["corpus_sha256"]
        chunks = [corpus[i:i + 4000] for i in range(0, len(corpus), 4000)]
        t0 = time.perf_counter()
        chunk_idx, paragraph_idx = BM25(chunks), BM25(group["paragraphs"])
        build_sec = time.perf_counter() - t0
        parent = parent_chunks(group["paragraphs"])
        existing = next((x for x in result["groups"] if x["group"] == gi), None)
        if existing is None:
            existing = {"group": gi, "corpus_sha256": group["corpus_sha256"],
                        "n_questions": group["n_questions"],
                        "both_index_sec": round(build_sec, 6), "records": []}
            result["groups"].append(existing)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        assert existing["corpus_sha256"] == group["corpus_sha256"]
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            chunk_rank = {cid: rank for rank, cid in enumerate(chunk_idx.search(query, len(chunks)))}
            paragraph_rank = {pid: rank for rank, pid in enumerate(paragraph_idx.search(query, len(group["paragraphs"]))) }
            scores = [1 / (10 + paragraph_rank[pid]) + 1 / (10 + chunk_rank[parent[pid]])
                      for pid in range(len(parent))]
            selected = sorted(sorted(range(len(scores)), key=lambda pid: (-scores[pid], pid))[:16])
            search_sec = time.perf_counter() - t0
            material = "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in selected)
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            arm = read(material, inst)
            done.append({"id": q["id"], "selected": selected,
                         "search_sec": round(search_sec, 6), "arm": arm})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{gi} q{qi + 1}/{group['n_questions']} dualscale16={arm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
