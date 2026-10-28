"""Run the frozen same-corpus multi-question CPU-index comparison."""
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
    assert hashlib.sha256((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_bytes()).hexdigest() == frozen["source_sha256"]
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024, "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for group in frozen["groups"]:
        gi = group["group"]
        corpus = "\n\n".join(group["paragraphs"])
        assert len(corpus) == group["corpus_chars"]
        assert hashlib.sha256(corpus.encode()).hexdigest() == group["corpus_sha256"]
        chunks = [corpus[i:i+4000] for i in range(0, len(corpus), 4000)]
        t0 = time.perf_counter()
        chunk_idx = BM25(chunks)
        chunk_build_sec = time.perf_counter()-t0
        t0 = time.perf_counter()
        paragraph_idx = BM25(group["paragraphs"])
        paragraph_build_sec = time.perf_counter()-t0
        existing = next((x for x in result["groups"] if x["group"] == gi), None)
        if existing is None:
            existing = {"group": gi, "paper_ids": group["paper_ids"],
                        "corpus_sha256": group["corpus_sha256"],
                        "corpus_chars": group["corpus_chars"],
                        "n_questions": group["n_questions"],
                        "chunk_index_sec": round(chunk_build_sec, 6),
                        "paragraph_index_sec": round(paragraph_build_sec, 6),
                        "records": []}
            result["groups"].append(existing)
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        assert existing["corpus_sha256"] == group["corpus_sha256"]
        done = existing["records"]
        assert [x["id"] for x in done] == [x["id"] for x in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            retrieval_query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            chunk_selected = chunk_idx.search(retrieval_query, min(4,len(chunks)))
            chunk_search_sec = time.perf_counter()-t0
            t0 = time.perf_counter()
            paragraph_selected = paragraph_idx.search(retrieval_query, min(12,len(group["paragraphs"])))
            paragraph_search_sec = time.perf_counter()-t0
            materials = {
                "chunk4": "Retrieved original source chunks:\n" + "\n\n".join(chunks[i] for i in chunk_selected),
                "paragraph12": "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in paragraph_selected),
            }
            inst = {"question": retrieval_query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            order = ("chunk4","paragraph12") if qi % 2 == 0 else ("paragraph12","chunk4")
            arms = {arm: read(materials[arm], inst) for arm in order}
            done.append({"id":q["id"],"paper_id":q["paper_id"],"order":order,
                         "chunk_selected":chunk_selected,"paragraph_selected":paragraph_selected,
                         "chunk_search_sec":round(chunk_search_sec,6),
                         "paragraph_search_sec":round(paragraph_search_sec,6),
                         "arms":arms})
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
            print(f"group{gi} q{qi+1}/{group['n_questions']} chunk={arms['chunk4']['f1']:.3f} paragraph={arms['paragraph12']['f1']:.3f}",flush=True)


if __name__ == "__main__":
    main()
