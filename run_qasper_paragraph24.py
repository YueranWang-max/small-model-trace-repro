"""Run the declared post hoc paragraph-24 sensitivity arm on every frozen Qasper question."""
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
    path = OUT / "results_paragraph24.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "analysis_status": "post_hoc_budget_sensitivity", "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for group in frozen["groups"]:
        gi = group["group"]
        corpus = "\n\n".join(group["paragraphs"])
        assert len(corpus) == group["corpus_chars"]
        assert hashlib.sha256(corpus.encode()).hexdigest() == group["corpus_sha256"]
        t0 = time.perf_counter()
        index = BM25(group["paragraphs"])
        build_sec = time.perf_counter() - t0
        existing = next((x for x in result["groups"] if x["group"] == gi), None)
        if existing is None:
            existing = {"group": gi, "corpus_sha256": group["corpus_sha256"],
                        "corpus_chars": group["corpus_chars"],
                        "n_questions": group["n_questions"],
                        "paragraph_index_sec": round(build_sec, 6), "records": []}
            result["groups"].append(existing)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        assert existing["corpus_sha256"] == group["corpus_sha256"]
        done = existing["records"]
        assert [x["id"] for x in done] == [x["id"] for x in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            selected = index.search(query, min(24, len(group["paragraphs"])))
            search_sec = time.perf_counter() - t0
            material = "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in selected)
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            arm = read(material, inst)
            done.append({"id": q["id"], "paper_id": q["paper_id"],
                         "selected": selected, "search_sec": round(search_sec, 6), "arm": arm})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{gi} q{qi+1}/{group['n_questions']} paragraph24={arm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
