"""Post hoc paragraph-top-16 readout control for the Laya-selected top-12 arm."""
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
    path = OUT / "results_paragraph16_control.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "analysis_status": "post_hoc_laya_budget_control", "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for group in frozen["groups"]:
        t0 = time.perf_counter()
        index = BM25(group["paragraphs"])
        index_sec = time.perf_counter() - t0
        existing = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if existing is None:
            existing = {"group": group["group"], "n_questions": group["n_questions"],
                        "index_sec": round(index_sec, 6), "records": []}
            result["groups"].append(existing)
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            selected = index.search(query, min(16, len(group["paragraphs"])))
            search_sec = time.perf_counter() - t0
            material = "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in selected)
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            arm = read(material, inst)
            done.append({"id": q["id"], "selected": selected,
                         "search_sec": round(search_sec, 6), "arm": arm})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} q{qi+1}/{group['n_questions']} paragraph16={arm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
