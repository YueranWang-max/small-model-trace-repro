"""Reread identical archived paragraph-24 evidence in source order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_realdoc_hybrid_pilot import read

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def main():
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected_path = OUT / "results_paragraph24.json"
    selected_data = json.loads(selected_path.read_text(encoding="utf-8"))
    assert selected_data["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    path = OUT / "results_paragraph24_source_order.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": selected_data["frozen_sha256"],
        "selection_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
        "analysis_status": "post_hoc_exploratory_order_test", "groups": []}
    assert result["selection_sha256"] == hashlib.sha256(selected_path.read_bytes()).hexdigest()
    for group, selected_group in zip(frozen["groups"], selected_data["groups"]):
        assert group["group"] == selected_group["group"]
        existing = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if existing is None:
            existing = {"group": group["group"], "n_questions": group["n_questions"], "records": []}
            result["groups"].append(existing)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, (q, sr) in enumerate(zip(group["questions"][len(done):], selected_group["records"][len(done):]), start=len(done)):
            assert q["id"] == sr["id"]
            ids = sorted(sr["selected"])
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            material = "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in ids)
            inst = {"question": query, "answer": q["gold_answers"][0],
                    "answer_aliases": q["gold_answers"][1:]}
            arm = read(material, inst)
            done.append({"id": q["id"], "same_selected_ids_source_order": ids, "arm": arm})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} q{qi+1}/{group['n_questions']} source_order={arm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
