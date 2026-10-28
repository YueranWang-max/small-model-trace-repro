"""Offline Laya typed decisions over archived Qasper BM25 paragraph candidates."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import laya

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"
MODEL = "convaiinnovations/laya"
SCHEMA = {"relevant": {"type": "choice",
                       "instructions": "Does this passage provide direct evidence to answer the question?",
                       "criteria": {"yes": "the passage contains direct evidence to answer the question",
                                    "no": "the passage does not contain direct evidence to answer the question"}}}


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    frozen_path = OUT / "frozen.json"
    candidate_path = OUT / "results_paragraph24.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert candidates["frozen_sha256"] == frozen_sha
    path = OUT / "results_laya_top12.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": frozen_sha, "candidate_sha256": candidate_sha,
        "laya_version": laya.__version__, "model": MODEL, "schema": SCHEMA,
        "analysis_status": "post_hoc_exploratory_method_search", "groups": []}
    assert result["frozen_sha256"] == frozen_sha and result["candidate_sha256"] == candidate_sha
    started = time.perf_counter()
    agent = laya.load(MODEL)
    load_sec = time.perf_counter() - started
    result.setdefault("model_load_sec", round(load_sec, 3))
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for group, cg in zip(frozen["groups"], candidates["groups"]):
        assert group["group"] == cg["group"]
        existing = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if existing is None:
            existing = {"group": group["group"], "n_questions": group["n_questions"], "records": []}
            result["groups"].append(existing)
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, (q, cr) in enumerate(zip(group["questions"][len(done):], cg["records"][len(done):]), start=len(done)):
            assert q["id"] == cr["id"]
            ids = cr["selected"]
            states = [{"question": q["question"], "passage": group["paragraphs"][pid]} for pid in ids]
            t0 = time.perf_counter()
            responses = agent.predict_batch(states, SCHEMA, batch_size=16, sort_by_length=True)
            wall_sec = time.perf_counter() - t0
            scores = [r["answers"]["relevant"]["probabilities"]["yes"] for r in responses]
            rank = sorted(range(len(ids)), key=lambda j: (-scores[j], j))
            selected = [ids[j] for j in rank[:12]]
            existing["records"].append({"id": q["id"], "candidate_ids": ids,
                                        "yes_scores": scores, "selected": selected,
                                        "laya_input_tokens": sum(r["usage"]["input_tokens"] for r in responses),
                                        "laya_wall_sec": round(wall_sec, 3),
                                        "predictions": responses})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} q{qi + 1}/{group['n_questions']} laya_sec={wall_sec:.3f}", flush=True)


if __name__ == "__main__":
    main()
