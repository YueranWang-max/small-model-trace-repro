"""Paper-level Qasper transfer check with frozen hard-negative retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from prepare_qasper_multiq import gold_answers
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_holdout"
SOURCE = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
TRAINING = ROOT / "logs/qasper_multiq/hardnegative_training.json"
MODEL_DIR = ROOT / "logs/qasper_multiq/hardnegative_model"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    papers = json.loads(SOURCE.read_text(encoding="utf-8"))
    ids = sorted(papers, key=lambda pid: (-len(papers[pid]["qas"]), pid))[9:18]
    old = json.loads((ROOT / "logs/qasper_multiq/frozen.json").read_text(encoding="utf-8"))
    assert not (set(ids) & {pid for g in old["groups"] for pid in g["paper_ids"]})
    groups = []
    for gi in range(3):
        group_ids = ids[gi*3:gi*3+3]
        paragraphs, questions = [], []
        for pid in group_ids:
            paper = papers[pid]
            prefix = f"Paper {pid}: {paper['title']}"
            paragraphs.append(prefix + "\nAbstract: " + paper["abstract"])
            for section in paper["full_text"]:
                paragraphs.extend(prefix + "\nSection: " + (section["section_name"] or "") + "\n" + s.strip()
                                  for s in section["paragraphs"] if s.strip())
            for q in paper["qas"]:
                questions.append({"id": q["question_id"], "paper_id": pid,
                                  "paper_title": paper["title"], "question": q["question"],
                                  "gold_answers": gold_answers(q)})
        random.Random(20261002 + gi).shuffle(questions)
        corpus = "\n\n".join(paragraphs)
        groups.append({"group": gi+1, "paper_ids": group_ids, "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
                       "n_questions": len(questions), "paragraphs": paragraphs, "questions": questions})
    return {"source_sha256": sha(SOURCE), "selection_rule": "dev paper ranks 10-18 by descending question count, ID tie break",
            "groups": groups}


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else prepare()
    if not frozen_path.exists():
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    assert sha(MODEL_DIR / "model.safetensors") == training["weights_sha256"]
    path = OUT / "selections.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": sha(frozen_path), "trained_weights_sha256": training["weights_sha256"], "groups": []}
    assert result["frozen_sha256"] == sha(frozen_path)
    start = time.perf_counter()
    model = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    result.setdefault("model_load_sec", round(time.perf_counter() - start, 6))
    for group in frozen["groups"]:
        t0 = time.perf_counter()
        bm25 = BM25(group["paragraphs"])
        bm25_index_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        embeddings = model.encode(group["paragraphs"], batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        dense_index_sec = time.perf_counter() - t0
        assert np.isfinite(embeddings).all()
        g = next((x for x in result["groups"] if x["group"] == group["group"]), None)
        if g is None:
            g = {"group": group["group"], "corpus_sha256": group["corpus_sha256"],
                 "bm25_index_sec": round(bm25_index_sec, 6), "dense_index_sec": round(dense_index_sec, 6), "records": []}
            result["groups"].append(g)
        assert g["corpus_sha256"] == group["corpus_sha256"]
        for q in group["questions"][len(g["records"]):]:
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            bm25_order = bm25.search(query, len(group["paragraphs"]))
            qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            dense_order = np.argsort(-(embeddings @ qvec), kind="stable").tolist()
            brank = {pid: r for r, pid in enumerate(bm25_order)}
            drank = {pid: r for r, pid in enumerate(dense_order)}
            fused = sorted(range(len(group["paragraphs"])),
                           key=lambda pid: (-(1/(10+brank[pid])+1/(10+drank[pid])), pid))
            baseline = bm25_order[:16]
            budget = len("\n\n".join(group["paragraphs"][i] for i in baseline))
            matched, used = [], 0
            for pid in fused:
                cost = len(group["paragraphs"][pid]) + (2 if matched else 0)
                if used + cost <= budget:
                    matched.append(pid)
                    used += cost
                if len(matched) == 16:
                    break
            assert matched
            g["records"].append({"id": q["id"], "paper_id": q["paper_id"], "bm25_16": baseline,
                                 "fused_16": fused[:16], "fused_budget": matched,
                                 "bm25_material_chars": budget, "budget_material_chars": used,
                                 "search_sec": round(time.perf_counter()-t0, 6)})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} {len(g['records'])}/{len(group['questions'])}", flush=True)


if __name__ == "__main__":
    main()
