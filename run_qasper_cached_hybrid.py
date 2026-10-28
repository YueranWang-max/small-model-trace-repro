"""CLM-inspired surrogate: reuse generic paragraph embeddings across Qasper questions."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    model_path = (Path.home() / ".cache" / "huggingface" / "hub" /
                  "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / MODEL_REVISION)
    assert model_path.is_dir()
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    path = OUT / "results_cached_hybrid16.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "fusion": "1/(10+bm25_rank)+1/(10+dense_rank)",
        "analysis_status": "post_hoc_clm_inspired_surrogate", "groups": []}
    assert result["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    t0 = time.perf_counter()
    model = SentenceTransformer(str(model_path), local_files_only=True)
    result.setdefault("model_load_sec", round(time.perf_counter() - t0, 3))
    for group in frozen["groups"]:
        corpus = "\n\n".join(group["paragraphs"])
        assert hashlib.sha256(corpus.encode()).hexdigest() == group["corpus_sha256"]
        t0 = time.perf_counter()
        bm25 = BM25(group["paragraphs"])
        bm25_index_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        embeddings = model.encode(group["paragraphs"], batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        dense_index_sec = time.perf_counter() - t0
        assert np.isfinite(embeddings).all()
        vector_path = OUT / f"cached_paragraph_vectors_group{group['group']}.npy"
        if not vector_path.exists():
            np.save(vector_path, embeddings)
        existing = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if existing is None:
            existing = {"group": group["group"], "corpus_sha256": group["corpus_sha256"],
                        "n_questions": group["n_questions"], "n_paragraphs": len(group["paragraphs"]),
                        "vector_shape": list(embeddings.shape),
                        "bm25_index_sec": round(bm25_index_sec, 6),
                        "dense_index_sec": round(dense_index_sec, 6), "records": []}
            result["groups"].append(existing)
        assert existing["corpus_sha256"] == group["corpus_sha256"]
        done = existing["records"]
        assert [r["id"] for r in done] == [q["id"] for q in group["questions"][:len(done)]]
        for qi, q in enumerate(group["questions"][len(done):], start=len(done)):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            t0 = time.perf_counter()
            query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            dense_scores = embeddings @ query_embedding
            dense_order = np.argsort(-dense_scores, kind="stable").tolist()
            dense_rank = {pid: rank for rank, pid in enumerate(dense_order)}
            bm25_order = bm25.search(query, len(group["paragraphs"]))
            bm25_rank = {pid: rank for rank, pid in enumerate(bm25_order)}
            fused = [1 / (10 + bm25_rank[pid]) + 1 / (10 + dense_rank[pid])
                     for pid in range(len(group["paragraphs"]))]
            selected = sorted(range(len(fused)), key=lambda pid: (-fused[pid], pid))[:16]
            search_sec = time.perf_counter() - t0
            done.append({"id": q["id"], "selected": selected,
                         "bm25_top16": bm25_order[:16], "dense_top16": dense_order[:16],
                         "search_sec": round(search_sec, 6),
                         "selected_material_chars": len("Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in selected))})
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"group{group['group']} q{qi + 1}/{group['n_questions']} hybrid_search_sec={search_sec:.3f}", flush=True)


if __name__ == "__main__":
    main()
