"""Frozen evidence gate for the learned Qasper question-to-fact ranker."""
from __future__ import annotations

import hashlib
import json
import os
import time
import argparse
from pathlib import Path

import numpy as np

from analyze_qasper_evidence_gaps import evidence_paragraph_ids
from run_qasper_atomic_facts import extract
from run_qasper_outline_route import build_paper
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_fact_ranker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=54, help="zero-based first ranked dev paper")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    training = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    assert training["dev_sha256"] == hashlib.sha256(DEV.read_bytes()).hexdigest()
    source = json.loads(DEV.read_text(encoding="utf-8"))
    ids = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))[args.start:args.start+9]
    assert len(ids) == 9
    t0 = time.perf_counter()
    model = SentenceTransformer(str(OUT / "model"), local_files_only=True)
    model.max_seq_length = 256
    model_load_sec = time.perf_counter() - t0
    all_papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        records, state_chars, extraction_sec = extract(paper)
        fact_text = [" ".join(r["tags"]) + " " + r["quote"] for r in records]
        t0 = time.perf_counter()
        vectors = model.encode(fact_text, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
        encoding_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        para_index = BM25(paper["paragraphs"])
        index_sec = time.perf_counter() - t0
        qrows = []
        for q in paper["questions"]:
            t0 = time.perf_counter()
            query = "Paper: " + paper["title"] + ". Question: " + q["question"]
            global_rank = para_index.search(query, len(paper["paragraphs"]))
            base = global_rank[:16]
            qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)[0]
            scores = vectors @ qvec
            rank = np.argsort(-scores).tolist()
            selection = global_rank[:12]
            for ri in rank:
                para_id = records[ri]["source_id"]
                if para_id not in selection:
                    selection.append(para_id)
                if len(selection) >= min(16, len(paper["paragraphs"])):
                    break
            for para_id in global_rank:
                if len(selection) >= min(16, len(paper["paragraphs"])):
                    break
                if para_id not in selection:
                    selection.append(para_id)
            query_sec = time.perf_counter() - t0
            original = next(x for x in source[pid]["qas"] if x["question_id"] == q["id"])
            evidence = evidence_paragraph_ids(original, paper["paragraphs"])
            qrows.append({"id": q["id"], "question": q["question"],
                          "baseline": base, "learned": selection, "evidence": evidence,
                          "baseline_chars": sum(len(paper["paragraphs"][i]) for i in base),
                          "learned_chars": sum(len(paper["paragraphs"][i]) for i in selection),
                          "query_sec": round(query_sec, 6)})
        all_papers.append({"paper_id": pid, "n_records": len(records), "state_chars": state_chars,
                           "source_chars": paper["source_chars"], "extraction_sec": round(extraction_sec, 6),
                           "encoding_sec": round(encoding_sec, 6), "index_sec": round(index_sec, 6),
                           "questions": qrows})
        print(pid, len(qrows), flush=True)
    result = {"status": "frozen_exploratory_dev_learned_fact_evidence_gate",
              "train_weights_sha256": training["weights_sha256"],
              "dev_sha256": training["dev_sha256"], "paper_ids": ids,
              "model_load_sec": round(model_load_sec, 6), "papers": all_papers}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    eligible = [(p, q) for p in all_papers for q in p["questions"] if q["evidence"]]
    def nhit(q, arm):
        return len(set(q["evidence"]) & set(q[arm]))
    summary = {"n_papers": len(all_papers), "n_questions": sum(len(p["questions"]) for p in all_papers),
               "mappable": len(eligible), "n_records": sum(p["n_records"] for p in all_papers),
               "state_chars": sum(p["state_chars"] for p in all_papers),
               "source_chars": sum(p["source_chars"] for p in all_papers),
               "training_sec": training["train_sec"], "model_load_sec": result["model_load_sec"],
               "extraction_sec": sum(p["extraction_sec"] for p in all_papers),
               "encoding_sec": sum(p["encoding_sec"] for p in all_papers),
               "index_sec": sum(p["index_sec"] for p in all_papers),
               "query_sec": sum(q["query_sec"] for p in all_papers for q in p["questions"]),
               "model_weights_bytes": training["weights_bytes"], "arms": {}}
    for arm in ("baseline", "learned"):
        summary["arms"][arm] = {"any_hit": sum(nhit(q, arm)>0 for _, q in eligible),
                                "all_hit": sum(nhit(q, arm)==len(q["evidence"]) for _, q in eligible),
                                "selected_chars": sum(q[arm+"_chars"] for p in all_papers for q in p["questions"])}
    summary["paired"] = {"wins": sum(nhit(q,"learned")>0 and nhit(q,"baseline")==0 for _,q in eligible),
                         "losses": sum(nhit(q,"baseline")>0 and nhit(q,"learned")==0 for _,q in eligible),
                         "ties": sum((nhit(q,"baseline")>0)==(nhit(q,"learned")>0) for _,q in eligible)}
    summary["per_paper"] = [{"paper_id": p["paper_id"],
                             "mappable": sum(bool(q["evidence"]) for q in p["questions"]),
                             "baseline_hit": sum(bool(nhit(q,"baseline")) for q in p["questions"] if q["evidence"]),
                             "learned_hit": sum(bool(nhit(q,"learned")) for q in p["questions"] if q["evidence"])}
                            for p in all_papers]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
