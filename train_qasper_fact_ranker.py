"""Train a question-to-extractive-fact dual encoder on Qasper train only."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_qasper_evidence_gaps import evidence_paragraph_ids
from run_qasper_atomic_facts import extract
from run_qasper_outline_route import build_paper
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/qasper_official/qasper-train-v0.3.json"
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_fact_ranker"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 20261002


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    assert not (set(train) & set(dev))
    triples = []
    for pid, raw in train.items():
        paper = build_paper(pid, raw)
        records, _, _ = extract(paper)
        if len(records) < 2:
            continue
        fact_text = [" ".join(r["tags"]) + " " + r["quote"] for r in records]
        bm25 = BM25(fact_text)
        for original, q in zip(raw["qas"], paper["questions"]):
            ev = evidence_paragraph_ids(original, paper["paragraphs"])
            if not ev:
                continue
            query = "Paper: " + paper["title"] + ". Question: " + q["question"]
            rank = bm25.search(query, len(records))
            pos = next((i for i in rank if records[i]["source_id"] in ev), None)
            neg = next((i for i in rank if records[i]["source_id"] not in ev), None)
            if pos is None or neg is None:
                continue
            triples.append({"id": q["id"], "paper_id": pid, "query": query,
                            "positive": fact_text[pos], "negative": fact_text[neg],
                            "positive_source_id": records[pos]["source_id"],
                            "negative_source_id": records[neg]["source_id"]})
    assert triples
    base = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots" / REVISION
    assert base.is_dir()
    started = time.perf_counter()
    model = SentenceTransformer(str(base), local_files_only=True)
    model.max_seq_length = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    order = list(range(len(triples)))
    random.Random(SEED).shuffle(order)
    losses = []
    for start in range(0, len(order), 16):
        batch = [triples[i] for i in order[start:start+16]]
        def encode(field):
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in model.tokenize([x[field] for x in batch]).items()}
            return F.normalize(model(features)["sentence_embedding"], dim=-1)
        query, positive, negative = encode("query"), encode("positive"), encode("negative")
        loss = F.softplus(((query * negative).sum(-1) - (query * positive).sum(-1)) / 0.05).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started
    saved = OUT / "model"
    model.save(str(saved))
    weights = saved / "model.safetensors"
    record = {"status": "train_only_weak_sentence_supervision", "train_sha256": sha(TRAIN),
              "dev_sha256": sha(DEV), "base_revision": REVISION, "seed": SEED,
              "n_train_triples": len(triples), "n_steps": len(losses),
              "first_loss": losses[0], "last_loss": losses[-1], "mean_loss": sum(losses)/len(losses),
              "train_sec": round(elapsed, 3), "device": str(device),
              "weights_sha256": sha(weights), "weights_bytes": weights.stat().st_size,
              "train_labels": [{k: x[k] for k in ("id", "paper_id", "positive_source_id", "negative_source_id")}
                               for x in triples]}
    (OUT / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "train_labels"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
