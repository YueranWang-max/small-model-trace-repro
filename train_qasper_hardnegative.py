"""Exploratory within-paper hard-negative training on Qasper train only."""
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
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_multiq"
SOURCE = ROOT / "data/qasper_official/qasper-train-v0.3.json"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 20261002


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train = json.loads(SOURCE.read_text(encoding="utf-8"))
    dev_path = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
    assert not (set(train) & set(json.loads(dev_path.read_text(encoding="utf-8"))))
    triples = []
    for pid, paper in train.items():
        prefix = f"Paper {pid}: {paper['title']}"
        paragraphs = [prefix + "\nAbstract: " + paper["abstract"]]
        for section in paper["full_text"]:
            paragraphs.extend(prefix + "\nSection: " + (section["section_name"] or "") + "\n" + s.strip()
                              for s in section["paragraphs"] if s.strip())
        bm25 = BM25(paragraphs)
        for q in paper["qas"]:
            ids = evidence_paragraph_ids(q, paragraphs)
            if not ids:
                continue
            query = "Paper: " + paper["title"] + ". Question: " + q["question"]
            negative_id = next((i for i in bm25.search(query, len(paragraphs)) if i not in ids), None)
            if negative_id is None:
                continue
            triples.append({"id": q["question_id"], "paper_id": pid, "query": query,
                            "positive_id": ids[0], "negative_id": negative_id,
                            "positive": paragraphs[ids[0]], "negative": paragraphs[negative_id]})
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
        batch = [triples[i] for i in order[start:start + 16]]
        def encode(key):
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in model.tokenize([x[key] for x in batch]).items()}
            return F.normalize(model(features)["sentence_embedding"], dim=-1)
        qvec, pvec, nvec = encode("query"), encode("positive"), encode("negative")
        loss = F.softplus(((qvec * nvec).sum(-1) - (qvec * pvec).sum(-1)) / 0.05).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if len(losses) % 20 == 0:
            print(f"step {len(losses)} loss={losses[-1]:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    saved = OUT / "hardnegative_model"
    model.save(str(saved))
    weights = saved / "model.safetensors"
    record = {"status": "post_hoc_within_paper_hard_negative", "train_source_sha256": sha256_file(SOURCE),
              "dev_source_sha256": sha256_file(dev_path), "base_revision": REVISION, "seed": SEED,
              "n_train_triples": len(triples),
              "train_mining_labels": [{k: x[k] for k in ("id", "paper_id", "positive_id", "negative_id")} for x in triples],
              "n_steps": len(losses), "first_loss": losses[0], "last_loss": losses[-1],
              "mean_loss": sum(losses) / len(losses), "train_sec": round(elapsed, 3),
              "device": str(device), "weights_sha256": sha256_file(weights),
              "weights_bytes": weights.stat().st_size}
    (OUT / "hardnegative_training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "train_mining_labels"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
