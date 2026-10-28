"""Train a small cached-candidate retriever on Qasper train evidence only."""
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"
SOURCE = ROOT / "data/qasper_official/qasper-train-v0.3.json"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
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
    dev = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    assert not (set(train) & set(dev))
    pairs = []
    for pid, paper in train.items():
        prefix = f"Paper {pid}: {paper['title']}"
        paragraphs = [prefix + "\nAbstract: " + paper["abstract"]]
        for section in paper["full_text"]:
            paragraphs.extend(prefix + "\nSection: " + (section["section_name"] or "") + "\n" + text.strip()
                              for text in section["paragraphs"] if text.strip())
        for q in paper["qas"]:
            ids = evidence_paragraph_ids(q, paragraphs)
            if ids:
                pairs.append({"id": q["question_id"], "paper_id": pid,
                              "query": "Paper: " + paper["title"] + ". Question: " + q["question"],
                              "positive": paragraphs[ids[0]]})
    assert pairs
    model_path = (Path.home() / ".cache" / "huggingface" / "hub" /
                  "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / BASE_REVISION)
    assert model_path.is_dir()
    started = time.perf_counter()
    model = SentenceTransformer(str(model_path), local_files_only=True)
    model.max_seq_length = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    order = list(range(len(pairs)))
    random.Random(SEED).shuffle(order)
    losses = []
    for start in range(0, len(order), 32):
        batch = [pairs[i] for i in order[start:start + 32]]
        if len(batch) < 2:
            break
        qfeat = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in model.tokenize([x["query"] for x in batch]).items()}
        pfeat = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in model.tokenize([x["positive"] for x in batch]).items()}
        qvec = F.normalize(model(qfeat)["sentence_embedding"], dim=-1)
        pvec = F.normalize(model(pfeat)["sentence_embedding"], dim=-1)
        logits = (qvec @ pvec.T) / 0.05
        target = torch.arange(len(batch), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if len(losses) % 10 == 0:
            print(f"step {len(losses)} loss={losses[-1]:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    saved = OUT / "contrastive_model"
    model.save(str(saved))
    weights = saved / "model.safetensors"
    assert weights.exists()
    record = {"status": "post_hoc_clm_inspired_surrogate_training",
              "train_source_sha256": sha256_file(SOURCE),
              "dev_source_sha256": sha256_file(ROOT / "data/qasper_official/qasper-dev-v0.3.json"),
              "base_model": "sentence-transformers/all-MiniLM-L6-v2",
              "base_revision": BASE_REVISION, "seed": SEED,
              "n_train_pairs": len(pairs), "train_question_ids": [x["id"] for x in pairs],
              "n_steps": len(losses), "first_loss": losses[0], "last_loss": losses[-1],
              "mean_loss": sum(losses) / len(losses), "train_sec": round(elapsed, 3),
              "device": str(device), "weights_sha256": sha256_file(weights),
              "weights_bytes": weights.stat().st_size}
    (OUT / "contrastive_training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "train_question_ids"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
