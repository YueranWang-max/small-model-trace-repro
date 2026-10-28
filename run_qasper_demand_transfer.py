"""Frozen official-test evidence transfer of the previously selected global prior."""
from __future__ import annotations
import json
import time
import random
import joblib
from run_qasper_prefix_vs_bm25 import ROOT, sha
from run_qasper_fingerprint import prepare, select, metrics, recall

OUT = ROOT / "logs/qasper_demand_transfer"
SOURCE = ROOT / "data/qasper_official/qasper-test-v0.3.json"
MODEL_PATH = ROOT / "logs/qasper_fingerprint/model.joblib"
ARMS = ("bm25", "global")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    fp = OUT / "frozen.json"
    if fp.exists():
        frozen = json.loads(fp.read_text(encoding="utf-8"))
        assert frozen["dataset_sha256"] == sha(SOURCE.read_bytes())
        assert frozen["model_sha256"] == sha(MODEL_PATH.read_bytes())
        assert frozen["prior_code_sha256"] == sha((ROOT / "run_qasper_fingerprint.py").read_bytes())
    else:
        t0 = time.perf_counter()
        model = joblib.load(MODEL_PATH)
        load_sec = time.perf_counter() - t0
        cfg = model["selection"]["global"]
        assert cfg["k"] == 1 and cfg["weight"] == 0.5
        density = model["models"][1][1][0]
        ids = sorted(raw, key=lambda pid: sha(pid.encode()))
        t0 = time.perf_counter()
        papers = [prepare(pid, raw[pid]) for pid in ids]
        prepare_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        choices = {p["id"]: {q["id"]: {"bm25": q["rank"][:8],
            "global": select(p, q, density, 0.5)} for q in p["questions"]} for p in papers}
        selection_sec = time.perf_counter() - t0
        frozen = {"dataset_sha256": sha(SOURCE.read_bytes()), "model_sha256": sha(MODEL_PATH.read_bytes()),
            "prior_code_sha256": sha((ROOT / "run_qasper_fingerprint.py").read_bytes()),
            "protocol_sha256": sha((ROOT / "protocol_qasper_demand_transfer_20261003.md").read_bytes()),
            "reader_paper_ids": ids[:40], "parameters": cfg, "papers": papers, "choices": choices,
            "model_load_sec": load_sec, "preparation_sec": prepare_sec, "selection_sec": selection_sec}
        fp.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    # First consultation of test evidence annotations occurs after all selections are frozen.
    records = metrics(frozen["papers"], frozen["choices"], raw)
    (OUT / "evidence_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {"status": "frozen_global_prior_official_test_transfer", "papers": len(records),
        "questions": sum(len(p["records"]) for p in records), "arms": {},
        "reader_papers": len(frozen["reader_paper_ids"]),
        "reader_questions": sum(len(p["questions"]) for p in frozen["papers"] if p["id"] in frozen["reader_paper_ids"]),
        "model_load_sec": frozen["model_load_sec"], "preparation_sec": frozen["preparation_sec"],
        "selection_sec": frozen["selection_sec"], "frozen_sha256": sha(fp.read_bytes())}
    for arm in ARMS:
        rows = [q["arms"][arm] for p in records for q in p["records"]]
        eligible = [r for r in rows if r["recall"] is not None]
        summary["arms"][arm] = {"eligible": len(eligible), "recall": recall(records, arm),
            "any_hit": sum(r["any"] for r in eligible), "all_hit": sum(r["all"] for r in eligible),
            "chars": sum(r["chars"] for r in rows)}
    rng = random.Random(20261003)
    boot = []
    for _ in range(10000):
        sample = [records[rng.randrange(len(records))] for _ in records]
        boot.append(recall(sample, "global") - recall(sample, "bm25"))
    boot.sort()
    summary["paired"] = {"recall_difference": recall(records, "global") - recall(records, "bm25"),
                         "paper_ci95": [boot[250], boot[9750]]}
    summary["evidence_gate_pass"] = boot[250] > 0 and summary["arms"]["global"]["any_hit"] >= summary["arms"]["bm25"]["any_hit"]
    (OUT / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
