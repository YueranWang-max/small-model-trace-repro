"""Paired, paper-clustered analysis of learned fact-ranker reader outputs."""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_fact_ranker"


def mean(values):
    return sum(values) / len(values) if values else None


def ci(values):
    if not values:
        return None
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT)
    args = parser.parse_args()
    out_dir = args.input
    evidence = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    readout = json.loads((out_dir / "readout.json").read_text(encoding="utf-8"))
    train = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    assert [p["paper_id"] for p in evidence["papers"]] == [p["paper_id"] for p in readout["papers"]]
    rows = []
    paper_rows = []
    for ep, rp in zip(evidence["papers"], readout["papers"]):
        assert [q["id"] for q in ep["questions"]] == [q["id"] for q in rp["records"]]
        prs = []
        for eq, rq in zip(ep["questions"], rp["records"]):
            assert set(rq["arms"]) == {"baseline", "learned"}
            row = {"paper_id": ep["paper_id"], "id": eq["id"], "question": eq["question"],
                   "evidence": eq["evidence"], "baseline_selected": eq["baseline"],
                   "learned_selected": eq["learned"], "baseline": rq["arms"]["baseline"],
                   "learned": rq["arms"]["learned"]}
            rows.append(row)
            prs.append(row)
        paper_rows.append(prs)
    summary = {"status": "exploratory_dev_paired_reader_after_evidence_gate",
               "n_papers": len(paper_rows), "n_questions": len(rows), "n_mappable": sum(bool(r["evidence"]) for r in rows),
               "train_triples": train["n_train_triples"], "train_sec": train["train_sec"],
               "model_weights_bytes": train["weights_bytes"], "arms": {}, "per_paper": []}
    for arm in ("baseline", "learned"):
        summary["arms"][arm] = {"f1": mean([r[arm]["f1"] for r in rows]),
                                "em": mean([r[arm]["em"] for r in rows]),
                                "prompt_tokens": sum(r[arm]["prompt_tokens"] for r in rows),
                                "generation_tokens": sum(r[arm]["generation_tokens"] for r in rows),
                                "reader_sec": sum(r[arm]["wall_sec"] for r in rows),
                                "malformed": sum(bool(r[arm]["malformed"]) for r in rows)}
    summary["per_paper"] = [{"paper_id": prs[0]["paper_id"], "n_questions": len(prs),
                             "baseline_f1": mean([r["baseline"]["f1"] for r in prs]),
                             "learned_f1": mean([r["learned"]["f1"] for r in prs])}
                            for prs in paper_rows]
    diffs = [r["learned"]["f1"] - r["baseline"]["f1"] for r in rows]
    rng = random.Random(20261002)
    boot_question = [mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(10000)]
    boot_paper = []
    for _ in range(10000):
        sample = [paper_rows[rng.randrange(len(paper_rows))] for _ in paper_rows]
        boot_paper.append(mean([r["learned"]["f1"]-r["baseline"]["f1"] for prs in sample for r in prs]))
    summary["paired"] = {"f1_difference": mean(diffs), "question_ci95": ci(boot_question),
                         "paper_ci95": ci(boot_paper),
                         "wins": sum(d > 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs),
                         "ties": sum(abs(d) <= 1e-9 for d in diffs)}
    summary["nonreader_cost"] = {
        "train_sec": train["train_sec"], "model_load_sec": evidence["model_load_sec"],
        "extraction_sec": sum(p["extraction_sec"] for p in evidence["papers"]),
        "fact_encoding_sec": sum(p["encoding_sec"] for p in evidence["papers"]),
        "index_sec": sum(p["index_sec"] for p in evidence["papers"]),
        "query_sec": sum(q["query_sec"] for p in evidence["papers"] for q in p["questions"]),
    }
    (out_dir / "readout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
