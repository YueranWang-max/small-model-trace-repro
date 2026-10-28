"""Answer, coverage and lifecycle accounting for source-linked outlines."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids, norm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_outline_route"
ARMS = ("bm25", "outline", "route")


def boot(values):
    rng = random.Random(20261002)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def main():
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert results["frozen_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    original = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    rows, papers = [], []
    for paper, rp in zip(frozen["papers"], results["papers"]):
        assert paper["paper_id"] == rp["paper_id"]
        assert len(paper["questions"]) == len(rp["records"])
        for card in paper["cards"]:
            assert card["quote"] in paper["paragraphs"][card["source_id"]]
            assert card["source_sha256"] == hashlib.sha256(paper["paragraphs"][card["source_id"]].encode()).hexdigest()
        for q, rr in zip(paper["questions"], rp["records"]):
            assert q["id"] == rr["id"] and all(a in rr["arms"] for a in ("bm25", "outline"))
            used = "outline" if rr["type"] == "overview" else "bm25"
            orig_q = next(x for x in original[paper["paper_id"]]["qas"] if x["question_id"] == q["id"])
            evidence = set(evidence_paragraph_ids(orig_q, paper["paragraphs"]))
            candidate_spans = [norm(s) for an in orig_q["answers"] for s in
                               (an["answer"].get("highlighted_evidence") or an["answer"].get("evidence") or [])
                               if len(norm(s)) >= 30]
            outline_hit = any(s in norm(paper["state"]) for s in candidate_spans) if evidence else None
            rows.append({"id": q["id"], "paper_id": paper["paper_id"], "type": rr["type"],
                         "mappable": bool(evidence), "bm25_hit": bool(evidence & set(rr["selected"])) if evidence else None,
                         "outline_hit": outline_hit,
                         "f1": {"bm25": rr["arms"]["bm25"]["f1"], "outline": rr["arms"]["outline"]["f1"],
                                "route": rr["arms"][used]["f1"]},
                         "em": {"bm25": rr["arms"]["bm25"]["em"], "outline": rr["arms"]["outline"]["em"],
                                "route": rr["arms"][used]["em"]},
                         "tokens": {"bm25": rr["arms"]["bm25"].get("prompt_tokens") or 0,
                                    "outline": rr["arms"]["outline"].get("prompt_tokens") or 0,
                                    "route": rr["arms"][used].get("prompt_tokens") or 0},
                         "generation": {"bm25": rr["arms"]["bm25"].get("generation_tokens") or 0,
                                        "outline": rr["arms"]["outline"].get("generation_tokens") or 0,
                                        "route": rr["arms"][used].get("generation_tokens") or 0},
                         "seconds": {"bm25": rr["arms"]["bm25"]["wall_sec"],
                                     "outline": rr["arms"]["outline"]["wall_sec"],
                                     "route": rr["arms"][used]["wall_sec"]}})
        local = [r for r in rows if r["paper_id"] == paper["paper_id"]]
        papers.append({"paper_id": paper["paper_id"], "n_questions": len(local),
                       "state_chars": paper["state_chars"], "source_chars": paper["source_chars"],
                       "n_cards": len(paper["cards"]),
                       "f1": {a: statistics.mean(r["f1"][a] for r in local) for a in ARMS}})
    pooled = {"n_papers": len(papers), "n_questions": len(rows),
              "type_counts": {t: sum(r["type"] == t for r in rows) for t in ("overview", "numeric_comparison", "detail")},
              "state_chars_total": sum(p["state_chars"] for p in papers),
              "source_chars_total": sum(p["source_chars"] for p in papers),
              "state_to_source_char_ratio": sum(p["state_chars"] for p in papers)/sum(p["source_chars"] for p in papers),
              "f1": {a: statistics.mean(r["f1"][a] for r in rows) for a in ARMS},
              "em": {a: statistics.mean(r["em"][a] for r in rows) for a in ARMS},
              "prompt_tokens": {a: sum(r["tokens"][a] for r in rows) for a in ARMS},
              "generation_tokens": {a: sum(r["generation"][a] for r in rows) for a in ARMS},
              "reader_sec": {a: round(sum(r["seconds"][a] for r in rows), 3) for a in ARMS},
              "mappable": sum(r["mappable"] for r in rows),
              "bm25_evidence_any_hit": sum(r["bm25_hit"] is True for r in rows),
              "outline_exact_span_any_hit": sum(r["outline_hit"] is True for r in rows),
              "state_construction_sec": round(sum(p["construction_sec"] for p in frozen["papers"]), 3),
              "bm25_index_query_sec": round(sum(rp["bm25_build_sec"]+sum(x["bm25_search_sec"] for x in rp["records"])
                                            for rp in results["papers"]), 3)}
    pooled["paired"] = {}
    for a in ("outline", "route"):
        diffs = [r["f1"][a]-r["f1"]["bm25"] for r in rows]
        pdiffs = [p["f1"][a]-p["f1"]["bm25"] for p in papers]
        pooled["paired"][a] = {"mean": statistics.mean(diffs), "question_ci95": boot(diffs),
                               "paper_mean": statistics.mean(pdiffs), "paper_ci95": boot(pdiffs),
                               "wins": sum(x>1e-9 for x in diffs), "losses": sum(x < -1e-9 for x in diffs)}
    pooled["by_type"] = {t: {"n": sum(r["type"] == t for r in rows),
                             "f1": {a: statistics.mean(r["f1"][a] for r in rows if r["type"] == t)
                                    for a in ARMS}} for t in ("overview", "numeric_comparison", "detail")
                         if any(r["type"] == t for r in rows)}
    summary = {"status": "exploratory_extractive_outline_type_route", "per_paper": papers, "pooled": pooled}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
