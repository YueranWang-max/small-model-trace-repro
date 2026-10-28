"""Summarize the frozen same-corpus lifecycle experiment without model calls."""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "logs" / "e2e_128k"


def avg(xs):
    return sum(xs) / len(xs)


def ci(xs, seed=20261001, n=10000):
    rng = random.Random(seed)
    boots = sorted(avg([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return [boots[int(0.025 * n)], boots[int(0.975 * n)]]


def main(budget):
    name = "results.json" if budget == 4096 else f"results_budget{budget}.json"
    data = json.loads((DIR / name).read_text(encoding="utf-8"))
    rows = data["records"]
    if len(rows) != 12:
        raise RuntimeError(f"Incomplete experiment: {len(rows)}/12 questions")
    ing = data["ingest"]
    state_f1 = [r["state"]["f1"] for r in rows]
    bm_f1 = [r["bm25_top4"]["f1"] for r in rows]
    delta = [a-b for a,b in zip(state_f1,bm_f1)]
    state_query_sec = [r["state"]["wall_sec"] + sum(c["wall_sec"] for c in r["alarm_calls"])
                       for r in rows]
    bm_query_sec = [r["bm25_top4"]["wall_sec"] + r["bm25_top4"]["retrieval_sec"]
                    for r in rows]
    sdri = ing["wall_sec"] + ing["sdr_index_sec"]
    bmi = ing["bm25_index_sec"]
    slope = avg(bm_query_sec) - avg(state_query_sec)
    breakeven = (sdri - bmi)/slope if slope > 0 else None
    summary = {
        "n": len(rows), "corpus_sha256": data["corpus_sha256"],
        "readout_num_predict": budget,
        "ingest_calls": ing["calls"], "ingest_wall_sec": ing["wall_sec"],
        "ingest_verified_edge_recall": ing["recall"],
        "state_f1": avg(state_f1), "bm25_f1": avg(bm_f1),
        "paired_f1_difference": avg(delta), "paired_f1_ci95_exploratory": ci(delta),
        "state_em": avg([r["state"]["em"] for r in rows]),
        "bm25_em": avg([r["bm25_top4"]["em"] for r in rows]),
        "det_pre_f1": avg([r["det_pre_f1"] for r in rows]),
        "det_post_f1": avg([r["det_post_f1"] for r in rows]),
        "raw_parser_f1": avg([r["raw_parser_f1"] for r in rows]),
        "alarm_calls": sum(len(r["alarm_calls"]) for r in rows),
        "alarm_repaired": sum(r["repaired"] for r in rows),
        "alarm_unrepaired": sum(r["unrepaired"] for r in rows),
        "state_query_calls": len(rows) + sum(len(r["alarm_calls"]) for r in rows),
        "bm25_query_calls": len(rows),
        "sdr_mean_query_sec": avg(state_query_sec),
        "bm25_mean_query_sec": avg(bm_query_sec),
        "sdr_total_12q_sec": sdri + sum(state_query_sec),
        "bm25_total_12q_sec": bmi + sum(bm_query_sec),
        "raw_parser_total_12q_sec": ing["raw_parser_build_sec"] + sum(r["raw_parser_query_sec"] for r in rows),
        "sdr_query_prompt_tokens": sum((r["state"]["prompt_tokens"] or 0)
                                       + sum(c["prompt_tokens"] or 0 for c in r["alarm_calls"])
                                       for r in rows),
        "bm25_query_prompt_tokens": sum(r["bm25_top4"]["prompt_tokens"] or 0 for r in rows),
        "sdr_query_generation_tokens": sum((r["state"]["generation_tokens"] or 0)
                                           + sum(c["generation_tokens"] or 0 for c in r["alarm_calls"])
                                           for r in rows),
        "bm25_query_generation_tokens": sum(r["bm25_top4"]["generation_tokens"] or 0 for r in rows),
        "extrapolated_break_even_questions": breakeven,
    }
    summary_name = "summary.json" if budget == 4096 else f"summary_budget{budget}.json"
    (DIR / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k,v in summary.items():
        print(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-predict", type=int, default=4096)
    args = parser.parse_args()
    main(args.num_predict)
