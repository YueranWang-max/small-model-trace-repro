"""Summarize explicit answer-span selection and pointer-first accuracy."""
from __future__ import annotations

import json
import random
from pathlib import Path

from analyze_qasper_pointer_oracle import vbytes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_known_question_span_v2"


def mean(x):
    return sum(x)/len(x) if x else None


def ci(x):
    s = sorted(x)
    return [s[int(len(s)*0.025)], s[int(len(s)*0.975)]]


def main():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert len(result["papers"]) == 9
    groups = []
    for p in result["papers"]:
        assert all(set(q["arms"]) == {"baseline", "compiler"} for q in p["records"])
        groups.append(p["records"])
    rows = [q for g in groups for q in g]
    arms = {}
    for arm in ("baseline", "compiler"):
        arms[arm] = {"f1": mean([q["arms"][arm]["f1"] for q in rows]),
                     "em": mean([q["arms"][arm]["em"] for q in rows]),
                     "prompt_tokens": sum(q["arms"][arm].get("prompt_tokens") or 0 for q in rows),
                     "generation_tokens": sum(q["arms"][arm].get("generation_tokens") or 0 for q in rows),
                     "wall_sec": sum(q["arms"][arm]["wall_sec"] for q in rows)}
    compiler = [q["arms"]["compiler"] for q in rows]
    arms["pointer_first"] = {"f1": mean([x["decoded_f1"] for x in compiler]),
                             "em": mean([x["decoded_em"] for x in compiler])}
    direct_cache_bytes = sum(len(q["question"].encode("utf-8"))+len(q["arms"]["compiler"]["answer"].encode("utf-8"))+1 for q in rows)
    pointer_core_bytes = 0
    for q in rows:
        handle = q["arms"]["compiler"]["handle"]
        pointer_core_bytes += len(q["question"].encode("utf-8")) + 1
        if handle["kind"] == "span":
            pointer_core_bytes += vbytes(handle["start"]) + vbytes(handle["length"])
        elif handle["kind"] == "inline":
            pointer_core_bytes += len(handle["value"].encode("utf-8"))
    pointer_with_evidence_bytes = sum(len(q["question"].encode("utf-8"))+q["arms"]["compiler"]["handle_payload_bytes"] for q in rows)
    diffs = [q["arms"]["compiler"]["f1"]-q["arms"]["baseline"]["f1"] for q in rows]
    rng = random.Random(20261003)
    boot = []
    for _ in range(10000):
        picked = [groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(mean([q["arms"]["compiler"]["f1"]-q["arms"]["baseline"]["f1"] for group in picked for q in group]))
    summary = {"status": "known_question_compiler_exploratory_dev_no_gold_compilation",
               "n_papers": len(groups), "n_questions": len(rows), "arms": arms,
               "paired": {"f1_diff": mean(diffs), "paper_ci95": ci(boot),
                          "wins": sum(x>1e-9 for x in diffs), "losses": sum(x < -1e-9 for x in diffs),
                          "ties": sum(abs(x)<=1e-9 for x in diffs)},
               "compiler": {"valid_quote": sum(x["quote_valid"] for x in compiler),
                            "explicit_answer_span": sum(bool(x.get("answer_span")) and x["answer_span"].lower()!="none" for x in compiler),
                            "handle_kinds": {k: sum(x["handle"]["kind"]==k for x in compiler)
                                             for k in ("span", "enum", "inline")},
                            "span_with_valid_evidence": sum(x["handle"]["kind"]=="span" and x["quote_valid"] for x in compiler),
                            "malformed_or_error": sum(not x["answer"] or "error" in x for x in compiler)},
               "storage_bytes": {"source_canonical": sum(p["source_bytes"] for p in result["papers"]),
                                 "direct_answer_cache_question_plus_answer": direct_cache_bytes,
                                 "pointer_tier_question_plus_answer_handle": pointer_core_bytes,
                                 "pointer_tier_question_plus_answer_and_evidence_handle": pointer_with_evidence_bytes,
                                 "doc_sha256_binary_per_paper": 32*len(result["papers"])},
               "nonmodel_index_search_sec": sum(p["index_sec"] for p in result["papers"])+sum(q["search_sec"] for q in rows),
               "cost_note": "Compiler cost is paid once per known question; online exact-key lookup/dereference is deterministic. Repeated-query crossover below is a projection from observed call times, not an executed repeated run."}
    if arms["baseline"]["wall_sec"] > 0:
        summary["projected_repeat_count_to_beat_repeated_baseline_walltime"] = arms["compiler"]["wall_sec"]/arms["baseline"]["wall_sec"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
