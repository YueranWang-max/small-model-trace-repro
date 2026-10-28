"""Paper-clustered answer, cost, and offline evidence analysis of reading ladder."""
from __future__ import annotations

import json
import random

from analyze_qasper_evidence_gaps import norm
from run_qasper_outline_route import build_paper
from run_qasper_state_reading_ladder import ARMS, DEV, OUT


def mean(x):
    return sum(x) / len(x) if x else None


def sec(x, key):
    return (x.get(key) or 0) / 1e9


def mapped_evidence_strings(question, paragraphs):
    pieces = []
    for annotation in question["answers"]:
        answer = annotation["answer"]
        highlighted = answer.get("highlighted_evidence", [])
        pieces.extend(s for s in (highlighted or answer.get("evidence", [])) if len(norm(s)) >= 30)
    return sorted({norm(s) for s in pieces if any(norm(s) in norm(p) for p in paragraphs)})


def main():
    source = json.loads(DEV.read_text(encoding="utf-8"))
    plan = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert len(data["papers"]) == len(plan["papers"]) == 6
    groups = [p["records"] for p in data["papers"]]
    assert all(len(g) == 4 and all(set(q["arms"]) == set(ARMS) for q in g) for g in groups)
    rows = [q for g in groups for q in g]
    summary = {"status": "exploratory_dev_verified_state_reading_ladder", "n_papers": 6,
               "n_questions": len(rows), "arms": {}, "paired": {}, "per_paper": [],
               "evidence_diagnostic": {a: {"eligible": 0, "any_hit": 0, "all_hit": 0} for a in ARMS}}
    for arm in ARMS:
        x = [q["arms"][arm] for q in rows]
        summary["arms"][arm] = {"f1": mean([v["f1"] for v in x]), "em": mean([v["em"] for v in x]),
            "prompt_tokens": sum(v.get("prompt_tokens") or 0 for v in x),
            "generation_tokens": sum(v.get("generation_tokens") or 0 for v in x),
            "prefill_sec": sum(sec(v, "prompt_eval_duration_ns") for v in x),
            "api_total_sec": sum(sec(v, "total_duration_ns") for v in x),
            "wall_sec": sum(v["wall_sec"] for v in x),
            "errors": sum("error" in v for v in x)}
    for pair in (("state_full", "full"), ("state_16", "bm25_16"),
                 ("state_8", "bm25_8"), ("state_8", "bm25_16"),
                 ("state_only", "state_8")):
        a, b = pair
        diff = [q["arms"][a]["f1"] - q["arms"][b]["f1"] for q in rows]
        rng = random.Random(20261003)
        boot = []
        for _ in range(10000):
            sample = [groups[rng.randrange(len(groups))] for _ in groups]
            boot.append(mean([q["arms"][a]["f1"] - q["arms"][b]["f1"] for g in sample for q in g]))
        boot.sort()
        summary["paired"][a + "_minus_" + b] = {"f1_difference": mean(diff),
            "paper_ci95": [boot[250], boot[9750]], "wins": sum(v > 1e-9 for v in diff),
            "losses": sum(v < -1e-9 for v in diff), "ties": sum(abs(v) <= 1e-9 for v in diff)}
    for fp, rp in zip(plan["papers"], data["papers"]):
        assert fp["paper_id"] == rp["paper_id"]
        paragraphs = build_paper(fp["paper_id"], source[fp["paper_id"]])["paragraphs"]
        entry = {"paper_id": fp["paper_id"], "state_chars": fp["state_chars"],
                 "source_chars": fp["source_chars"], "arm_order": rp["arm_order"],
                 "f1": {a: mean([q["arms"][a]["f1"] for q in rp["records"]]) for a in ARMS}}
        summary["per_paper"].append(entry)
        for fq in fp["questions"]:
            original = next(q for q in source[fp["paper_id"]]["qas"] if q["question_id"] == fq["id"])
            evidence = mapped_evidence_strings(original, paragraphs)
            if not evidence:
                continue
            for arm in ARMS:
                if arm == "state_only":
                    text = fp["state"]
                elif arm in ("full", "state_full"):
                    text = "\n\n".join(paragraphs) + (fp["state"] if arm == "state_full" else "")
                elif arm in ("bm25_16", "state_16"):
                    text = "\n\n".join(paragraphs[i] for i in fq["selected_16"]) + (fp["state"] if arm == "state_16" else "")
                else:
                    text = "\n\n".join(paragraphs[i] for i in fq["selected_8"]) + (fp["state"] if arm == "state_8" else "")
                normalized_text = norm(text)
                hits = sum(span in normalized_text for span in evidence)
                tally = summary["evidence_diagnostic"][arm]
                tally["eligible"] += 1
                tally["any_hit"] += bool(hits)
                tally["all_hit"] += hits == len(evidence)
    summary["construction_index_search_sec"] = sum(p["state_construction_sec"] + p["index_sec"] +
        sum(q["search_sec"] for q in p["questions"]) for p in plan["papers"])
    summary["primary_noninferiority_margin"] = -0.05
    summary["primary_gate_pass"] = summary["paired"]["state_8_minus_bm25_16"]["paper_ci95"][0] > -0.05
    summary["caveat"] = "Source-verified extractive outline is not a reasoning trace. Six dev papers and sequential Ollama timing; evidence diagnostic uses gold only after readout."
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
