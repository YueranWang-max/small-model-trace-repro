"""Post hoc, gold-only diagnostic of evidence coverage; never used for retrieval."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "qasper_multiq"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_paragraph_ids(question, paragraphs):
    pieces = []
    for annotation in question["answers"]:
        answer = annotation["answer"]
        # Highlighted spans are more likely to survive Qasper serialization.
        pieces.extend(s for s in answer.get("highlighted_evidence", []) if len(norm(s)) >= 30)
        if not answer.get("highlighted_evidence"):
            pieces.extend(s for s in answer.get("evidence", []) if len(norm(s)) >= 30)
    candidates = [norm(s) for s in pieces]
    return sorted({i for i, paragraph in enumerate(paragraphs)
                   if any(s in norm(paragraph) for s in candidates)})


def main():
    source = json.loads((ROOT / "data/qasper_official/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    para24 = json.loads((OUT / "results_paragraph24.json").read_text(encoding="utf-8"))
    dual = json.loads((OUT / "results_dualscale16.json").read_text(encoding="utf-8"))
    totals = {a: {"eligible": 0, "any_hit": 0, "all_hit": 0, "f1_when_any_hit": [], "f1_when_no_hit": []}
              for a in ("chunk4", "paragraph12", "paragraph24", "dualscale16")}
    group_rows = []
    for g, bg, pg, dg in zip(frozen["groups"], base["groups"], para24["groups"], dual["groups"]):
        assert g["group"] == bg["group"] == pg["group"] == dg["group"]
        corpus = "\n\n".join(g["paragraphs"])
        chunks = [corpus[i:i+4000] for i in range(0, len(corpus), 4000)]
        group_summary = {a: {"eligible": 0, "any_hit": 0, "all_hit": 0} for a in totals}
        for q, br, pr, dr in zip(g["questions"], bg["records"], pg["records"], dg["records"]):
            assert q["id"] == br["id"] == pr["id"] == dr["id"]
            original = next(x for x in source[q["paper_id"]]["qas"] if x["question_id"] == q["id"])
            ids = evidence_paragraph_ids(original, g["paragraphs"])
            if not ids:
                continue
            materials = {
                "chunk4": " ".join(chunks[i] for i in br["chunk_selected"]),
                "paragraph12": " ".join(g["paragraphs"][i] for i in br["paragraph_selected"]),
                "paragraph24": " ".join(g["paragraphs"][i] for i in pr["selected"]),
                "dualscale16": " ".join(g["paragraphs"][i] for i in dr["selected"]),
            }
            f1 = {"chunk4": br["arms"]["chunk4"]["f1"],
                  "paragraph12": br["arms"]["paragraph12"]["f1"],
                  "paragraph24": pr["arm"]["f1"], "dualscale16": dr["arm"]["f1"]}
            for arm, text in materials.items():
                # Full paragraph presence is conservative for chunk arms that cut through a paragraph.
                hits = sum(norm(g["paragraphs"][i]) in norm(text) for i in ids)
                group_summary[arm]["eligible"] += 1
                group_summary[arm]["any_hit"] += bool(hits)
                group_summary[arm]["all_hit"] += hits == len(ids)
                totals[arm]["eligible"] += 1
                totals[arm]["any_hit"] += bool(hits)
                totals[arm]["all_hit"] += hits == len(ids)
                totals[arm]["f1_when_any_hit" if hits else "f1_when_no_hit"].append(f1[arm])
        group_rows.append({"group": g["group"], "arms": group_summary})
    summary = {"description": "post hoc annotation mapping; exact normalized highlighted-span to paragraph; full paragraph coverage criterion; diagnostic only",
               "groups": group_rows, "pooled": {}}
    for arm, x in totals.items():
        summary["pooled"][arm] = {
            "eligible": x["eligible"], "any_hit": x["any_hit"], "all_hit": x["all_hit"],
            "any_hit_rate": x["any_hit"] / x["eligible"] if x["eligible"] else None,
            "all_hit_rate": x["all_hit"] / x["eligible"] if x["eligible"] else None,
            "mean_f1_when_any_hit": sum(x["f1_when_any_hit"]) / len(x["f1_when_any_hit"]) if x["f1_when_any_hit"] else None,
            "mean_f1_when_no_hit": sum(x["f1_when_no_hit"]) / len(x["f1_when_no_hit"]) if x["f1_when_no_hit"] else None,
        }
    (OUT / "evidence_gap_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
