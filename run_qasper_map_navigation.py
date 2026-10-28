"""Frozen map-navigation evidence gate on a new Qasper dev cohort."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids
from run_qasper_outline_route import build_paper
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_map_navigation"


def section_groups(raw, cards):
    # Paragraph numbering follows build_paper: P0 is the abstract.
    groups = []
    idx = 1
    for sec in raw["full_text"]:
        ids = []
        for p in sec["paragraphs"]:
            if p.strip():
                ids.append(idx)
                idx += 1
        if not ids:
            continue
        name = sec["section_name"] or ""
        quotes = [c["quote"] for c in cards if c["source_id"] in ids]
        groups.append({"name": name, "ids": ids, "map_text": name + "\n" + "\n".join(quotes)})
    return groups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    ids = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))[36:45]
    assert len(ids) == 9
    rows = []
    for pid in ids:
        t0 = time.perf_counter()
        p = build_paper(pid, source[pid])
        sections = section_groups(source[pid], p["cards"])
        para_index = BM25(p["paragraphs"])
        map_index = BM25([s["map_text"] for s in sections])
        index_sec = time.perf_counter() - t0
        qrows = []
        for q in p["questions"]:
            t0 = time.perf_counter()
            query = "Paper: " + p["title"] + ". Question: " + q["question"]
            global_rank = para_index.search(query, len(p["paragraphs"]))
            base = global_rank[:16]
            sec_rank = map_index.search(query, min(2, len(sections)))
            selected = global_rank[:12]
            for si in sec_rank:
                for para_id in global_rank:
                    if para_id in sections[si]["ids"] and para_id not in selected:
                        selected.append(para_id)
                        break
            # Repeat over the two candidate sections until four extra slots fill.
            for si in sec_rank:
                for para_id in global_rank:
                    if len(selected) >= 16:
                        break
                    if para_id in sections[si]["ids"] and para_id not in selected:
                        selected.append(para_id)
            for para_id in global_rank:
                if len(selected) >= 16:
                    break
                if para_id not in selected:
                    selected.append(para_id)
            assert len(selected) == min(16, len(p["paragraphs"]))
            select_sec = time.perf_counter() - t0
            original = next(x for x in source[pid]["qas"] if x["question_id"] == q["id"])
            evidence = evidence_paragraph_ids(original, p["paragraphs"])
            qrows.append({"id": q["id"], "question": q["question"],
                          "baseline": base, "navigation": selected,
                          "map_sections": [sections[i]["name"] for i in sec_rank],
                          "evidence": evidence,
                          "selection_sec": round(select_sec, 6),
                          "baseline_chars": sum(len(p["paragraphs"][i]) for i in base),
                          "navigation_chars": sum(len(p["paragraphs"][i]) for i in selected)})
        rows.append({"paper_id": pid, "questions": qrows, "n_sections": len(sections),
                     "n_cards": len(p["cards"]), "state_chars": p["state_chars"],
                     "index_sec": round(index_sec, 6)})
        print(pid, len(qrows), flush=True)
    data = {"status": "exploratory_frozen_dev_map_navigation_evidence_gate",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "paper_ids": ids, "papers": rows}
    (OUT / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    eligible = [(p, q) for p in rows for q in p["questions"] if q["evidence"]]
    def hits(q, arm):
        return len(set(q["evidence"]) & set(q[arm]))
    summary = {"papers": len(rows), "questions": sum(len(p["questions"]) for p in rows),
               "mapped_questions": len(eligible), "arms": {}}
    for arm in ("baseline", "navigation"):
        summary["arms"][arm] = {
            "any_hit": sum(hits(q, arm) > 0 for _, q in eligible),
            "all_hit": sum(hits(q, arm) == len(q["evidence"]) for _, q in eligible),
            "total_selected_chars": sum(q[arm + "_chars"] for p in rows for q in p["questions"]),
        }
    summary["paired_win_loss_tie_any_hit"] = {
        "wins": sum(hits(q, "navigation") > 0 and hits(q, "baseline") == 0 for _, q in eligible),
        "losses": sum(hits(q, "baseline") > 0 and hits(q, "navigation") == 0 for _, q in eligible),
        "ties": sum((hits(q, "baseline") > 0) == (hits(q, "navigation") > 0) for _, q in eligible),
    }
    summary["index_sec"] = sum(p["index_sec"] for p in rows)
    summary["query_sec"] = sum(q["selection_sec"] for p in rows for q in p["questions"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
