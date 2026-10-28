"""Frozen, question-independent extractive facts and evidence-coverage gate."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from analyze_qasper_evidence_gaps import evidence_paragraph_ids
from run_qasper_outline_route import build_paper, sentences
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_atomic_facts"
TAGS = {
    "number": r"\b\d+(?:\.\d+)?\s*(?:%|percent|hours?|minutes?|seconds?|datasets?|samples?|examples?|layers?|epochs?|points?)?\b",
    "data": r"\b(?:dataset|corpus|data source|training data|test set|benchmark|sample|subjects?)\b",
    "metric": r"\b(?:metric|accuracy|precision|recall|f1|bleu|rouge|score|auc|error rate|likert|evaluate)\b",
    "comparison": r"\b(?:baseline|compare|compared|versus|vs\.?|outperform|better than|lower than|higher than|improv)\w*\b",
    "method": r"\b(?:method|model|algorithm|module|architecture|classifier|embedding|train|propose)\w*\b",
    "result": r"\b(?:result|achieve|perform|find|show|demonstrate|observe)\w*\b",
    "limitation": r"\b(?:limit|future work|however|although|except|only when|fail)\w*\b",
}
QUERY_WORDS = {
    "numeric_comparison": "number result score compare baseline percentage",
    "data_metric": "dataset corpus data metric evaluation accuracy score",
    "method_model": "method model architecture algorithm module classifier",
    "other": "",
}


def tags_for(sentence):
    return [k for k, pattern in TAGS.items() if re.search(pattern, sentence, re.I)]


def qtype(text):
    s = text.lower()
    if re.search(r"\bhow (many|much)\b|\b(percent|percentage|difference|compar|higher|lower|better|worse)\w*\b", s):
        return "numeric_comparison"
    if re.search(r"\b(dataset|data|corpus|sample|metric|measure|evaluat|accuracy|score)\w*\b", s):
        return "data_metric"
    if re.search(r"\b(method|model|algorithm|module|architecture|classifier|embedding|train)\w*\b", s):
        return "method_model"
    return "other"


def extract(paper):
    start = time.perf_counter()
    candidates = []
    for pid, paragraph in enumerate(paper["paragraphs"]):
        section = "Abstract" if pid == 0 else paragraph.split("\nSection: ", 1)[-1].split("\n", 1)[0]
        body = paragraph.split("\n", 2)[-1]
        for sent in sentences(body):
            if not 25 <= len(sent) <= 550 or sent not in paragraph:
                continue
            tags = tags_for(sent)
            if not tags:
                continue
            score = len(tags) + int("number" in tags) + int("comparison" in tags)
            candidates.append({"source_id": pid, "section": section, "quote": sent,
                               "tags": tags, "score": score,
                               "source_sha256": hashlib.sha256(paragraph.encode()).hexdigest()})
    candidates.sort(key=lambda c: (-c["score"], c["source_id"], c["quote"]))
    records, section_counts, chars = [], {}, 0
    for c in candidates:
        if len(records) >= 160:
            break
        if section_counts.get(c["section"], 0) >= 12:
            continue
        cost = len(c["quote"]) + len(c["section"]) + 24
        if chars + cost > 24000:
            continue
        records.append(c)
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1
        chars += cost
    return records, chars, time.perf_counter() - start


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    ids = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))[45:54]
    assert len(ids) == 9
    all_papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        records, state_chars, construction_sec = extract(paper)
        for rec in records:
            para = paper["paragraphs"][rec["source_id"]]
            assert rec["quote"] in para
            assert rec["source_sha256"] == hashlib.sha256(para.encode()).hexdigest()
        t0 = time.perf_counter()
        para_index = BM25(paper["paragraphs"])
        fact_index = BM25([" ".join(r["tags"]) + " " + r["quote"] for r in records])
        index_sec = time.perf_counter() - t0
        qrows = []
        for q in paper["questions"]:
            t0 = time.perf_counter()
            query = "Paper: " + paper["title"] + ". Question: " + q["question"]
            global_rank = para_index.search(query, len(paper["paragraphs"]))
            base = global_rank[:16]
            qt = qtype(q["question"])
            fact_rank = fact_index.search(query + " " + QUERY_WORDS[qt], min(len(records), 40))
            selection = global_rank[:12]
            for ri in fact_rank:
                para_id = records[ri]["source_id"]
                if para_id not in selection:
                    selection.append(para_id)
                if len(selection) >= min(16, len(paper["paragraphs"])):
                    break
            for para_id in global_rank:
                if len(selection) >= min(16, len(paper["paragraphs"])):
                    break
                if para_id not in selection:
                    selection.append(para_id)
            assert len(selection) == min(16, len(paper["paragraphs"]))
            select_sec = time.perf_counter() - t0
            original = next(x for x in source[pid]["qas"] if x["question_id"] == q["id"])
            evidence = evidence_paragraph_ids(original, paper["paragraphs"])
            qrows.append({"id": q["id"], "question": q["question"], "type": qt,
                          "baseline": base, "facts": selection, "evidence": evidence,
                          "baseline_chars": sum(len(paper["paragraphs"][i]) for i in base),
                          "facts_chars": sum(len(paper["paragraphs"][i]) for i in selection),
                          "selection_sec": round(select_sec, 6)})
        all_papers.append({"paper_id": pid, "records": records, "state_chars": state_chars,
                           "source_chars": paper["source_chars"], "construction_sec": round(construction_sec, 6),
                           "index_sec": round(index_sec, 6), "questions": qrows})
        print(pid, "records", len(records), "questions", len(qrows), flush=True)
    result = {"status": "exploratory_dev_extractive_atomic_fact_evidence_gate",
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "paper_ids": ids, "papers": all_papers}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    eligible = [(p, q) for p in all_papers for q in p["questions"] if q["evidence"]]
    def nhit(q, arm):
        return len(set(q["evidence"]) & set(q[arm]))
    summary = {"n_papers": len(all_papers), "n_questions": sum(len(p["questions"]) for p in all_papers),
               "mappable": len(eligible), "n_records": sum(len(p["records"]) for p in all_papers),
               "state_chars": sum(p["state_chars"] for p in all_papers),
               "source_chars": sum(p["source_chars"] for p in all_papers),
               "construction_sec": sum(p["construction_sec"] for p in all_papers),
               "index_sec": sum(p["index_sec"] for p in all_papers),
               "query_sec": sum(q["selection_sec"] for p in all_papers for q in p["questions"]),
               "arms": {}}
    for arm in ("baseline", "facts"):
        summary["arms"][arm] = {"any_hit": sum(nhit(q, arm) > 0 for _, q in eligible),
                                "all_hit": sum(nhit(q, arm) == len(q["evidence"]) for _, q in eligible),
                                "selected_chars": sum(q[arm + "_chars"] for p in all_papers for q in p["questions"])}
    summary["paired"] = {"wins": sum(nhit(q, "facts") > 0 and nhit(q, "baseline") == 0 for _, q in eligible),
                         "losses": sum(nhit(q, "baseline") > 0 and nhit(q, "facts") == 0 for _, q in eligible),
                         "ties": sum((nhit(q, "baseline") > 0) == (nhit(q, "facts") > 0) for _, q in eligible)}
    summary["per_paper"] = [{"paper_id": p["paper_id"], "mappable": sum(bool(q["evidence"]) for q in p["questions"]),
                             "baseline_hit": sum(bool(nhit(q, "baseline")) for q in p["questions"] if q["evidence"]),
                             "facts_hit": sum(bool(nhit(q, "facts")) for q in p["questions"] if q["evidence"])}
                            for p in all_papers]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
