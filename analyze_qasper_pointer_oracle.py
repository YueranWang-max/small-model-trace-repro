"""Gold-only upper-bound diagnostic for known-question byte-span memories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/qasper_official/qasper-train-v0.3.json"
OUT = ROOT / "logs/qasper_pointer_oracle"


def vbytes(n):
    assert n >= 0
    size = 1
    while n >= 128:
        n >>= 7
        size += 1
    return size


def serialize_paper(p):
    parts = [p["title"], p["abstract"]]
    for section in p["full_text"]:
        parts.append(section["section_name"] or "")
        parts.extend(x.strip() for x in section["paragraphs"] if x.strip())
    return "\n".join(parts).encode("utf-8")


def find_ptr(blob, string):
    if not string or not string.strip():
        return None
    raw = string.encode("utf-8")
    start = blob.find(raw)
    if start < 0:
        return None
    assert blob[start:start+len(raw)] == raw
    return [start, len(raw)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    totals = {k: 0 for k in ("papers", "questions", "answer_extractive", "answer_yesno",
                             "answer_unanswerable", "answer_freeform", "answer_pointer",
                             "answer_pointer_in_evidence", "answer_yesno_or_unknown_direct",
                             "evidence_any", "evidence_all", "evidence_spans",
                             "evidence_pointer_count", "source_bytes", "question_bytes",
                             "answer_text_bytes", "answer_pointer_payload_bytes",
                             "evidence_text_bytes", "evidence_pointer_payload_bytes")}
    rows = []
    for pid, paper in source.items():
        blob = serialize_paper(paper)
        totals["papers"] += 1
        totals["source_bytes"] += len(blob)
        for q in paper["qas"]:
            if not q["answers"]:
                continue
            ans = q["answers"][0]["answer"]
            totals["questions"] += 1
            totals["question_bytes"] += len(q["question"].encode("utf-8"))
            if ans.get("unanswerable"):
                typ, answer = "unanswerable", ""
            elif ans.get("yes_no") is not None:
                typ, answer = "yesno", str(ans["yes_no"])
            elif ans.get("extractive_spans"):
                typ, answer = "extractive", " ".join(ans["extractive_spans"])
            else:
                typ, answer = "freeform", ans.get("free_form_answer") or ""
            totals["answer_" + typ] += 1
            totals["answer_text_bytes"] += len(answer.encode("utf-8"))
            if typ in ("yesno", "unanswerable"):
                totals["answer_yesno_or_unknown_direct"] += 1
            evidence = [s for s in ans.get("highlighted_evidence", []) if s and s.strip()]
            if not evidence:
                evidence = [s for s in ans.get("evidence", []) if s and s.strip()]
            evidence = list(dict.fromkeys(evidence))
            eptrs = [find_ptr(blob, s) for s in evidence]
            mapped = [x for x in eptrs if x is not None]
            if mapped:
                totals["evidence_any"] += 1
            if evidence and len(mapped) == len(evidence):
                totals["evidence_all"] += 1
            totals["evidence_spans"] += len(evidence)
            totals["evidence_pointer_count"] += len(mapped)
            totals["evidence_text_bytes"] += sum(len(s.encode("utf-8")) for s in evidence)
            totals["evidence_pointer_payload_bytes"] += sum(vbytes(a)+vbytes(b) for a,b in mapped)
            candidates = ans.get("extractive_spans", []) if typ == "extractive" else [answer]
            answer_ptrs = [find_ptr(blob, x) for x in candidates if x]
            if candidates and all(x is not None for x in answer_ptrs) and answer_ptrs:
                totals["answer_pointer"] += 1
                totals["answer_pointer_payload_bytes"] += sum(vbytes(a)+vbytes(b) for a,b in answer_ptrs)
                if any(x in ev for x in candidates for ev in evidence):
                    totals["answer_pointer_in_evidence"] += 1
            rows.append({"paper_id": pid, "question_id": q["question_id"], "answer_type": typ,
                         "answer_pointer": bool(answer_ptrs and all(x is not None for x in answer_ptrs)),
                         "answer_pointer_in_evidence": any(x in ev for x in candidates for ev in evidence),
                         "evidence_count": len(evidence), "evidence_pointer_count": len(mapped)})
    summary = {"status": "gold_annotation_oracle_capacity_only_not_deployable_QA",
               "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "unit": "first human answer annotation per question; canonical UTF-8 paper serialization",
               "totals": totals,
               "rates": {"answer_pointer": totals["answer_pointer"]/totals["questions"],
                         "answer_pointer_in_evidence": totals["answer_pointer_in_evidence"]/totals["questions"],
                         "evidence_any": totals["evidence_any"]/totals["questions"],
                         "evidence_all": totals["evidence_all"]/totals["questions"]}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "per_question.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
