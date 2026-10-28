"""Compile known Qasper questions with explicit exact answer-span selection."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from analyze_qasper_pointer_oracle import serialize_paper, vbytes
from prepare_qasper_multiq import gold_answers
from repro_lib import ollama_chat
from run_qasper_outline_route import build_paper
from run_realdoc import score
from run_realdoc_hybrid_pilot import read
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/qasper_known_question_span_v2"
MODEL = "qwen3.5:4b"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compile_answer(material, question, source_bytes, inst):
    prompt = ("Answer using only the passages below. Copy one exact, contiguous supporting quote from a passage, "
              "without changing any character. Choose the shortest exact contiguous substring of that quote "
              "that directly answers the question; if no such substring exists, write none. "
              "If no exact quote supports the answer, write none. Give a short ordinary answer. "
              "End with exactly these three lines:\n"
              "Supporting Quote: <exact quote or none>\n"
              "Answer Span: <exact substring of quote or none>\n"
              "Final Answer: <short answer or unknown>\n\n"
              + material + "\n\nQuestion: " + question)
    t0 = time.perf_counter()
    try:
        g = ollama_chat(MODEL, "Answer research-paper questions from supplied evidence.", prompt,
                        num_ctx=32768, think=False, temperature=0.0, num_predict=1024)
        found_a = re.findall(r"(?im)^Final Answer:\s*(.*?)\s*$", g.content)
        found_q = re.findall(r"(?im)^Supporting Quote:\s*(.*?)\s*$", g.content)
        found_s = re.findall(r"(?im)^Answer Span:\s*(.*?)\s*$", g.content)
        answer = found_a[-1].strip() if found_a else ""
        quote = found_q[-1].strip() if found_q else ""
        answer_span = found_s[-1].strip() if found_s else ""
        qbytes = quote.encode("utf-8")
        qstart = source_bytes.find(qbytes) if quote.lower() != "none" and qbytes else -1
        qptr = [qstart, len(qbytes)] if qstart >= 0 else None
        if answer.lower() in ("yes", "no", "unknown"):
            handle = {"kind": "enum", "value": answer.lower(), "evidence": qptr}
        elif qptr and answer_span and answer_span.lower() != "none" and answer_span.encode("utf-8") in qbytes:
            abytes = answer_span.encode("utf-8")
            start = qstart + qbytes.find(abytes)
            handle = {"kind": "span", "start": start, "length": len(abytes), "evidence": qptr}
        else:
            handle = {"kind": "inline", "value": answer, "evidence": qptr}
        if handle["kind"] == "span":
            decoded = source_bytes[handle["start"]:handle["start"]+handle["length"]].decode("utf-8")
        elif handle["kind"] == "enum":
            decoded = handle["value"]
        else:
            decoded = handle["value"]
        f1, em = score([answer] if answer else [], inst)
        decoded_f1, decoded_em = score([decoded] if decoded else [], inst)
        raw = g.raw
        payload = 1
        if handle["kind"] == "span":
            payload += vbytes(handle["start"]) + vbytes(handle["length"])
        elif handle["kind"] == "inline":
            payload += len(answer.encode("utf-8"))
        if qptr:
            payload += vbytes(qptr[0]) + vbytes(qptr[1])
        return {"answer": answer, "decoded_answer": decoded, "answer_span": answer_span, "quote": quote,
                "quote_valid": bool(qptr), "handle": handle, "handle_payload_bytes": payload,
                "f1": f1, "em": em, "decoded_f1": decoded_f1, "decoded_em": decoded_em,
                "prompt_tokens": raw.get("prompt_eval_count"),
                "generation_tokens": raw.get("eval_count"),
                "wall_sec": round(time.perf_counter()-t0, 3), "raw_content": g.content,
                "response_sha256": sha(json.dumps(raw, sort_keys=True).encode())}
    except Exception as exc:
        return {"answer": "", "decoded_answer": "", "answer_span": "", "quote": "", "quote_valid": False,
                "handle": {"kind": "inline", "value": "", "evidence": None},
                "handle_payload_bytes": 1, "f1": 0.0, "em": 0.0, "decoded_f1": 0.0, "decoded_em": 0.0,
                "wall_sec": round(time.perf_counter()-t0, 3), "error": repr(exc)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DEV.read_text(encoding="utf-8"))
    ids = sorted(data, key=lambda pid: (-len(data[pid]["qas"]), pid))[81:90]
    assert len(ids) == 9
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen["paper_ids"] == ids
    else:
        frozen = {"source_sha256": sha(DEV.read_bytes()), "paper_ids": ids,
                  "question_ids": {pid: [q["question_id"] for q in data[pid]["qas"]] for pid in ids}}
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_known_questions_no_gold_compilation", "frozen_sha256": sha(frozen_path.read_bytes()),
        "model": MODEL, "papers": []}
    assert result["frozen_sha256"] == sha(frozen_path.read_bytes())
    global_q = 0
    for pid in ids:
        paper = build_paper(pid, data[pid])
        blob = serialize_paper(data[pid])
        t0 = time.perf_counter()
        bm25 = BM25(paper["paragraphs"])
        index_sec = time.perf_counter()-t0
        rp = next((x for x in result["papers"] if x["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "source_sha256": sha(blob), "source_bytes": len(blob),
                  "index_sec": round(index_sec, 6), "records": []}
            result["papers"].append(rp)
        for original in data[pid]["qas"]:
            qid = original["question_id"]
            rr = next((x for x in rp["records"] if x["id"] == qid), None)
            if rr is None:
                t0 = time.perf_counter()
                query = "Paper: " + paper["title"] + ". Question: " + original["question"]
                selected = bm25.search(query, 16)
                rr = {"id": qid, "question": original["question"], "selected": selected,
                      "search_sec": round(time.perf_counter()-t0, 6),
                      "order": ["baseline", "compiler"] if global_q % 2 == 0 else ["compiler", "baseline"],
                      "arms": {}}
                rp["records"].append(rr)
            gold = gold_answers(original)
            inst = {"question": "Paper: " + paper["title"] + ". Question: " + original["question"],
                    "answer": gold[0], "answer_aliases": gold[1:]}
            material = "Retrieved evidence:\n" + "\n\n".join(paper["paragraphs"][i] for i in rr["selected"])
            for arm in rr["order"]:
                if arm in rr["arms"]:
                    continue
                rr["arms"][arm] = read(material, inst) if arm == "baseline" else compile_answer(material, inst["question"], blob, inst)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, qid, arm, round(rr["arms"][arm]["f1"], 3), flush=True)
            global_q += 1


if __name__ == "__main__":
    main()
