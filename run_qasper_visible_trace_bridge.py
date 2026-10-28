"""Exploratory bounded visible-trace prepend/append bridge."""
from __future__ import annotations

import json
import re
import time

from prepare_qasper_multiq import gold_answers
from repro_lib import ollama_chat
from run_qasper_outline_route import build_paper
from run_qasper_prefix_vs_bm25 import DEV, MODEL, ROOT, sha
from run_realdoc import score

OUT = ROOT / "logs/qasper_visible_trace_bridge"
SYSTEM = "Answer research-paper questions using only the supplied paper. Give a short final answer."
ARMS = ("reread", "trace_pre", "trace_post", "quote_pre")


def query_text(title, question):
    return "Paper: " + title + ". Question: " + question


def call(prompt, inst, *, think, cap):
    t0 = time.perf_counter()
    try:
        g = ollama_chat(MODEL, SYSTEM, prompt, num_ctx=32768, think=think,
                        temperature=0.0, num_predict=cap)
        lines = re.findall(r"(?im)^Final Answer:\s*(.*?)\s*$", g.content)
        answer = lines[-1].strip() if lines else ""
        f1, em = score([answer] if answer else [], inst)
        raw = g.raw
        return {"answer": answer, "f1": f1, "em": em,
                "thinking": g.thinking, "content": g.content,
                "prompt_sha256": sha(prompt.encode("utf-8")),
                "prompt_chars": len(prompt), "prompt_tokens": raw.get("prompt_eval_count"),
                "generation_tokens": raw.get("eval_count"),
                "prefill_ns": raw.get("prompt_eval_duration"),
                "eval_ns": raw.get("eval_duration"),
                "api_total_ns": raw.get("total_duration"),
                "load_ns": raw.get("load_duration"),
                "wall_sec": round(time.perf_counter() - t0, 3),
                "response_sha256": sha(json.dumps(raw, sort_keys=True).encode("utf-8"))}
    except Exception as exc:
        return {"answer": "", "f1": 0.0, "em": 0.0,
                "error": repr(exc), "wall_sec": round(time.perf_counter() - t0, 3)}


def freeze(source):
    ranked = sorted(source, key=lambda pid: (-len(source[pid]["qas"]), pid))
    ids = ranked[134:140]
    assert len(ids) == 6 and sum(len(source[pid]["qas"]) for pid in ids) == 20
    papers = []
    for pid in ids:
        paper = build_paper(pid, source[pid])
        text = "\n\n".join(paper["paragraphs"])
        papers.append({"paper_id": pid, "title": paper["title"],
                       "source_sha256": sha(text.encode("utf-8")),
                       "source_chars": len(text),
                       "question_ids": [q["question_id"] for q in source[pid]["qas"]]})
    return {"dataset_sha256": sha(DEV.read_bytes()), "selection": "Qasper dev ranks 135-140, all questions",
            "model": MODEL, "arms": list(ARMS), "papers": papers}


def extract_quote(content, paragraphs):
    candidates = re.findall(r"(?im)^Evidence Quote:\s*(.*?)\s*$", content)
    if not candidates:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) == 3:
            candidates = [lines[1]]
    for candidate in candidates:
        candidate = candidate.strip().strip('"“”')
        if len(candidate) >= 20:
            for i, paragraph in enumerate(paragraphs):
                if candidate in paragraph:
                    return candidate, i
    return "", None


def extract_trace(content):
    lines = re.findall(r"(?im)^Trace:\s*(.*?)\s*$", content)
    if not lines:
        positional = [line.strip() for line in content.splitlines() if line.strip()]
        if len(positional) == 3:
            lines = [positional[0]]
    value = lines[-1].strip() if lines else ""
    return value[:1200], len(value) > 1200


def second_prompt(arm, text, query, trace, quote):
    warning = "First-pass notes may be wrong. Verify against the paper.\n"
    block = "<first_run_reasoning_trace>\n" + (trace or "[missing trace]") + "\n</first_run_reasoning_trace>\n"
    quote_block = "<source_verified_quote>\n" + (quote or "[no verified quote]") + "\n</source_verified_quote>\n"
    source = "Original paper:\n" + text + "\n"
    if arm == "trace_pre":
        body = warning + block + source
    elif arm == "trace_post":
        body = source + warning + block
    elif arm == "quote_pre":
        body = quote_block + source
    else:
        body = source
    return body + "\nQuestion: " + query + "\nEnd with exactly one line: Final Answer: <short answer>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    frozen_path = OUT / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen == freeze(source)
    else:
        frozen = freeze(source)
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    path = OUT / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_visible_trace_placement_bridge",
        "frozen_sha256": sha(frozen_path.read_bytes()), "papers": []}
    assert results["frozen_sha256"] == sha(frozen_path.read_bytes())
    for pi, fp in enumerate(frozen["papers"]):
        pid = fp["paper_id"]
        paper = build_paper(pid, source[pid])
        paragraphs = paper["paragraphs"]
        text = "\n\n".join(paragraphs)
        assert sha(text.encode("utf-8")) == fp["source_sha256"]
        rp = next((p for p in results["papers"] if p["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "arm_order": list(ARMS[pi:] + ARMS[:pi]), "records": []}
            results["papers"].append(rp)
        # Every first-pass output is fixed before any second-pass call on this paper.
        for original in source[pid]["qas"]:
            qid = original["question_id"]
            rr = next((r for r in rp["records"] if r["id"] == qid), None)
            if rr is None:
                rr = {"id": qid, "arms": {}}
                rp["records"].append(rr)
            if "first" in rr:
                if not rr["feedback"]["trace"]:
                    first = rr["first"]
                    trace, trace_truncated = extract_trace(first.get("content", ""))
                    quote, quote_pid = extract_quote(first.get("content", ""), paragraphs)
                    rr["feedback"] = {"trace": trace, "trace_truncated": trace_truncated,
                                      "verified_quote": quote, "quote_paragraph_id": quote_pid}
                    lines = [line.strip() for line in first.get("content", "").splitlines() if line.strip()]
                    if len(lines) == 3 and not first.get("answer"):
                        answer = lines[-1]
                        gold = gold_answers(original)
                        inst = {"question": query_text(paper["title"], original["question"]),
                                "answer": gold[0], "answer_aliases": gold[1:]}
                        first["answer"] = answer
                        first["f1"], first["em"] = score([answer], inst)
                    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                continue
            query = query_text(paper["title"], original["question"])
            gold = gold_answers(original)
            inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
            prompt = ("Original paper:\n" + text + "\nQuestion: " + query +
                      "\nRespond with exactly three short lines. Trace: <what fact or relation is needed and where to look>; "
                      "Evidence Quote: <one exact original-source sentence>; Final Answer: <short answer>. "
                      "Keep the trace under 80 words. Do not add other lines.")
            first = call(prompt, inst, think=False, cap=1024)
            lines = [line.strip() for line in first.get("content", "").splitlines() if line.strip()]
            if len(lines) == 3 and not first.get("answer"):
                first["answer"] = lines[-1]
                first["f1"], first["em"] = score([lines[-1]], inst)
            quote, quote_pid = extract_quote(first.get("content", ""), paragraphs)
            trace, trace_truncated = extract_trace(first.get("content", ""))
            rr["first"] = first
            rr["feedback"] = {"trace": trace,
                              "trace_truncated": trace_truncated,
                              "verified_quote": quote, "quote_paragraph_id": quote_pid}
            path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(pid, qid, "first", round(first["f1"], 3), "trace", len(rr["feedback"]["trace"]),
                  "quote", bool(quote), flush=True)
        for arm in rp["arm_order"]:
            for original in source[pid]["qas"]:
                rr = next(r for r in rp["records"] if r["id"] == original["question_id"])
                if arm in rr["arms"]:
                    continue
                query = query_text(paper["title"], original["question"])
                gold = gold_answers(original)
                inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
                feedback = rr["feedback"]
                prompt = second_prompt(arm, text, query, feedback["trace"], feedback["verified_quote"])
                rr["arms"][arm] = call(prompt, inst, think=False, cap=1024)
                path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pid, rr["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)


if __name__ == "__main__":
    main()
