"""Frozen same-evidence examination-workflow reader comparison."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from prepare_qasper_multiq import gold_answers
from repro_lib import ollama_chat
from run_realdoc import score
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs/qasper_exam_reader"
SOURCE = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
MODEL = "qwen3.5:4b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    papers = json.loads(SOURCE.read_text(encoding="utf-8"))
    sorted_ids = sorted(papers, key=lambda pid: (-len(papers[pid]["qas"]), pid))
    ids = sorted_ids[18:27]
    assert not (set(ids) & set(sorted_ids[:18]))
    groups = []
    for gi in range(3):
        group_ids = ids[gi*3:gi*3+3]
        paragraphs, questions = [], []
        for pid in group_ids:
            paper = papers[pid]
            prefix = f"Paper {pid}: {paper['title']}"
            paragraphs.append(prefix + "\nAbstract: " + paper["abstract"])
            for section in paper["full_text"]:
                paragraphs.extend(prefix + "\nSection: " + (section["section_name"] or "") + "\n" + p.strip()
                                  for p in section["paragraphs"] if p.strip())
            for q in paper["qas"]:
                questions.append({"id": q["question_id"], "paper_id": pid, "paper_title": paper["title"],
                                  "question": q["question"], "gold_answers": gold_answers(q)})
        random.Random(20261002 + gi).shuffle(questions)
        groups.append({"group": gi+1, "paper_ids": group_ids, "paragraphs": paragraphs,
                       "corpus_sha256": hashlib.sha256("\n\n".join(paragraphs).encode()).hexdigest(),
                       "questions": questions})
    return {"source_sha256": sha(SOURCE), "selection_rule": "dev paper ranks 19-27, descending question count then ID",
            "groups": groups}


def call(prompt, inst):
    start = time.perf_counter()
    try:
        g = ollama_chat(MODEL, "You answer academic reading questions from supplied evidence.", prompt,
                        num_ctx=32768, think=False, temperature=0.0, num_predict=1024)
        answer_match = re.findall(r"(?im)^Final Answer:\s*(.*?)\s*$", g.content)
        answer = answer_match[-1].strip() if answer_match else ""
        evidence_match = re.findall(r"(?im)^Evidence:\s*(.*?)\s*$", g.content)
        f1, em = score([answer] if answer else [], inst)
        return {"answer": answer, "f1": f1, "em": em, "malformed": not bool(answer),
                "evidence_line": evidence_match[-1].strip() if evidence_match else "",
                "prompt_tokens": g.raw.get("prompt_eval_count"), "generation_tokens": g.raw.get("eval_count"),
                "wall_sec": round(time.perf_counter()-start, 3), "raw_content": g.content,
                "response_sha256": hashlib.sha256(json.dumps(g.raw, sort_keys=True).encode()).hexdigest()}
    except Exception as exc:
        return {"answer": "", "f1": 0.0, "em": 0.0, "malformed": True, "evidence_line": "",
                "wall_sec": round(time.perf_counter()-start, 3), "error": repr(exc)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else prepare()
    if not frozen_path.exists():
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": sha(frozen_path), "model": MODEL, "think": False, "num_predict": 1024, "groups": []}
    assert result["frozen_sha256"] == sha(frozen_path)
    global_index = 0
    for group in frozen["groups"]:
        t0 = time.perf_counter()
        bm25 = BM25(group["paragraphs"])
        build_sec = time.perf_counter()-t0
        rg = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if rg is None:
            rg = {"group": group["group"], "corpus_sha256": group["corpus_sha256"],
                  "bm25_build_sec": round(build_sec, 6), "records": []}
            result["groups"].append(rg)
        assert rg["corpus_sha256"] == group["corpus_sha256"]
        for qi, q in enumerate(group["questions"]):
            query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
            rr = next((r for r in rg["records"] if r["id"] == q["id"]), None)
            if rr is None:
                t0 = time.perf_counter()
                selected = bm25.search(query, 16)
                search_sec = time.perf_counter()-t0
                rr = {"id": q["id"], "paper_id": q["paper_id"], "selected": selected,
                      "search_sec": round(search_sec, 6), "arms": {}}
                rg["records"].append(rr)
            material = "\n\n".join(f"[P{j+1}] {group['paragraphs'][pid]}" for j, pid in enumerate(rr["selected"]))
            common = "Numbered original source paragraphs:\n" + material + "\n\nQuestion: " + query
            prompts = {
                "baseline": "Answer the question using only the evidence below. If it is insufficient, say unknown.\n" + common +
                            "\nEnd with exactly one line: Final Answer: <short answer>",
                "exam": "Answer the academic reading question using only the numbered source paragraphs.\n" + common +
                        "\nFollow these checks briefly: (1) identify exactly what the question asks, including negation, comparison, units and scope; "
                        "(2) find paragraphs that directly support that answer; (3) reject any tentative answer that is contradicted, too strong, or merely related but does not answer the question. "
                        "If direct support is insufficient, answer unknown. End with exactly two lines:\nEvidence: <supporting paragraph IDs, or none>\nFinal Answer: <short answer>"}
            order = ("baseline", "exam") if global_index % 2 == 0 else ("exam", "baseline")
            rr["order"] = list(order)
            inst = {"question": query, "answer": q["gold_answers"][0], "answer_aliases": q["gold_answers"][1:]}
            for arm in order:
                if arm in rr["arms"]:
                    continue
                rr["arms"][arm] = call(prompts[arm], inst)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"group{group['group']} q{qi+1}/{len(group['questions'])} {arm} f1={rr['arms'][arm]['f1']:.3f}", flush=True)
            global_index += 1


if __name__ == "__main__":
    main()
