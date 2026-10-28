"""Freeze three multi-question corpora from the official Qasper v0.3 dev set."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "qasper_official" / "qasper-dev-v0.3.json"
OUT = ROOT / "logs" / "qasper_multiq"
SEED = 20261001


def gold_answers(q):
    answers = []
    for annotation in q["answers"]:
        a = annotation["answer"]
        if a["unanswerable"]:
            answers.append("unknown")
        elif a["yes_no"] is not None:
            answers.append("yes" if a["yes_no"] else "no")
        elif a["extractive_spans"]:
            answers.append(", ".join(a["extractive_spans"]))
        elif a["free_form_answer"].strip():
            answers.append(a["free_form_answer"].strip())
    return list(dict.fromkeys(answers))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    papers = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = sorted(papers, key=lambda pid: (-len(papers[pid]["qas"]), pid))[:9]
    groups = []
    for gi in range(3):
        ids = selected[gi*3:gi*3+3]
        units = []
        questions = []
        for pid in ids:
            paper = papers[pid]
            prefix = f"Paper {pid}: {paper['title']}"
            units.append(prefix + "\nAbstract: " + paper["abstract"])
            for section in paper["full_text"]:
                for paragraph in section["paragraphs"]:
                    paragraph = paragraph.strip()
                    if paragraph:
                        units.append(prefix + "\nSection: " + section["section_name"] + "\n" + paragraph)
            for q in paper["qas"]:
                questions.append({"id": q["question_id"], "paper_id": pid,
                                  "paper_title": paper["title"],
                                  "question": q["question"],
                                  "gold_answers": gold_answers(q)})
        rng = random.Random(SEED + gi)
        rng.shuffle(questions)
        corpus = "\n\n".join(units)
        groups.append({"group": gi+1, "paper_ids": ids, "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
                       "corpus_chars": len(corpus), "n_paragraphs": len(units),
                       "n_questions": len(questions), "paragraphs": units, "questions": questions})
    obj={"source": "Qasper v0.3 official development split",
         "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         "selection_rule": "top nine papers by number of questions, tie paper ID ascending; consecutive groups of three; all original questions; shuffle within group",
         "seed": SEED, "groups": groups}
    target=OUT / "frozen.json"
    target.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    print([(g["group"],g["corpus_chars"],g["n_questions"]) for g in groups])


if __name__ == "__main__":
    main()
