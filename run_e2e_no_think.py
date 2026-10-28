"""Post hoc matched decoder pilot; see protocol_e2e_no_think.md."""
from __future__ import annotations

import hashlib
import json
import time
import argparse

from repro_lib import extract_answer, f1_score
from run_distill3 import chunk_edges
from run_e2e_cost import MODEL, OUT, call, evidence, graph


def read(material, q, model, budget):
    prompt = ("Use only the provided directed edges to answer the operation.\n"
              + material + "\n\nOperation: " + q["question"]
              + "\nReturn exactly one line: Final Answer: [node1, node2]. Use [] if empty.")
    started = time.perf_counter()
    try:
        g, meta = call(model, "You solve directed-graph algorithm problems.", prompt,
                       think=False, temperature=0.0, num_predict=budget)
        answer, malformed = extract_answer(g.content, "")
        meta.update({"answer": answer, "malformed": malformed,
                     "f1": f1_score(answer, q["answer"]),
                     "em": int(set(answer) == set(q["answer"])),
                     "raw_content": g.content,
                     "response_sha256": hashlib.sha256(json.dumps(g.raw, sort_keys=True).encode()).hexdigest()})
        return meta
    except Exception as exc:
        return {"answer": [], "malformed": True, "f1": 0.0, "em": 0,
                "raw_content": "", "wall_sec": time.perf_counter()-started,
                "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--budget", type=int, default=2048)
    args = parser.parse_args()
    frozen = json.loads((OUT / "questions.json").read_text(encoding="utf-8"))
    primary = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    sample, _, digest = graph()
    assert frozen["corpus_sha256"] == digest == primary["corpus_sha256"]
    chunks = chunk_edges(sample, 4000)
    db = set(map(tuple, json.loads((OUT / "ingest_db.json").read_text(encoding="utf-8"))["db"]))
    suffix = "" if args.model == MODEL else "_" + args.model.replace("/", "_").replace(":", "_")
    budget_suffix = "" if args.budget == 2048 else f"_budget{args.budget}"
    path = OUT / ("results_no_think" + suffix + budget_suffix + ".json")
    records = json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []
    assert [x["id"] for x in records] == [x["id"] for x in frozen["questions"][:len(records)]]
    for i, (q, old) in enumerate(zip(frozen["questions"], primary["records"])):
        db.update(map(tuple, old["new_verified_edges"]))
        if i < len(records):
            continue
        state_material = evidence(db, q)
        selected = old["bm25_top4"]["selected_chunks"]
        bm_material = "Retrieved source chunks:\n" + "\n\n".join(chunks[j] for j in selected)
        if q["id"] % 2:
            state, bm = read(state_material, q, args.model, args.budget), read(bm_material, q, args.model, args.budget)
        else:
            bm, state = read(bm_material, q, args.model, args.budget), read(state_material, q, args.model, args.budget)
        records.append({"id": q["id"], "state": state, "bm25_top4": bm})
        path.write_text(json.dumps({"corpus_sha256": digest,
                                    "questions_sha256": primary["questions_sha256"],
                                    "model": args.model, "think": False, "num_predict": args.budget,
                                    "records": records}, indent=2), encoding="utf-8")
        print(f"q{q['id']}/12 state={state['f1']:.3f} bm25={bm['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
