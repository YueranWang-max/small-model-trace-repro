"""Fresh-paper readout of frozen question-directed structural navigation."""
from __future__ import annotations
import json
from prepare_qasper_multiq import gold_answers
from run_qasper_prefix_vs_bm25 import DEV, ROOT, sha
from run_qasper_genre_structure import freeze
from run_qasper_visible_trace_replication import call, query_text

OUT = ROOT / "logs/qasper_genre_structure_readout"
ARMS = ("flat8", "structure8")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(DEV.read_text(encoding="utf-8"))
    fp = OUT / "frozen.json"
    if fp.exists():
        plan = json.loads(fp.read_text(encoding="utf-8"))
        assert plan["dataset_sha256"] == sha(DEV.read_bytes())
        assert plan["selection"] == "dev ranks247-276 inclusive"
    else:
        plan = freeze(source, 247, 30)
        fp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    path = OUT / "results.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "exploratory_dev_question_directed_structure_readout",
        "frozen_sha256": sha(fp.read_bytes()), "papers": []}
    assert data["frozen_sha256"] == sha(fp.read_bytes())
    for pi, p in enumerate(plan["papers"]):
        rp = next((r for r in data["papers"] if r["paper_id"] == p["paper_id"]), None)
        if rp is None:
            rp = {"paper_id": p["paper_id"], "arm_order": list(ARMS if pi % 2 == 0 else ARMS[::-1]), "records": []}
            data["papers"].append(rp)
        for arm in rp["arm_order"]:
            for q in p["questions"]:
                rr = next((r for r in rp["records"] if r["id"] == q["id"]), None)
                if rr is None:
                    rr = {"id": q["id"], "arms": {}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                original = next(x for x in source[p["paper_id"]]["qas"] if x["question_id"] == q["id"])
                gold = gold_answers(original)
                query = query_text(source[p["paper_id"]]["title"], q["question"])
                material = "\n\n".join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n" + p["paragraphs"][i]["text"] for i in q[arm])
                prompt = ("Answer using only the evidence below. If insufficient, say unknown.\nEvidence:\n" + material +
                          "\n\nQuestion: " + query + "\nEnd with exactly one line: Final Answer: <short answer>")
                inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = call(prompt, inst, think=False, cap=1024)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(p["paper_id"], q["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)

if __name__ == "__main__":
    main()
