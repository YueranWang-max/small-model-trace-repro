"""Readout on the preselected official-test cohort, conditional on the frozen evidence gate."""
from __future__ import annotations
import json
from prepare_qasper_multiq import gold_answers
from run_qasper_prefix_vs_bm25 import ROOT, sha
from run_qasper_demand_transfer import OUT, SOURCE, ARMS
from run_qasper_visible_trace_replication import call, query_text

def main():
    gate = json.loads((OUT / "evidence_summary.json").read_text(encoding="utf-8"))
    assert gate["evidence_gate_pass"], "Frozen evidence gate failed; do not advance to readout."
    fp = OUT / "frozen.json"
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    assert gate["frozen_sha256"] == sha(fp.read_bytes())
    assert frozen["dataset_sha256"] == sha(SOURCE.read_bytes())
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    path = OUT / "reader_results.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status": "frozen_global_prior_official_test_reader", "frozen_sha256": sha(fp.read_bytes()), "papers": []}
    assert data["frozen_sha256"] == sha(fp.read_bytes())
    for pi, pid in enumerate(frozen["reader_paper_ids"]):
        p = next(p for p in frozen["papers"] if p["id"] == pid)
        rp = next((r for r in data["papers"] if r["paper_id"] == pid), None)
        if rp is None:
            rp = {"paper_id": pid, "arm_order": list(ARMS if pi % 2 == 0 else ARMS[::-1]), "records": []}
            data["papers"].append(rp)
        for arm in rp["arm_order"]:
            for q in p["questions"]:
                rr = next((r for r in rp["records"] if r["id"] == q["id"]), None)
                if rr is None:
                    rr = {"id": q["id"], "arms": {}}
                    rp["records"].append(rr)
                if arm in rr["arms"]:
                    continue
                original = next(x for x in source[pid]["qas"] if x["question_id"] == q["id"])
                gold = gold_answers(original)
                query = query_text(source[pid]["title"], q["text"])
                selected = frozen["choices"][pid][q["id"]][arm]
                material = "\n\n".join(f"[P{i}] Section: {p['paragraphs'][i]['heading']}\n" + p["paragraphs"][i]["text"] for i in selected)
                prompt = ("Answer using only the evidence below. If insufficient, say unknown.\nEvidence:\n" + material +
                          "\n\nQuestion: " + query + "\nEnd with exactly one line: Final Answer: <short answer>")
                inst = {"question": query, "answer": gold[0], "answer_aliases": gold[1:]}
                rr["arms"][arm] = call(prompt, inst, think=False, cap=1024)
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(pi+1, pid, q["id"], arm, round(rr["arms"][arm]["f1"], 3), flush=True)

if __name__ == "__main__":
    main()
