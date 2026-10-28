"""Read all frozen Qasper holdout arms with identical Qwen settings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_realdoc_hybrid_pilot import read

OUT = Path(__file__).resolve().parent / "logs/qasper_holdout"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen_path, selection_path = OUT / "frozen.json", OUT / "selections.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["frozen_sha256"] == sha(frozen_path)
    path = OUT / "readout.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "frozen_sha256": sha(frozen_path), "selection_sha256": sha(selection_path),
        "model": "qwen3.5:4b", "think": False, "num_predict": 1024, "groups": []}
    assert result["selection_sha256"] == sha(selection_path)
    for group, sg in zip(frozen["groups"], selection["groups"]):
        assert group["group"] == sg["group"]
        outg = next((g for g in result["groups"] if g["group"] == group["group"]), None)
        if outg is None:
            outg = {"group": group["group"], "records": []}
            result["groups"].append(outg)
        for qi, (q, sr) in enumerate(zip(group["questions"], sg["records"]), start=1):
            assert q["id"] == sr["id"]
            rr = next((r for r in outg["records"] if r["id"] == q["id"]), None)
            if rr is None:
                rr = {"id": q["id"], "paper_id": q["paper_id"], "arms": {}}
                outg["records"].append(rr)
            for arm in ("bm25_16", "fused_16", "fused_budget"):
                if arm in rr["arms"]:
                    continue
                query = "Paper: " + q["paper_title"] + ". Question: " + q["question"]
                material = "Retrieved original source paragraphs:\n" + "\n\n".join(group["paragraphs"][i] for i in sr[arm])
                inst = {"question": query, "answer": q["gold_answers"][0], "answer_aliases": q["gold_answers"][1:]}
                rr["arms"][arm] = read(material, inst)
                path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"group{group['group']} q{qi}/{len(group['questions'])} {arm} f1={rr['arms'][arm]['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
