"""Post hoc BM25 top-2-only control for the frozen document pilot."""
from __future__ import annotations

import hashlib
import json

from run_realdoc_hybrid_pilot import OUT, ROOT, read


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    by_dataset = {}
    for name in manifest["dataset_sha256"]:
        p = ROOT / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest["dataset_sha256"][name]
        by_dataset[name] = {x["id"]: x for x in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if x}
    path = OUT / "results_top2_control.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []
    assert [x["id"] for x in records] == [x["id"] for x in manifest["selected"][:len(records)]]
    for i, item in enumerate(manifest["selected"][len(records):], start=len(records)):
        inst = by_dataset[item["dataset"]][item["id"]]
        assert old["records"][i]["id"] == item["id"]
        top2 = old["records"][i]["top2"]
        material = "Retrieved original passages:\n" + "\n\n".join(inst["chunks"][j] for j in top2)
        answer = read(material, inst)
        records.append({"id": item["id"], "dataset": item["dataset"], "top2": top2, "bm25_top2": answer})
        path.write_text(json.dumps({"manifest_sha256": old["manifest_sha256"],
                                    "model": old["model"], "think": False,
                                    "num_predict": 1024, "records": records},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{i+1}/16 {item['id']} top2={answer['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
