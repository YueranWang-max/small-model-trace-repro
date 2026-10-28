"""Post hoc CPU paragraph-BM25 control on the frozen sixteen corpora."""
from __future__ import annotations

import hashlib
import json
import time

from run_realdoc_hybrid_pilot import OUT, ROOT, read
from run_stageC import BM25


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    by_dataset = {}
    for name in manifest["dataset_sha256"]:
        p = ROOT / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest["dataset_sha256"][name]
        by_dataset[name] = {x["id"]: x for x in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if x}
    path = OUT / "results_paragraph12.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []
    assert [x["id"] for x in records] == [x["id"] for x in manifest["selected"][:len(records)]]
    for i, item in enumerate(manifest["selected"][len(records):], start=len(records)):
        inst = by_dataset[item["dataset"]][item["id"]]
        units, owners = [], []
        for ci, chunk in enumerate(inst["chunks"]):
            for paragraph in chunk.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    units.append(paragraph)
                    owners.append(ci)
        started = time.perf_counter()
        bm = BM25(units)
        index_sec = time.perf_counter() - started
        started = time.perf_counter()
        selected = bm.search(inst["question"], min(12, len(units)))
        search_sec = time.perf_counter() - started
        material = "Retrieved original paragraphs:\n" + "\n\n".join(units[j] for j in selected)
        answer = read(material, inst)
        gold = set(inst["support_chunk_ids"])
        records.append({"id": item["id"], "dataset": item["dataset"],
                        "n_paragraphs": len(units), "selected_paragraphs": selected,
                        "selected_chunk_ids": [owners[j] for j in selected],
                        "support_chunk_recall": len(gold & {owners[j] for j in selected}) / len(gold),
                        "index_sec": round(index_sec, 6), "search_sec": round(search_sec, 6),
                        "paragraph12": answer})
        path.write_text(json.dumps({"manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
                                    "model": "qwen3.5:4b", "think": False, "num_predict": 1024,
                                    "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{i+1}/16 {item['id']} paragraph12={answer['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
