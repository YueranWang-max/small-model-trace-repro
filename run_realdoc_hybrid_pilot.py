"""Frozen matched natural-document readout pilot; see protocol_realdoc_hybrid_pilot.md."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from repro_lib import ollama_chat
from run_realdoc import build_e, score
from run_stageC import BM25

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "realdoc_hybrid_pilot"
MODEL = "qwen3.5:4b"
ARMS = ("state", "state_source2", "bm25_top6")


def read(material: str, inst: dict) -> dict:
    prompt = ("Answer the question using only the evidence below. If it is insufficient, say unknown.\n"
              + material + "\n\nQuestion: " + inst["question"]
              + "\nEnd with exactly one line: Final Answer: <short answer>")
    started = time.perf_counter()
    try:
        g = ollama_chat(MODEL, "You answer multi-hop questions from supplied evidence.", prompt,
                        num_ctx=32768, think=False, temperature=0.0, num_predict=1024)
        wall_sec = time.perf_counter() - started
        found = re.findall(r"(?im)^Final Answer:\s*(.*?)\s*$", g.content)
        answer = found[-1].strip() if found else ""
        bad = not bool(answer)
        f1, em = score([answer] if answer else [], inst)
        raw = g.raw
        return {"answer": answer, "malformed": bad, "f1": f1, "em": em,
                "wall_sec": round(wall_sec, 3), "prompt_chars": len(prompt),
                "prompt_tokens": raw.get("prompt_eval_count"),
                "generation_tokens": raw.get("eval_count"),
                "raw_content": g.content,
                "response_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()}
    except Exception as exc:
        return {"answer": "", "malformed": True, "f1": 0.0, "em": 0.0,
                "wall_sec": round(time.perf_counter()-started, 3),
                "prompt_chars": len(prompt), "error": repr(exc)}


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    cache_path = ROOT / "logs" / "stageC_distill_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    by_dataset = {}
    for dataset in manifest["dataset_sha256"]:
        path = ROOT / dataset
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"][dataset]
        by_dataset[dataset] = {x["id"]: x for x in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if x}
    output = OUT / "results.json"
    records = json.loads(output.read_text(encoding="utf-8"))["records"] if output.exists() else []
    assert [r["id"] for r in records] == [x["id"] for x in manifest["selected"][:len(records)]]
    for index, item in enumerate(manifest["selected"][len(records):], start=len(records)):
        inst = by_dataset[item["dataset"]][item["id"]]
        cached = cache[item["id"] + "::qwen3:1.7b"]
        fbc = {int(k): v for k, v in cached["fbc"].items()}
        state = build_e(fbc)
        t0 = time.perf_counter()
        bm = BM25(inst["chunks"])
        build_sec = time.perf_counter()-t0
        t0 = time.perf_counter()
        top6 = bm.search(inst["question"], 6)
        search_sec = time.perf_counter()-t0
        top2 = top6[:2]
        materials = {
            "state": "Verified source quotes:\n" + state,
            "state_source2": "Verified source quotes:\n" + state + "\n\nRetrieved original passages:\n" + "\n\n".join(inst["chunks"][i] for i in top2),
            "bm25_top6": "Retrieved original passages:\n" + "\n\n".join(inst["chunks"][i] for i in top6),
        }
        order = ARMS[index % 3:] + ARMS[:index % 3]
        arms = {arm: read(materials[arm], inst) for arm in order}
        gold = set(inst["support_chunk_ids"])
        records.append({"id": item["id"], "dataset": item["dataset"],
                        "hops": inst["hops"], "order": order,
                        "state_chars": len(state), "state_extraction_calls_archived": cached["nc"],
                        "bm25_index_sec": round(build_sec, 6), "bm25_search_sec": round(search_sec, 6),
                        "top2": top2, "top6": top6,
                        "top2_support_recall": len(gold & set(top2))/len(gold),
                        "top6_support_recall": len(gold & set(top6))/len(gold),
                        "arms": arms})
        output.write_text(json.dumps({"manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
                                      "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                                      "model": MODEL, "think": False, "num_predict": 1024,
                                      "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{index+1}/16 {item['id']} " + " ".join(f"{arm}={arms[arm]['f1']:.3f}" for arm in ARMS), flush=True)


if __name__ == "__main__":
    main()
