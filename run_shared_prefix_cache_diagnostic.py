"""Measure local Ollama prefill counters with shared versus shifted prefixes."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import argparse
from pathlib import Path

from analyze_qasper_pointer_oracle import serialize_paper
from repro_lib import ollama_chat

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/qasper_official/qasper-dev-v0.3.json"
OUT = ROOT / "logs/shared_prefix_cache"
SYSTEM = "Answer the question from the supplied paper text with one short sentence."
MODEL = "qwen3.5:4b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",type=int,default=91,help="one-based ranked dev paper")
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--reverse",action="store_true",help="run question-first pair before document-first pairs")
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(DEV.read_text(encoding="utf-8"))
    pid = sorted(data,key=lambda p:(-len(data[p]["qas"]),p))[args.rank-1]
    paper = data[pid]
    assert len(paper["qas"]) >= 3
    source = serialize_paper(paper).decode("utf-8")[:12000]
    assert len(source) == 12000
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    version = json.loads(opener.open("http://127.0.0.1:11434/api/version",timeout=5).read().decode())
    q = [x["question"] for x in paper["qas"][:2]]
    salt = "PREFIX-CACHE-DIAGNOSTIC-B-20261003\n"
    calls = [("A1","Paper:\n"+source+"\nQuestion: "+q[0]),
             ("A2","Paper:\n"+source+"\nQuestion: "+q[1]),
             ("B1",salt+"Paper:\n"+source+"\nQuestion: "+q[0]),
             ("B2",salt+"Paper:\n"+source+"\nQuestion: "+q[1]),
             ("C1","Question: "+q[0]+"\nPaper:\n"+source),
             ("C2","Question: "+q[1]+"\nPaper:\n"+source)]
    if args.reverse:
        calls = calls[4:] + calls[:4]
    path = out_dir / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "status":"local_ollama_prefix_cache_diagnostic", "ollama_version":version,
        "model":MODEL, "paper_id":pid, "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "source_chars":len(source), "calls":[]}
    assert result["paper_id"] == pid
    for label,prompt in calls:
        if any(x["label"]==label for x in result["calls"]):
            continue
        t0 = time.perf_counter()
        g = ollama_chat(MODEL,SYSTEM,prompt,num_ctx=32768,think=False,temperature=0.0,num_predict=32)
        raw = g.raw
        row = {"label":label,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
               "prompt_chars":len(prompt),"prompt_eval_count":raw.get("prompt_eval_count"),
               "eval_count":raw.get("eval_count"),
               "prompt_eval_duration_ns":raw.get("prompt_eval_duration"),
               "eval_duration_ns":raw.get("eval_duration"),
               "total_duration_ns":raw.get("total_duration"),
               "load_duration_ns":raw.get("load_duration"),
               "wall_sec":round(time.perf_counter()-t0,6),
               "response_sha256":hashlib.sha256(g.content.encode()).hexdigest()}
        result["calls"].append(row)
        path.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(label, "prompt_eval_count",row["prompt_eval_count"],
              "prefill_sec", None if row["prompt_eval_duration_ns"] is None else round(row["prompt_eval_duration_ns"]/1e9,3),
              "wall_sec",row["wall_sec"],flush=True)


if __name__ == "__main__":
    main()
